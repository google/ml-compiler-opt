# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract IR for training.

Extract IR for training, either from a compile_commands.json file produced by
cmake, or a linker parameter list file.

Only run with
'python compiler_opt/tools/extract_ir.py ...'

The compilation is assumed to have been performed with clang, using
-fembed-bitcode=all passed to cc1 (i.e. pass clang -Xclang=-fembed-bitcode=all)

In a distributed ThinLTO case, the compilation is assumed to have been performed
specifying -mllvm -lto-embed-bitcode=post-merge-pre-opt.

In a local ThinLTO case, the compilation is assumedto have been performed
specifying -Wl,--save-temps=import -Wl,--thinlto-emit-index-files
"""

import json
import multiprocessing
import os
import pathlib
import re
import shutil
import subprocess

from typing import Dict, List, Optional

from absl import app
from absl import flags
from absl import logging

from compiler_opt.rl import constant

flags.DEFINE_string(
    'input', None,
    'Input file - either compile_commands.json or a linker parameter list')
flags.DEFINE_enum(
    'input_type', 'json', ['json', 'params'],
    'Input file type - json or params. The latter refers to lld params.')
flags.DEFINE_string('output_dir', None, 'Output directory')
flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel workers for objcopy. `None` for maximum available.')
flags.DEFINE_string('llvm_objcopy_path', 'llvm-objcopy', 'Path to llvm-objcopy')
flags.DEFINE_string(
    'obj_base_dir', '',
    'Base directory for object files. Defaults to current working dir.')
flags.DEFINE_string(
    'cmd_filter', None,
    'Include only those modules with a command line matching this regexp. '
    'Setting it to None for not filtering. Note that the regexp is applied '
    'independently for each separate command line option. For example, ^-Oz$ '
    'will match Oz - built binaries. Does not work with thinlto_build=lld.')
flags.DEFINE_enum(
    'thinlto_build', None, ['distributed', 'local'],
    'Set if the build was performed with either \'distributed\' or '
    '\'local\' ThinLTO. This ensures the thinlto.bc files are also copied. '
    'The build is assumed to have had '
    '-mllvm -lto-embed-bitcode=post-merge-pre-opt passed in the distributed '
    'case, or -Wl,--save-temps=import and -Wl,--thinlto-emit-index-files '
    'passed in the local case.')

FLAGS = flags.FLAGS


# TODO(ml-compiler-opt): maybe we can also convert here the cmdline file,from a
# \0 - separated list of strings, to a \n one.
def should_include_module(cmdline: str, match_regexp: Optional[str]) -> bool:
  """Determine if the module should be included."""
  if match_regexp is None:
    return True
  lines = cmdline.split('\0')
  return any(len(re.findall(match_regexp, l)) for l in lines)


def get_thinlto_index(cmdline: str, basedir: str) -> Optional[str]:
  opts = cmdline.split('\0')
  for option in opts:
    if option.startswith('-fthinlto-index'):
      return os.path.join(basedir, option.split('=')[1])
  return None


class TrainingIRExtractor:
  """IR and command line extraction from an object file.

  The object file is assumed to have the .llvmbc and .llvmcmd sections.
  """

  def __init__(self, obj_relative_path, output_base_dir, obj_base_dir=None):
    """Set up a TrainingIRExtractor.

    Args:
      obj_relative_path: relative path to the input object file. It will be also
        used to construct the absolute path of the output IR and cmd files, by
        appending it to output_base_dir.
      output_base_dir: the directory under which the output will be produced.
      obj_base_dir: the base directory for all the input object files.
    """
    self._obj_relative_path = obj_relative_path
    self._output_base_dir = output_base_dir
    self._obj_base_dir = obj_base_dir if obj_base_dir is not None else ''

  def obj_base_dir(self):
    return self._obj_base_dir

  def output_base_dir(self):
    return self._output_base_dir

  def relative_output_path(self):
    return self._obj_relative_path

  def input_obj(self):
    return os.path.join(self.obj_base_dir(), self._obj_relative_path)

  def lld_src_bc(self):
    # .3.import.bc is the suffix attached to post-merge-pre-opt ('postimport')
    # IR bitcode saved by lld. It is hardcoded into lld.
    return os.path.join(self._obj_base_dir,
                        self._obj_relative_path + '.3.import.bc')

  def lld_src_thinlto(self):
    return os.path.join(self._obj_base_dir,
                        self._obj_relative_path + '.thinlto.bc')

  def dest_dir(self):
    return os.path.join(self.output_base_dir(),
                        os.path.dirname(self._obj_relative_path))

  def module_name(self):
    return os.path.basename(self._obj_relative_path)

  def cmd_file(self):
    return os.path.join(self.dest_dir(), self.module_name() + '.cmd')

  def bc_file(self):
    return os.path.join(self.dest_dir(), self.module_name() + '.bc')

  def thinlto_index_file(self):
    return os.path.join(self.dest_dir(), self.module_name() + '.thinlto.bc')

  def _get_extraction_cmd_command(self, llvm_objcopy_path):
    """Call llvm_objcopy to extract the .llvmcmd section in self._cmd_file."""
    return [
        llvm_objcopy_path, '--dump-section=.llvmcmd=' + self.cmd_file(),
        self.input_obj(), '/dev/null'
    ]

  def _get_extraction_bc_command(self, llvm_objcopy_path):
    """Call llvm_objcopy to extract the .llvmbc section in self._bc_file."""
    return [
        llvm_objcopy_path, '--dump-section=.llvmbc=' + self.bc_file(),
        self.input_obj(), '/dev/null'
    ]

  def _extract_clang_artifacts(self, llvm_objcopy_path: str, cmd_filter: str,
                               is_thinlto: bool) -> Optional[str]:
    """Run llvm-objcopy to extract the .bc and command line."""
    if not os.path.exists(self.input_obj()):
      logging.info('%s does not exist.', self.input_obj())
      return None
    os.makedirs(self.dest_dir(), exist_ok=True)
    try:
      subprocess.run(
          self._get_extraction_cmd_command(llvm_objcopy_path), check=True)
      if cmd_filter is not None or is_thinlto:
        with open(self.cmd_file(), encoding='utf-8') as f:
          lines = f.readlines()
        assert len(lines) == 1
        cmdline = lines[0]
        if not should_include_module(cmdline, cmd_filter):
          logging.info(
              'Excluding module %s because it does not match the filter',
              self.input_obj())
          os.remove(self.cmd_file())
          return None
        if is_thinlto:
          index_file = get_thinlto_index(cmdline, self.obj_base_dir())
          shutil.copy(index_file, self.thinlto_index_file())

      subprocess.run(
          self._get_extraction_bc_command(llvm_objcopy_path), check=True)
    except subprocess.CalledProcessError as e:
      # This may happen if  .o file was build from asm (.S source).
      logging.warning('%s was not processed: %s', self.input_obj(), e)
      return None
    assert (os.path.exists(self.cmd_file()) and
            os.path.exists(self.bc_file()) and
            (not is_thinlto or os.path.exists(self.thinlto_index_file())))
    return self.relative_output_path()

  def _extract_lld_artifacts(self) -> Optional[str]:
    """Extract the .bc file with ThinLTO index from an lld ThinLTO invocation.
    """
    if not os.path.exists(self.lld_src_bc()):
      logging.info('%s does not exist.', self.lld_src_bc())
      return None
    if not os.path.exists(self.lld_src_thinlto()):
      logging.info('%s does not exist.', self.lld_src_thinlto())
      return None
    os.makedirs(self.dest_dir(), exist_ok=True)

    # Copy over the files
    shutil.copy(self.lld_src_bc(), self.bc_file())
    shutil.copy(self.lld_src_thinlto(), self.thinlto_index_file())

    assert os.path.exists(self.bc_file())
    assert os.path.exists(self.thinlto_index_file())
    return self._obj_relative_path

  def extract(self,
              llvm_objcopy_path: Optional[str] = None,
              cmd_filter: Optional[str] = None,
              thinlto_build: Optional[str] = None) -> Optional[str]:
    if self._obj_relative_path is None:
      return None
    if thinlto_build == 'local':
      return self._extract_lld_artifacts()
    return self._extract_clang_artifacts(
        llvm_objcopy_path=llvm_objcopy_path,
        cmd_filter=cmd_filter,
        is_thinlto=thinlto_build == 'distributed')


def convert_compile_command_to_objectfile(command: Dict[str, str],
                                          output_dir: str):
  obj_base_dir = command['directory']
  cmd = command['command']

  cmd_parts = cmd.split()
  try:
    obj_index = cmd_parts.index('-o') + 1
  except ValueError:
    logging.info('Command has no -o option: %s', cmd)
    return TrainingIRExtractor(None, None, None)
  obj_rel_path = cmd_parts[obj_index]
  # TODO(mtrofin): is the obj_base_dir correct for thinlto index bc files?
  return TrainingIRExtractor(
      obj_relative_path=obj_rel_path,
      output_base_dir=output_dir,
      obj_base_dir=obj_base_dir)


def load_from_compile_commands(json_array: List[Dict[str, str]],
                               output_dir: str) -> List[TrainingIRExtractor]:
  return [
      convert_compile_command_to_objectfile(cmd, output_dir)
      for cmd in json_array
  ]


def load_from_lld_params(params_array: List[str], obj_base_dir: str,
                         output_dir: str) -> List[TrainingIRExtractor]:
  """Create an ObjectFile array based on lld's parameters."""
  # yank out -o and the output. After that, anything not starting with '-', and
  # ending in a '.o', is an object file.
  try:
    minus_o_idx = params_array.index('-o')
    del params_array[minus_o_idx:minus_o_idx + 2]
    just_obj_paths = [
        o for o in params_array if not o.startswith('-') and o.endswith('.o')
    ]
  except ValueError:
    logging.info('This params file does not have an explicit -o option.')
    just_obj_paths = params_array

  def make_obj(obj_file: str) -> TrainingIRExtractor:
    return TrainingIRExtractor(
        obj_relative_path=obj_file,
        output_base_dir=output_dir,
        obj_base_dir=obj_base_dir)

  return [make_obj(obj_file) for obj_file in just_obj_paths]


def load_for_lld_thinlto(obj_base_dir: str,
                         output_dir: str) -> List[TrainingIRExtractor]:
  # .3.import.bc is the suffix attached to post-merge-pre-opt ('postimport')
  # IR bitcode saved by lld. It is hardcoded into lld. ThinLTO index files
  # are also emitted next to the postimport bitcode, with the suffix
  # .thinlto.bc instead
  paths = [str(p) for p in pathlib.Path(obj_base_dir).glob('**/*.3.import.bc')]

  def make_spec(obj_file: str):
    return TrainingIRExtractor(
        # Cut away .3.import.bc
        obj_relative_path=os.path.relpath(obj_file, start=obj_base_dir)[:-12],
        output_base_dir=output_dir,
        obj_base_dir=obj_base_dir)

  return [make_spec(path) for path in paths]


# This is here just for readability, lint complains if the pooling expression is
# over 3 lines; and it needs to be a non-local so it may be pickled.
def extract_artifacts(obj: TrainingIRExtractor) -> Optional[str]:
  return obj.extract(FLAGS.llvm_objcopy_path, FLAGS.cmd_filter,
                     FLAGS.thinlto_build)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flags_as_required(['output_dir'])

  objs = []
  if FLAGS.input is None:
    if FLAGS.thinlto_build != 'local':
      raise ValueError('--input or --thinlto_build=local must be provided')
    objs = load_for_lld_thinlto(FLAGS.obj_base_dir, FLAGS.output_dir)
  elif FLAGS.input_type == 'json':
    with open(FLAGS.input, encoding='utf-8') as f:
      objs = load_from_compile_commands(json.load(f), FLAGS.output_dir)
  elif FLAGS.input_type == 'params':
    if not FLAGS.obj_base_dir:
      logging.info(
          '-obj_base_dir is unspecified, assuming current directory.'
          'If no objects are found, use this option to specify the root'
          'directory for the object file paths in the input file.')
    with open(FLAGS.input, encoding='utf-8') as f:
      objs = load_from_lld_params([l.strip() for l in f.readlines()],
                                  FLAGS.obj_base_dir, FLAGS.output_dir)
  else:
    logging.error('Unknown input type: %s', FLAGS.input_type)

  with multiprocessing.Pool(FLAGS.num_workers) as pool:
    relative_output_paths = pool.map(extract_artifacts, objs)

  # This comes first rather than later so global_command_override is at the top
  # of the .json after being written
  if FLAGS.thinlto_build == 'local':
    corpus_description = {
        'global_command_override': constant.UNSPECIFIED_OVERRIDE
    }
  else:
    corpus_description = {}

  corpus_description.update({
      'has_thinlto': FLAGS.thinlto_build is not None,
      'modules': [path for path in relative_output_paths if path is not None]
  })

  with open(
      os.path.join(FLAGS.output_dir, 'corpus_description.json'),
      'w',
      encoding='utf-8') as f:
    json.dump(corpus_description, f, indent=2)

    logging.info('Converted %d files out of %d',
                 len(objs) - relative_output_paths.count(None), len(objs))


if __name__ == '__main__':
  app.run(main)
