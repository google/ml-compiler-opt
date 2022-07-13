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
"""ModuleSpec definition and utility command line parsing functions."""

from absl import logging
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional
import json
import os
import tensorflow as tf


@dataclass(frozen=True)
class ModuleSpec:
  """Dataclass describing an input module and its compilation command options.
  """
  exec_cmd: Tuple[str, ...]
  extra_opts: dict[str, Tuple[str, ...]]
  name: str

  def cmd(self, add_flags: List[Tuple[str, ...]] = None) -> List[str]:
    """Retrieves the compiler execution options,
    optionally adding configurable options.
    """
    ret_cmd: List[str] = list(self.exec_cmd)
    if add_flags is None:
      return ret_cmd

    for opt in add_flags:
      if len(opt) == 0 or len(opt) > 2:
        logging.error('Additional option given of invalid length %d', len(opt))
        raise ValueError
      if len(opt) == 2:
        if opt[0] == '-mllvm':
          format_strs = self.extra_opts['mllvm']
        else:
          logging.error(
              'Additional option of length 2 doesn\'t start with -mllvm')
          raise ValueError
      else:
        format_strs = self.extra_opts['std']
      ret_cmd += [s.format(opt=opt[-1]) for s in format_strs]
    return ret_cmd


def read(data_path: str, additional_flags: Tuple[str, ...],
         delete_flags: Tuple[str, ...]) -> List[ModuleSpec]:
  module_paths: List[str] = _load_module_paths(data_path)

  is_thinlto: bool = _has_thinlto_index(module_paths)
  has_cmd: bool = _has_cmd(module_paths)

  # TODO: (b/233935329) Per-corpus *fdo profile paths can be read into
  # {additional|delete}_flags here
  meta = _load_metadata(os.path.join(data_path, 'metadata.json'))

  extra_options = {  # In preparation for allowing -Wl,-flags
      'mllvm': ('-mllvm', '{opt:s}'),
      'std': ('{opt:s}',)
  }

  module_specs: List[ModuleSpec] = []

  # Future: cmd_override could have per-module overrides if needed
  if 'global_command_override' in meta:
    cmd_override = tuple(meta['global_command_override'])
    if len(additional_flags) > 0:
      logging.warning("Additional flags are specified together with override.")
    if len(delete_flags) > 0:
      logging.warning("Delete flags are specified together with override.")
  else:
    cmd_override = None

  # This takes ~7s for 30k modules
  for module_path in module_paths:
    exec_cmd = _load_and_parse_command(
        cmd_file=(module_path + '.cmd') if has_cmd else None,
        ir_file=module_path + '.bc',
        thinlto_file=(module_path + '.thinlto.bc') if is_thinlto else None,
        additional_flags=additional_flags,
        delete_flags=delete_flags,
        cmd_override=cmd_override)
    module_specs.append(ModuleSpec(tuple(exec_cmd), extra_options, module_path))

  return module_specs


def _has_thinlto_index(module_paths: Iterable[str]) -> bool:
  return tf.io.gfile.exists(next(iter(module_paths)) + '.thinlto.bc')


def _has_cmd(module_paths: Iterable[str]) -> bool:
  return tf.io.gfile.exists(next(iter(module_paths)) + '.cmd')


def _load_module_paths(data_path) -> List[str]:
  module_paths_path = os.path.join(data_path, 'module_paths')
  with open(module_paths_path, 'r', encoding='utf-8') as f:
    ret = [os.path.join(data_path, name.rstrip('\n')) for name in f]
    if len(ret) == 0:
      logging.error('%s is empty.', module_paths_path)
      raise ValueError
    return ret


def _load_metadata(metadata_path: str) -> Dict[any, any]:
  try:
    with open(metadata_path, 'r', encoding='utf-8') as f:
      return json.load(f)
  except FileNotFoundError:
    logging.info('%s couldn\'t be found.', metadata_path)
    return {}


def _load_and_parse_command(cmd_file: Optional[str],
                            ir_file: str,
                            thinlto_file: Optional[str] = None,
                            additional_flags: Tuple[str, ...] = (),
                            delete_flags: Tuple[str, ...] = (),
                            cmd_override: Tuple[str, ...] = None) -> List[str]:
  """Loads and cleans up base command line.

  Remove certain unnecessary flags, and add the .bc file to compile and, if
  given, the thinlto index.

  Args:
    cmd_file: Path to a .cmd file (from corpus).
    ir_file: The path to the ir file to compile.
    thinlto_file: The path to the thinlto index, or None.
    additional_flags: Tuple of flags to add.
    delete_flags: Tuple of flags to remove.

  Returns:
    The argument list to pass to the compiler process.
  """
  if cmd_override is not None:
    option_iterator = iter(cmd_override)
  elif cmd_file is not None:
    with open(cmd_file, encoding='utf-8') as f:
      option_iterator = iter(f.read().split('\0'))
  else:
    logging.error('.cmd file not available and no command override specified.')
    raise ValueError

  cmdline = []
  option = next(option_iterator, None)
  while option:
    if any(option.startswith(flag) for flag in delete_flags):
      if '=' not in option:
        next(option_iterator, None)
    else:
      cmdline.append(option)
    option = next(option_iterator, None)

  cmdline.extend(['-x', 'ir', ir_file])

  if thinlto_file:
    cmdline.append('-fthinlto-index=' + thinlto_file)
    cmdline += ['-mllvm', '-thinlto-assume-merged']

  cmdline.extend(additional_flags)

  if cmd_file is not None and cmd_override is None:
    # Ensure that -cc1 is always present
    if cmdline[0] != '-cc1':
      cmdline = ['-cc1'] + cmdline

  return cmdline
