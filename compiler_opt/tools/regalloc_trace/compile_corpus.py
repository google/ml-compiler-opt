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
"""Compiles a corpus for further downstream processing."""

import time

from absl import app
from absl import flags
from absl import logging
import gin

from compiler_opt.rl import corpus
from compiler_opt.es.regalloc_trace import regalloc_trace_worker
from compiler_opt.rl import registry

_CORPUS_PATH = flags.DEFINE_string(
    'corpus_path', None, 'The path to the corpus.', required=True)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'The path to the output.', required=True)
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files to use.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [], 'Gin bindings to override the values in config files.')
_TFLITE_POLICY_PATH = flags.DEFINE_string(
    'tflite_policy_path', None,
    'The path to the TFLite policy to use, if there is one.')
_MODE = flags.DEFINE_enum('mode', 'full', ['full', 'bc', 'asm'],
                          'How far to compile the corpus.')


def main(_) -> None:
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, _GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  problem_config = registry.get_configuration()
  additional_compilation_flags = problem_config.flags_to_add()
  delete_compilation_flags = problem_config.flags_to_delete()
  replace_compilation_flags = problem_config.flags_to_replace()

  if _MODE.value == 'full':
    # We do not need to change any flags here because this is what the
    # compiler is set up to do normally.
    pass
  elif _MODE.value == 'bc':
    additional_compilation_flags = additional_compilation_flags + (
        '-emit-llvm-bc',)
  elif _MODE.value == 'asm':
    additional_compilation_flags = additional_compilation_flags + (
        '-disable-llvm-passes',)
    # When compiling from bitcode to an object file, we also need to remove all
    # the flags that can load profiles or ThinLTO indices as they are embedded
    # within the BC at this stage of compilation.
    delete_compilation_flags = delete_compilation_flags + (
        '-fprofile-sample-use', '-fprofile-instrument-use-path',
        'fthinlto-index')
  else:
    raise ValueError('Invalid mode')

  train_corpus = corpus.Corpus(
      data_path=_CORPUS_PATH.value,
      additional_flags=additional_compilation_flags,
      delete_flags=delete_compilation_flags,
      replace_flags=replace_compilation_flags,
  )

  logging.info("Compiling corpus.")
  compile_start = time.time()
  worker = regalloc_trace_worker.RegallocTraceWorker(
      gin_config=gin.operative_config_str())
  worker._build_corpus(train_corpus.module_specs, _OUTPUT_PATH.value,
                       _TFLITE_POLICY_PATH.value)
  compile_end = time.time()
  compile_duration = compile_end - compile_start
  logging.info(f'Compilation took {compile_duration}s')


if __name__ == '__main__':
  app.run(main)
