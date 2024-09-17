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
"""Local trace data collector."""

from absl import logging
import os
import subprocess
import functools
import json
import pathlib
import shutil
import multiprocessing

PROFILE_PATH = "/usr/local/google/home/aidengrossman/opt_mlregalloc_pgo/combined.profdata"

def compile_module(module_path, corpus_path, clang_path, tflite_dir,
                   output_path):
  module_full_input_path = os.path.join(corpus_path, module_path) + '.bc'
  module_full_output_path = os.path.join(output_path, module_path) + '.bc.o'
  module_command_full_path = os.path.join(corpus_path, module_path) + '.cmd'
  thinlto_index_path = os.path.join(corpus_path, module_path) + '.thinlto.bc'
  pathlib.Path(os.path.dirname(module_full_output_path)).mkdir(
      parents=True, exist_ok=True)
  with open(module_command_full_path) as module_command_handle:
    module_command_line = tuple(module_command_handle.read().replace(
        r'{', r'{{').replace(r'}', r'}}').split('\0'))

  command_vector = [clang_path]
  command_vector.extend(module_command_line)
  command_vector.extend([module_full_input_path, '-o', module_full_output_path])

  if tflite_dir is not None:
    command_vector.extend(['-mllvm', '-regalloc-enable-advisor=development'])
    command_vector.extend(['-mllvm', '-regalloc-model=' + tflite_dir])
  
  try:
    advisor_flag_index = command_vector.index("-regalloc-enable-advisor=release")
    command_vector.pop(advisor_flag_index - 1)
    command_vector.pop(advisor_flag_index - 1)
  except ValueError:
    pass

  try:
    split_dwarf_file_index = command_vector.index("-split-dwarf-file")
    command_vector.pop(split_dwarf_file_index)
    command_vector.pop(split_dwarf_file_index)
  except ValueError:
    pass

  try:
    split_dwarf_output_index = command_vector.index("-split-dwarf-output")
    command_vector.pop(split_dwarf_output_index)
    command_vector.pop(split_dwarf_output_index)
  except ValueError:
    pass

  for index, cli_option in enumerate(command_vector):
    if cli_option.startswith("-fprofile-instrument-use-path="):
      command_vector[index] = f"-fprofile-instrument-use-path={PROFILE_PATH}"
    if cli_option.startswith("-fthinlto-index="):
      command_vector[index] = f"-fthinlto-index={thinlto_index_path}"
  
  command_vector.extend(["-mllvm", "-thinlto-assume-merged"])

  subprocess.run(command_vector, check=True)
  logging.info(f'Just finished compiling {module_full_output_path}')


def compile_corpus(corpus_path,
                   output_path,
                   clang_path,
                   tflite_dir=None,
                   thread_count=multiprocessing.cpu_count()):
  with open(
      os.path.join(corpus_path, 'corpus_description.json'),
      encoding='utf-8') as corpus_description_handle:
    corpus_description = json.load(corpus_description_handle)

  to_compile = []

  for module_index, module_path in enumerate(corpus_description['modules']):
    # Compile each module.
    to_compile.append(module_path)

  with multiprocessing.Pool(thread_count) as pool:
    pool.map(
        functools.partial(
            compile_module,
            corpus_path=corpus_path,
            clang_path=clang_path,
            tflite_dir=tflite_dir,
            output_path=output_path), to_compile)

  shutil.copy(
      os.path.join(corpus_path, 'corpus_description.json'),
      os.path.join(output_path, 'corpus_description.json'))


def evaluate_compiled_corpus(compiled_corpus_path,
                             trace_path,
                             function_index_path,
                             bb_trace_model_path,
                             thread_count=multiprocessing.cpu_count()):
  corpus_description_path = os.path.join(compiled_corpus_path,
                                         'corpus_description.json')
  command_vector = [
      bb_trace_model_path, '--bb_trace_path=' + trace_path,
      '--corpus_path=' + corpus_description_path, '--cpu_name=skylake-avx512',
      '--function_index_path=' + function_index_path,
      '--thread_count=' + str(thread_count)
  ]

  process_return = subprocess.run(
      command_vector, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  output = process_return.stdout.decode('utf-8')

  total_cost = 0.0

  for line in output.split('\n'):
    if line == '':
      continue
    total_cost += float(line)

  return total_cost
