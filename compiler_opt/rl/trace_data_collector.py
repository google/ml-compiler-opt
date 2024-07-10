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
import json
import pathlib
import shutil


def compile_corpus(corpus_path, output_path, clang_path, tflite_dir = None):
  with open(
      os.path.join(corpus_path, 'corpus_description.json'),
      encoding='utf-8') as corpus_description_handle:
    corpus_description = json.load(corpus_description_handle)

  for module_index, module_path in enumerate(corpus_description['modules']):
    # Compile each module.
    module_full_input_path = os.path.join(corpus_path, module_path) + '.bc'
    module_full_output_path = os.path.join(output_path, module_path) + '.bc.o'
    pathlib.Path(os.path.dirname(module_full_output_path)).mkdir(
        parents=True, exist_ok=True)

    module_command_full_path = os.path.join(corpus_path, module_path) + '.cmd'
    with open(module_command_full_path) as module_command_handle:
      module_command_line = tuple(module_command_handle.read().replace(
          r'{', r'{{').replace(r'}', r'}}').split('\0'))

    command_vector = [clang_path]
    command_vector.extend(module_command_line)
    command_vector.extend(
        [module_full_input_path, '-o', module_full_output_path])
    
    if tflite_dir is not None:
      command_vector.extend(['-mllvm', '-regalloc-enable-advisor=development'])
      command_vector.extend(['-mllvm', '-ml-regalloc-model=' + tflite_dir])

    subprocess.run(command_vector)
    logging.info(
        f'Just finished compiling {module_full_output_path} ({module_index + 1}/{len(corpus_description["modules"])})'
    )

  shutil.copy(
      os.path.join(corpus_path, 'corpus_description.json'),
      os.path.join(output_path, 'corpus_description.json'))
