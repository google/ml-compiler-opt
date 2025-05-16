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
"""A library that contains utilities for grouping functions."""

import dataclasses
import os
import math
import subprocess
import json

from compiler_opt.rl import corpus


@dataclasses.dataclass
class FunctionPathAndSize:
  path: str
  size: int


def _get_functions_chunked_by_command_line(
    function_folder: str, delete_flags: tuple[str, ...] = ()
) -> dict[tuple[str], list[str]]:
  function_corpus = corpus.Corpus(
      data_path=function_folder,
      delete_flags=delete_flags,
      construct_cmd_for_compilation=False)
  command_lines = {}

  for module_spec in function_corpus.module_specs:
    function_path = os.path.join(function_folder, module_spec.name + '.bc')
    function_size = module_spec.size
    function_path_and_size = FunctionPathAndSize(function_path, function_size)
    module_command_line = list(module_spec.command_line)
    module_command_line = tuple(module_command_line)
    if module_command_line in command_lines:
      command_lines[module_command_line].append(function_path_and_size)
    else:
      command_lines[module_command_line] = [function_path_and_size]

  for command_line in command_lines:
    command_lines[command_line] = sorted(
        command_lines[command_line],
        key=lambda function_path_and_size: function_path_and_size.size,
        reverse=True)

  final_command_lines = {}
  for command_line, sorted_functions in command_lines.items():
    final_command_lines[command_line] = [
        function_path_and_size.path
        for function_path_and_size in sorted_functions
    ]

  return final_command_lines


def _partition_functions(
    functions_per_command_line: dict[tuple[str], list[str]],
    max_functions_per_chunk: int) -> dict[tuple[str], list[list[str]]]:
  corpus_chunks = {}
  for command_line in functions_per_command_line:
    corpus_chunks[command_line] = []
    chunks_for_command_line = math.ceil(
        len(functions_per_command_line[command_line]) / max_functions_per_chunk)
    for chunk_index in range(0, chunks_for_command_line):
      current_index = chunk_index
      current_chunk = []
      while current_index < len(functions_per_command_line[command_line]):
        function_path = functions_per_command_line[command_line][current_index]
        current_chunk.append(function_path)
        current_index += chunks_for_command_line
      corpus_chunks[command_line].append(current_chunk)
  return corpus_chunks


def get_chunks(
    function_folder: str, delete_flags: tuple[str, ...],
    max_functions_per_chunk: int) -> dict[tuple[str], list[list[str]]]:
  chunked_functions = _get_functions_chunked_by_command_line(
      function_folder, delete_flags)
  partitioned_functions = _partition_functions(chunked_functions,
                                               max_functions_per_chunk)
  return partitioned_functions


def combine_chunks(function_chunks: dict[tuple[str], list[list[str]]],
                   llvm_link_path: str, output_folder: str):
  corpus_chunk_index = 0
  for command_line in function_chunks:
    for function_chunk in function_chunks[command_line]:
      output_file = os.path.join(output_folder, f'{corpus_chunk_index}.bc')
      command_vector = [llvm_link_path, '-o', output_file]
      command_vector.extend(function_chunk)
      subprocess.run(command_vector, capture_output=True, check=True)

      output_cmd_file = os.path.join(output_folder, f'{corpus_chunk_index}.cmd')
      with open(
          output_cmd_file, 'w', encoding='utf-8') as output_cmd_file_handle:
        output_cmd_file_handle.write('\0'.join(command_line))
      corpus_chunk_index += 1

  with open(
      os.path.join(output_folder, 'corpus_description.json'),
      'w',
      encoding='utf-8') as corpus_description_handle:
    corpus_description = {
        'has_thinlto': False,
        'modules': [str(index) for index in range(0, corpus_chunk_index + 1)]
    }
    json.dump(corpus_description, corpus_description_handle)
