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
"""A library that contains utilities for extracting functions.

This library contains utilities to find what functions exist within a specific
bitcode module as well as to extract functions of interest to separate bitcode
files for use in the training process.
"""

import subprocess
import os
import shutil
import json
import concurrent.futures
from collections.abc import Sequence


def _get_function_names_in_file(bc_file_path: str,
                                llvm_nm_path: str) -> list[str]:
  """Gets all function names defined in a file.

  This function returns all the (mangled) function names present in a bitcode
  file that have external or weak linkage.

  Args:
    bc_file_path: The path to the bitcode file to find functions in.
    llvm_nm_path: The path to the llvm-nm binary that is used to extract all of
      the symbol names within the function.

  Returns:
    A list of strings representing the functions in the file.
  """
  command_vector = [
      llvm_nm_path,
      "--defined-only",
      "--format=posix",
      bc_file_path,
  ]
  result = subprocess.run(command_vector, capture_output=True, check=True)

  functions_list = []
  for symbol in result.stdout.decode("utf-8").split("\n")[:-1]:
    symbol_parts = symbol.split(" ")
    if (symbol_parts[1] == "t" or symbol_parts[1] == "T" or
        symbol_parts[1] == "w" or symbol_parts[1] == "W"):
      functions_list.append(symbol_parts[0])

  return functions_list


def _extract_function_from_file(
    bc_file_path: str,
    output_file_path: str,
    function_name: str,
    llvm_extract_path: str,
    opt_path: str,
) -> None:
  """Extracts a function from a file.

  Args:
    bc_file_path: The path to the bitcode file to extract the function from.
    output_folder: The folder to dump the extracted function into.
    function_name: The (mangled) name of the function to extract.
    llvm_extract_path: The path to the llvm-extract binary to use to extract
      the function.
    opt_path: The path to the opt binary to use to strip debug information.
  """
  command_vector = [
      llvm_extract_path,
      "-func",
      function_name,
      "-o",
      output_file_path + ".fat",
      bc_file_path,
  ]
  subprocess.run(command_vector, capture_output=True, check=True)

  opt_command_vector = [
      opt_path,
      "-strip-debug",
      "-o",
      output_file_path,
      output_file_path + ".fat",
  ]
  subprocess.run(opt_command_vector, capture_output=True, check=True)
  os.remove(output_file_path + ".fat")

  orig_cmd_file_path = os.path.splitext(bc_file_path)[0] + ".cmd"
  output_cmd_file_path = os.path.splitext(output_file_path)[0] + ".cmd"
  shutil.copy(orig_cmd_file_path, output_cmd_file_path)


def get_function_module_map(corpus_path: str,
                            llvm_nm_path: str) -> dict[str, str]:
  """Gets a mapping from function names to module paths.

  Args:
    corpus_path: The path to the corpus to obtain the mapping from.
    llvm_nm_path: The path to the llvm-nm binary to obtain symbols from
      bitcode files.

  Returns:
    A dictionary mapping (mangled) function names to module paths.
  """
  function_to_module_map = {}

  with open(
      os.path.join(corpus_path, "corpus_description.json"),
      encoding="utf-8") as corpus_description_handle:
    corpus_description = json.load(corpus_description_handle)

  for module in corpus_description["modules"]:
    module_path = os.path.join(corpus_path, module) + ".bc"
    for function_name in _get_function_names_in_file(module_path, llvm_nm_path):
      function_to_module_map[function_name] = module_path

  return function_to_module_map


def extract_functions(functions_to_extract: Sequence[str],
                      function_to_module: dict[str, str],
                      llvm_extract_path: str, opt_path: str, thread_count: int,
                      output_dir: str) -> None:
  """Extracts all the functions specified.

  Args:
    functions_to_extract: A string list containing (mangled) names of all the
      functions that should be extracted.
    function_to_module: A dictionary mapping (mangled) function names to module
      paths.
    llvm_extract_path: The path to the llvm-extract binary to use to extract
      function bodies from bitcode files.
    opt_path: The path to the opt binary to use for stripping debug info.
    thread_count: The number of threads to use for extracting functions.
    output_dir: The path to the new corpus where all the extracted functions
      will be placed.
  """
  module_paths = []

  with concurrent.futures.ThreadPoolExecutor(thread_count) as thread_pool:
    extract_futures = []

    for index, function_to_extract in enumerate(functions_to_extract):
      bc_file = function_to_module[function_to_extract]
      output_path = os.path.join(output_dir, f"{index}.bc")
      module_paths.append(str(index))
      extract_futures.append(
          thread_pool.submit(_extract_function_from_file, bc_file, output_path,
                             function_to_extract, llvm_extract_path, opt_path))

    for future in extract_futures:
      if future.exception() is not None:
        raise future.exception()

  corpus_description = {"modules": module_paths, "has_thinlto": False}

  with open(
      os.path.join(output_dir, "corpus_description.json"),
      "w",
      encoding="utf-8") as corpus_description_handle:
    json.dump(corpus_description, corpus_description_handle)
