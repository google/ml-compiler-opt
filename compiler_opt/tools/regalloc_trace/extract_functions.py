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
"""Extracts functions specified in a list from a corpus."""

import multiprocessing
import pathlib

from absl import flags
from absl import app
from absl import logging

from compiler_opt.tools.regalloc_trace import extract_functions_lib

_CORPUS_PATH = flags.DEFINE_string(
    "optimized_corpus", None,
    "The path to the optimized bitcode corpus that has been run through the "
    "middle-end pipeline.")
_FUNCTION_LIST = flags.DEFINE_string(
    "function_list", None, "The list of functions to extract", required=True)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "The folder to place the extracted functions in.",
    required=True)
_LLVM_NM_PATH = flags.DEFINE_string(
    "llvm_nm_path", None, "The path to llvm-nm", required=True)
_LLVM_EXTRACT_PATH = flags.DEFINE_string(
    "llvm_extract_path", None, "The path to llvm-extract", required=True)
_OPT_PATH = flags.DEFINE_string(
    "opt_path", None, "The path to opt", required=True)
_THREAD_COUNT = flags.DEFINE_integer(
    "thread_count", multiprocessing.cpu_count(),
    "The number of threads to use when processing")


def main(_):
  logging.info("Loading the function to module mapping.")
  function_to_module = extract_functions_lib.get_function_module_map(
      _CORPUS_PATH.value, _LLVM_NM_PATH.value)

  logging.info("Loading functions to extract.")
  with open(_FUNCTION_LIST.value, encoding="utf-8") as function_list_handle:
    functions_to_extract = [
        function_name.strip()
        for function_name in function_list_handle.readlines()
    ]

  logging.info("Extracting functions.")
  pathlib.Path(_OUTPUT_PATH.value).mkdir(parents=True, exist_ok=True)
  extract_functions_lib.extract_functions(functions_to_extract,
                                          function_to_module,
                                          _LLVM_EXTRACT_PATH.value,
                                          _OPT_PATH.value, _THREAD_COUNT.value,
                                          _OUTPUT_PATH.value)


if __name__ == "__main__":
  app.run(main)
