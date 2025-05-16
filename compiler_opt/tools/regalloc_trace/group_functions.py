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
"""Groups functions for more efficient compilation.

This script takes a corpus of extracted functions produced by the
extract_functions.py script and then groups them according to some
parameters. It has automatic handling for things such as variant command
lines.
"""

import os
import math

from absl import app
from absl import flags
from absl import logging

from compiler_opt.tools.regalloc_trace import group_functions_lib

_FUNCTION_FOLDER = flags.DEFINE_string(
    "function_folder",
    None,
    "The path to the folder containing all the functions.",
    required=True)
_OUTPUT_FOLDER = flags.DEFINE_string(
    "output_folder", None, "The path to the output folder.", required=True)
_LLVM_LINK_PATH = flags.DEFINE_string(
    "llvm_link_path",
    None,
    "The path to the llvm-link binary to use.",
    required=True)
_CHUNK_COUNT = flags.DEFINE_string(
    "chunk_count", 256, "The approximate number of chunks to produce.")


def main(_) -> None:
  # These are flags that do not matter at this point in the pipeline that are
  # different per translation unit and thus would require individual chunks.
  delete_compilation_flags = (
      "-fthinlto-index",
      "-main-file-name",
      "-split-dwarf-file",
      "-split-dwarf-output",
      "-fcoverage-compilation-dir",
  )

  # We have a .bc and a .cmd for each extracted function along with a
  # corpus_description.json file.
  functions_in_corpus = (os.listdir(_FUNCTION_FOLDER.value) - 1) / 2
  max_functions_per_chunk = math.ceil(functions_in_corpus / _CHUNK_COUNT.value)
  logging.info("Getting chunks.")
  corpus_chunks = group_functions_lib.get_chunks(_FUNCTION_FOLDER.value,
                                                 delete_compilation_flags,
                                                 max_functions_per_chunk)
  logging.info("Combining chunks.")
  group_functions_lib.combine_chunks(corpus_chunks, _LLVM_LINK_PATH.value,
                                     _OUTPUT_FOLDER.value)


if __name__ == "__main__":
  app.run(main)
