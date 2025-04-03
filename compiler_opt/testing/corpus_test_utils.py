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
"""Test utilities broadly related to corpora."""

from collections.abc import Sequence
import os
import json
import pathlib
import textwrap
import stat

from compiler_opt.rl import corpus


def setup_corpus(corpus_dir: str,
                 has_thinlto: bool = False,
                 cli_flags: tuple = ()) -> list[corpus.ModuleSpec]:
  modules = [
      corpus.ModuleSpec("module_a.o", 1, ("-fmodule-a", *cli_flags), True),
      corpus.ModuleSpec("module_b.o", 1, ("-fmodule-b", *cli_flags), True)
  ]

  corpus_description = {
      "has_thinlto": has_thinlto,
      "modules": [module.name for module in modules]
  }

  with open(
      os.path.join(corpus_dir, "corpus_description.json"),
      "w",
      encoding="utf-8") as corpus_description_handle:
    json.dump(corpus_description, corpus_description_handle)

  for module in ["module_a.o", "module_b.o"]:
    extensions = [".cmd", ".bc"]
    if has_thinlto:
      extensions.append(".thinlto.bc")

    for extension in extensions:
      module_path = os.path.join(corpus_dir, module + extension)
      pathlib.Path(module_path).touch()

  return modules


def create_test_binary(binary_path: str,
                       output_path: str,
                       commands_to_run: Sequence[str] = []):
  test_binary = textwrap.dedent(f"""\
  #!/bin/bash
  echo "$@" >> {output_path}
  """)

  for command in commands_to_run:
    test_binary = test_binary + f"{command}\n"

  with open(binary_path, "w", encoding="utf-8") as binary_handle:
    binary_handle.write(test_binary)
  binary_stat = os.stat(binary_path)
  os.chmod(binary_path, binary_stat.st_mode | stat.S_IEXEC)
