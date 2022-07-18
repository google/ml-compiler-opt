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

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import tensorflow as tf


@dataclass(frozen=True)
class ModuleSpec:
  """Dataclass describing an input module and its compilation command options.
  """
  name: str
  has_thinlto: bool = False

  def cmd(
      self,
      additional_flags: Tuple[str, ...] = (),
      delete_flags: Tuple[str, ...] = ()
  ) -> List[str]:
    """Retrieves the compiler execution options.
    """
    return _load_and_parse_command(
        self.name + '.cmd',
        self.name + '.bc',
        (self.name + '.thinlto.bc') if self.has_thinlto else None,
        additional_flags=additional_flags,
        delete_flags=delete_flags)


def has_thinlto_index(module_paths: Iterable[str]) -> bool:
  return tf.io.gfile.exists(next(iter(module_paths)) + '.thinlto.bc')


def _load_and_parse_command(
    cmd_file: str,
    ir_file: str,
    thinlto_file: Optional[str] = None,
    additional_flags: Tuple[str, ...] = (),
    delete_flags: Tuple[str, ...] = ()
) -> List[str]:
  """Cleans up base command line.

  Remove certain unnecessary flags, and add the .bc file to compile and, if
  given, the thinlto index.

  Args:
    cmd_file: Path to a .cmd file (from corpus).
    ir_file: The path to the ir file to compile.
    thinlto_file: The path to the thinlto index, or None.
    additional_flags: Tuple of clang flags to add.
    delete_flags: Tuple of clang flags to remove.

  Returns:
    The argument list to pass to the compiler process.
  """
  cmdline = []

  with open(cmd_file, encoding='utf-8') as f:
    option_iterator = iter(f.read().split('\0'))
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
    cmdline.extend(
        [f'-fthinlto-index={thinlto_file}', '-mllvm', '-thinlto-assume-merged'])

  cmdline.extend(additional_flags)

  return cmdline
