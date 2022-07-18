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

import os
import tensorflow as tf


@dataclass(frozen=True)
class ModuleSpec:
  """Dataclass describing an input module and its compilation command options.
  """
  name: str
  _exec_cmd: Tuple[str, ...] = ()

  def cmd(self) -> List[str]:
    """Retrieves the compiler execution options.
    """
    # Using a getter as this will be modified in the future
    return list(self._exec_cmd)


def build_modulespecs_from_datapath(
    data_path: str,
    additional_flags: Tuple[str, ...] = (),
    delete_flags: Tuple[str, ...] = ()
) -> List[ModuleSpec]:
  module_paths: List[str] = _load_module_paths(data_path)

  has_thinlto: bool = _has_thinlto_index(module_paths)

  module_specs: List[ModuleSpec] = []

  # This takes ~7s for 30k modules
  for module_path in module_paths:
    exec_cmd = _load_and_parse_command(
        ir_file=module_path + '.bc',
        cmd_file=(module_path + '.cmd'),
        thinlto_file=(module_path + '.thinlto.bc') if has_thinlto else None,
        additional_flags=additional_flags,
        delete_flags=delete_flags)
    module_specs.append(ModuleSpec(name=module_path, _exec_cmd=tuple(exec_cmd)))

  return module_specs


def _has_thinlto_index(module_paths: Iterable[str]) -> bool:
  return tf.io.gfile.exists(next(iter(module_paths)) + '.thinlto.bc')


def _load_module_paths(data_path) -> List[str]:
  module_paths_path = os.path.join(data_path, 'module_paths')
  with open(module_paths_path, 'r', encoding='utf-8') as f:
    ret = [os.path.join(data_path, name.rstrip('\n')) for name in f]
    if len(ret) == 0:
      raise ValueError(f'{module_paths_path} is empty.')
    return ret


def _load_and_parse_command(
    ir_file: str,
    cmd_file: str,
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
    while option is not None:
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
