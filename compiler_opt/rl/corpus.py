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

from absl import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import json
import os

from compiler_opt.rl import constant


@dataclass(frozen=True)
class ModuleSpec:
  """Dataclass describing an input module and its compilation command options.
  """
  name: str
  exec_cmd: Tuple[str, ...] = ()


def build_modulespecs_from_datapath(
    data_path: str,
    additional_flags: Tuple[str, ...] = (),
    delete_flags: Tuple[str, ...] = ()
) -> List[ModuleSpec]:
  # TODO: (b/233935329) Per-corpus *fdo profile paths can be read into
  # {additional|delete}_flags here
  with open(
      os.path.join(data_path, 'corpus_description.json'), 'r',
      encoding='utf-8') as f:
    corpus_description: Dict[str, Any] = json.load(f)

  module_paths = corpus_description['modules']
  if len(module_paths) == 0:
    raise ValueError(f'{data_path}\'s corpus_description contains no modules.')

  has_thinlto: bool = corpus_description['has_thinlto']

  cmd_override = ()
  if 'global_command_override' in corpus_description:
    if corpus_description[
        'global_command_override'] == constant.UNSPECIFIED_OVERRIDE:
      raise ValueError(
          'global_command_override in corpus_description.json not filled.')
    cmd_override = tuple(corpus_description['global_command_override'])
    if len(additional_flags) > 0:
      logging.warning('Additional flags are specified together with override.')
    if len(delete_flags) > 0:
      logging.warning('Delete flags are specified together with override.')

  module_specs: List[ModuleSpec] = []

  # This takes ~7s for 30k modules
  for module_path in module_paths:
    exec_cmd = _load_and_parse_command(
        module_path=os.path.join(data_path, module_path),
        has_thinlto=has_thinlto,
        additional_flags=additional_flags,
        delete_flags=delete_flags,
        cmd_override=cmd_override)
    module_specs.append(ModuleSpec(name=module_path, exec_cmd=tuple(exec_cmd)))

  return module_specs


def _load_and_parse_command(
    module_path: str,
    has_thinlto: bool,
    additional_flags: Tuple[str, ...] = (),
    delete_flags: Tuple[str, ...] = (),
    cmd_override: Tuple[str, ...] = ()
) -> List[str]:
  """Cleans up base command line.

  Remove certain unnecessary flags, and add the .bc file to compile and, if
  given, the thinlto index.

  Args:
    module_path: Absolute path to the module without extension (from corpus).
    has_thinlto: Whether to add thinlto flags.
    additional_flags: Tuple of clang flags to add.
    delete_flags: Tuple of clang flags to remove.
    cmd_override: Tuple of strings to use as the base command line.

  Returns:
    The argument list to pass to the compiler process.
  """
  cmdline = []

  if cmd_override:
    option_iterator = iter(cmd_override)
  else:
    with open(module_path + '.cmd', encoding='utf-8') as f:
      option_iterator = iter(f.read().split('\0'))
  option = next(option_iterator, None)

  while option is not None:
    if any(option.startswith(flag) for flag in delete_flags):
      if '=' not in option:
        next(option_iterator, None)
    else:
      cmdline.append(option)
    option = next(option_iterator, None)
  cmdline.extend(['-x', 'ir', module_path + '.bc'])

  if has_thinlto:
    cmdline.extend([
        f'-fthinlto-index={module_path}.thinlto.bc', '-mllvm',
        '-thinlto-assume-merged'
    ])

  cmdline.extend(additional_flags)

  # The options read from a .cmd file must be run with -cc1
  if not cmd_override and cmdline[0] != '-cc1':
    cmdline = ['-cc1'] + cmdline

  return cmdline
