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

from __future__ import annotations  # for typing .get()
from absl import logging
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional
import abc
import json
import os
import tensorflow as tf


@dataclass(frozen=True)
class ModuleSpec:
  """Dataclass describing an input module and its compilation command options.
  """
  _exec_cmd: Tuple[str, ...]
  _xopts: dict[str, Tuple[str, ...]]
  name: str

  def cmd(self, **kwargs) -> List[str]:
    """Retrieves the compiler execution options,
    optionally adding configurable, pre-set options.
    """
    ret_cmd: List[str] = list(self._exec_cmd)
    for k, substitutions in kwargs.items():
      if k in self._xopts:
        if substitutions is not None:
          ret_cmd += [opt.format(**substitutions) for opt in self._xopts[k]]
      else:
        logging.fatal(
            'Command line addition of \'%s\' requested '
            'but no such extra option exists.', k)
        raise ValueError
    return ret_cmd

  @classmethod
  def _get(cls, data_path: str, additional_flags: Tuple[str, ...],
           delete_flags: Tuple[str, ...], xopts: Dict[str, Tuple[str, ...]]):
    module_paths: List[str] = _load_module_paths(
        data_path, os.path.join(data_path, 'module_paths'))

    is_thinlto: bool = _has_thinlto_index(module_paths)
    has_cmd: bool = _has_cmd(module_paths)

    # TODO: (b/233935329) Per-corpus *fdo profile paths can be read into
    # {additional|delete}_flags here
    meta = _load_metadata(os.path.join(data_path, 'metadata.json'))

    xopts.update({'output': ('-o', '{path:s}')})

    module_specs: List[ModuleSpec] = []

    # Future: cmd_override could have per-module overrides if needed
    if 'global_command_override' in meta:
      cmd_override = tuple(meta['global_command_override'])
    else:
      cmd_override = None

    # This takes ~7s for 30k modules
    for module_path in module_paths:
      exec_cmd = _load_and_parse_command(
          cmd_file=(module_path + '.cmd') if has_cmd else None,
          ir_file=module_path + '.bc',
          thinlto_file=(module_path + '.thinlto.bc') if is_thinlto else None,
          additional_flags=additional_flags,
          delete_flags=delete_flags,
          cmd_override=cmd_override)
      module_specs.append(cls(tuple(exec_cmd), xopts, module_path))

    return module_specs

  @classmethod
  @abc.abstractmethod
  def get(cls, data_path: str, additional_flags: Tuple[str, ...],
          delete_flags: Tuple[str, ...]) -> List[ModuleSpec]:
    """Fetch a list of ModuleSpecs for the corpus at data_path

    Args:
      data_path: base directory of corpus
      additional_flags: tuple of clang flags to add.
      delete_flags: tuple of clang flags to remove.
    """
    raise NotImplementedError


def _has_thinlto_index(module_paths: Iterable[str]) -> bool:
  return tf.io.gfile.exists(next(iter(module_paths)) + '.thinlto.bc')


def _has_cmd(module_paths: Iterable[str]) -> bool:
  return tf.io.gfile.exists(next(iter(module_paths)) + '.cmd')


def _load_module_paths(data_path, module_paths_path: str) -> List[str]:
  with open(module_paths_path, 'r', encoding='utf-8') as f:
    ret = [os.path.join(data_path, name.rstrip('\n')) for name in f]
    if len(ret) == 0:
      logging.fatal('%s is empty.', module_paths_path)
      raise ValueError
    return ret


def _load_metadata(metadata_path: str) -> Dict[any, any]:
  try:
    with open(metadata_path, 'r', encoding='utf-8') as f:
      return json.load(f)
  except FileNotFoundError:
    logging.info('%s couldn\'t be found.', metadata_path)
    return {}


def _load_and_parse_command(cmd_file: Optional[str],
                            ir_file: str,
                            thinlto_file: Optional[str] = None,
                            additional_flags: Tuple[str, ...] = (),
                            delete_flags: Tuple[str, ...] = (),
                            cmd_override: Tuple[str] = None) -> List[str]:
  """Loads and cleans up base command line.

  Remove certain unnecessary flags, and add the .bc file to compile and, if
  given, the thinlto index.

  Args:
    cmd_file: Path to a .cmd file (from corpus).
    ir_file: The path to the ir file to compile.
    thinlto_file: The path to the thinlto index, or None.
    additional_flags: Tuple of flags to add.
    delete_flags: Tuple of flags to remove.

  Returns:
    The argument list to pass to the compiler process.
  """
  cmdline = []
  option = None
  if cmd_override is not None:
    cmdline = list(cmd_override)
  elif cmd_file is not None:
    with open(cmd_file, encoding='utf-8') as f:
      option_iterator = iter(f.read().split('\0'))
      option = next(option_iterator, None)
  else:
    logging.fatal('.cmd file not available and no command override specified.')
    raise ValueError

  while option:
    if any(option.startswith(flag) for flag in delete_flags):
      if '=' not in option:
        next(option_iterator, None)
    else:
      cmdline.append(option)
    option = next(option_iterator, None)

  cmdline.extend(['-x', 'ir', ir_file])

  if thinlto_file:
    cmdline.append('-fthinlto-index=' + thinlto_file)
    cmdline += ['-mllvm', '-thinlto-assume-merged']

  if cmd_override is None:
    cmdline.extend(additional_flags)
    # Ensure that -cc1 is always present
    if cmdline[0] != '-cc1':
      cmdline = ['-cc1'] + cmdline

  return cmdline
