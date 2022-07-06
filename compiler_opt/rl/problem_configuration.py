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
"""Compilation problem component model.

A 'compilation problem' is an optimization problem with a specific way of
invoking clang and specific features and tensorflow topologies. The component
model requires all these be exported in a class implementing
ProblemConfiguration below, however, to avoid cycle dependencies in Bazel
environments, do not explicitly inherit from it.

Internally, all the module's implementation parameters are expected to be
gin-initialized.

To use:

* just get a ProblemConfiguration object in your python:

`config = config.get_configuration()`

* when running the tool, pass:

  `--gin_bindings=config_registry.get_configuration.implementation=\
    @the.implementation.name`

  for example:

  `--gin_bindings=config_registry.get_configuration.implementation=\
    @configs.InliningConfig`

=================
Conventions
=================

* to avoid long binding names, use the 'runners' module name for the
  CompilationRunner implementation, and use the 'configs' module name for the
  implementation of ProblemConfiguration.

* the CompilationRunner gin initialization should initialize to None, and use,
  the 'clang_path' and 'launcher_path' macros
  (https://github.com/google/gin-config#syntax-quick-reference):

  clang_path = None
  launcher_path = None
  runners.MyCompilationRunner.clang_path = %clang_path
  runners.MyCompilationRunner.launcher_path = %launcher_path

  Use a similar pattern for problem-specific additional flags (see 'inlining'
  and 'llvm_size_path' for example).

  When running tools, this allows the user pass common flags transparently wrt
  the underlying runner - i.e. if swapping 2 runners, the clang flag stays the
  same:

  `--gin_bindings=clang_path="'/foo/bar/clang'"`

"""

import abc
import json
import logging
import os
import sys
from typing import Callable, Dict, Iterable, Tuple, List, Optional, Union

import tensorflow as tf
import tf_agents as tfa

# used for type annotation in a string (for 3.8 compat)
# pylint: disable=unused-import
from compiler_opt.rl import compilation_runner
from compiler_opt.rl.adt import ModuleSpec

types = tfa.typing.types


class ProblemConfiguration(metaclass=abc.ABCMeta):
  """Abstraction of the APIs accessing a problem-specific configuration."""

  @abc.abstractmethod
  def get_signature_spec(
      self) -> Tuple[types.NestedTensorSpec, types.NestedTensorSpec]:
    raise NotImplementedError

  @abc.abstractmethod
  def get_preprocessing_layer_creator(
      self) -> Callable[[types.TensorSpec], tf.keras.layers.Layer]:
    raise NotImplementedError

  def get_nonnormalized_features(self) -> Iterable[str]:
    return []

  @abc.abstractmethod
  def get_runner_type(self) -> 'type[compilation_runner.CompilationRunner]':
    raise NotImplementedError

  @abc.abstractmethod
  def get_module_specs(
      self,
      data_path: str,
      additional_flags: Tuple[str, ...] = (),
      delete_flags: Tuple[str, ...] = ()
  ) -> List[ModuleSpec]:
    """Fetch a list of ModuleSpecs for the corpus at data_path

    Args:
      data_path: base directory of corpus
      additional_flags: tuple of clang flags to add.
      delete_flags: tuple of clang flags to remove.
    """
    raise NotImplementedError

  @staticmethod
  def _get_module_specs(data_path: str, additional_flags, delete_flags,
                        xopts) -> List[ModuleSpec]:
    module_paths: List[str] = get_module_paths(data_path)

    is_thinlto: bool = is_thinlto_fn(module_paths)
    has_cmd: bool = has_cmd_fn(module_paths)

    # TODO: (b/233935329) Per-corpus *fdo profile paths can be read into
    # {additional|delete}_flags here
    meta = load_metadata(data_path)

    xopts.update({'output': ['-o', '{path:s}']})

    module_specs: List[ModuleSpec] = []

    # Future: cmd_override could have per-module overrides if needed
    cmd_override = get_cmd_override(meta, critical=not has_cmd)

    for module_path in module_paths:
      exec_cmd = get_command_line_for_bundle(
          cmd_file=(module_path + '.cmd') if has_cmd else None,
          ir_file=module_path + '.bc',
          thinlto_file=(module_path + '.thinlto.bc') if is_thinlto else None,
          additional_flags=additional_flags,
          delete_flags=delete_flags,
          cmd_override=list(cmd_override) if cmd_override is not None else None)
      module_specs.append(ModuleSpec(exec_cmd, xopts, module_path))

    return module_specs


def is_thinlto_fn(module_paths: Iterable[str]) -> bool:
  return tf.io.gfile.exists(next(iter(module_paths)) + '.thinlto.bc')


def has_cmd_fn(module_paths: Iterable[str]) -> bool:
  return tf.io.gfile.exists(next(iter(module_paths)) + '.cmd')


def get_module_paths(data_path: str) -> List[str]:
  with open(
      os.path.join(data_path, 'module_paths'), 'r', encoding='utf-8') as f:
    return [os.path.join(data_path, name.rstrip('\n')) for name in f]


def load_metadata(data_path: str) -> Dict[any, any]:
  try:
    with open(
        os.path.join(data_path, 'metadata.json'), 'r', encoding='utf-8') as f:
      return json.load(f)
  except FileNotFoundError:
    logging.info('metadata.json couldn\'t be found.')
    return {}


def get_cmd_override(meta: Dict[any, any],
                     critical=False) -> Union[List[str], None]:
  if 'global_command_override' in meta:
    logging.info('global_command_override will be used instead of .cmd files')
    return meta['global_command_override']
  elif critical:
    logging.fatal('global_command_override couldn\'t be found but is needed')
    sys.exit(1)
  else:
    logging.info('global_command_override couldn\'t be found. Using .cmd files')
    return None


def get_command_line_for_bundle(cmd_file: str,
                                ir_file: str,
                                thinlto_file: Optional[str] = None,
                                additional_flags: Tuple[str, ...] = (),
                                delete_flags: Tuple[str, ...] = (),
                                cmd_override: List[str] = None) -> List[str]:
  """Cleans up base command line.

  Remove certain unnecessary flags, and add the .bc file to compile and, if
  given, the thinlto index.

  Args:
    cmd_file: Path to a .cmd file (from corpus).
    ir_file: The path to the ir file to compile.
    thinlto: The path to the thinlto index, or None.
    additional_flags: Tuple of clang flags to add.
    delete_flags: Tuple of clang flags to remove.

  Returns:
    The argument list to pass to the compiler process.
  """

  if cmd_override is not None:
    cmdline = cmd_override
    option = None
  elif cmd_file is not None:
    cmdline = []
    with open(cmd_file, encoding='utf-8') as f:
      option_iterator = iter(f.read().split('\0'))
      option = next(option_iterator, None)
  else:
    logging.fatal('.cmd file not available and no command override specified')
    sys.exit(1)

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
