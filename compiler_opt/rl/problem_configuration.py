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
import gin
from typing import Callable, Dict, Iterable, Optional, Tuple

import tensorflow as tf
import tf_agents as tfa

# used for type annotation in a string (for 3.8 compat)
# pylint: disable=unused-import
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import env

types = tfa.typing.types


class ProblemConfiguration(metaclass=abc.ABCMeta):
  """Abstraction of the APIs accessing a problem-specific configuration."""

  @abc.abstractmethod
  def get_env(self) -> env.MLGOEnvironmentBase:
    raise NotImplementedError

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

  # TODO(b/233935329): The following clang flags need to be tied to a corpus
  # rather than to a training tool invocation.

  # List of flags to add to clang compilation command.
  @gin.configurable(module='problem_config')
  def flags_to_add(self, add_flags=()) -> Tuple[str, ...]:
    return add_flags

  # List of flags to remove from clang compilation command. The flag names
  # should match the actual flags provided to clang.'
  @gin.configurable(module='problem_config')
  def flags_to_delete(self, delete_flags=()) -> Tuple[str, ...]:
    return delete_flags

  # List of flags to replace in the clang compilation command. The flag names
  # should match the actual flags provided to clang. An example for AFDO
  # reinjection:
  # replace_flags={
  #     '-fprofile-sample-use':'/path/to/gwp.afdo',
  #     '-fprofile-remapping-file':'/path/to/prof_remap.txt'
  # }
  # return replace_flags
  @gin.configurable(module='problem_config')
  def flags_to_replace(self, replace_flags=None) -> Optional[Dict[str, str]]:
    return replace_flags
