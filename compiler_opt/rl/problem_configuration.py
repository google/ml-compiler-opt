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
from typing import Callable, Iterable, Tuple

import tensorflow as tf
import tf_agents as tfa

# used for type annotation in a string (for 3.8 compat)
# pylint: disable=unused-import
from compiler_opt.rl import compilation_runner

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
