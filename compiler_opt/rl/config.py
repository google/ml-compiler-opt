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

"""Configuration utility helpers for RL training."""

from typing import Callable, Text, Tuple

import tensorflow as tf
import tf_agents as tfa

from compiler_opt.rl import compilation_runner

from compiler_opt.rl.inlining import config as inlining_config
from compiler_opt.rl.inlining import inlining_runner
from compiler_opt.rl.regalloc import config as regalloc_config


types = tfa.typing.types

# TODO(b/214316645): get rid of the if-else statement by defining a class for
# each problem type instead.


def get_signature_spec(
    problem_type: Text
) -> Tuple[types.NestedTensorSpec, types.NestedTensorSpec]:
  """Get the signature spec for the given problem type."""
  if problem_type == 'inlining':
    return inlining_config.get_inlining_signature_spec()
  elif problem_type == 'regalloc':
    return regalloc_config.get_regalloc_signature_spec()
  else:
    raise ValueError('Unknown problem_type: {}'.format(problem_type))


def get_preprocessing_layer_creator(
    problem_type: Text,
) -> Callable[[types.TensorSpec], tf.keras.layers.Layer]:
  """Get the observation processing layer creator for the given problem type."""
  if problem_type == 'inlining':
    return inlining_config.get_observation_processing_layer_creator()
  elif problem_type == 'regalloc':
    return regalloc_config.get_observation_processing_layer_creator()
  else:
    raise ValueError('Unknown problem_type: {}'.format(problem_type))


def get_compilation_runner(
    problem_type: str, clang_path: str, llvm_size_path: str, launcher_path: str,
    moving_average_decay_rate: float) -> compilation_runner.CompilationRunner:
  """Gets the compile function for the given problem type."""
  if problem_type == 'inlining':
    return inlining_runner.InliningRunner(clang_path, llvm_size_path,
                                          launcher_path,
                                          moving_average_decay_rate)
  elif problem_type == 'regalloc':
    # TODO(yundi): add in the next cl.
    raise ValueError('RegAlloc Compile Function not Supported.')
  else:
    raise ValueError('Unknown problem_type: {}'.format(problem_type))
