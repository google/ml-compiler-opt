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

from compiler_opt.rl.inlining import config as inlining_config
from compiler_opt.rl.regalloc import config as regalloc_config


types = tfa.typing.types


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
