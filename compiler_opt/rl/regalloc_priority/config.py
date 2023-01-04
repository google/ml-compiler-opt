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
"""RegAlloc priority prediction training config."""

import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from compiler_opt.rl import feature_ops


@gin.configurable()
def get_regalloc_signature_spec():
  observation_spec = dict(
      (key, tf.TensorSpec(dtype=tf.int64, shape=(), name=key))
      for key in ('li_size', 'stage'))
  observation_spec['weight'] = tf.TensorSpec(
      dtype=tf.float32, shape=(), name='weight')

  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)

  action_spec = tensor_spec.TensorSpec(
      dtype=tf.float32, shape=(), name='priority')

  return time_step_spec, action_spec


@gin.configurable
def get_observation_processing_layer_creator():

  def observation_processing_layer(obs_spec):
    """Creates the layer to process observation given obs_spec."""

    if obs_spec.name in ('li_size', 'stage', 'weight'):
      return tf.keras.layers.Lambda(feature_ops.identity_fn)

    # Make sure all features have a preprocessing function.
    raise KeyError('Missing preprocessing function for some feature.')

  return observation_processing_layer
