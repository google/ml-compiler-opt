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
"""Loop unroll training config."""

import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from compiler_opt.rl import feature_ops


# pylint: disable=g-complex-comprehension
@gin.configurable()
def get_unroll_signature_spec():
  """Returns (time_step_spec, action_spec) for LLVM loop unroll."""
  # LINT.IfChange
  observation_spec = dict(
      (key, tf.TensorSpec(dtype=tf.int64, shape=(), name=key))
      for key in ('loop_size', 'trip_count', 'is_innermost_loop',
                  'preheader_blocksize', 'bb_count', 'num_of_loop_latch',
                  'load_inst_count', 'store_inst_count', 'logical_inst_count',
                  'cast_inst_count'))
  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)
  action_spec = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64, shape=(), name='unroll_count')

  return time_step_spec, action_spec


@gin.configurable
def get_observation_processing_layer_creator(quantile_file_dir=None,
                                             with_sqrt=True,
                                             with_z_score_normalization=True,
                                             eps=1e-8):
  """Wrapper for observation_processing_layer."""
  quantile_map = feature_ops.build_quantile_map(quantile_file_dir)

  def observation_processing_layer(obs_spec):
    """Creates the layer to process observation given obs_spec."""

    # I guess we discard rewards when observation?
    if obs_spec.name in ('icache_pressure', 'latency'):
      return tf.keras.layers.Lambda(feature_ops.discard_fn)

    # for boolean features, use feature_ops.identity_fn
    if obs_spec.name in ('is_innermost_loop'):
      return tf.keras.layers.Lambda(feature_ops.identity_fn)

    # Do we need to define some layer here to normalize 'loop_size'
    # and instruction count features (e.g. 'load_inst_count').
    # Bigger loops expect more instruction counts, and we need to
    # normalize this?

    quantile = quantile_map[obs_spec.name]
    return tf.keras.layers.Lambda(
        feature_ops.get_normalize_fn(quantile, with_sqrt,
                                     with_z_score_normalization, eps))

  return observation_processing_layer


def get_nonnormalized_features():
  return ['reward', 'is_innermost_loop']
