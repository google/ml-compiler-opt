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
"""Register allocation training config."""

import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from compiler_opt.rl import feature_ops


def get_num_registers():
  return 33


# pylint: disable=g-complex-comprehension
@gin.configurable()
def get_regalloc_signature_spec():
  """Returns (time_step_spec, action_spec) for LLVM register allocation."""
  num_registers = get_num_registers()

  observation_spec = dict(
      (key, tf.TensorSpec(dtype=tf.int64, shape=(num_registers), name=key))
      for key in ('mask', 'is_hint', 'is_local', 'is_free'))
  observation_spec.update(
      dict((key,
            tensor_spec.BoundedTensorSpec(
                dtype=tf.int64,
                shape=(num_registers),
                name=key,
                minimum=0,
                maximum=6)) for key in ('max_stage', 'min_stage')))
  observation_spec.update(
      dict((key,
            tf.TensorSpec(dtype=tf.float32, shape=(num_registers), name=key))
           for key in ('weighed_reads_by_max', 'weighed_writes_by_max',
                       'weighed_read_writes_by_max', 'weighed_indvars_by_max',
                       'hint_weights_by_max', 'start_bb_freq_by_max',
                       'end_bb_freq_by_max', 'hottest_bb_freq_by_max',
                       'liverange_size', 'use_def_density', 'nr_defs_and_uses',
                       'nr_broken_hints', 'nr_urgent', 'nr_rematerializable')))
  observation_spec['progress'] = tensor_spec.BoundedTensorSpec(
      dtype=tf.float32, shape=(), name='progress', minimum=0, maximum=1)

  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)

  action_spec = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64,
      shape=(),
      name='index_to_evict',
      minimum=0,
      maximum=num_registers - 1)

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
    if obs_spec.name in ('mask', 'nr_urgent'):
      return tf.keras.layers.Lambda(feature_ops.discard_fn)

    if obs_spec.name in ('is_hint', 'is_local', 'is_free'):
      return tf.keras.layers.Lambda(feature_ops.identity_fn)

    if obs_spec.name in ('max_stage', 'min_stage'):
      return tf.keras.layers.Embedding(7, 4)

    normalize_fn = log_normalize_fn = None
    if obs_spec.name not in get_nonnormalized_features():
      quantile = quantile_map[obs_spec.name]

      first_non_zero = 0
      for x in quantile:
        if x > 0:
          first_non_zero = x
          break

      normalize_fn = feature_ops.get_normalize_fn(quantile, with_sqrt,
                                                  with_z_score_normalization,
                                                  eps)
      log_normalize_fn = feature_ops.get_normalize_fn(
          quantile,
          with_sqrt,
          with_z_score_normalization,
          eps,
          preprocessing_fn=lambda x: tf.math.log(x + first_non_zero))

    if obs_spec.name in ['nr_rematerializable', 'nr_broken_hints']:
      return tf.keras.layers.Lambda(normalize_fn)

    if obs_spec.name in ('liverange_size', 'nr_defs_and_uses'
                        ) or obs_spec.name.endswith('by_max'):
      return tf.keras.layers.Lambda(log_normalize_fn)

    if obs_spec.name == 'use_def_density':

      def use_def_density_processing_fn(obs):
        features = [tf.where(tf.math.is_inf(tf.expand_dims(obs, -1)), 1.0, 0.0)]
        obs = tf.where(tf.math.is_inf(obs), 1.0, obs)
        features.append(log_normalize_fn(obs))
        # pylint: disable=unexpected-keyword-arg
        return tf.concat(features, axis=-1)

      return tf.keras.layers.Lambda(use_def_density_processing_fn)

    if obs_spec.name == 'progress':

      def progress_processing_fn(obs):
        obs = tf.expand_dims(obs, -1)
        obs = tf.tile(obs, [1, get_num_registers()])
        obs = normalize_fn(obs)
        return obs

      return tf.keras.layers.Lambda(progress_processing_fn)

    # Make sure all features have a preprocessing function.
    raise KeyError('Missing preprocessing function for some feature.')

  return observation_processing_layer


def get_nonnormalized_features():
  return [
      'mask', 'nr_urgent', 'is_hint', 'is_local', 'is_free', 'max_stage',
      'min_stage', 'reward'
  ]
