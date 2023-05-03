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
"""Inlining Training config."""

import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from compiler_opt.rl import feature_ops


# pylint: disable=g-complex-comprehension
@gin.configurable()
def get_inlining_signature_spec():
  """Returns (time_step_spec, action_spec) for LLVM inlining."""
  observation_spec = dict(
      (key, tf.TensorSpec(dtype=tf.int64, shape=(), name=key)) for key in (
          # Base features
          'caller_basic_block_count',
          'caller_conditionally_executed_blocks',
          'caller_users',
          'callee_basic_block_count',
          'callee_conditionally_executed_blocks',
          'callee_users',
          'nr_ctant_params',
          'node_count',
          'edge_count',
          'callsite_height',
          'cost_estimate',

          # Expanded cost features
          'sroa_savings',
          'sroa_losses',
          'load_elimination',
          'call_penalty',
          'call_argument_setup',
          'load_relative_intrinsic',
          'lowered_call_arg_setup',
          'indirect_call_penalty',
          'jump_table_penalty',
          'case_cluster_penalty',
          'switch_penalty',
          'unsimplified_common_instructions',
          'num_loops',
          'dead_blocks',
          'simplified_instructions',
          'constant_args',
          'constant_offset_ptr_args',
          'callsite_cost',
          'cold_cc_penalty',
          'last_call_to_static_bonus',
          'is_multiple_blocks',
          'nested_inlines',
          'nested_inline_cost_estimate',
          'threshold',

          # inlining_default is not used as feature in training.
          'inlining_default'))
  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)
  action_spec = tensor_spec.BoundedTensorSpec(
      dtype=tf.int64, shape=(), name='inlining_decision', minimum=0, maximum=1)

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
    if obs_spec.name == 'inlining_default':
      return tf.keras.layers.Lambda(feature_ops.discard_fn)

    quantile = quantile_map[obs_spec.name]
    return tf.keras.layers.Lambda(
        feature_ops.get_normalize_fn(quantile, with_sqrt,
                                     with_z_score_normalization, eps))

  return observation_processing_layer


def get_nonnormalized_features():
  return ['reward', 'inlining_default', 'inlining_decision']
