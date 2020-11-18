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

"""Training config."""

import collections

import tensorflow as tf

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step


Config = collections.namedtuple("Config",
                                ("feature_keys", "action_key", "reward_key"))

# pylint: disable=g-complex-comprehension
CONFIG = Config(
    feature_keys=tuple(
        tf.TensorSpec(dtype=tf.int64, shape=(), name=name) for name in (
            "caller_basic_block_count",
            "caller_conditionally_executed_blocks",
            "caller_users",
            "callee_basic_block_count",
            "callee_conditionally_executed_blocks",
            "callee_users",
            "nr_ctant_params",
            "node_count",
            "edge_count",
            "callsite_height",
            "cost_estimate",
            # inlining_default is not used as feature in training.
            "inlining_default",
        )),
    action_key=tf.TensorSpec(
        dtype=tf.int64, shape=(), name="inlining_decision"),
    reward_key=tf.TensorSpec(dtype=tf.float32, shape=(), name="reward"),
)
# pylint: enable=g-complex-comprehension


def create_signature_specs(config):
  """Returns (time_step_spec, action_spec) for LLVM inlining."""

  observation_spec = dict(
      (key.name, tf.TensorSpec(dtype=key.dtype, shape=key.shape, name=key.name))
      for key in config.feature_keys)

  time_step_spec = time_step.time_step_spec(observation_spec)
  action_spec = tensor_spec.BoundedTensorSpec(
      shape=config.action_key.shape,
      dtype=config.action_key.dtype,
      minimum=0,
      maximum=1,
      name=config.action_key.name)
  return (time_step_spec, action_spec)
