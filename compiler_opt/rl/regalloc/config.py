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


# pylint: disable=g-complex-comprehension
@gin.configurable()
def get_regalloc_signature_spec():
  """Returns (time_step_spec, action_spec) for LLVM register allocation."""
  observation_spec = dict(
      (key, tf.TensorSpec(dtype=tf.int64, shape=(), name=key))
      for key in ('is_local_split', 'nr_defs_and_uses', 'nr_implicit_defs',
                  'nr_identity_copies', 'liverange_size',
                  'is_rematerializable'))
  observation_spec.update(
      dict((key, tf.TensorSpec(dtype=tf.float32, shape=(), name=key))
           for key in ('weighed_reads', 'weighed_writes', 'weighed_indvars',
                       'hint_weights', 'start_bb_freq', 'end_bb_freq',
                       'hottest_bb_freq', 'weighed_read_writes')))
  reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
  time_step_spec = time_step.time_step_spec(observation_spec, reward_spec)
  action_spec = tensor_spec.BoundedTensorSpec(
      dtype=tf.float32,
      shape=(),
      name='live_interval_weight',
      minimum=-100,
      maximum=20)

  return time_step_spec, action_spec
