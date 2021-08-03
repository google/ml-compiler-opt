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
