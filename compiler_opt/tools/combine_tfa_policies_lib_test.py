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
"""Tests for the feature_importance_utils.py module"""

from absl.testing import absltest

import tensorflow as tf
from compiler_opt.tools import combine_tfa_policies_lib
from tf_agents.trajectories import time_step
import tf_agents
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
import hashlib
import numpy as np


class AddOnePolicy(tf_agents.policies.TFPolicy):

  def __init__(self):
    observation_spec = {
        'obs': tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)
    }
    time_step_spec = time_step.time_step_spec(observation_spec)

    action_spec = tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)

    super(AddOnePolicy, self).__init__(
        time_step_spec=time_step_spec, action_spec=action_spec)

  def _distribution(self, time_step):
    pass

  def _variables(self):
    return ()

  def _action(self, time_step, policy_state, seed):
    observation = time_step.observation['obs'][0]
    action = tf.reshape(observation + 1, (1,))
    return policy_step.PolicyStep(action, policy_state)


class SubtractOnePolicy(tf_agents.policies.TFPolicy):

  def __init__(self):
    observation_spec = {
        'obs': tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)
    }
    time_step_spec = time_step.time_step_spec(observation_spec)

    action_spec = tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)

    super(SubtractOnePolicy, self).__init__(
        time_step_spec=time_step_spec, action_spec=action_spec)

  def _distribution(self, time_step):
    pass

  def _variables(self):
    return ()

  def _action(self, time_step, policy_state, seed):
    observation = time_step.observation['obs'][0]
    action = tf.reshape(observation - 1, (1,))
    return policy_step.PolicyStep(action, policy_state)


observation_spec = time_step.time_step_spec({
    'obs':
        tf.TensorSpec(dtype=tf.int32, shape=(), name='obs'),
    'model_selector':
        tf.TensorSpec(shape=(2,), dtype=tf.uint64, name='model_selector')
})

action_spec = tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)


class FeatureImportanceTest(absltest.TestCase):

  def test_select_add_policy(self):
    policy1 = AddOnePolicy()
    policy2 = SubtractOnePolicy()
    combined_policy = combine_tfa_policies_lib.CombinedTFPolicy(
        tf_policies={
            'add_one': policy1,
            'subtract_one': policy2
        },
        time_step_spec=observation_spec,
        action_spec=action_spec)

    m = hashlib.md5()
    m.update('add_one'.encode('utf-8'))
    high = int.from_bytes(m.digest()[8:], 'little')
    low = int.from_bytes(m.digest()[:8], 'little')
    model_selector = tf.constant([[high, low]], dtype=tf.uint64)

    state = tf_agents.trajectories.TimeStep(
        discount=tf.constant(np.array([0.]), dtype=tf.float32),
        observation={
            'obs': tf.constant(np.array([0]), dtype=tf.int64),
            'model_selector': model_selector
        },
        reward=tf.constant(np.array([0]), dtype=tf.float64),
        step_type=tf.constant(np.array([0]), dtype=tf.int64))

    self.assertEqual(
        combined_policy.action(state).action, tf.constant(1, dtype=tf.int64))

  def test_select_subtract_policy(self):
    policy1 = AddOnePolicy()
    policy2 = SubtractOnePolicy()
    combined_policy = combine_tfa_policies_lib.CombinedTFPolicy(
        tf_policies={
            'add_one': policy1,
            'subtract_one': policy2
        },
        time_step_spec=observation_spec,
        action_spec=action_spec)

    m = hashlib.md5()
    m.update('subtract_one'.encode('utf-8'))
    high = int.from_bytes(m.digest()[8:], 'little')
    low = int.from_bytes(m.digest()[:8], 'little')
    model_selector = tf.constant([[high, low]], dtype=tf.uint64)

    state = tf_agents.trajectories.TimeStep(
        discount=tf.constant(np.array([0.]), dtype=tf.float32),
        observation={
            'obs': tf.constant(np.array([0]), dtype=tf.int64),
            'model_selector': model_selector
        },
        reward=tf.constant(np.array([0]), dtype=tf.float64),
        step_type=tf.constant(np.array([0]), dtype=tf.int64))

    self.assertEqual(
        combined_policy.action(state).action, tf.constant(-1, dtype=tf.int64))
