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
"""Tests for the combine_tfa_policies_lib.py module"""

from absl.testing import absltest

import tensorflow as tf
from compiler_opt.tools import combine_tfa_policies_lib
from tf_agents.trajectories import time_step as ts
import tf_agents
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.typing import types
import hashlib
import numpy as np


def client_side_model_selector_calculation(policy_name: str) -> types.Tensor:
  m = hashlib.md5()
  m.update(policy_name.encode('utf-8'))
  high = int.from_bytes(m.digest()[8:], 'little')
  low = int.from_bytes(m.digest()[:8], 'little')
  model_selector = tf.constant([[high, low]], dtype=tf.uint64)
  return model_selector


class AddOnePolicy(tf_agents.policies.TFPolicy):
  """Test policy which increments the obs feature."""

  def __init__(self):
    obs_spec = {'obs': tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)}
    time_step_spec = ts.time_step_spec(obs_spec)

    act_spec = tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)

    super().__init__(time_step_spec=time_step_spec, action_spec=act_spec)

  def _distribution(self, time_step):
    """Boilerplate function for TFPolicy."""
    pass

  def _variables(self):
    """Boilerplate function for TFPolicy."""
    return ()

  def _action(self, time_step, policy_state, seed):
    """Boilerplate function for TFPolicy."""
    observation = time_step.observation['obs'][0]
    action = tf.reshape(observation + 1, (1,))
    return policy_step.PolicyStep(action, policy_state)


class SubtractOnePolicy(tf_agents.policies.TFPolicy):
  """Test policy which decrements the obs feature."""

  def __init__(self):
    obs_spec = {'obs': tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)}
    time_step_spec = ts.time_step_spec(obs_spec)

    act_spec = tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)

    super().__init__(time_step_spec=time_step_spec, action_spec=act_spec)

  def _distribution(self, time_step):
    """Boilerplate function for TFPolicy."""
    pass

  def _variables(self):
    """Boilerplate function for TFPolicy."""
    return ()

  def _action(self, time_step, policy_state, seed):
    """Boilerplate function for TFPolicy."""
    observation = time_step.observation['obs'][0]
    action = tf.reshape(observation - 1, (1,))
    return policy_step.PolicyStep(action, policy_state)


observation_spec = ts.time_step_spec({
    'obs':
        tf.TensorSpec(dtype=tf.int32, shape=(), name='obs'),
    'model_selector':
        tf.TensorSpec(shape=(2,), dtype=tf.uint64, name='model_selector')
})

action_spec = tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64)


class CombinedTFPolicyTest(absltest.TestCase):
  """Test for CombinedTFPolicy."""

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

    model_selector = client_side_model_selector_calculation('add_one')

    state = ts.TimeStep(
        discount=tf.constant(np.array([0.]), dtype=tf.float32),
        observation={
            'obs': tf.constant(np.array([42]), dtype=tf.int64),
            'model_selector': model_selector
        },
        reward=tf.constant(np.array([0]), dtype=tf.float64),
        step_type=tf.constant(np.array([0]), dtype=tf.int64))

    self.assertEqual(
        combined_policy.action(state).action, tf.constant(43, dtype=tf.int64))

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

    model_selector = client_side_model_selector_calculation('subtract_one')

    state = ts.TimeStep(
        discount=tf.constant(np.array([0.]), dtype=tf.float32),
        observation={
            'obs': tf.constant(np.array([42]), dtype=tf.int64),
            'model_selector': model_selector
        },
        reward=tf.constant(np.array([0]), dtype=tf.float64),
        step_type=tf.constant(np.array([0]), dtype=tf.int64))

    self.assertEqual(
        combined_policy.action(state).action, tf.constant(41, dtype=tf.int64))
