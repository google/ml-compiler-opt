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
"""Tests for compiler_opt.rl.generate_bc_trajectories."""

from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step

from compiler_opt.rl import generate_bc_trajectories

_eps = 1e-5


def _get_state_list() -> List[time_step.TimeStep]:

  state_0 = time_step.TimeStep(
      discount=tf.constant(np.array([0.]), dtype=tf.float32),
      observation={
          'feature_1': tf.constant(np.array([0]), dtype=tf.int64),
          'feature_2': tf.constant(np.array([50]), dtype=tf.int64),
          'feature_3': tf.constant(np.array([0]), dtype=tf.int64),
      },
      reward=tf.constant(np.array([0]), dtype=tf.float32),
      step_type=tf.constant(np.array([0]), dtype=tf.int32))
  state_1 = time_step.TimeStep(
      discount=tf.constant(np.array([0.]), dtype=tf.float32),
      observation={
          'feature_1': tf.constant(np.array([1]), dtype=tf.int64),
          'feature_2': tf.constant(np.array([25]), dtype=tf.int64),
          'feature_3': tf.constant(np.array([0]), dtype=tf.int64),
      },
      reward=tf.constant(np.array([0]), dtype=tf.float32),
      step_type=tf.constant(np.array([0]), dtype=tf.int32))
  state_2 = time_step.TimeStep(
      discount=tf.constant(np.array([0.]), dtype=tf.float32),
      observation={
          'feature_1': tf.constant(np.array([0]), dtype=tf.int64),
          'feature_2': tf.constant(np.array([25]), dtype=tf.int64),
          'feature_3': tf.constant(np.array([1]), dtype=tf.int64),
      },
      reward=tf.constant(np.array([0]), dtype=tf.float32),
      step_type=tf.constant(np.array([0]), dtype=tf.int32))
  state_3 = time_step.TimeStep(
      discount=tf.constant(np.array([0.]), dtype=tf.float32),
      observation={
          'feature_1': tf.constant(np.array([0]), dtype=tf.int64),
          'feature_2': tf.constant(np.array([25]), dtype=tf.int64),
          'feature_3': tf.constant(np.array([0]), dtype=tf.int64),
      },
      reward=tf.constant(np.array([0]), dtype=tf.float32),
      step_type=tf.constant(np.array([0]), dtype=tf.int32))

  return [state_0, state_1, state_2, state_3]


def _policy(state: time_step.TimeStep) -> np.ndarray:
  feature_sum = np.array([0])
  for feature in state.observation.values():
    feature_sum += feature.numpy()
  return np.mod(feature_sum, 5)


def _explore_policy(state: time_step.TimeStep) -> policy_step.PolicyStep:
  probs = [
      0.5 * float(state.observation['feature_3'].numpy()),
      1 - 0.5 * float(state.observation['feature_3'].numpy())
  ]
  logits = [[0.0, tf.math.log(probs[1] / (1.0 - probs[1] + _eps))]]
  return policy_step.PolicyStep(
      action=tfp.distributions.Categorical(logits=logits))


class ExplorationWithPolicyTest(tf.test.TestCase):

  def test_explore_policy(self):
    prob = 1.
    state = _get_state_list()[3]
    logits = [[0.0, tf.math.log(prob / (1.0 - prob + _eps))]]
    action = tfp.distributions.Categorical(logits=logits)
    self.assertAllClose(action.logits, _explore_policy(state).action.logits)

  def test_explore_with_gap(self):
    explore_with_policy = generate_bc_trajectories.ExplorationWithPolicy(
        replay_prefix=[np.array([1])],
        policy=_policy,
        explore_policy=_explore_policy,
    )
    for state in _get_state_list():
      _ = explore_with_policy.get_advice(state)[0]

    self.assertAllClose(0, explore_with_policy.gap, atol=2 * _eps)
    self.assertEqual(2, explore_with_policy.explore_step)

    explore_with_policy = generate_bc_trajectories.ExplorationWithPolicy(
        replay_prefix=[np.array([1]),
                       np.array([1]),
                       np.array([1])],
        policy=_policy,
        explore_policy=_explore_policy,
    )
    for state in _get_state_list():
      _ = explore_with_policy.get_advice(state)[0]

    self.assertAllClose(1, explore_with_policy.gap, atol=2 * _eps)
    self.assertEqual(3, explore_with_policy.explore_step)

  def test_explore_with_feature(self):

    def explore_on_feature_1_val(feature_val):
      return feature_val.numpy()[0] > 0

    def explore_on_feature_2_val(feature_val):
      return feature_val.numpy()[0] > 25

    explore_on_features = {
        'feature_1': explore_on_feature_1_val,
        'feature_2': explore_on_feature_2_val
    }

    explore_with_policy = generate_bc_trajectories.ExplorationWithPolicy(
        replay_prefix=[],
        policy=_policy,
        explore_policy=_explore_policy,
        explore_on_features=explore_on_features)
    for state in _get_state_list():
      _ = explore_with_policy.get_advice(state)[0]
    self.assertEqual(0, explore_with_policy.explore_step)

    explore_with_policy = generate_bc_trajectories.ExplorationWithPolicy(
        replay_prefix=[np.array([1])],
        policy=_policy,
        explore_policy=_explore_policy,
        explore_on_features=explore_on_features,
    )

    for state in _get_state_list():
      _ = explore_with_policy.get_advice(state)[0]
    self.assertEqual(1, explore_with_policy.explore_step)
