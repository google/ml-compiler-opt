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
"""Tests for random_network_distillation."""

from absl.testing import parameterized
import tensorflow as tf
from tf_agents.networks import encoding_network
from tf_agents.networks import expand_dims_layer
from tf_agents.trajectories import time_step
from tf_agents.trajectories import trajectory

from compiler_opt.rl import random_net_distillation


def _processing_layer_creator():
  expand_dims_op = expand_dims_layer.ExpandDims(-1)

  def observation_processing_layer(_):
    """Creates the toy layer to process observation."""

    def discard_feature(obs):
      expanded_obs = expand_dims_op(obs)
      return tf.ones_like(expanded_obs, dtype=tf.float32)

    func = discard_feature
    return tf.keras.layers.Lambda(func)

  return observation_processing_layer


def _create_test_data(batch_size, sequence_length):
  test_trajectory = trajectory.Trajectory(
      step_type=tf.fill([batch_size, sequence_length], 1),
      observation={
          'edge_count':
              tf.fill([batch_size, sequence_length],
                      tf.constant(10, dtype=tf.int64))
      },
      action=tf.fill([batch_size, sequence_length],
                     tf.constant(1, dtype=tf.int64)),
      policy_info=(),
      next_step_type=tf.fill([batch_size, sequence_length], 1),
      reward=tf.fill([batch_size, sequence_length], 2.0),
      discount=tf.fill([batch_size, sequence_length], 1.0),
  )

  def test_data_iterator():
    while True:
      yield test_trajectory

  return test_data_iterator()


class RandomNetworkDistillationTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._update_frequency = 1
    observation_spec = {
        'edge_count': tf.TensorSpec(
            dtype=tf.int64, shape=(), name='edge_count')
    }
    self._time_step_spec = time_step.time_step_spec(observation_spec)

  def test_train_on_batch(self):
    data_iterator = _create_test_data(batch_size=3, sequence_length=3)

    # initialize the random_network_distillation instance
    random_network_distillation = \
        random_net_distillation.RandomNetworkDistillation(
          time_step_spec=self._time_step_spec,
          preprocessing_layer_creator=_processing_layer_creator(),
          encoding_network=encoding_network.EncodingNetwork,
          update_frequency=self._update_frequency)

    experience = next(data_iterator)
    # test the RND train function return type
    for _ in range(5):
      new_experience = random_network_distillation.train(experience)
    self.assertIsInstance(new_experience, trajectory.Trajectory)
    # the rest of experience should remain the same except reward
    self.assertAllEqual(experience.step_type, new_experience.step_type)
    self.assertAllEqual(experience.observation, new_experience.observation)
    self.assertAllEqual(experience.action, new_experience.action)
    self.assertAllEqual(experience.policy_info, new_experience.policy_info)
    self.assertAllEqual(experience.next_step_type, experience.next_step_type)
    self.assertAllEqual(experience.discount, new_experience.discount)
    # reward should have same shape
    self.assertAllEqual(experience.reward.shape, new_experience.reward.shape)
    # new reward should has finite value
    self.assertFalse(tf.math.is_inf(tf.reduce_sum(new_experience.reward)))


if __name__ == '__main__':
  tf.test.main()
