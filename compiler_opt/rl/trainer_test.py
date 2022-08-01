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
"""Tests for compiler_opt.rl.trainer."""

# pylint: disable=protected-access

import tensorflow as tf
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.networks import q_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.trajectories import trajectory
from unittest import mock

from compiler_opt.rl import trainer


def _create_test_data(batch_size, sequence_length):
  test_trajectory = trajectory.Trajectory(
      step_type=tf.fill([batch_size, sequence_length], 1),
      observation={
          'inlining_default':
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


class TrainerTest(tf.test.TestCase):

  def setUp(self):
    observation_spec = {
        'inlining_default':
            tf.TensorSpec(dtype=tf.int64, shape=(), name='inlining_default')
    }
    self._time_step_spec = time_step.time_step_spec(observation_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int64,
        shape=(),
        minimum=0,
        maximum=1,
        name='inlining_decision')
    self._network = q_rnn_network.QRnnNetwork(
        input_tensor_spec=self._time_step_spec.observation,
        action_spec=self._action_spec,
        lstm_size=(40,),
        preprocessing_layers={
            'inlining_default': tf.keras.layers.Lambda(lambda x: x)
        })
    super().setUp()

  def test_trainer_initialization(self):
    test_agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        self._network,
        tf.compat.v1.train.AdamOptimizer(),
        num_outer_dims=2)
    test_trainer = trainer.Trainer(
        root_dir=self.get_temp_dir(), agent=test_agent)
    self.assertEqual(0, test_trainer._global_step.numpy())

  def test_training(self):
    test_agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        self._network,
        tf.compat.v1.train.AdamOptimizer(),
        num_outer_dims=2)
    test_trainer = trainer.Trainer(
        root_dir=self.get_temp_dir(),
        agent=test_agent,
        summary_log_interval=1,
        summary_export_interval=10)
    self.assertEqual(0, test_trainer._global_step.numpy())

    dataset_iter = _create_test_data(batch_size=3, sequence_length=3)
    monitor_dict = {'default': {'test': 1}}

    with mock.patch.object(
        tf.summary, 'scalar', autospec=True) as mock_scalar_summary:
      test_trainer.train(dataset_iter, monitor_dict, num_iterations=100)
      self.assertEqual(
          10,
          sum(1 for c in mock_scalar_summary.mock_calls
              if c[2]['name'] == 'test'))
      self.assertEqual(100, test_trainer._global_step.numpy())
      self.assertEqual(100, test_trainer.global_step_numpy())

  def test_training_with_multiple_times(self):
    test_agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        self._network,
        tf.compat.v1.train.AdamOptimizer(),
        num_outer_dims=2)
    test_trainer = trainer.Trainer(
        root_dir=self.get_temp_dir(), agent=test_agent)
    self.assertEqual(0, test_trainer._global_step.numpy())

    dataset_iter = _create_test_data(batch_size=3, sequence_length=3)
    monitor_dict = {'default': {'test': 1}}
    test_trainer.train(dataset_iter, monitor_dict, num_iterations=10)
    self.assertEqual(10, test_trainer._global_step.numpy())

    dataset_iter = _create_test_data(batch_size=6, sequence_length=4)
    test_trainer.train(dataset_iter, monitor_dict, num_iterations=10)
    self.assertEqual(20, test_trainer._global_step.numpy())

  def test_inference(self):
    test_agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec,
        self._action_spec,
        self._network,
        tf.compat.v1.train.AdamOptimizer(),
        num_outer_dims=2)
    test_trainer = trainer.Trainer(
        root_dir=self.get_temp_dir(), agent=test_agent)

    inference_batch_size = 1
    random_time_step = tensor_spec.sample_spec_nest(
        self._time_step_spec, outer_dims=(inference_batch_size,))

    initial_policy_state = test_trainer._agent.policy.get_initial_state(
        inference_batch_size)

    action_outputs = test_trainer._agent.policy.action(random_time_step,
                                                       initial_policy_state)
    self.assertAllEqual([inference_batch_size], action_outputs.action.shape)

    action_outputs = test_trainer._agent.policy.action(random_time_step,
                                                       action_outputs.state)
    self.assertAllEqual([inference_batch_size], action_outputs.action.shape)


if __name__ == '__main__':
  tf.test.main()
