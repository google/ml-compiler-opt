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
"""Tests for compiler_opt.rl.data_reader."""

# pylint: disable=protected-access

import os

from absl.testing import parameterized
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.trajectories import trajectory

from compiler_opt.rl import agent_config
from compiler_opt.rl import data_reader


def _define_sequence_example(agent_config_type, is_action_discrete):
  example = tf.train.SequenceExample()
  for _ in range(10):
    example.feature_lists.feature_list['feature_key'].feature.add(
    ).int64_list.value.append(1)
    if is_action_discrete:
      example.feature_lists.feature_list['inlining_decision'].feature.add(
      ).int64_list.value.append(0)
    else:
      example.feature_lists.feature_list['live_interval_weight'].feature.add(
      ).float_list.value.append(1.23)
    example.feature_lists.feature_list['reward'].feature.add(
    ).float_list.value.append(2.3)
    if agent_config_type in (agent_config.PPOAgentConfig,
                             agent_config.DistributedPPOAgentConfig):
      if is_action_discrete:
        example.feature_lists.feature_list[
            'CategoricalProjectionNetwork_logits'].feature.add(
            ).float_list.value.extend([1.2, 3.4])
      else:
        example.feature_lists.feature_list[
            'NormalProjectionNetwork_scale'].feature.add(
            ).float_list.value.extend([1.2])
        example.feature_lists.feature_list[
            'NormalProjectionNetwork_loc'].feature.add(
            ).float_list.value.extend([3.4])
  return example


def _write_tmp_tfrecord(file_path, example, num_examples):
  with tf.io.TFRecordWriter(file_path) as file_writer:
    for _ in range(num_examples):
      file_writer.write(example.SerializeToString())


class DataReaderTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    observation_spec = {
        'feature_key':
            tf.TensorSpec(dtype=tf.int64, shape=(), name='feature_key')
    }
    reward_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='reward')
    self._time_step_spec = time_step.time_step_spec(observation_spec,
                                                    reward_spec)
    self._discrete_action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int64,
        shape=(),
        minimum=0,
        maximum=1,
        name='inlining_decision')
    self._continuous_action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.float32,
        shape=(),
        name='live_interval_weight',
        minimum=-100,
        maximum=20)
    super().setUp()

  def _create_sequence_example_datasource(self, example):
    return [example.SerializeToString()] * 100

  def _create_tfrecord_datasource(self, example):
    data_source = os.path.join(self.get_temp_dir(), 'data_tfrecord')
    _write_tmp_tfrecord(data_source, example, 100)
    return data_source

  _test_config = (('SequenceExampleDatasetFn',
                   data_reader.create_sequence_example_dataset_fn,
                   agent_config.PPOAgentConfig,
                   _create_sequence_example_datasource),
                  ('TFRecordDatasetFn', data_reader.create_tfrecord_dataset_fn,
                   agent_config.PPOAgentConfig, _create_tfrecord_datasource))

  @parameterized.named_parameters(*_test_config)
  def test_create_dataset_fn(self, test_fn, _, data_source_fn):
    agent_type_override = agent_config.DQNAgentConfig
    example = _define_sequence_example(
        agent_type_override, is_action_discrete=True)

    data_source = data_source_fn(self, example)
    dataset_fn = test_fn(
        agent_cfg=agent_type_override(
            time_step_spec=self._time_step_spec,
            action_spec=self._discrete_action_spec),
        batch_size=2,
        train_sequence_length=3)
    data_iterator = iter(dataset_fn(data_source).repeat())

    experience = next(data_iterator)
    self.assertIsInstance(experience, trajectory.Trajectory)
    self.assertAllEqual([2, 3], experience.step_type.shape)
    self.assertCountEqual(['feature_key'], experience.observation.keys())
    self.assertAllEqual([[1, 1, 1], [1, 1, 1]],
                        experience.observation['feature_key'])
    self.assertAllEqual([[0, 0, 0], [0, 0, 0]], experience.action)
    self.assertEmpty(experience.policy_info)
    self.assertAllEqual([2, 3], experience.next_step_type.shape)
    self.assertAllClose([[2.3, 2.3, 2.3], [2.3, 2.3, 2.3]], experience.reward)
    self.assertAllEqual([[1, 1, 1], [1, 1, 1]], experience.discount)

  _distrib_test_config = (('SequenceExampleDatasetFnDistributed',
                           data_reader.create_sequence_example_dataset_fn,
                           agent_config.DistributedPPOAgentConfig,
                           _create_sequence_example_datasource),
                          ('TFRecordDatasetFnDistributed',
                           data_reader.create_tfrecord_dataset_fn,
                           agent_config.DistributedPPOAgentConfig,
                           _create_tfrecord_datasource))

  @parameterized.named_parameters(*(_test_config + _distrib_test_config))
  def test_ppo_policy_info_discrete(self, test_fn, agent_config_type,
                                    data_source_fn):
    example = _define_sequence_example(
        agent_config_type, is_action_discrete=True)

    data_source = data_source_fn(self, example)

    dataset_fn = test_fn(
        agent_cfg=agent_config_type(
            time_step_spec=self._time_step_spec,
            action_spec=self._discrete_action_spec),
        batch_size=2,
        train_sequence_length=3)
    data_iterator = iter(dataset_fn(data_source).repeat())

    experience = next(data_iterator)
    self.assertAllEqual(['feature_key'], list(experience.observation.keys()))
    self.assertAllClose([[[1.2, 3.4], [1.2, 3.4], [1.2, 3.4]],
                         [[1.2, 3.4], [1.2, 3.4], [1.2, 3.4]]],
                        experience.policy_info['dist_params']['logits'])

  @parameterized.named_parameters(*(_test_config + _distrib_test_config))
  def test_ppo_policy_info_continuous(self, test_fn, agent_config_type,
                                      data_source_fn):
    example = _define_sequence_example(
        agent_config_type, is_action_discrete=False)

    data_source = data_source_fn(self, example)

    dataset_fn = test_fn(
        agent_cfg=agent_config_type(
            time_step_spec=self._time_step_spec,
            action_spec=self._continuous_action_spec),
        batch_size=2,
        train_sequence_length=3)
    data_iterator = iter(dataset_fn(data_source).repeat())

    experience = next(data_iterator)
    self.assertAllEqual(['feature_key'], list(experience.observation.keys()))
    self.assertAllClose([[1.23, 1.23, 1.23], [1.23, 1.23, 1.23]],
                        experience.action)
    self.assertAllClose([[1.2, 1.2, 1.2], [1.2, 1.2, 1.2]],
                        experience.policy_info['dist_params']['scale'])
    self.assertAllClose([[3.4, 3.4, 3.4], [3.4, 3.4, 3.4]],
                        experience.policy_info['dist_params']['loc'])


if __name__ == '__main__':
  tf.test.main()
