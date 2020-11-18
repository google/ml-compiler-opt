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

import tensorflow as tf

from tf_agents.trajectories import trajectory

from compiler_opt.rl import config
from compiler_opt.rl import data_reader

_TEST_CONFIG = config.Config(
    feature_keys=(tf.TensorSpec(dtype=tf.int64, shape=(), name='feature_key'),),
    action_key=tf.TensorSpec(
        dtype=tf.int64, shape=(), name='inlining_decision'),
    reward_key=tf.TensorSpec(dtype=tf.float32, shape=(), name='reward'),
)


def _define_sequence_example(agent_name, inlining_decision):
  example = tf.train.SequenceExample()
  for _ in range(10):
    example.feature_lists.feature_list['feature_key'].feature.add(
    ).int64_list.value.append(1)
    example.feature_lists.feature_list['inlining_decision'].feature.add(
    ).int64_list.value.append(inlining_decision)
    example.feature_lists.feature_list['reward'].feature.add(
    ).float_list.value.append(2.3)
    if agent_name == 'ppo':
      example.feature_lists.feature_list[
          'CategoricalProjectionNetwork_logits'].feature.add(
          ).float_list.value.extend([1.2, 3.4])
  return example


class DataReaderTest(tf.test.TestCase):

  def setUp(self):
    self._agent_name = 'dqn'
    super(DataReaderTest, self).setUp()

  def test_create_iterator_fn(self):
    example = _define_sequence_example(self._agent_name, inlining_decision=0)
    sequence_examples = [example.SerializeToString() for _ in range(100)]

    iterator_fn = data_reader.create_sequence_example_iterator_fn(
        agent_name=self._agent_name,
        config=_TEST_CONFIG,
        batch_size=2,
        train_sequence_length=3)
    data_iterator = iterator_fn(sequence_examples)

    experience = next(data_iterator)
    self.assertIsInstance(experience, trajectory.Trajectory)
    self.assertAllEqual([2, 3], experience.step_type.shape)
    self.assertCountEqual(['feature_key'],
                          experience.observation.keys())
    self.assertAllEqual([[1, 1, 1], [1, 1, 1]],
                        experience.observation['feature_key'])
    self.assertAllEqual([[0, 0, 0], [0, 0, 0]], experience.action)
    self.assertEmpty(experience.policy_info)
    self.assertAllEqual([2, 3], experience.next_step_type.shape)
    self.assertAllClose([[2.3, 2.3, 2.3], [2.3, 2.3, 2.3]], experience.reward)
    self.assertAllEqual([[1, 1, 1], [1, 1, 1]], experience.discount)

  def test_ppo_policy_info(self):
    self._agent_name = 'ppo'

    example = _define_sequence_example(self._agent_name, inlining_decision=0)
    sequence_examples = [example.SerializeToString() for _ in range(100)]

    iterator_fn = data_reader.create_sequence_example_iterator_fn(
        agent_name=self._agent_name,
        config=_TEST_CONFIG,
        batch_size=2,
        train_sequence_length=3)
    data_iterator = iterator_fn(sequence_examples)

    experience = next(data_iterator)
    self.assertAllEqual(['feature_key'], list(experience.observation.keys()))
    self.assertAllClose([[[1.2, 3.4], [1.2, 3.4], [1.2, 3.4]],
                         [[1.2, 3.4], [1.2, 3.4], [1.2, 3.4]]],
                        experience.policy_info['dist_params']['logits'])

if __name__ == '__main__':
  tf.test.main()
