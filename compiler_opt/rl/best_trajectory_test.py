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
"""Tests for compiler_opt.rl.best_trajectory."""

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf

from compiler_opt.rl import best_trajectory

_ACTION_NAME = 'mock'


def _get_test_repo_1():
  repo = best_trajectory.BestTrajectoryRepo(action_name=_ACTION_NAME)
  # pylint: disable=protected-access
  repo._best_trajectories['module_1'] = {
      'function_1':
          best_trajectory.BestTrajectory(reward=3.4, action_list=[1, 3, 5]),
      'function_2':
          best_trajectory.BestTrajectory(reward=1.2, action_list=[9, 7, 5])
  }
  # pylint: enable=protected-access
  return repo


def _get_test_repo_2():
  repo = best_trajectory.BestTrajectoryRepo(action_name=_ACTION_NAME)
  # pylint: disable=protected-access
  repo._best_trajectories['module_1'] = {
      'function_1':
          best_trajectory.BestTrajectory(reward=2.3, action_list=[1, 3]),
      'function_2':
          best_trajectory.BestTrajectory(reward=3.4, action_list=[9, 7])
  }
  repo._best_trajectories['module_2'] = {
      'function_1':
          best_trajectory.BestTrajectory(reward=7.8, action_list=[2, 4, 6]),
  }
  # pylint: enable=protected-access
  return repo


def _get_combined_repo():
  repo = best_trajectory.BestTrajectoryRepo(action_name=_ACTION_NAME)
  # pylint: disable=protected-access
  repo._best_trajectories['module_1'] = {
      'function_1':
          best_trajectory.BestTrajectory(reward=2.3, action_list=[1, 3]),
      'function_2':
          best_trajectory.BestTrajectory(reward=1.2, action_list=[9, 7, 5])
  }
  repo._best_trajectories['module_2'] = {
      'function_1':
          best_trajectory.BestTrajectory(reward=7.8, action_list=[2, 4, 6]),
  }
  # pylint: enable=protected-access
  return repo


def _create_sequence_example(action_list):
  example = tf.train.SequenceExample()
  for action in action_list:
    example.feature_lists.feature_list[_ACTION_NAME].feature.add(
    ).int64_list.value.append(action)
  return example.SerializeToString()


class BestTrajectoryTest(parameterized.TestCase):

  @parameterized.named_parameters(('repo_1', _get_test_repo_1()),
                                  ('repo_2', _get_test_repo_2()))
  def test_sink_load_json_file(self, repo):
    path = self.create_tempfile().full_path
    repo.sink_to_json_file(path)
    loaded_repo = best_trajectory.BestTrajectoryRepo(action_name=_ACTION_NAME)
    loaded_repo.load_from_json_file(path)
    self.assertDictEqual(repo.best_trajectories, loaded_repo.best_trajectories)

  def test_sink_to_csv_file(self):
    path = self.create_tempfile().full_path
    repo = _get_test_repo_1()
    repo.sink_to_csv_file(path)
    with open(path, 'r', encoding='utf-8') as f:
      text = f.read()

    self.assertEqual(text,
                     'module_1,function_1,1,3,5\nmodule_1,function_2,9,7,5\n')

  @parameterized.named_parameters(
      {
          'testcase_name': 'repo_1_combine_2',
          'base_repo': _get_test_repo_1(),
          'second_repo': _get_test_repo_2()
      }, {
          'testcase_name': 'repo_2_combine_1',
          'base_repo': _get_test_repo_2(),
          'second_repo': _get_test_repo_1()
      })
  def test_combine_with_other_repo(self, base_repo, second_repo):
    base_repo.combine_with_other_repo(second_repo)
    self.assertDictEqual(base_repo.best_trajectories,
                         _get_combined_repo().best_trajectories)

  def test_update_if_better_trajectory(self):
    repo = _get_test_repo_1()
    repo.update_if_better_trajectory(
        'module_1', 'function_1', 2.3,
        _create_sequence_example(action_list=[1, 3]))
    repo.update_if_better_trajectory(
        'module_1', 'function_2', 3.4,
        _create_sequence_example(action_list=[9, 7]))
    repo.update_if_better_trajectory(
        'module_2', 'function_1', 7.8,
        _create_sequence_example(action_list=[2, 4, 6]))
    self.assertDictEqual(repo.best_trajectories,
                         _get_combined_repo().best_trajectories)


if __name__ == '__main__':
  absltest.main()
