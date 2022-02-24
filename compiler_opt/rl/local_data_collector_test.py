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

"""Tests for compiler_opt.rl.local_data_collector."""

import collections
import time
from unittest import mock

import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import local_data_collector


class LocalDataCollectorTest(tf.test.TestCase):

  def test_local_data_collector(self):
    mock_compilation_runner = mock.create_autospec(
        compilation_runner.CompilationRunner)

    def mock_collect_data(file_paths, tf_policy_dir, reward_stat):
      assert file_paths == ('a', 'b')
      assert tf_policy_dir == 'policy'
      assert reward_stat is None or reward_stat == {
          'default':
              compilation_runner.RewardStat(
                  default_reward=1, moving_average_reward=2)
      }
      if reward_stat is None:
        return [1], {
            'default':
                compilation_runner.RewardStat(
                    default_reward=1, moving_average_reward=2)
        }, [1.2]
      else:
        return [2], {
            'default':
                compilation_runner.RewardStat(
                    default_reward=1, moving_average_reward=3)
        }, [3.4]

    mock_compilation_runner.collect_data = mock_collect_data

    def create_test_iterator_fn():
      def _test_iterator_fn(data_list):
        assert data_list in ([1] * 9, [2] * 9)
        if data_list == [1] * 9:
          return iter(tf.data.Dataset.from_tensor_slices([1, 2, 3]))
        else:
          return iter(tf.data.Dataset.from_tensor_slices([4, 5, 6]))

      return _test_iterator_fn

    collector = local_data_collector.LocalDataCollector(
        file_paths=tuple([('a', 'b')] * 100),
        num_workers=4,
        num_modules=9,
        runner=mock_compilation_runner,
        parser=create_test_iterator_fn(),
        reward_stat_map=collections.defaultdict(lambda: None))

    data_iterator, monitor_dict = collector.collect_data(policy_path='policy')
    data = list(data_iterator)
    self.assertEqual([1, 2, 3], data)
    expected_monitor_dict_subset = {'default': {'success_modules': 9}}
    self.assertDictContainsSubset(expected_monitor_dict_subset, monitor_dict)

    data_iterator, monitor_dict = collector.collect_data(policy_path='policy')
    data = list(data_iterator)
    self.assertEqual([4, 5, 6], data)
    expected_monitor_dict_subset = {'default': {'success_modules': 9}}
    self.assertDictContainsSubset(expected_monitor_dict_subset, monitor_dict)

    collector.close_pool()

  def test_local_data_collector_task_management(self):

    class OverloadHandler:

      def __init__(self):
        self.counts = []

      def reset(self):
        self.counts.clear()

      def handler(self, count):
        self.counts.append(count)

    mock_compilation_runner = mock.create_autospec(
        compilation_runner.CompilationRunner)

    def mock_collect_data(file_path, *_):
      _, t = file_path.split('_')
      # avoid lint warnings
      t = int(t)
      time.sleep(t)
      return file_path, {}, [float(t)]

    mock_compilation_runner.collect_data = mock_collect_data

    def parser(data_list):
      assert data_list

    overload_handler = OverloadHandler()
    # Set the max_unfinished_tasks so we may schedule first some very long
    # running work that occupies some, but not all the worker processes of the
    # pool. This ensures there are workers able to pick up the short-running
    # work and clear it.
    collector = local_data_collector.LocalDataCollector(
        file_paths=['wait_0' for _ in range(0, 200)],
        num_workers=4,
        num_modules=4,
        runner=mock_compilation_runner,
        parser=parser,
        reward_stat_map=collections.defaultdict(lambda: None),
        max_unfinished_tasks=2,
        overload_handler=overload_handler.handler)  # pytype: disable=wrong-arg-types

    collector.collect_data(policy_path='policy')
    while [r for _, r in collector._unfinished_work if not r.ready()]:
      time.sleep(1)

    collector.inject_unfinished_work_for_test([
        ('policy', r) for r in collector._schedule_jobs(
            'policy', ['wait_5', 'wait_5', 'wait_10'])
    ])
    collector.collect_data(policy_path='policy')
    self.assertNotEmpty(overload_handler.counts)
    # We set the overload threshold (_max_unfinished_tasks) at 2, so the
    # overload handler should have seen a '3' after the short running tasks have
    # cleared.
    self.assertIn(3, overload_handler.counts)
    # The really long running task would not have cleared yet.
    self.assertLen(
        [r for _, r in collector.unfinished_work if not r.ready()], 1)
    collector.close_pool()

if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
