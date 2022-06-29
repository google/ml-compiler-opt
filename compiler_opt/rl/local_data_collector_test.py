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

import multiprocessing as mp
import string
import subprocess
from unittest import mock

import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import data_collector
from compiler_opt.rl import local_data_collector

# This is https://github.com/google/pytype/issues/764
from google.protobuf import text_format  # pytype: disable=pyi-error


def _get_sequence_example(feature_value):
  sequence_example_text = string.Template("""
    feature_lists {
      feature_list {
        key: "feature_0"
        value {
          feature { int64_list { value: $feature_value } }
          feature { int64_list { value: $feature_value } }
        }
      }
    }""").substitute(feature_value=feature_value)
  return text_format.Parse(sequence_example_text, tf.train.SequenceExample())


def mock_collect_data(file_paths, tf_policy_dir, reward_stat, _):
  assert file_paths == ('a', 'b')
  assert tf_policy_dir == 'policy'
  assert reward_stat is None or reward_stat == {
      'default':
          compilation_runner.RewardStat(
              default_reward=1, moving_average_reward=2)
  }
  if reward_stat is None:
    return compilation_runner.CompilationResult(
        sequence_examples=[_get_sequence_example(feature_value=1)],
        reward_stats={
            'default':
                compilation_runner.RewardStat(
                    default_reward=1, moving_average_reward=2)
        },
        rewards=[1.2],
        keys=['default'])
  else:
    return compilation_runner.CompilationResult(
        sequence_examples=[_get_sequence_example(feature_value=2)],
        reward_stats={
            'default':
                compilation_runner.RewardStat(
                    default_reward=1, moving_average_reward=3)
        },
        rewards=[3.4],
        keys=['default'])


class Sleeper(compilation_runner.CompilationRunner):
  """Test CompilationRunner that just sleeps."""

  # pylint: disable=super-init-not-called
  def __init__(self, killed, timedout, living):
    self._killed = killed
    self._timedout = timedout
    self._living = living
    self._lock = mp.Manager().Lock()

  def collect_data(self, file_paths, tf_policy_path, reward_stat,
                   cancellation_token):
    _ = reward_stat
    cancellation_manager = self._get_cancellation_manager(cancellation_token)
    try:
      compilation_runner.start_cancellable_process(['sleep', '3600s'], 3600,
                                                   cancellation_manager)
    except compilation_runner.ProcessKilledError as e:
      with self._lock:
        self._killed.value += 1
      raise e
    except subprocess.TimeoutExpired as e:
      with self._lock:
        self._timedout.value += 1
      raise e
    with self._lock:
      self._living.value += 1
    return compilation_runner.CompilationResult(
        sequence_examples=[], reward_stats={}, rewards=[], keys=[])


class LocalDataCollectorTest(tf.test.TestCase):

  def test_local_data_collector(self):
    mock_compilation_runner = mock.create_autospec(
        compilation_runner.CompilationRunner)

    mock_compilation_runner.collect_data = mock_collect_data

    def create_test_iterator_fn():

      def _test_iterator_fn(data_list):
        assert data_list in (
            [_get_sequence_example(feature_value=1).SerializeToString()] * 9,
            [_get_sequence_example(feature_value=2).SerializeToString()] * 9)
        if data_list == [
            _get_sequence_example(feature_value=1).SerializeToString()
        ] * 9:
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
    expected_monitor_dict_subset = {
        'default': {
            'success_modules': 9,
            'total_trajectory_length': 18,
        }
    }
    self.assertDictContainsSubset(expected_monitor_dict_subset, monitor_dict)

    data_iterator, monitor_dict = collector.collect_data(policy_path='policy')
    data = list(data_iterator)
    self.assertEqual([4, 5, 6], data)
    expected_monitor_dict_subset = {
        'default': {
            'success_modules': 9,
            'total_trajectory_length': 18,
        }
    }
    self.assertDictContainsSubset(expected_monitor_dict_subset, monitor_dict)

    collector.close_pool()

  def test_local_data_collector_task_management(self):
    killed = mp.Manager().Value('i', value=0)
    timedout = mp.Manager().Value('i', value=0)
    living = mp.Manager().Value('i', value=0)

    mock_compilation_runner = Sleeper(killed, timedout, living)

    def parser(_):
      pass

    class QuickExiter(data_collector.EarlyExitChecker):

      def __init__(self, num_modules):
        data_collector.EarlyExitChecker.__init__(self, num_modules=num_modules)

      def wait(self, _):
        return False

    collector = local_data_collector.LocalDataCollector(
        file_paths=tuple([('a', 'b')] * 200),
        num_workers=4,
        num_modules=4,
        runner=mock_compilation_runner,
        parser=parser,
        reward_stat_map=collections.defaultdict(lambda: None),
        exit_checker_ctor=QuickExiter)
    collector.collect_data(policy_path='policy')
    # close the pool first, so we are forced to wait for the workers to process
    # their cancellation.
    collector.close_pool()
    self.assertEqual(killed.value, 4)
    self.assertEqual(living.value, 0)
    self.assertEqual(timedout.value, 0)


if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
