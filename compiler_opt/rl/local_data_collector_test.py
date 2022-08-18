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

# pylint: disable=protected-access
import collections

import string
import sys

import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.distributed.local.local_worker_manager import LocalWorkerPool
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
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


def mock_collect_data(module_spec, tf_policy_dir, reward_stat):
  assert module_spec.name == 'dummy'
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

  def collect_data(self, module_spec, tf_policy_path, reward_stat):
    _ = module_spec, tf_policy_path, reward_stat
    compilation_runner.start_cancellable_process(['sleep', '3600s'], 3600,
                                                 self._cancellation_manager)

    return compilation_runner.CompilationResult(
        sequence_examples=[], reward_stats={}, rewards=[], keys=[])


class MyRunner(compilation_runner.CompilationRunner):

  def collect_data(self, *args, **kwargs):
    return mock_collect_data(*args, **kwargs)


class LocalDataCollectorTest(tf.test.TestCase):

  def test_local_data_collector(self):

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

    with LocalWorkerPool(worker_class=MyRunner, count=4) as lwp:
      collector = local_data_collector.LocalDataCollector(
          cps=corpus.Corpus.from_module_specs(
              module_specs=[corpus.ModuleSpec(name='dummy')] * 100),
          num_modules=9,
          worker_pool=lwp,
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
      # Issue #38
      if sys.version_info.minor >= 9:
        self.assertEqual(monitor_dict,
                         monitor_dict | expected_monitor_dict_subset)
      else:
        self.assertEqual(monitor_dict, {
            **monitor_dict,
            **expected_monitor_dict_subset
        })

      data_iterator, monitor_dict = collector.collect_data(policy_path='policy')
      data = list(data_iterator)
      self.assertEqual([4, 5, 6], data)
      expected_monitor_dict_subset = {
          'default': {
              'success_modules': 9,
              'total_trajectory_length': 18,
          }
      }
      # Issue #38
      if sys.version_info.minor >= 9:
        self.assertEqual(monitor_dict,
                         monitor_dict | expected_monitor_dict_subset)
      else:
        self.assertEqual(monitor_dict, {
            **monitor_dict,
            **expected_monitor_dict_subset
        })

      collector.close_pool()

  def test_local_data_collector_task_management(self):

    def parser(_):
      pass

    class QuickExiter(data_collector.EarlyExitChecker):

      def __init__(self, num_modules):
        data_collector.EarlyExitChecker.__init__(self, num_modules=num_modules)

      def wait(self, _):
        return False

    with LocalWorkerPool(worker_class=Sleeper, count=4) as lwp:
      collector = local_data_collector.LocalDataCollector(
          cps=corpus.Corpus.from_module_specs(
              module_specs=[corpus.ModuleSpec(name='dummy')] * 200),
          num_modules=4,
          worker_pool=lwp,
          parser=parser,
          reward_stat_map=collections.defaultdict(lambda: None),
          exit_checker_ctor=QuickExiter)
      collector.collect_data(policy_path='policy')
      collector._join_pending_jobs()
      killed = 0
      for w in collector._current_futures:
        self.assertRaises(compilation_runner.ProcessKilledError, w.result)
        killed += 1
      self.assertEqual(killed, 4)
      collector.close_pool()


if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
