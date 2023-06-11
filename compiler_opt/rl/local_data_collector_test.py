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
from typing import List, Tuple

import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing

# This is https://github.com/google/pytype/issues/764
from google.protobuf import text_format  # pytype: disable=pyi-error
from compiler_opt.distributed.local.local_worker_manager import LocalWorkerPoolManager
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import data_collector
from compiler_opt.rl import local_data_collector
from compiler_opt.rl import policy_saver

_policy_str = 'policy'.encode(encoding='utf-8')

_mock_policy = policy_saver.Policy(output_spec=bytes(), policy=_policy_str)


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


def mock_collect_data(loaded_module_spec: corpus.LoadedModuleSpec, policy,
                      reward_stat, model_id):
  assert loaded_module_spec.name.startswith('dummy')
  assert policy.policy == _policy_str
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
        policy_rewards=[36],
        keys=['default'],
        model_id=model_id)
  else:
    return compilation_runner.CompilationResult(
        sequence_examples=[_get_sequence_example(feature_value=2)],
        reward_stats={
            'default':
                compilation_runner.RewardStat(
                    default_reward=1, moving_average_reward=3)
        },
        rewards=[3.4],
        policy_rewards=[18],
        keys=['default'],
        model_id=model_id)


class Sleeper(compilation_runner.CompilationRunner):
  """Test CompilationRunner that just sleeps."""

  def collect_data(self,
                   loaded_module_spec,
                   policy=None,
                   reward_stat=None,
                   model_id=None):
    _ = loaded_module_spec, policy, reward_stat
    compilation_runner.start_cancellable_process(['sleep', '3600s'], 3600,
                                                 self._cancellation_manager)

    return compilation_runner.CompilationResult(
        sequence_examples=[],
        reward_stats={},
        rewards=[],
        policy_rewards=[],
        keys=[],
        model_id=model_id)


class MyRunner(compilation_runner.CompilationRunner):

  def collect_data(self, *args, **kwargs):
    return mock_collect_data(*args, **kwargs)


class DeterministicSampler(corpus.Sampler):
  """A corpus sampler that returns modules in order, and can also be reset."""

  def __init__(self, module_specs: Tuple[corpus.ModuleSpec]):
    super().__init__(module_specs)
    self._cur_pos = 0

  def __call__(self, k: int, n: int = 20) -> List[corpus.ModuleSpec]:
    ret = []
    for _ in range(k):
      ret.append(self._module_specs[self._cur_pos % len(self._module_specs)])
      self._cur_pos += 1
    return ret

  def reset(self):
    self._cur_pos = 0


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

    with LocalWorkerPoolManager(worker_class=MyRunner, count=4) as lwp:
      cps = corpus.create_corpus_for_testing(
          location=self.create_tempdir(),
          elements=[
              corpus.ModuleSpec(name=f'dummy{i}', size=i) for i in range(100)
          ],
          sampler_type=DeterministicSampler)
      collector = local_data_collector.LocalDataCollector(
          cps=cps,
          num_modules=9,
          worker_pool=lwp,
          parser=create_test_iterator_fn(),
          reward_stat_map=collections.defaultdict(lambda: None),
          best_trajectory_repo=None)

      # reset the sampler, so the next time we collect, we collect the same
      # modules. We do it before the collect_data call, because that's when
      # we'll re-sample to prefetch the next batch.
      cps.reset()

      data_iterator, monitor_dict = collector.collect_data(
          policy=_mock_policy, model_id=0)
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
      data_iterator, monitor_dict = collector.collect_data(
          policy=_mock_policy, model_id=0)
      data = list(data_iterator)
      # because we reset the sampler, these are the same modules
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

    with LocalWorkerPoolManager(worker_class=Sleeper, count=4) as lwp:
      collector = local_data_collector.LocalDataCollector(
          cps=corpus.create_corpus_for_testing(
              location=self.create_tempdir(),
              elements=[
                  corpus.ModuleSpec(name=f'dummy{i}', size=1)
                  for i in range(200)
              ]),
          num_modules=4,
          worker_pool=lwp,
          parser=parser,
          reward_stat_map=collections.defaultdict(lambda: None),
          best_trajectory_repo=None,
          exit_checker_ctor=QuickExiter)
      collector.collect_data(policy=_mock_policy, model_id=0)
      collector._join_pending_jobs()
      killed = 0
      for w in collector._current_futures:
        self.assertRaises(compilation_runner.ProcessKilledError, w.result)
        killed += 1
      self.assertEqual(killed, 4)
      collector.close_pool()


if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
