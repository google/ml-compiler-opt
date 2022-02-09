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

"""Module for collecting data locally."""

import collections
import itertools
import random
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from absl import logging
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.trajectories import trajectory

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import data_collector


# How much work is allowed to be unfinished, relative to number of tasks
# requested in a data collection session, before we stop accepting new work and
# start waiting for it to drain.
_UNFINISHED_WORK_RATIO = 0.5


def default_overload_handler(total_unfinished_work):
  logging.warn('Too much unfinished work: %d unfinished!',
               total_unfinished_work)


class LocalDataCollector(data_collector.DataCollector):
  """class for local data collection."""

  def __init__(self,
               file_paths: Tuple[Tuple[str, ...], ...],
               num_workers: int,
               num_modules: int,
               runner: compilation_runner.CompilationRunner,
               parser: Callable[[List[str]], Iterator[trajectory.Trajectory]],
               use_stale_results: bool = False,
               max_unfinished_tasks: Optional[int] = None,
               overload_handler: Callable[[int],
                                          None] = default_overload_handler):
    super(LocalDataCollector, self).__init__()

    self._file_paths = file_paths
    self._num_modules = num_modules
    self._runner = runner
    self._parser = parser

    self._unfinished_work = []
    self._pool = multiprocessing.get_context('spawn').Pool(num_workers)

    self._max_unfinished_tasks = max_unfinished_tasks
    if not self._max_unfinished_tasks:
      self._max_unfinished_tasks = _UNFINISHED_WORK_RATIO * num_modules
    self._use_stale_results = use_stale_results

    self._reward_stat_map = collections.defaultdict(lambda: None)
    self._overloaded_workers_handler = overload_handler

  def close_pool(self):
    if self._pool:
      # Stop accepting new work
      self._pool.close()
      self._pool.join()
      self._pool = None

  def inject_unfinished_work_for_test(self, work):
    self._unfinished_work = work

  @property
  def unfinished_work(self):
    return self._unfinished_work

  def _schedule_jobs(self, policy_path, sampled_file_paths):
    jobs = [(file_paths, policy_path, self._reward_stat_map[file_paths])
            for file_paths in sampled_file_paths]
    return [
        self._pool.apply_async(self._runner.collect_data, job) for job in jobs
    ]

  def collect_data(
      self, policy_path: str
  ) -> Tuple[Iterator[trajectory.Trajectory], Dict[str, Dict[str, float]]]:
    """Collect data for a given policy.

    Args:
      policy_path: the path to the policy directory to collect data with.

    Returns:
      An iterator of batched trajectory.Trajectory that are ready to be fed to
        training.
      A dict of extra monitoring information, e.g., how many modules succeeded.
      They will be reported using `tf.scalar.summary` by the trainer so these
      information is viewable in TensorBoard.
    """
    sampled_file_paths = random.sample(self._file_paths, k=self._num_modules)
    results = self._schedule_jobs(policy_path, sampled_file_paths)

    def wait_for_termination():
      early_exit = data_collector.EarlyExitChecker(
          num_modules=self._num_modules)

      def get_num_finished_work():
        finished_work = sum(res.ready() for res in results)
        unfinished_work = len(results) - finished_work
        prev_unfinished_work = sum(
            not res.ready() for _, res in self._unfinished_work)
        # Handle overworked workers
        total_unfinished_work = unfinished_work + prev_unfinished_work
        if total_unfinished_work >= self._max_unfinished_tasks:
          self._overloaded_workers_handler(total_unfinished_work)
          return -1
        return finished_work

      return early_exit.wait(get_num_finished_work)

    wait_seconds = wait_for_termination()

    current_work = [
        (paths, res) for paths, res in zip(sampled_file_paths, results)
    ]
    finished_work = [(paths, res) for paths, res in current_work if res.ready()]
    unfinished_current_work = list(set(current_work) - set(finished_work))
    successful_work = [
        (paths, res) for paths, res in finished_work if res.successful()
    ]
    stale_results = [(paths, res)
                     for paths, res in self._unfinished_work
                     if res.ready() and res.successful()]
    self._unfinished_work = unfinished_current_work + [
        (paths, res) for paths, res in self._unfinished_work if not res.ready()
    ]

    logging.info(('%d of %d modules finished in %d seconds (%d failures).'
                  ' Currently %d unfinished work'), len(finished_work),
                 self._num_modules, wait_seconds,
                 len(finished_work) - len(successful_work),
                 len(self._unfinished_work))

    if self._use_stale_results:
      logging.info('Using %d stale results and %d fresh ones.',
                   len(stale_results), len(successful_work))
      successful_work += stale_results

    sequence_examples = list(
        itertools.chain.from_iterable(
            [res.get()[0] for (_, res) in successful_work]))
    self._reward_stat_map.update(
        {file_paths: res.get()[1] for (file_paths, res) in successful_work})

    monitor_dict = {}
    monitor_dict['default'] = {'success_modules': len(finished_work)}
    rewards = list(
        itertools.chain.from_iterable(
            [res.get()[2] for (_, res) in successful_work]))
    monitor_dict[
        'reward_distribution'] = data_collector.build_distribution_monitor(
            rewards)

    return self._parser(sequence_examples), monitor_dict

  def on_dataset_consumed(self,
                          dataset_iterator: Iterator[trajectory.Trajectory]):
    pass
