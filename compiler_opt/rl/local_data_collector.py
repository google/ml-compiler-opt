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
import random
import time
from typing import Callable, Iterator, List, Tuple, Iterable

from absl import logging
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.trajectories import trajectory

from compiler_opt.rl import data_collector

# Deadline for data collection.
_DEADLINE_IN_SECONDS = 120

# We don't wait for all data collection to finish --- it continues if either of
# the wait_termination_conditions is met.
# (0.8, 0.5) means it won't wait for more data collection to finish if 80% of
# the data collection have finished and it has waited 50% of
# _DEADLINE_IN_SECONDS time.
_WAIT_TERMINATION = ((0.98, 0), (0.95, 0.25), (0.9, 0.5), (0, 1))

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
               file_paths: List[Iterable[str]],
               num_workers: int,
               num_modules: int,
               runner: Callable[[str, str, int], Tuple[str, int]],
               parser: Callable[[List[str]], Iterator[trajectory.Trajectory]],
               use_stale_results=False,
               max_unfinished_tasks=None,
               overload_handler=default_overload_handler):
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

    self._default_policy_size_map = collections.defaultdict(lambda: None)
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
    jobs = [(file_paths, policy_path, self._default_policy_size_map[file_paths])
            for file_paths in sampled_file_paths]
    return [self._pool.apply_async(self._runner, job) for job in jobs]

  def collect_data(self, policy_path: str) -> Iterator[trajectory.Trajectory]:
    """Collect data for a given policy.

    Args:
      policy_path: the path to the policy directory to collect data with.

    Returns:
      An iterator of batched trajectory.Trajectory that are ready to be fed to
        training.
    """
    sampled_file_paths = random.sample(self._file_paths, k=self._num_modules)
    results = self._schedule_jobs(policy_path, sampled_file_paths)

    def wait_for_termination():
      wait_seconds = 0
      while True:
        finished_work = sum(res.ready() for res in results)
        unfinised_work = len(results) - finished_work
        prev_unfinished_work = sum(
            not res.ready() for _, res in self._unfinished_work)
        total_unfinished_work = unfinised_work + prev_unfinished_work
        for (data_collection_threshold,
             wait_time_threshold) in _WAIT_TERMINATION:
          if ((finished_work >= self._num_modules * data_collection_threshold)
              and (wait_seconds >= _DEADLINE_IN_SECONDS * wait_time_threshold)):
            if total_unfinished_work >= self._max_unfinished_tasks:
              self._overloaded_workers_handler(total_unfinished_work)
              break
            else:
              return wait_seconds
        wait_seconds += 1
        time.sleep(1)

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

    sequence_examples = [res.get()[0] for (_, res) in successful_work]
    self._default_policy_size_map.update(
        {file_paths: res.get()[1] for (file_paths, res) in successful_work})

    return self._parser(sequence_examples)

  def on_dataset_consumed(self,
                          dataset_iterator: Iterator[trajectory.Trajectory]):
    pass
