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


class LocalDataCollector(data_collector.DataCollector):
  """class for local data collection."""

  def __init__(self, file_paths: List[Iterable[str]], num_workers: int,
               num_modules: int, runner: Callable[[str, str, int], Tuple[str,
                                                                         int]],
               parser: Callable[[List[str]], Iterator[trajectory.Trajectory]]):
    super(LocalDataCollector, self).__init__()

    self._file_paths = file_paths
    self._num_modules = num_modules
    self._runner = runner
    self._parser = parser

    ctx = multiprocessing.get_context('spawn')
    self._pool = ctx.Pool(num_workers)

    self._default_policy_size_map = collections.defaultdict(lambda: None)

  def collect_data(self, policy_path: str) -> Iterator[trajectory.Trajectory]:
    """Collect data for a given policy.

    Args:
      policy_path: the path to the policy directory to collect data with.

    Returns:
      An iterator of batched trajectory.Trajectory that are ready to be fed to
        training.
    """
    sampled_file_paths = random.sample(self._file_paths, k=self._num_modules)
    jobs = [(file_paths, policy_path, self._default_policy_size_map[file_paths])
            for file_paths in sampled_file_paths]
    results = [self._pool.apply_async(self._runner, job) for job in jobs]

    def wait_for_termination():
      wait_seconds = 0
      while True:
        finished_data_collection = sum([x.ready() for x in results])
        for (data_collection_threshold,
             wait_time_threshold) in _WAIT_TERMINATION:
          if ((finished_data_collection >=
               self._num_modules * data_collection_threshold) and
              (wait_seconds >= _DEADLINE_IN_SECONDS * wait_time_threshold)):
            return finished_data_collection, wait_seconds
        wait_seconds += 1
        time.sleep(1)

    finished_data_collection, wait_seconds = wait_for_termination()
    successful_tuples = [(paths, res)
                         for (paths, res) in zip(sampled_file_paths, results)
                         if res.ready() and res.successful()]
    logging.info('%d of %d modules finished with %d seconds (%d failures)',
                 finished_data_collection, self._num_modules, wait_seconds,
                 finished_data_collection - len(successful_tuples))

    sequence_examples = [res.get()[0] for (_, res) in successful_tuples]
    self._default_policy_size_map.update(
        {file_paths: res.get()[1] for (file_paths, res) in successful_tuples})

    return self._parser(sequence_examples)

  def on_dataset_consumed(self,
                          dataset_iterator: Iterator[trajectory.Trajectory]):
    pass
