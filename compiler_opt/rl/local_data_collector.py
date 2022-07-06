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

import itertools
import random
import time
from typing import Callable, Dict, Iterator, List, Tuple, Optional

from absl import logging
import multiprocessing  # for Pool
from tf_agents.trajectories import trajectory

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import data_collector
from compiler_opt.rl.adt import ModuleSpec


class LocalDataCollector(data_collector.DataCollector):
  """class for local data collection."""

  def __init__(
      self,
      module_specs: List[ModuleSpec],
      num_workers: int,
      num_modules: int,
      runner: compilation_runner.CompilationRunner,
      parser: Callable[[List[str]], Iterator[trajectory.Trajectory]],
      reward_stat_map: Dict[str, Optional[Dict[str,
                                               compilation_runner.RewardStat]]],
      exit_checker_ctor=data_collector.EarlyExitChecker):
    # TODO(mtrofin): type exit_checker_ctor when we get typing.Protocol support
    super().__init__()

    self._module_specs = module_specs
    self._num_modules = num_modules
    self._runner = runner
    self._parser = parser
    self._pool = multiprocessing.get_context().Pool(
        num_workers, initializer=compilation_runner.CompilationRunner.init_pool)
    self._reward_stat_map = reward_stat_map

    self._exit_checker_ctor = exit_checker_ctor
    self._pending_work = None
    # hold on to the token so it won't get GCed before all its wait()
    # complete
    self._last_token = None

  def close_pool(self):
    self._join_pending_jobs()
    if self._pool:
      # Stop accepting new work
      self._pool.close()
      self._pool.join()
      self._pool = None

  def _join_pending_jobs(self):
    if self._pending_work:
      t1 = time.time()
      for w in self._pending_work:
        w.wait()

      self._pending_work = None
      # this should have taken negligible time, normally, since all the work
      # has been cancelled and the workers had time to process the cancellation
      # while training was unfolding.
      logging.info('Waiting for pending work from last iteration took %f',
                   time.time() - t1)
    self._last_token = None

  def _schedule_jobs(self, policy_path: str, sampled_modules: List[ModuleSpec]):
    # by now, all the pending work, which was signaled to cancel, must've
    # finished
    self._join_pending_jobs()
    cancellation_token = compilation_runner.ProcessCancellationToken()
    jobs = [(module_spec, policy_path,
             self._reward_stat_map[module_spec.name], cancellation_token)
            for module_spec in sampled_modules]

    # Make sure we're not missing failures in workers. All but
    # ProcessKilledError, which we want to ignore.
    def error_callback(e):
      if isinstance(e, compilation_runner.ProcessKilledError):
        return
      logging.exception('Error in worker: %r', e)

    return (cancellation_token, [
        self._pool.apply_async(
            self._runner.collect_data, job, error_callback=error_callback)
        for job in jobs
    ])

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
    sampled_modules = random.sample(self._module_specs, k=self._num_modules)
    ct, results = self._schedule_jobs(policy_path, sampled_modules)

    def wait_for_termination():
      early_exit = self._exit_checker_ctor(num_modules=self._num_modules)

      def get_num_finished_work():
        finished_work = sum(res.ready() for res in results)
        return finished_work

      return early_exit.wait(get_num_finished_work)

    wait_seconds = wait_for_termination()
    # signal whatever work is left to finish
    ct.signal()
    current_work = zip(sampled_modules, results)
    finished_work = [(paths, res) for paths, res in current_work if res.ready()]
    successful_work = [
        (paths, res) for paths, res in finished_work if res.successful()
    ]
    failures = len(finished_work) - len(successful_work)

    logging.info(('%d of %d modules finished in %d seconds (%d failures).'),
                 len(finished_work), self._num_modules, wait_seconds, failures)

    sequence_examples = list(
        itertools.chain.from_iterable([
            res.get().serialized_sequence_examples
            for (_, res) in successful_work
        ]))
    total_trajectory_length = sum(
        res.get().length for (_, res) in successful_work)
    self._reward_stat_map.update({
        module_spec.name: res.get().reward_stats
        for (module_spec, res) in successful_work
    })

    monitor_dict = {}
    monitor_dict['default'] = {
        'success_modules': len(successful_work),
        'total_trajectory_length': total_trajectory_length,
    }
    rewards = list(
        itertools.chain.from_iterable(
            [res.get().rewards for (_, res) in successful_work]))
    monitor_dict[
        'reward_distribution'] = data_collector.build_distribution_monitor(
            rewards)

    parsed = self._parser(sequence_examples)

    self._pending_work = [res for res in results if not res.ready()]
    # if some of the cancelled work hasn't yet processed the signal, let's let
    # it do that while we process the data. We also need to hold on to the
    # current token, so its Event object doesn't get GC-ed here.
    if self._pending_work:
      self._last_token = ct
    return parsed, monitor_dict

  def on_dataset_consumed(self,
                          dataset_iterator: Iterator[trajectory.Trajectory]):
    pass
