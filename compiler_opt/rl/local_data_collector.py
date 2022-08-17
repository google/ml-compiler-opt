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

import concurrent.futures
import itertools
import time
from typing import Callable, Dict, Iterator, List, Tuple, Optional

from absl import logging
from tf_agents.trajectories import trajectory

from compiler_opt.distributed import worker
from compiler_opt.distributed.local import buffered_scheduler
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import data_collector


class LocalDataCollector(data_collector.DataCollector):
  """class for local data collection."""

  def __init__(
      self,
      cps: corpus.Corpus,
      num_modules: int,
      worker_pool: List[compilation_runner.CompilationRunnerStub],
      parser: Callable[[List[str]], Iterator[trajectory.Trajectory]],
      reward_stat_map: Dict[str, Optional[Dict[str,
                                               compilation_runner.RewardStat]]],
      exit_checker_ctor=data_collector.EarlyExitChecker):
    # TODO(mtrofin): type exit_checker_ctor when we get typing.Protocol support
    super().__init__()

    self._corpus = cps
    self._num_modules = num_modules
    self._parser = parser
    self._worker_pool = worker_pool
    self._reward_stat_map = reward_stat_map
    self._exit_checker_ctor = exit_checker_ctor
    # _reset_workers is a future that resolves when post-data collection cleanup
    # work completes, i.e. cancelling all work and re-enabling the workers.
    # We remove this activity from the critical path by running it concurrently
    # with the training phase - i.e. whatever happens between successive data
    # collection calls. Subsequent runs will wait for these to finish.
    self._reset_workers: Optional[concurrent.futures.Future] = None
    self._current_futures: List[worker.WorkerFuture] = []
    self._pool = concurrent.futures.ThreadPoolExecutor()

  def close_pool(self):
    self._join_pending_jobs()
    for p in self._worker_pool:
      p.cancel_all_work()
    self._worker_pool = None

  def _join_pending_jobs(self):
    t1 = time.time()
    if self._reset_workers:
      concurrent.futures.wait([self._reset_workers])

    self._reset_workers = None
    # this should have taken negligible time, normally, since all the work
    # has been cancelled and the workers had time to process the cancellation
    # while training was unfolding.
    logging.info('Waiting for pending work from last iteration took %f',
                 time.time() - t1)

  def _schedule_jobs(
      self, policy_path: str, sampled_modules: List[corpus.ModuleSpec]
  ) -> List[worker.WorkerFuture[compilation_runner.CompilationResult]]:
    # by now, all the pending work, which was signaled to cancel, must've
    # finished
    self._join_pending_jobs()
    jobs = [(module_spec, policy_path, self._reward_stat_map[module_spec.name])
            for module_spec in sampled_modules]

    def work_factory(job):

      def work(w):
        return w.collect_data(*job)

      return work

    work = [work_factory(job) for job in jobs]
    return buffered_scheduler.schedule(work, self._worker_pool, buffer=10)

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
    sampled_modules = self._corpus.sample(k=self._num_modules, sort=False)
    self._current_futures = self._schedule_jobs(policy_path, sampled_modules)

    def wait_for_termination():
      early_exit = self._exit_checker_ctor(num_modules=self._num_modules)

      def get_num_finished_work():
        finished_work = sum(res.done() for res in self._current_futures)
        return finished_work

      return early_exit.wait(get_num_finished_work)

    wait_seconds = wait_for_termination()
    current_work = list(zip(sampled_modules, self._current_futures))
    finished_work = [(spec, res) for spec, res in current_work if res.done()]
    successful_work = [(spec, res.result())
                       for spec, res in finished_work
                       if not worker.get_exception(res)]
    failures = len(finished_work) - len(successful_work)

    logging.info(('%d of %d modules finished in %d seconds (%d failures).'),
                 len(finished_work), self._num_modules, wait_seconds, failures)

    # signal whatever work is left to finish, and re-enable workers.
    def wrapup():
      cancel_futures = [wkr.cancel_all_work() for wkr in self._worker_pool]
      worker.wait_for(cancel_futures)
      # now that the workers killed pending compilations, make sure the workers
      # drained their working queues first - they should all complete quickly
      # since the cancellation manager is killing immediately any process starts
      worker.wait_for(self._current_futures)
      worker.wait_for([wkr.enable() for wkr in self._worker_pool])

    self._reset_workers = self._pool.submit(wrapup)

    sequence_examples = list(
        itertools.chain.from_iterable(
            [res.serialized_sequence_examples for (_, res) in successful_work]))
    total_trajectory_length = sum(res.length for (_, res) in successful_work)
    self._reward_stat_map.update(
        {spec.name: res.reward_stats for (spec, res) in successful_work})

    monitor_dict = {}
    monitor_dict['default'] = {
        'success_modules': len(successful_work),
        'total_trajectory_length': total_trajectory_length,
    }
    rewards = list(
        itertools.chain.from_iterable(
            [res.rewards for (_, res) in successful_work]))
    monitor_dict[
        'reward_distribution'] = data_collector.build_distribution_monitor(
            rewards)

    parsed = self._parser(sequence_examples)

    return parsed, monitor_dict

  def on_dataset_consumed(self,
                          dataset_iterator: Iterator[trajectory.Trajectory]):
    pass
