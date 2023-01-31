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
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from absl import logging
from tf_agents.trajectories import trajectory

from compiler_opt.distributed import worker
from compiler_opt.distributed import buffered_scheduler
from compiler_opt.rl import best_trajectory
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import data_collector
from compiler_opt.rl import policy_saver


class LocalDataCollector(data_collector.DataCollector):
  """class for local data collection."""

  def __init__(
      self,
      cps: corpus.Corpus,
      num_modules: int,
      worker_pool: worker.WorkerPool,
      parser: Callable[[List[str]], Iterator[trajectory.Trajectory]],
      reward_stat_map: Dict[str, Optional[Dict[str,
                                               compilation_runner.RewardStat]]],
      best_trajectory_repo: Optional[best_trajectory.BestTrajectoryRepo]):
    super().__init__()

    self._corpus = cps
    self._num_modules = num_modules
    self._parser = parser
    self._worker_pool = worker_pool
    self._reward_stat_map = reward_stat_map
    self._best_trajectory_repo = best_trajectory_repo
    self._current_futures: List[worker.WorkerFuture] = []
    self._prefetch_pool = concurrent.futures.ThreadPoolExecutor()
    self._next_sample: List[
        concurrent.futures.Future] = self._prefetch_next_sample()

  def _prefetch_next_sample(self):
    t1 = time.time()
    sample = self._corpus.sample(k=self._num_modules, sort=False)
    ret = [
        self._prefetch_pool.submit(self._corpus.load_module_spec, element)
        for element in sample
    ]
    logging.info('prefetching took %d', time.time() - t1)
    return ret

  def close_pool(self):
    # if the pool lost some workers, that's fine - we don't need to tell them
    # anything anymore. To the new ones, the call is redundant (fine).
    for p in self._worker_pool.get_currently_active():
      p.cancel_all_work()
    self._worker_pool = None

  def _schedule_jobs(
      self, policy: policy_saver.Policy, model_id: int,
      sampled_modules: List[corpus.LoadedModuleSpec]
  ) -> List[worker.WorkerFuture[compilation_runner.CompilationResult]]:
    # by now, all the pending work, which was signaled to cancel, must've
    # finished
    jobs = [(loaded_module_spec, policy,
             self._reward_stat_map[loaded_module_spec.name])
            for loaded_module_spec in sampled_modules]

    def work_factory(job):

      def work(w: compilation_runner.CompilationRunnerStub):
        return w.collect_data(*job, model_id=model_id)

      return work

    work = [work_factory(job) for job in jobs]
    return self._worker_pool.schedule(work)

  def collect_data(
      self, policy: policy_saver.Policy, model_id: int
  ) -> Tuple[Iterator[trajectory.Trajectory], Dict[str, Dict[str, float]]]:
    """Collect data for a given policy.

    Args:
      policy: a policy_saver.Policy object to collect data with.

    Returns:
      An iterator of batched trajectory.Trajectory that are ready to be fed to
        training.
      A dict of extra monitoring information, e.g., how many modules succeeded.
      They will be reported using `tf.scalar.summary` by the trainer so these
      information is viewable in TensorBoard.
    """
    time1 = time.time()
    sampled_modules: List[corpus.LoadedModuleSpec] = [
        s.result() for s in self._next_sample
    ]
    logging.info('resolving prefetched sample took: %d seconds',
                 time.time() - time1)
    self._next_sample = self._prefetch_next_sample()

    time_before_schedule = time.time()
    self._current_futures = self._schedule_jobs(policy, model_id, sampled_modules)

    # Wait for all futures to complete. We don't do any early-exit checking as
    # that functionality has been moved to the
    # data_collector.EarlyExitWorkerPool abstraction.
    worker.wait_for(self._current_futures)

    current_work = list(zip(sampled_modules, self._current_futures))

    def is_cancelled(fut):
      if not fut.done():
        return False
      if e := worker.get_exception(fut):
        return isinstance(e, data_collector.CancelledForEarlyExitException)
      return False

    finished_work = [(spec, res) for spec, res in current_work if res.done()]
    successful_work = [(spec, res.result())
                       for spec, res in finished_work
                       if not worker.get_exception(res)]
    cancelled_work = [res for res in self._current_futures if is_cancelled(res)]
    failures = len(finished_work) - len(successful_work) - len(cancelled_work)

    logging.info(('%d of %d modules finished in %d seconds (%d failures).'),
                 len(finished_work) - len(cancelled_work), self._num_modules,
                 time.time() - time_before_schedule, failures)

    sequence_examples = list(
        itertools.chain.from_iterable(
            [res.serialized_sequence_examples for (_, res) in successful_work]))
    total_trajectory_length = sum(res.length for (_, res) in successful_work)
    self._reward_stat_map.update(
        {spec.name: res.reward_stats for (spec, res) in successful_work})

    if self._best_trajectory_repo is not None:
      for spec, res in successful_work:
        module_name = spec.name
        for (identifier, reward,
             sequence_example) in zip(res.keys, res.policy_rewards,
                                      res.serialized_sequence_examples):
          self._best_trajectory_repo.update_if_better_trajectory(
              module_name, identifier, reward, sequence_example)

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
