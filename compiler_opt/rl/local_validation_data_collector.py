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
"""Validation data collection module."""
import concurrent.futures
import threading
import time
from typing import Dict, Optional, List, Tuple

from absl import logging

from compiler_opt.distributed import worker
from compiler_opt.distributed.local import buffered_scheduler
from compiler_opt.distributed.local import cpu_affinity
from compiler_opt.distributed.local.local_worker_manager import LocalWorkerPool

from compiler_opt.rl import corpus


class LocalValidationDataCollector(worker.ContextAwareWorker):
  """Local implementation of a validation data collector
  Args:
    module_specs: List of module specs to use
    worker_pool_args: Pool of workers to use
  """

  def __init__(self, cps: corpus.Corpus, worker_pool_args, reward_stat_map,
               max_cpus):
    self._num_modules = len(cps) if cps is not None else 0
    self._corpus: corpus.Corpus = cps
    self._default_rewards = {}

    self._running_policy = None
    self._default_futures: List[worker.WorkerFuture] = []
    self._current_work: List[Tuple[corpus.ModuleSpec, worker.WorkerFuture]] = []
    self._last_time = None
    self._elapsed_time = 0

    self._context_local = True

    # Check a bit later so some expected vars have been set first.
    if not cps:
      return

    affinities = cpu_affinity.set_and_get(
        is_main_process=False, max_cpus=max_cpus)

    # Add some runner specific flags.
    logging.info('Validation data collector using %d workers.', len(affinities))
    worker_pool_args['count'] = len(affinities)
    worker_pool_args['moving_average_decay_rate'] = 1
    worker_pool_args['compilation_timeout'] = 1200

    # Borrow from the external reward_stat_map in case it got loaded from disk
    # and already has some values. On a fresh run this will be recalculated
    # from scratch in the main data collector and here. It would be ideal if
    # both shared the same dict, but that would be too complex to implement.
    for name, data in reward_stat_map.items():
      if name not in self._default_rewards:
        self._default_rewards[name] = {}
      for identifier, reward_stat in data.items():
        self._default_rewards[name][identifier] = reward_stat.default_reward

    self._pool = LocalWorkerPool(**worker_pool_args)
    self._worker_pool = self._pool.stubs

    for i, p in zip(affinities, self._worker_pool):
      p.set_nice(19)
      p.set_affinity([i])

  # BEGIN: ContextAwareWorker methods
  @classmethod
  def is_priority_method(cls, _: str) -> bool:
    # Everything is a priority: this is essentially a synchronous RPC endpoint.
    return True

  def set_context(self, local: bool):
    self._context_local = local

  # END: ContextAwareWorker methods

  def _schedule_jobs(self, policy_path, module_specs):
    default_jobs = []
    for module_spec in module_specs:
      if module_spec.name not in self._default_rewards:
        # The bool is reward_only, None is cancellation_manager
        default_jobs.append((module_spec, '', True, None))

    default_rewards_lock = threading.Lock()

    def create_update_rewards(spec_name):

      def updater(f: concurrent.futures.Future):
        if f.exception() is not None:
          reward_stat = f.result()
          for identifier, (_, default_reward) in reward_stat:
            with default_rewards_lock:
              self._default_rewards[spec_name][identifier] = default_reward

      return updater

    # The bool is reward_only, None is cancellation_manager
    policy_jobs = [
        (module_spec, policy_path, True, None) for module_spec in module_specs
    ]

    def work_factory(job):

      def work(w):
        return w.compile_fn(*job)

      return work

    work = [work_factory(job) for job in default_jobs]
    work += [work_factory(job) for job in policy_jobs]

    futures = buffered_scheduler.schedule(work, self._worker_pool, buffer=10)

    self._default_futures = futures[:len(default_jobs)]
    policy_futures = futures[len(default_jobs):]

    for job, future in zip(default_jobs, self._default_futures):
      future.add_done_callback(create_update_rewards(job[0]))

    return policy_futures

  def collect_data_async(
      self,
      policy_path: str,
      step: int = 0) -> Optional[Dict[tuple, Dict[str, float]]]:
    """Collect data for a given policy.

    Args:
      policy_path: the path to the policy directory to collect data with.
      step: the step number associated with the policy_path

    Returns:
      Either returns data in the form of a dictionary, or returns None if the
      data is not ready yet.
    """
    if self._num_modules == 0:
      return None

    # Resume immediately, so that if new jobs are scheduled,
    # they run while processing last batch's results
    self.resume_children()
    finished_work = [
        (spec, res) for spec, res in self._current_work if res.done()
    ]

    # Check if there are default rewards being collected.
    if len(self._default_futures) > 0:
      finished_default_work = sum(res.done() for res in self._default_futures)
      if finished_default_work != len(self._default_futures):
        logging.info('%d out of %d default-rewards modules are finished.',
                     finished_default_work, len(self._default_futures))
        return None

    if len(finished_work) != len(self._current_work):  # on 1st iter both are 0
      logging.info('%d out of %d modules are finished.', len(finished_work),
                   len(self._current_work))
      return None
    module_specs = self._corpus.modules
    results = self._schedule_jobs(policy_path, module_specs)
    self._current_work = list(zip(module_specs, results))
    prev_policy = self._running_policy
    self._running_policy = step

    if len(finished_work) == 0:  # 1st iteration this is 0
      return None

    # Since all work is done: reset clock. Essential if processes never paused.
    if self._last_time is not None:
      cur_time = time.time()
      self._elapsed_time += cur_time - self._last_time
      self._last_time = cur_time

    successful_work = [(spec, res.result())
                       for spec, res in finished_work
                       if not worker.get_exception(res)]
    failures = len(finished_work) - len(successful_work)

    logging.info('%d of %d modules finished in %d seconds (%d failures).',
                 len(finished_work), self._num_modules, self._elapsed_time,
                 failures)

    sum_policy = 0
    sum_default = 0
    for spec, res in successful_work:
      # res format: {_DEFAULT_IDENTIFIER: (None, native_size)}
      for identifier, (_, policy_reward) in res:
        sum_policy += policy_reward
        sum_default += self._default_rewards[spec.name][identifier]

    if sum_default <= 0:
      raise ValueError('Sum of default rewards is 0.')
    reward = 1 - sum_policy / sum_default

    monitor_dict = {
        prev_policy: {
            'success_modules': len(successful_work),
            'compile_wall_time': self._elapsed_time,
            'sum_reward': reward
        }
    }
    self._elapsed_time = 0  # Only on completion this is reset
    return monitor_dict

  def pause_children(self):
    if not self._context_local or self._running_policy is None:
      return

    for p in self._worker_pool:
      p.pause_all_work()

    if self._last_time is not None:
      self._elapsed_time += time.time() - self._last_time
      self._last_time = None

  def resume_children(self):
    last_time_was_none = False
    if self._last_time is None:
      last_time_was_none = True
      self._last_time = time.time()

    if not self._context_local or self._running_policy is None:
      return

    # Only pause changes last_time to None.
    if last_time_was_none:
      for p in self._worker_pool:
        p.resume_all_work()
