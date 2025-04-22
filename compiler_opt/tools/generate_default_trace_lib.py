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
"""Collects information from a corpus and a specified heuristic."""

import concurrent.futures
import contextlib
import functools
import re

from absl import app
from absl import flags
from absl import logging
import gin

import tensorflow as tf

from compiler_opt.distributed import worker
from compiler_opt.distributed import worker_manager
from compiler_opt.distributed import buffered_scheduler
from compiler_opt.distributed.local import local_worker_manager

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import policy_saver
from compiler_opt.rl import registry


class FilteringWorker(worker.Worker):
  """Worker that performs a computation and optionally filters the result.


  Args:
    policy_path: the policy_path to generate trace with.
    key_filter: regex filter for key names to include, or None to include all.
  """

  def __init__(self, policy_path: str | None, key_filter: str | None,
               runner_type: 'type[compilation_runner.CompilationRunner]',
               runner_kwargs):
    self._policy_path = policy_path
    self._key_filter = re.compile(key_filter) if key_filter else None
    self._runner = runner_type(**runner_kwargs)
    self._policy = policy_saver.Policy.from_filesystem(
        policy_path) if policy_path else None

  def compile_and_filter(
      self, loaded_module_spec: corpus.LoadedModuleSpec
  ) -> tuple[str, list[str], dict[str, compilation_runner.RewardStat],
             list[str]]:
    data = self._runner.collect_data(
        loaded_module_spec=loaded_module_spec,
        policy=self._policy,
        reward_stat=None,
        model_id=0)
    if self._key_filter is None:
      return (loaded_module_spec.name, data.serialized_sequence_examples,
              data.reward_stats, data.keys)
    new_reward_stats = {}
    new_sequence_examples = []
    new_keys = []
    for k, sequence_example in zip(data.keys,
                                   data.serialized_sequence_examples):
      if not self._key_filter.match(k):
        continue
      new_reward_stats[k] = data.reward_stats[k]
      new_sequence_examples.append(sequence_example)
      new_keys.append(k)
    return (loaded_module_spec.name, new_sequence_examples, new_reward_stats,
            new_keys)


def generate_trace(data_path: str,
                   output_path: str | None,
                   output_performance_path: str | None,
                   num_workers: int | None,
                   sampling_rate: float,
                   module_filter_str: str | None,
                   key_filter: str | None,
                   keys_file_path: str | None,
                   policy_path: str,
                   worker_manager_class: type[
                       worker_manager.WorkerManager] = local_worker_manager
                   .LocalWorkerPoolManager):

  config = registry.get_configuration()

  logging.info('Loading module specs from corpus.')
  module_filter = re.compile(module_filter_str) if module_filter_str else None

  cps = corpus.Corpus(
      data_path=data_path,
      module_filter=lambda name: True
      if not module_filter else module_filter.match(name),
      additional_flags=config.flags_to_add(),
      delete_flags=config.flags_to_delete(),
      replace_flags=config.flags_to_replace())
  logging.info('Done loading module specs from corpus.')

  # Sampling if needed.
  sampled_modules = int(len(cps) * sampling_rate)
  # sort files by size, to process the large files upfront, hopefully while
  # other smaller files are processed in parallel
  corpus_elements = cps.sample(k=sampled_modules, sort=True)

  tfrecord_context = (
      tf.io.TFRecordWriter(output_path)
      if output_path else contextlib.nullcontext())
  performance_context = (
      tf.io.gfile.GFile(output_performance_path, 'w')
      if output_performance_path else contextlib.nullcontext())
  work = [
      cps.load_module_spec(corpus_element) for corpus_element in corpus_elements
  ]
  all_keys = []

  runner_type = config.get_runner_type()
  with tfrecord_context as tfrecord_writer:
    with performance_context as performance_writer:
      with worker_manager_class(
          FilteringWorker,
          count=num_workers,
          worker_kwargs=dict(
              policy_path=policy_path,
              key_filter=key_filter,
              runner_type=runner_type,
              runner_kwargs=worker.get_full_worker_args(
                  runner_type, moving_average_decay_rate=0))) as lwm:

        _, result_futures = buffered_scheduler.schedule_on_worker_pool(
            action=lambda w, j: w.compile_and_filter(j),
            jobs=work,
            worker_pool=lwm)
        total_successful_examples = 0
        total_work = len(corpus_elements)
        total_failed_examples = 0
        total_training_examples = 0
        not_done = result_futures
        while not_done:
          (done, not_done) = concurrent.futures.wait(not_done, 10)
          succeeded = [
              r for r in done if not r.cancelled() and r.exception() is None
          ]
          total_successful_examples += len(succeeded)
          total_failed_examples += (len(done) - len(succeeded))
          for r in succeeded:
            module_name, records, reward_stat, keys = r.result()
            all_keys.extend(keys)
            if tfrecord_writer:
              total_training_examples += len(records)
              for r in records:
                tfrecord_writer.write(r)
            if performance_writer:
              for key, value in reward_stat.items():
                performance_writer.write(
                    f'{module_name},{key},{value.default_reward},'
                    f'{value.moving_average_reward}\n')
          logging.info('%d success, %d failed out of %d',
                       total_successful_examples, total_failed_examples,
                       total_work)

  print(f'{total_successful_examples} of {len(corpus_elements)} modules '
        f'succeeded, and {total_training_examples} trainining examples '
        'written')

  if keys_file_path is not None:
    with open(keys_file_path, 'w', encoding='utf-8') as keys_file:
      keys_file.write('\n'.join(str(key) for key in all_keys) + '\n')
