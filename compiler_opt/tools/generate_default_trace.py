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
"""Generate initial training data from the behavior of the current heuristic."""

import contextlib
import functools
import os
import queue
import re
import subprocess
from typing import Dict, List, Optional, Union, Tuple  # pylint:disable=unused-import

from absl import app
from absl import flags
from absl import logging
import gin
import multiprocessing
import tensorflow as tf

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import registry

# see https://bugs.python.org/issue33315 - we do need these types, but must
# currently use them as string annotations

_DATA_PATH = flags.DEFINE_string('data_path', None,
                                 'Path to folder containing IR files.')
_POLICY_PATH = flags.DEFINE_string(
    'policy_path', '', 'Path to the policy to generate trace with.')
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'Path to the output tfrecord file if not None.')
_OUTPUT_PERFORMANCE_PATH = flags.DEFINE_string(
    'output_performance_path', None,
    'Path to the output performance file if not None.')
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel workers for compilation. `None` for maximum available.')
_SAMPLING_RATE = flags.DEFINE_float(
    'sampling_rate', 1,
    'Sampling rate of modules, 0.5 means 50% sampling rate that generates data '
    'for half modules.')
_MODULE_FILTER = flags.DEFINE_string(
    'module_filter', None,
    'Regex for module names to include, do not provide one to include all')
_KEY_FILTER = flags.DEFINE_string(
    'key_filter', None,
    'Regex for key names to include, do not provide one to include all')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

ResultsQueueEntry = Union[Optional[Tuple[str, List[str],
                                         Dict[str,
                                              compilation_runner.RewardStat]]],
                          BaseException]


def get_runner() -> compilation_runner.CompilationRunner:
  problem_config = registry.get_configuration()
  return problem_config.get_runner_type()(moving_average_decay_rate=0)


def worker(policy_path: str, work_queue: 'queue.Queue[corpus.ModuleSpec]',
           results_queue: 'queue.Queue[ResultsQueueEntry]',
           key_filter: Optional[str]):
  """Describes the job each paralleled worker process does.

  The worker picks a workitem from the work_queue, process it, and deposits
  a result on the results_queue, in either success or failure cases.
  The results_queue items are tuples (workitem, result). On failure, the result
  is None.

  Args:
    runner: the data collector.
    policy_path: the policy_path to generate trace with.
    work_queue: the queue of unprocessed work items.
    results_queue: the queue where results are deposited.
    key_filter: regex filter for key names to include, or None to include all.
  """
  try:
    runner = get_runner()
    m = re.compile(key_filter) if key_filter else None

    while True:
      try:
        module_spec = work_queue.get_nowait()
      except queue.Empty:
        return
      try:
        data = runner.collect_data(
            module_spec=module_spec,
            tf_policy_path=policy_path,
            reward_stat=None)
        if not m:
          results_queue.put(
              (module_spec.name, data.serialized_sequence_examples,
               data.reward_stats))
          continue
        new_reward_stats = {}
        new_sequence_examples = []
        for k, sequence_example in zip(data.keys,
                                       data.serialized_sequence_examples):
          if not m.match(k):
            continue
          new_reward_stats[k] = data.reward_stats[k]
          new_sequence_examples.append(sequence_example)
        results_queue.put(
            (module_spec.name, new_sequence_examples, new_reward_stats))
      except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
              RuntimeError):
        logging.error('Failed to compile %s.', module_spec.name)
        results_queue.put(None)
  except BaseException as e:  # pylint: disable=broad-except
    results_queue.put(e)


def main(_):

  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  config = registry.get_configuration()

  logging.info('Loading module specs from corpus.')
  cps = corpus.Corpus(
      _DATA_PATH.value,
      additional_flags=config.flags_to_add(),
      delete_flags=config.flags_to_delete())
  logging.info('Done loading module specs from corpus.')

  if _MODULE_FILTER.value:
    m = re.compile(_MODULE_FILTER.value)
    cps.filter(m)

  # Sampling if needed.
  sampled_modules = int(len(cps) * _SAMPLING_RATE.value)
  # sort files by size, to process the large files upfront, hopefully while
  # other smaller files are processed in parallel
  module_specs = cps.sample(k=sampled_modules, sort=True)

  worker_count = (
      min(os.cpu_count(), _NUM_WORKERS.value)
      if _NUM_WORKERS.value else os.cpu_count())

  tfrecord_context = (
      tf.io.TFRecordWriter(_OUTPUT_PATH.value)
      if _OUTPUT_PATH.value else contextlib.nullcontext())
  performance_context = (
      tf.io.gfile.GFile(_OUTPUT_PERFORMANCE_PATH.value, 'w')
      if _OUTPUT_PERFORMANCE_PATH.value else contextlib.nullcontext())

  with tfrecord_context as tfrecord_writer:
    with performance_context as performance_writer:
      ctx = multiprocessing.get_context()
      m = ctx.Manager()
      results_queue: 'queue.Queue[ResultsQueueEntry]' = m.Queue()
      work_queue: 'queue.Queue[corpus.ModuleSpec]' = m.Queue()
      for module_spec in module_specs:
        work_queue.put(module_spec)

      # pylint:disable=g-complex-comprehension
      processes = [
          ctx.Process(
              target=functools.partial(worker, _POLICY_PATH.value, work_queue,
                                       results_queue, _KEY_FILTER.value))
          for _ in range(0, worker_count)
      ]
      # pylint:enable=g-complex-comprehension

      for p in processes:
        p.start()

      total_successful_examples = 0
      total_work = len(module_specs)
      total_failed_examples = 0
      total_training_examples = 0
      for _ in range(total_work):
        logging.log_every_n_seconds(logging.INFO,
                                    '%d success, %d failed out of %d', 10,
                                    total_successful_examples,
                                    total_failed_examples, total_work)

        results = results_queue.get()
        if isinstance(results, BaseException):
          logging.fatal(results)
        if not results:
          total_failed_examples += 1
          continue

        total_successful_examples += 1
        module_name, records, reward_stat = results
        if tfrecord_writer:
          total_training_examples += len(records)
          for r in records:
            tfrecord_writer.write(r)
        if performance_writer:
          for key, value in reward_stat.items():
            performance_writer.write(
                (f'{module_name},{key},{value.default_reward},'
                 f'{value.moving_average_reward}\n'))

      print((f'{total_successful_examples} of {len(module_specs)} modules '
             f'succeeded, and {total_training_examples} trainining examples '
             'written'))
      for p in processes:
        p.join()


if __name__ == '__main__':
  flags.mark_flag_as_required('data_path')
  app.run(main)
