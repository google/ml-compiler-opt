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
import random
import subprocess
# see https://bugs.python.org/issue33315 - we do need these types, but must
# currently use them as string annotations
from typing import Dict, List, Optional, Tuple  # pylint:disable=unused-import

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl import compilation_runner
from compiler_opt.rl.inlining import inlining_runner

_DATA_PATH = flags.DEFINE_string('data_path', None,
                                 'Path to folder containing IR files.')
_POLICY_PATH = flags.DEFINE_string(
    'policy_path', '', 'Path to the policy to generate trace with.')
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'Path to the output tfrecord file if not None.')
_OUTPUT_PERFORMANCE_PATH = flags.DEFINE_string(
    'output_performance_path', None,
    'Path to the output performance file if not None.')
_COMPILE_TASK = flags.DEFINE_enum(
    'compile_task', 'inlining', ['inlining'],
    'compile task to generate tfrecord with, only support '
    'inlining currently.')
_CLANG_PATH = flags.DEFINE_string('clang_path', 'clang',
                                  'Path to clang binary.')
_LLVM_SIZE_PATH = flags.DEFINE_string('llvm_size_path', 'llvm-size',
                                      'Path to llvm_size binary.')
_LAUNCHER_PATH = flags.DEFINE_string('launcher_path', None,
                                     'Path to launcher binary.')
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel workers for compilation. `None` for maximum available.')
_SAMPLING_RATE = flags.DEFINE_float(
    'sampling_rate', 1,
    'Sampling rate of modules, 0.5 means 50% sampling rate that generates data '
    'for half modules.')

ResultsQueueEntry = Optional[Tuple[str, List[str],
                                   Dict[str, compilation_runner.RewardStat]]]


def worker(runner: compilation_runner.CompilationRunner, policy_path: str,
           work_queue: 'queue.Queue[Tuple[str, ...]]',
           results_queue: 'queue.Queue[Optional[List[str]]]'):
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
  """
  while True:
    try:
      module_triple = work_queue.get_nowait()
    except queue.Empty:
      return
    try:
      (records, reward_stat, _) = runner.collect_data(module_triple,
                                                      policy_path, None)
      results_queue.put((module_triple[0], records, reward_stat))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            RuntimeError):
      logging.error('Failed to compile %s.', module_triple)
      results_queue.put(None)


def main(_):
  # Initialize runner and file_suffix according to compile_task.
  if _COMPILE_TASK.value == 'inlining':
    runner = inlining_runner.InliningRunner(
        clang_path=_CLANG_PATH.value,
        llvm_size_path=_LLVM_SIZE_PATH.value,
        launcher_path=_LAUNCHER_PATH.value,
        moving_average_decay_rate=0)
    file_suffix = ['.bc', '.cmd']

  with open(os.path.join(_DATA_PATH.value, 'module_paths'), 'r') as f:
    module_paths = [
        os.path.join(_DATA_PATH.value, name.rstrip('\n')) for name in f
    ]

  # Sampling if needed.
  if _SAMPLING_RATE.value < 1:
    sampled_modules = int(len(module_paths) * _SAMPLING_RATE.value)
    module_paths = random.sample(module_paths, k=sampled_modules)

  # sort files by size, to process the large files upfront, hopefully while
  # other smaller files are processed in parallel
  sizes_and_paths = [(os.path.getsize(p + '.bc'), p) for p in module_paths]
  sizes_and_paths.sort(reverse=True)
  sorted_module_paths = [p for _, p in sizes_and_paths]
  module_specs = [
      tuple([p + suffix for suffix in file_suffix]) for p in sorted_module_paths
  ]

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
      results_queue: ('queue.Queue[ResultsQueueEntry]') = m.Queue()
      work_queue: 'queue.Queue[Tuple[str, ...]]' = m.Queue()
      for module_spec in module_specs:
        work_queue.put(module_spec)

      # pylint:disable=g-complex-comprehension
      processes = [
          ctx.Process(
              target=functools.partial(worker, runner, _POLICY_PATH.value,
                                       work_queue, results_queue))
          for _ in range(0, worker_count)
      ]
      # pylint:enable=g-complex-comprehension

      for p in processes:
        p.start()

      total_successful_examples = 0
      total_work = len(module_specs)
      total_failed_examples = 0
      for _ in range(total_work):
        logging.log_every_n_seconds(logging.INFO,
                                    '%d success, %d failed out of %d', 10,
                                    total_successful_examples,
                                    total_failed_examples, total_work)

        results = results_queue.get()
        if not results:
          total_failed_examples += 1
          continue

        total_successful_examples += 1
        module_name, records, reward_stat = results
        if tfrecord_writer:
          for r in records:
            tfrecord_writer.write(r)
        if performance_writer:
          for key, value in reward_stat.items():
            performance_writer.write('%s,%s,%f,%f\n' %
                                     (module_name, key, value.default_reward,
                                      value.moving_average_reward))

      print('%d of %d modules succeeded.' %
            (total_successful_examples, len(module_specs)))
      for p in processes:
        p.join()


if __name__ == '__main__':
  flags.mark_flag_as_required('data_path')
  multiprocessing.handle_main(functools.partial(app.run, main))
