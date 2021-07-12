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

import functools
import os
import queue
import random

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl import inlining_runner

flags.DEFINE_string('data_path', None, 'Path to folder containing IR files.')
flags.DEFINE_string('output_path', None, 'Path to the output tfrecord file.')
flags.DEFINE_enum(
    'compile_task', 'inlining', ['inlining'],
    'compile task to generate tfrecord with, only support '
    'inlining currently.')
flags.DEFINE_string('clang_path', 'clang', 'Path to clang binary.')
flags.DEFINE_string('llvm_size_path', 'llvm-size', 'Path to llvm_size binary.')
flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel workers for compilation. `None` for maximum available.')
flags.DEFINE_float(
    'sampling_rate', 1,
    'Sampling rate of modules, 0.5 means 50% sampling rate that generates data '
    'for half modules.')

FLAGS = flags.FLAGS


def worker(runner, work_queue: queue.Queue, results_queue: queue.Queue):
  """What each worker process does.

  Each worker picks a workitem from the work_queue, process it, and deposits
  a result on the results_queue, in either success or failure cases.
  The results_queue items are tuples (workitem, result). On failure, the result
  is None.

  Args:
    runner: the data collector.
    work_queue: the queue of unprocessed work items.
    results_queue: the queue where results are deposited.
  """
  while True:
    try:
      module_triple = work_queue.get_nowait()
    except queue.Empty:
      return
    try:
      record = runner.collect_data(module_triple, '', None)
      results_queue.put((module_triple, record))
    except:  # pylint: disable=bare-except
      logging.error('Failed to compile %s.', module_triple)
      results_queue.put((module_triple, None))


def main(_):
  # Initialize runner and file_suffix according to compile_task.
  if FLAGS.compile_task == 'inlining':
    runner = inlining_runner.InliningRunner(
        clang_path=FLAGS.clang_path, llvm_size_path=FLAGS.llvm_size_path)
    file_suffix = ['.bc', '.cmd']

  with open(os.path.join(FLAGS.data_path, 'module_paths'), 'r') as f:
    module_paths = [
        os.path.join(FLAGS.data_path, name.rstrip('\n')) for name in f
    ]

  # Sampling if needed.
  if FLAGS.sampling_rate < 1:
    sampled_modules = int(len(module_paths) * FLAGS.sampling_rate)
    module_paths = random.sample(module_paths, k=sampled_modules)

  # sort files by size, to process the large files upfront, hopefully while
  # other smaller files are processed in parallel
  sizes_and_paths = [(os.path.getsize(p + '.bc'), p) for p in module_paths]
  sizes_and_paths.sort(reverse=True)
  sorted_module_paths = [p for _, p in sizes_and_paths]
  file_paths = [
      tuple([p + suffix for suffix in file_suffix]) for p in sorted_module_paths
  ]

  worker_count = (
      min(os.cpu_count(), FLAGS.num_workers)
      if FLAGS.num_workers else os.cpu_count())
  with tf.io.TFRecordWriter(FLAGS.output_path) as file_writer:
    ctx = multiprocessing.get_context()
    m = ctx.Manager()
    results_queue = m.Queue()
    work_queue = m.Queue()
    for path in file_paths:
      work_queue.put(path)
    processes = [
        ctx.Process(
            target=functools.partial(worker, runner, work_queue, results_queue))
        for _ in range(0, worker_count)
    ]

    for p in processes:
      p.start()

    total_successful_examples = 0
    total_work = len(file_paths)
    total_failed_examples = 0
    for _ in range(0, total_work):
      _, record = results_queue.get()
      if record:
        total_successful_examples += 1
        file_writer.write(record[0])
      else:
        total_failed_examples += 1

      logging.log_every_n_seconds(logging.INFO,
                                  '%d success, %d failed out of %d', 10,
                                  total_successful_examples,
                                  total_failed_examples, total_work)

    print('%d of %d modules succeeded.' %
          (total_successful_examples, len(file_paths)))
    for p in processes:
      p.join()


if __name__ == '__main__':
  flags.mark_flag_as_required('data_path')
  flags.mark_flag_as_required('output_path')
  multiprocessing.handle_main(functools.partial(app.run, main))
