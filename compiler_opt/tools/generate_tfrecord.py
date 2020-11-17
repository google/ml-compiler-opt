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

"""Generate training data in tfrecord format."""

import functools
import os
import random
import time

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
flags.DEFINE_integer('num_workers', -1,
                     'Number of parallel workers for compilation. Set to '
                     'multiprocessing.cpu_count() if set -1.')
flags.DEFINE_float(
    'sampling_rate', 1,
    'Sampling rate of modules, 0.5 means 50% sampling rate that generates data '
    'for half modules.')

FLAGS = flags.FLAGS

_BATCH_SIZE = 1000


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Initialize runner and file_suffix according to compile_task.
  if FLAGS.compile_task == 'inlining':
    runner = inlining_runner.InliningRunner(
        clang_path=FLAGS.clang_path, llvm_size_path=FLAGS.llvm_size_path)
    file_suffix = ['.bc', '.cmd']

  with open(os.path.join(FLAGS.data_path, 'module_paths'), 'r') as f:
    module_paths = [
        os.path.join(FLAGS.data_path, name.rstrip('\n')) for name in f
    ]
    file_paths = [
        tuple([p + suffix for suffix in file_suffix]) for p in module_paths
    ]

  # Sampling if needed.
  if FLAGS.sampling_rate < 1:
    sampled_modules = int(len(file_paths) * FLAGS.sampling_rate)
    file_paths = random.sample(file_paths, k=sampled_modules)

  ctx = multiprocessing.get_context()
  num_workers = FLAGS.num_workers
  if num_workers == -1:
    num_workers = multiprocessing.cpu_count()
  pool = ctx.Pool(num_workers)

  index = 0
  total_successful_examples = 0
  with tf.io.TFRecordWriter(FLAGS.output_path) as file_writer:
    while index < len(file_paths):
      # Shard data collection and sink to tfrecord periodically to avoid OOM.
      next_index = min(index + _BATCH_SIZE, len(file_paths))
      sharded_file_paths = file_paths[index:next_index]
      index = next_index

      results = [
          pool.apply_async(runner.collect_data, (path, '', None))
          for path in sharded_file_paths
      ]

      # Wait till all jobs finish.
      waiting_time = 0
      while True:
        if sum([not r.ready() for r in results]) == 0:
          break
        logging.info('%d/%d: %d of %d modules finished in %d seconds.', index,
                     len(file_paths), sum([r.ready() for r in results]),
                     len(sharded_file_paths), waiting_time)
        time.sleep(1)
        waiting_time += 1

      # Write successful examples to tfrecord.
      successful_count = len(
          [file_writer.write(r.get()[0]) for r in results if r.successful()])
      logging.info('%d/%d: %d of %d modules succeeded.', index, len(file_paths),
                   successful_count, len(sharded_file_paths))
      total_successful_examples += successful_count

  logging.info('%d of %d modules succeeded in total.',
               total_successful_examples, len(file_paths))


if __name__ == '__main__':
  flags.mark_flag_as_required('data_path')
  flags.mark_flag_as_required('output_path')
  multiprocessing.handle_main(functools.partial(app.run, main))
