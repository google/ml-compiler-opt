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
"""Collect experiences for PPO training"""

import gin
import functools

from absl import app
from absl import flags
from absl import logging

from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl.distributed import ppo_collect_lib
from compiler_opt.distributed.local.local_worker_manager import LocalWorkerPoolManager

flags.DEFINE_string('root_dir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('corpus_path', None, 'Path to the training corpus.')
flags.DEFINE_string('replay_buffer_server_address', None,
                    'Replay buffer server address.')
flags.DEFINE_string('variable_container_server_address', None,
                    'Variable container server address.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')
flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel data collection workers. `None` for max available')

FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
  logging.info(gin.config_str())

  ppo_collect_lib.run_collect(
      root_dir=FLAGS.root_dir,
      corpus_path=FLAGS.corpus_path,
      replay_buffer_server_address=FLAGS.replay_buffer_server_address,
      variable_container_server_address=FLAGS.variable_container_server_address,
      num_workers=FLAGS.num_workers,
      worker_manager_class=LocalWorkerPoolManager)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir', 'corpus_path', 'replay_buffer_server_address',
      'variable_container_server_address'
  ])
  multiprocessing.handle_main(functools.partial(app.run, main))
