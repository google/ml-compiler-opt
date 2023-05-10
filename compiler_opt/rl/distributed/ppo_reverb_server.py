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
"""Main binary to launch a stand alone Reverb RB server."""

from absl import app
from absl import flags
from absl import logging

import gin

from compiler_opt.rl.distributed import ppo_reverb_server_lib
from compiler_opt.rl import registry  # pylint: disable=unused-import
from compiler_opt.rl import agent_config  # pylint: disable=unused-import

flags.DEFINE_string('root_dir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('replay_buffer_capacity', 1000000,
                     'Capacity of the replay buffer table.')
flags.DEFINE_integer('port', None, 'Port to start the server on.')
flags.DEFINE_multi_string('gin_files', [],
                          'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=FLAGS.gin_bindings, skip_unknown=False)
  logging.info(gin.config_str())

  ppo_reverb_server_lib.run_reverb_server(
      FLAGS.root_dir,
      port=FLAGS.port,
      replay_buffer_capacity=FLAGS.replay_buffer_capacity)


if __name__ == '__main__':
  flags.mark_flags_as_required(['root_dir', 'port'])
  app.run(main)
