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

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory for writing logs/summaries/checkpoints.',
    required=True)
_REPLAY_BUFFER_CAPACITY = flags.DEFINE_integer(
    'replay_buffer_capacity', 1000000, 'Capacity of the replay buffer table.')
_PORT = flags.DEFINE_integer(
    'port', None, 'Port to start the server on.', required=True)
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')


def main(_):
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  ppo_reverb_server_lib.run_reverb_server(
      _ROOT_DIR.value,
      port=_PORT.value,
      replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY.value)


if __name__ == '__main__':
  app.run(main)
