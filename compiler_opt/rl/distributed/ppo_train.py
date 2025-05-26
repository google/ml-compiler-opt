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
"""Binary to train a policy with PPO"""

import functools
import os

from absl import app
from absl import flags
from absl import logging

import gin

from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import strategy_utils

from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl.distributed import ppo_train_lib

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
    required=True)
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address',
    None,
    'Replay buffer server address.',
    required=True)
_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.',
    required=True)
_GIN_FILES = flags.DEFINE_multi_string('gin_files', None,
                                       'Paths to the gin-config files.')
_GIN_BINDINGS = flags.DEFINE_multi_string('gin_bindings', None,
                                          'Gin binding parameters.')

FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(_GIN_FILES.value, _GIN_BINDINGS.value)
  logging.info(gin.config_str())

  strategy = strategy_utils.get_strategy(FLAGS.tpu, FLAGS.use_gpu)

  ppo_train_lib.train(
      _ROOT_DIR.value,
      strategy=strategy,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value
  )


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
