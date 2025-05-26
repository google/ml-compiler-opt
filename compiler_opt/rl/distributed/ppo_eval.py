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
"""Evaluate the current policy stored in reverb"""

import functools

import gin

from absl import app
from absl import flags
from absl import logging

from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl.distributed import ppo_eval_lib
from compiler_opt.distributed.local.local_worker_manager import LocalWorkerPoolManager

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory for writing logs/summaries/checkpoints.',
    required=True)
_CORPUS_PATH = flags.DEFINE_string('corpus_path', None,
                                   'Path to the training corpus.')
_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.',
    required=True)
_GIN_FILES = flags.DEFINE_multi_string('gin_files', None,
                                       'Paths to the gin-config files.')
_GIN_BINDINGS = flags.DEFINE_multi_string('gin_bindings', None,
                                          'Gin binding parameters.')
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel data collection workers. `None` for max available')


def main(_):
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(_GIN_FILES.value, _GIN_BINDINGS.value)
  logging.info(gin.config_str())

  ppo_eval_lib.run_evaluate(
      root_dir=_ROOT_DIR.value,
      corpus_path=_CORPUS_PATH.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS
      .value,
      num_workers=_NUM_WORKERS.value,
      worker_manager_class=LocalWorkerPoolManager)


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
