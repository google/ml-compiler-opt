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
r"""Train and Eval with local_data_collector."""

import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin

from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import train_locally_lib

_ROOT_DIR = flags.DEFINE_string(
    'root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.')
_DATA_PATH = flags.DEFINE_string(
    'data_path',
    None,
    'Path to directory containing the corpus.',
    required=True)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel data collection workers. `None` for max available')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')


def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  train_locally_lib.train_eval(_ROOT_DIR.value, _DATA_PATH.value,
                               _NUM_WORKERS.value)


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
