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

from absl import app
from absl import flags
from absl import logging
import gin
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.tools import generate_default_trace_lib

_DATA_PATH = flags.DEFINE_string(
    'data_path', None, 'Path to folder containing IR files.', required=True)
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
_KEYS_FILE = flags.DEFINE_string(
    'keys_file', None,
    'The path to the file to write out the keys encountered.')


def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())
  generate_default_trace_lib.generate_trace(
      _DATA_PATH.value, _OUTPUT_PATH.value, _OUTPUT_PERFORMANCE_PATH.value,
      _NUM_WORKERS.value, _SAMPLING_RATE.value, _MODULE_FILTER.value,
      _KEY_FILTER.value, _KEYS_FILE.value, _POLICY_PATH.value)


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
