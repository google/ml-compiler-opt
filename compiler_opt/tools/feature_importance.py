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
"""A tool for analyzing which features a model uses to make a decision."""

from absl import app
from absl import flags
from absl import logging
import gin

from compiler_opt.rl import data_reader
from compiler_opt.rl import constant
from compiler_opt.rl import registry

_DATA_PATH = flags.DEFINE_multi_string(
  'data_path', [],
  'Path to TFRecord file(s) containing trace data.')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()

  tfrecord_dataset_fn = data_reader.create_tfrecord_dataset_fn(
    agent_name=constant.AgentName.BEHAVIORAL_CLONE,
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    batch_size=1,
    train_sequence_length=1)
  
  dataset_iter = iter(tfrecord_dataset_fn(_DATA_PATH.value).repeat())

  experience = next(dataset_iter)

  logging.info(experience)

if __name__ == '__main__':
  app.run(main)