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

r"""Train behavioral cloning policy for LLVM Inliner decision rule."""

import os

from absl import app
from absl import flags
from absl import logging
import gin

from compiler_opt.rl import agent_creators
from compiler_opt.rl import config
from compiler_opt.rl import constant
from compiler_opt.rl import data_reader
from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import policy_saver
from compiler_opt.rl import trainer

_ROOT_DIR = flags.DEFINE_string(
    'root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.')
_DATA_PATH = flags.DEFINE_multi_string(
    'data_path', [],
    'Path to TFRecord file(s) containing training data. Skip training and dump'
    'an untrained model with random weights (for testing purpose) if unspecified.'
)
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')


@gin.configurable
def train_eval(agent_name=constant.AgentName.BEHAVIORAL_CLONE,
               problem_type=None,
               num_iterations=100,
               batch_size=64,
               train_sequence_length=1):
  """Train for LLVM inliner."""
  root_dir = os.path.expanduser(_ROOT_DIR.value)
  root_dir = os.path.normpath(root_dir)

  time_step_spec, action_spec = config.get_signature_spec(
      problem_type)
  preprocessing_layer_creator = config.get_preprocessing_layer_creator(
      problem_type)

  # Initialize trainer and policy saver.
  tf_agent = agent_creators.create_agent(agent_name, time_step_spec,
                                         action_spec,
                                         preprocessing_layer_creator)
  llvm_trainer = trainer.Trainer(root_dir=root_dir, agent=tf_agent)
  policy_dict = {
      'saved_policy':
          tf_agent.policy,
      'saved_collect_policy':
          tf_agent.collect_policy,
  }
  saver = policy_saver.PolicySaver(policy_dict=policy_dict)

  tfrecord_dataset_fn = data_reader.create_tfrecord_dataset_fn(
      agent_name=agent_name,
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      batch_size=batch_size,
      train_sequence_length=train_sequence_length)

  # Train.
  if _DATA_PATH.value:
    dataset_iter = iter(tfrecord_dataset_fn(_DATA_PATH.value).repeat())
    monitor_dict = {}
    llvm_trainer.train(dataset_iter, monitor_dict, num_iterations)

  # Save final policy.
  saver.save(root_dir)


def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  train_eval()


if __name__ == '__main__':
  app.run(main)
