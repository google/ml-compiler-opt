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
from compiler_opt.rl import data_reader
from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import policy_saver
from compiler_opt.rl import trainer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('data_path', '',
                    'Path to TFRecord file(s) containing training data.')
flags.DEFINE_multi_string('gin_files', [],
                          'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(agent_name='behavioral_cloning',
               num_iterations=100,
               batch_size=64,
               train_sequence_length=1):
  """Train for LLVM inliner."""
  root_dir = os.path.expanduser(FLAGS.root_dir)
  root_dir = os.path.normpath(root_dir)

  # Initialize trainer and policy saver.
  time_step_spec, action_spec = config.create_signature_specs(config.CONFIG)
  tf_agent = agent_creators.create_agent(agent_name, time_step_spec,
                                         action_spec)
  llvm_trainer = trainer.Trainer(root_dir=root_dir, agent=tf_agent)
  policy_dict = {
      'saved_policy':
          tf_agent.policy,
      'saved_collect_policy':
          tf_agent.collect_policy,
  }
  saver = policy_saver.PolicySaver(policy_dict=policy_dict)

  tfrecord_iterator_fn = data_reader.create_tfrecord_iterator_fn(
      agent_name=agent_name,
      config=config.CONFIG,
      batch_size=batch_size,
      train_sequence_length=train_sequence_length)

  # Train.
  dataset_iter = tfrecord_iterator_fn(FLAGS.data_path)
  llvm_trainer.train(dataset_iter, num_iterations)

  # Save final policy.
  saver.save(root_dir)


def main(_):
  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=FLAGS.gin_bindings, skip_unknown=False)
  logging.info(gin.config_str())

  train_eval()


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  flags.mark_flag_as_required('data_path')
  app.run(main)
