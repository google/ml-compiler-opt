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
"""Generate test model given a problem and an algorithm."""

import os

from absl import app
from absl import flags
from absl import logging

import gin

from compiler_opt.rl import agent_config
from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import policy_saver
from compiler_opt.rl import registry

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing saved models.')
flags.DEFINE_multi_string('gin_files', [],
                          'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


@gin.configurable
def generate_test_model(agent_config_type=agent_config.PPOAgentConfig):
  """Generate test model."""
  root_dir = FLAGS.root_dir

  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()
  preprocessing_layer_creator = problem_config.get_preprocessing_layer_creator()

  # Initialize trainer and policy saver.
  tf_agent = agent_config.create_agent(
      agent_config_type(time_step_spec=time_step_spec, action_spec=action_spec),
      preprocessing_layer_creator=preprocessing_layer_creator)

  policy_dict = {
      'saved_policy': tf_agent.policy,
      'saved_collect_policy': tf_agent.collect_policy,
  }
  saver = policy_saver.PolicySaver(policy_dict=policy_dict)

  # Save policy.
  saver.save(root_dir)


def main(_):
  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=FLAGS.gin_bindings, skip_unknown=True)
  logging.info(gin.config_str())

  generate_test_model()


if __name__ == '__main__':
  app.run(main)
