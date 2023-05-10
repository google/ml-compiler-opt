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
"""Tests for generate_test_model.

An integration test for model saving, to detect TFLite model conversion.
"""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import gin

from compiler_opt.tools import generate_test_model

flags.FLAGS['gin_files'].allow_override = True
flags.FLAGS['gin_bindings'].allow_override = True


def _get_test_settings():
  test_setting = []

  agent_config_type_dict = {
      'ppo': '@agents.PPOAgentConfig',
      'behavioral_cloning': '@agents.BCAgentConfig',
  }

  for problem in ('inlining', 'regalloc'):
    for algorithm in ('ppo', 'behavioral_cloning'):
      test_name = f'{problem}_{algorithm}'
      gin_file = (
          f'compiler_opt/rl/{problem}/gin_configs/{algorithm}_nn_agent.gin')
      gin_binding = ('generate_test_model.agent_config_type=' +
                     agent_config_type_dict[algorithm])
      test_setting.append((test_name, gin_file, gin_binding))

  return test_setting


class GenerateTestModelTest(parameterized.TestCase):

  @parameterized.named_parameters(*_get_test_settings())
  def test_generate_test_model(self, gin_file, gin_binding):

    tmp_dir = self.create_tempdir()

    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          config_files=[gin_file], bindings=[gin_binding], skip_unknown=True)

    with flagsaver.flagsaver(root_dir=tmp_dir.full_path):
      generate_test_model.generate_test_model()


if __name__ == '__main__':
  absltest.main()
