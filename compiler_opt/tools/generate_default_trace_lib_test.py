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
"""Tests for generate_default_trace."""
import json
import os
from unittest import mock

from absl.testing import absltest
import gin
import tensorflow as tf

# This is https://github.com/google/pytype/issues/764
from google.protobuf import text_format  # pytype: disable=pyi-error
from compiler_opt.rl import compilation_runner
from compiler_opt.tools import generate_default_trace_lib

from tf_agents.system import system_multiprocessing as multiprocessing


@gin.configurable(module='runners')
class MockCompilationRunner(compilation_runner.CompilationRunner):
  """A compilation runner just for test."""

  def __init__(self, moving_average_decay_rate: float, sentinel=None):
    del moving_average_decay_rate  # Unused.
    assert sentinel == 42
    super().__init__()

  def collect_data(self,
                   loaded_module_spec,
                   policy=None,
                   reward_stat=None,
                   model_id=None):
    sequence_example_text = """
      feature_lists {
        feature_list {
          key: "feature_0"
          value {
            feature { int64_list { value: 1 } }
            feature { int64_list { value: 1 } }
          }
        }
      }"""
    sequence_example = text_format.Parse(sequence_example_text,
                                         tf.train.SequenceExample())

    key = f'key_{os.getpid()}'
    return compilation_runner.CompilationResult(
        sequence_examples=[sequence_example],
        reward_stats={
            key:
                compilation_runner.RewardStat(
                    default_reward=1, moving_average_reward=2)
        },
        rewards=[1.2],
        policy_rewards=[18],
        keys=[key],
        model_id=model_id)


class GenerateDefaultTraceTest(absltest.TestCase):

  def setUp(self):
    with gin.unlock_config():
      gin.parse_config_files_and_bindings(
          config_files=['compiler_opt/rl/inlining/gin_configs/common.gin'],
          bindings=['runners.MockCompilationRunner.sentinel=42'])
    return super().setUp()

  @mock.patch('compiler_opt.rl.inlining.InliningConfig.get_runner_type')
  def test_generate_trace(self, mock_get_runner):

    tmp_dir = self.create_tempdir()
    module_names = ['a', 'b', 'c', 'd']

    with tf.io.gfile.GFile(
        os.path.join(tmp_dir.full_path, 'corpus_description.json'), 'w') as f:
      json.dump({'modules': module_names, 'has_thinlto': False}, f)

    for module_name in module_names:
      with tf.io.gfile.GFile(
          os.path.join(tmp_dir.full_path, module_name + '.bc'), 'w') as f:
        f.write(module_name)

      with tf.io.gfile.GFile(
          os.path.join(tmp_dir.full_path, module_name + '.cmd'), 'w') as f:
        f.write('-cc1')

    mock_compilation_runner = MockCompilationRunner
    mock_get_runner.return_value = mock_compilation_runner

    generate_default_trace_lib.generate_trace(
        tmp_dir.full_path,
        os.path.join(tmp_dir.full_path, 'output'),
        os.path.join(tmp_dir.full_path, 'output_performance'),
        num_workers=2,
        sampling_rate=1,
        module_filter_str=None,
        key_filter=None,
        keys_file_path=os.path.join(tmp_dir.full_path, 'keys_file'),
        policy_path='')

    with open(
        os.path.join(tmp_dir.full_path, 'keys_file'),
        encoding='utf-8') as keys_file:
      keys = [key_line.strip() for key_line in keys_file.readlines()]
      for key in keys:
        self.assertStartsWith(key, 'key_')


if __name__ == '__main__':
  multiprocessing.handle_test_main(absltest.main)
