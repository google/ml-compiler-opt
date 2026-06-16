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
"""Tests for compiler_opt.rl.compilation_runner."""

import math
import string
import subprocess
from unittest import mock

import tensorflow as tf

# This is https://github.com/google/pytype/issues/764
from google.protobuf import text_format  # pytype: disable=pyi-error

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import constant
from compiler_opt.rl import corpus
from compiler_opt.rl import policy_saver

_DEFAULT_FEATURE_VALUE = 12
_POLICY_FEATURE_VALUE = 34

_DEFAULT_REWARD = 10
_POLICY_REWARD = 8

_MOVING_AVERAGE_DECAY_RATE = 0.8


def _get_sequence_example_with_reward(feature_value, reward):
  sequence_example_text = string.Template("""
    feature_lists {
      feature_list {
        key: "feature_0"
        value {
          feature { int64_list { value: $feature_value } }
          feature { int64_list { value: $feature_value } }
        }
      }
      feature_list {
        key: "reward"
        value {
          feature { float_list { value: $reward } }
          feature { float_list { value: $reward } }
        }
      }
    }""").substitute(
      feature_value=feature_value, reward=reward)
  return text_format.Parse(sequence_example_text, tf.train.SequenceExample())


def _get_sequence_example(feature_value):
  sequence_example_text = string.Template("""
    feature_lists {
      feature_list {
        key: "feature_0"
        value {
          feature { int64_list { value: $feature_value } }
          feature { int64_list { value: $feature_value } }
        }
      }
    }""").substitute(feature_value=feature_value)
  return text_format.Parse(sequence_example_text, tf.train.SequenceExample())


def _mock_compile_fn(file_paths, tf_policy_path, reward_only, workdir):  # pylint: disable=unused-argument
  del file_paths
  if tf_policy_path:
    sequence_example = _get_sequence_example(_POLICY_FEATURE_VALUE)
    native_size = _POLICY_REWARD
  else:
    sequence_example = _get_sequence_example(_DEFAULT_FEATURE_VALUE)
    native_size = _DEFAULT_REWARD

  if reward_only:
    return {'default': (None, native_size)}
  else:
    return {'default': (sequence_example, native_size)}


_mock_policy = policy_saver.Policy(b'', b'')

_mock_loaded_module_spec = corpus.LoadedModuleSpec(name='dummy', loaded_ir=b'')


class CompilationRunnerTest(tf.test.TestCase):

  def assertListProtoEqual(self, a, b):
    self.assertEqual(len(a), len(b))
    a.sort()
    b.sort()
    for (x, y) in zip(a, b):
      self.assertProtoEquals(x, y)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.compilation_runner.CompilationRunner.compile_fn')
  def test_policy(self, mock_compile_fn):
    mock_compile_fn.side_effect = _mock_compile_fn
    runner = compilation_runner.CompilationRunner(
        moving_average_decay_rate=_MOVING_AVERAGE_DECAY_RATE)
    data = runner.collect_data(
        loaded_module_spec=_mock_loaded_module_spec, policy=_mock_policy)
    self.assertEqual(2, mock_compile_fn.call_count)

    expected_example = _get_sequence_example_with_reward(
        _POLICY_FEATURE_VALUE, 0.1998)
    self.assertListProtoEqual([expected_example], [
        tf.train.SequenceExample.FromString(x)
        for x in data.serialized_sequence_examples
    ])
    self.assertEqual(2, data.length)
    self.assertCountEqual(
        {
            'default':
                compilation_runner.RewardStat(
                    default_reward=_DEFAULT_REWARD,
                    moving_average_reward=_DEFAULT_REWARD *
                    _MOVING_AVERAGE_DECAY_RATE + _POLICY_REWARD *
                    (1 - _MOVING_AVERAGE_DECAY_RATE))
        }, data.reward_stats)
    self.assertAllClose([0.1998002], data.rewards)
    self.assertAllClose([8], data.policy_rewards)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.compilation_runner.CompilationRunner.compile_fn')
  def test_default(self, mock_compile_fn):
    mock_compile_fn.side_effect = _mock_compile_fn
    runner = compilation_runner.CompilationRunner(
        moving_average_decay_rate=_MOVING_AVERAGE_DECAY_RATE)

    data = runner.collect_data(loaded_module_spec=_mock_loaded_module_spec)
    # One call when we ask for the default policy, because it can provide both
    # trace and default size.
    self.assertEqual(1, mock_compile_fn.call_count)

    expected_example = _get_sequence_example_with_reward(
        _DEFAULT_FEATURE_VALUE, 0)
    self.assertListProtoEqual([expected_example], [
        tf.train.SequenceExample.FromString(x)
        for x in data.serialized_sequence_examples
    ])
    self.assertEqual(2, data.length)
    self.assertCountEqual(
        {
            'default':
                compilation_runner.RewardStat(
                    default_reward=_DEFAULT_REWARD,
                    moving_average_reward=_DEFAULT_REWARD)
        }, data.reward_stats)
    self.assertAllClose([0], data.rewards)
    self.assertAllClose([10], data.policy_rewards)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.compilation_runner.CompilationRunner.compile_fn')
  def test_given_default_size(self, mock_compile_fn):
    mock_compile_fn.side_effect = _mock_compile_fn
    runner = compilation_runner.CompilationRunner(
        moving_average_decay_rate=_MOVING_AVERAGE_DECAY_RATE)

    data = runner.collect_data(
        loaded_module_spec=_mock_loaded_module_spec,
        policy=_mock_policy,
        reward_stat={
            'default':
                compilation_runner.RewardStat(
                    default_reward=_DEFAULT_REWARD, moving_average_reward=7)
        })
    self.assertEqual(1, mock_compile_fn.call_count)

    expected_example = _get_sequence_example_with_reward(
        _POLICY_FEATURE_VALUE,
        1 - (_POLICY_REWARD + constant.DELTA) / (7 + constant.DELTA))
    self.assertListProtoEqual([expected_example], [
        tf.train.SequenceExample.FromString(x)
        for x in data.serialized_sequence_examples
    ])
    self.assertEqual(2, data.length)
    self.assertCountEqual(
        {
            'default':
                compilation_runner.RewardStat(
                    default_reward=_DEFAULT_REWARD,
                    moving_average_reward=7 * _MOVING_AVERAGE_DECAY_RATE +
                    _POLICY_REWARD * (1 - _MOVING_AVERAGE_DECAY_RATE))
        }, data.reward_stats)
    self.assertAllClose([0.199800], data.rewards)
    self.assertAllClose([8], data.policy_rewards)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.compilation_runner.CompilationRunner.compile_fn')
  def test_exception_handling(self, mock_compile_fn):
    mock_compile_fn.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd='error')
    runner = compilation_runner.CompilationRunner(
        moving_average_decay_rate=_MOVING_AVERAGE_DECAY_RATE)

    with self.assertRaisesRegex(subprocess.CalledProcessError, 'error'):
      _ = runner.collect_data(
          loaded_module_spec=_mock_loaded_module_spec,
          policy=_mock_policy,
          reward_stat=None)
    self.assertEqual(1, mock_compile_fn.call_count)


  def test_calculate_reward_zero_delta(self):
    reward = compilation_runner.calculate_reward(3, 0)
    self.assertTrue(math.isfinite(reward))

  def test_calculate_reward(self):
    reward = compilation_runner.calculate_reward(1, 2)
    self.assertAlmostEqual(reward, 0.5, 2)


if __name__ == '__main__':
  tf.test.main()
