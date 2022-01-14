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

"""Tests for compiler_opt.rl.compilation_runner."""

import string
import subprocess
from unittest import mock

import tensorflow as tf

from google.protobuf import text_format
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import constant

_DEFAULT_FEATURE_VALUE = 12
_POLICY_FEATURE_VALUE = 34

_DEFAULT_REWARD = 10
_POLICY_REWARD = 8


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


def _mock_compile_fn(file_paths, tf_policy_path, reward_only):
  del file_paths
  if tf_policy_path:
    sequence_example = _get_sequence_example(_POLICY_FEATURE_VALUE)
    native_size = _POLICY_REWARD
  else:
    sequence_example = _get_sequence_example(_DEFAULT_FEATURE_VALUE)
    native_size = _DEFAULT_REWARD

  if reward_only:
    return None, native_size
  else:
    return sequence_example, native_size


class CompilationRunnerTest(tf.test.TestCase):

  @mock.patch(constant.BASE_MODULE_DIR +
              '.compilation_runner.CompilationRunner._compile_fn')
  def test_policy(self, mock_compile_fn):
    mock_compile_fn.side_effect = _mock_compile_fn
    runner = compilation_runner.CompilationRunner('', '')
    example, default_reward = runner.collect_data(
        file_paths=('bc', 'cmd'),
        tf_policy_path='policy_path',
        default_reward=None)
    self.assertEqual(2, mock_compile_fn.call_count)

    expected_example = _get_sequence_example_with_reward(
        _POLICY_FEATURE_VALUE, 0.1999999)
    self.assertProtoEquals(expected_example,
                           tf.train.SequenceExample.FromString(example))
    self.assertEqual(_DEFAULT_REWARD, default_reward)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.compilation_runner.CompilationRunner._compile_fn')
  def test_default(self, mock_compile_fn):
    mock_compile_fn.side_effect = _mock_compile_fn
    runner = compilation_runner.CompilationRunner('', '')

    example, default_reward = runner.collect_data(
        file_paths=('bc', 'cmd'), tf_policy_path='', default_reward=None)
    # One call when we ask for the default policy, because it can provide both
    # trace and default size.
    self.assertEqual(1, mock_compile_fn.call_count)

    expected_example = _get_sequence_example_with_reward(
        _DEFAULT_FEATURE_VALUE, 0)
    self.assertProtoEquals(expected_example,
                           tf.train.SequenceExample.FromString(example))
    self.assertEqual(_DEFAULT_REWARD, default_reward)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.compilation_runner.CompilationRunner._compile_fn')
  def test_given_default_size(self, mock_compile_fn):
    mock_compile_fn.side_effect = _mock_compile_fn
    runner = compilation_runner.CompilationRunner('', '')

    example, default_reward = runner.collect_data(
        file_paths=('bc', 'cmd'),
        tf_policy_path='policy_path',
        default_reward=_DEFAULT_REWARD)
    self.assertEqual(1, mock_compile_fn.call_count)

    expected_example = _get_sequence_example_with_reward(
        _POLICY_FEATURE_VALUE, 0.1999999)
    self.assertProtoEquals(expected_example,
                           tf.train.SequenceExample.FromString(example))
    self.assertEqual(_DEFAULT_REWARD, default_reward)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.compilation_runner.CompilationRunner._compile_fn')
  def test_exception_handling(self, mock_compile_fn):
    mock_compile_fn.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd='error')
    runner = compilation_runner.CompilationRunner('', '')

    with self.assertRaisesRegex(subprocess.CalledProcessError, 'error'):
      _, _ = runner.collect_data(
          file_paths=('bc', 'cmd'),
          tf_policy_path='policy_path',
          default_reward=None)
    self.assertEqual(1, mock_compile_fn.call_count)


if __name__ == '__main__':
  tf.test.main()
