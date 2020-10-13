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

"""Tests for compiler_opt.rl.inline_runner."""

import string
import subprocess
from unittest import mock

import tensorflow as tf

from google.protobuf import text_format
from compiler_opt.rl import constant
from compiler_opt.rl import inline_runner

_DEFAULT_FEATURE_VALUE = 12
_POLICY_FEATURE_VALUE = 34

_DEFAULT_NATIVE_SIZE = 10
_POLICY_NATIVE_SIZE = 8


def _get_sequence_example_with_delta_size(feature_value, delta_size):
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
        key: "delta_size"
        value {
          feature { int64_list { value: $delta_size } }
          feature { int64_list { value: $delta_size } }
        }
      }
    }""").substitute(
        feature_value=feature_value, delta_size=delta_size)
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


def _inline(input_ir_path, tf_policy_path, size_only):
  del input_ir_path
  if tf_policy_path:
    sequence_example = _get_sequence_example(_POLICY_FEATURE_VALUE)
    native_size = _POLICY_NATIVE_SIZE
  else:
    sequence_example = _get_sequence_example(_DEFAULT_FEATURE_VALUE)
    native_size = _DEFAULT_NATIVE_SIZE

  if size_only:
    return None, native_size
  else:
    return sequence_example, native_size


class InlineRunnerTest(tf.test.TestCase):

  @mock.patch(constant.BASE_MODULE_DIR +
              '.inline_runner.InlineRunner._run_inlining')
  def test_policy(self, mock_run_inlining):
    mock_run_inlining.side_effect = _inline
    inliner = inline_runner.InlineRunner('', '', '')

    example, default_size = inliner.collect_data(
        ir_path='ir', tf_policy_path='policy_path', default_policy_size=None)
    self.assertEqual(2, mock_run_inlining.call_count)

    expected_example = _get_sequence_example_with_delta_size(
        _POLICY_FEATURE_VALUE, -1999)
    self.assertProtoEquals(expected_example,
                           tf.train.SequenceExample.FromString(example))
    self.assertEqual(_DEFAULT_NATIVE_SIZE, default_size)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.inline_runner.InlineRunner._run_inlining')
  def test_default(self, mock_run_inlining):
    mock_run_inlining.side_effect = _inline
    inliner = inline_runner.InlineRunner('', '', '')

    example, default_size = inliner.collect_data(
        ir_path='ir', tf_policy_path='', default_policy_size=None)
    self.assertEqual(2, mock_run_inlining.call_count)

    expected_example = _get_sequence_example_with_delta_size(
        _DEFAULT_FEATURE_VALUE, 0)
    self.assertProtoEquals(expected_example,
                           tf.train.SequenceExample.FromString(example))
    self.assertEqual(_DEFAULT_NATIVE_SIZE, default_size)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.inline_runner.InlineRunner._run_inlining')
  def test_given_default_size(self, mock_run_inlining):
    mock_run_inlining.side_effect = _inline
    inliner = inline_runner.InlineRunner('', '', '')

    example, default_size = inliner.collect_data(
        ir_path='ir',
        tf_policy_path='policy_path',
        default_policy_size=_DEFAULT_NATIVE_SIZE)
    self.assertEqual(1, mock_run_inlining.call_count)

    expected_example = _get_sequence_example_with_delta_size(
        _POLICY_FEATURE_VALUE, -1999)
    self.assertProtoEquals(expected_example,
                           tf.train.SequenceExample.FromString(example))
    self.assertEqual(_DEFAULT_NATIVE_SIZE, default_size)

  @mock.patch(constant.BASE_MODULE_DIR +
              '.inline_runner.InlineRunner._run_inlining')
  def test_exception_handling(self, mock_run_inlining):
    mock_run_inlining.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd='error')
    inliner = inline_runner.InlineRunner('', '', '')

    with self.assertRaisesRegexp(subprocess.CalledProcessError, 'error'):
      _, _ = inliner.collect_data(
          ir_path='ir', tf_policy_path='policy_path', default_policy_size=None)
    self.assertEqual(1, mock_run_inlining.call_count)


if __name__ == '__main__':
  tf.test.main()
