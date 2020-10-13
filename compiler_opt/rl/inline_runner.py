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

"""Module for collect data of inlining-for-size.

unit test at experimental/ml_compiler/unittests/inliner_test.py
"""

import io
import os
import subprocess
import tempfile

import tensorflow as tf

from google.protobuf import text_format


class InlineRunner(object):
  """Class for collecting data for inlining-for-size.

  Usage:
  inliner = InlineRunner(opt_path, llc_path, llvm_size_path)
  serialized_sequence_example, default_policy_size = inliner.collect_data(
      ir_path, tf_policy_path, default_policy_size)
  """

  def __init__(self, opt_path, llc_path, llvm_size_path):
    """Initialization of InlineRunner class.

    Args:
      opt_path: path to the opt binary.
      llc_path: path to the llc binary.
      llvm_size_path: path to the llvm-size binary.
    """
    self._opt_path = opt_path
    self._llc_path = llc_path
    self._llvm_size_path = llvm_size_path

  def collect_data(self, ir_path, tf_policy_path, default_policy_size):
    """Collect data for the given IR file and policy.

    Args:
      ir_path: path to the IR file.
      tf_policy_path: path to the inlining policy.
      default_policy_size: native size under default inlining, None if unknown.

    Returns:
      A tuple containing:
        sequence_example: A serialized tf.SequenceExample proto.
        default_policy_size: Native size under default inlining policy.

    Raises:
      subprocess.CalledProcessError if process fails.
    """
    try:
      if default_policy_size is None:
        _, default_policy_size = self._run_inlining(
            ir_path, tf_policy_path='', size_only=True)

      # Return empty example if the default policy size is 0 since it is a data
      # only module and we can do nothing about it.
      if default_policy_size == 0:
        return (tf.train.SequenceExample().SerializeToString(),
                default_policy_size)

      sequence_example, native_size = self._run_inlining(
          ir_path, tf_policy_path, size_only=False)
    except subprocess.CalledProcessError as e:
      raise e

    if not sequence_example.HasField('feature_lists'):
      return tf.train.SequenceExample().SerializeToString(), default_policy_size

    sequence_example = self._postprocessing_sequence_example(
        sequence_example, default_policy_size, native_size)

    return sequence_example.SerializeToString(), default_policy_size

  def _run_inlining(self, input_ir_path, tf_policy_path, size_only):
    """Run inlining for the given IR file under the given policy.

    Args:
      input_ir_path: path to IR file on local disk.
      tf_policy_path: path to TF policy direcoty on local disk.
      size_only: whether only return native size.

    Returns:
      A tuple containing:
        sequence_example: A tf.SequenceExample proto describing inlining trace.
        native_size: Native size of the final native code.

    Raises:
      subprocess.CalledProcessError if process fails.
    """
    working_dir = tempfile.mkdtemp()

    output_ir_path = os.path.join(working_dir, 'output_ir')
    log_path = os.path.join(working_dir, 'log')
    output_native_path = os.path.join(working_dir, 'native')

    try:
      command_line = [
          self._opt_path, '-enable-ml-inliner=development',
          '-aa-pipeline=default',
          '-passes=scc-oz-module-inliner,oz-module-optimizer', input_ir_path,
          '-training-log', log_path, '-o', output_ir_path
      ]
      if tf_policy_path:
        command_line.extend(
            ['-ml-inliner-model-under-training', tf_policy_path])
      subprocess.check_call(command_line)

      command_line = [
          self._llc_path, '-O2', '-filetype=obj', output_ir_path, '-o',
          output_native_path
      ]
      subprocess.check_call(command_line)

      command_line = [self._llvm_size_path, output_native_path]
      output = subprocess.check_output(command_line).decode('utf-8')

      tmp = output.split('\n')
      if len(tmp) != 3:
        raise RuntimeError('Wrong llvm-size output %s' % output)
      tmp = tmp[1].split('\t')
      native_size = int(tmp[0])

      if size_only:
        tf.io.gfile.rmtree(working_dir)
        return None, native_size

      with io.open(log_path, 'r') as f:
        sequence_example = text_format.MergeLines(f, tf.train.SequenceExample())

      tf.io.gfile.rmtree(working_dir)
    except (subprocess.CalledProcessError, tf.errors.OpError) as e:
      raise e

    return sequence_example, native_size

  def _postprocessing_sequence_example(self, sequence_example,
                                       default_policy_size, tf_policy_size):
    """Post-processing of the trace (sequence_example).

    It computes the ratio of the native size shrinkage of the TF policy inlining
    compared with the default inlining, and uses this ratio as the whole
    trajectory reward to overwrite the original reward after each action.

    Args:
      sequence_example: A tf.SequenceExample proto describing inlining trace.
      default_policy_size: The native size under default inlining.
      tf_policy_size: The native size under the TF policy inlining.

    Returns:
      The tf.SequenceExample proto after post-processing.
    """
    total_trajectory_reward = int(
        (tf_policy_size / default_policy_size - 1) * 10000)

    sequence_length = len(
        next(iter(
            sequence_example.feature_lists.feature_list.values())).feature)

    delta_size_feature_list = sequence_example.feature_lists.feature_list[
        'delta_size']
    for _ in range(sequence_length):
      added_feature = delta_size_feature_list.feature.add()
      added_feature.int64_list.value.append(total_trajectory_reward)

    return sequence_example
