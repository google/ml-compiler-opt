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
"""Module for collect data of regalloc-for-performance."""

import base64
import io
import os
import tempfile
from typing import Dict, Optional, Tuple

import gin
import tensorflow as tf

# This is https://github.com/google/pytype/issues/764
from google.protobuf import struct_pb2  # pytype: disable=pyi-error
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus


@gin.configurable(module='runners')
class RegAllocRunner(compilation_runner.CompilationRunner):
  """Class for collecting data for regalloc-for-performance.

  Usage:
  runner = RegAllocRunner(
               clang_path, launcher_path, moving_average_decay_rate)
  serialized_sequence_example, default_reward, moving_average_reward,
  policy_reward = inliner.collect_data(
      ir_path, tf_policy_path, default_reward, moving_average_reward)
  """

  def _compile_fn(
      self, module_spec: corpus.ModuleSpec, tf_policy_path: str,
      reward_only: bool, cancellation_manager: Optional[
          compilation_runner.WorkerCancellationManager]
  ) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:
    """Run inlining for the given IR file under the given policy.

    Args:
      module_spec: a ModuleSpec.
      tf_policy_path: path to TF policy direcoty on local disk.
      reward_only: whether only return reward.
      cancellation_manager: handler for early termination by killing any running
        processes

    Returns:
      A dict mapping from example identifier to tuple containing:
        sequence_example: A tf.SequenceExample proto describing compilation
          trace, None if reward_only == True.
        reward: reward of register allocation.

    Raises:
      subprocess.CalledProcessError: if process fails.
      compilation_runner.ProcessKilledError: (which it must pass through) on
        cancelled work.
      RuntimeError: if llvm-size produces unexpected output.
    """
    working_dir = tempfile.mkdtemp()

    log_path = os.path.join(working_dir, 'log')
    output_native_path = os.path.join(working_dir, 'native')

    result = {}
    try:
      command_line = []
      if self._launcher_path:
        command_line.append(self._launcher_path)
      command_line.append(self._clang_path)
      additional_flags = [('-mllvm', '-regalloc-enable-advisor=development'),
                          ('-mllvm',
                           f'-regalloc-training-log={output_native_path}')]
      if tf_policy_path:
        additional_flags.append(('-mllvm', f'-regalloc-model={tf_policy_path}'))
      command_line.extend(module_spec.cmd(additional_flags))

      compilation_runner.start_cancellable_process(command_line,
                                                   self._compilation_timeout,
                                                   cancellation_manager)

      sequence_example = struct_pb2.Struct()

      with io.open(log_path, 'rb') as f:
        sequence_example.ParseFromString(f.read())

      for key, value in sequence_example.fields.items():
        e = tf.train.SequenceExample()
        e.ParseFromString(base64.b64decode(value.string_value))
        if not e.HasField('feature_lists'):
          continue
        r = (
            e.feature_lists.feature_list['reward'].feature[-1].float_list
            .value[0])
        if reward_only:
          result[key] = (None, r)
        else:
          del e.feature_lists.feature_list['reward']
          result[key] = (e, r)

    finally:
      tf.io.gfile.rmtree(working_dir)

    return result
