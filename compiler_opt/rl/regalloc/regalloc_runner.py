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

import os
import tempfile
from typing import Dict, Tuple

import gin
import tensorflow as tf

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader


@gin.configurable(module='runners')
class RegAllocRunner(compilation_runner.CompilationRunner):
  """Class for collecting data for regalloc-for-performance.

  Usage:
  runner = RegAllocRunner(
               clang_path, launcher_path, moving_average_decay_rate)
  serialized_sequence_example, default_reward, moving_average_reward,
  policy_reward = runner.collect_data(
      ir_path, tf_policy_path, default_reward, moving_average_reward)
  """

  # TODO: refactor file_paths parameter to ensure correctness during
  # construction
  def compile_fn(
      self, command_line: corpus.FullyQualifiedCmdLine, tf_policy_path: str,
      reward_only: bool,
      workdir: str) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:
    """Run the compiler for the given IR file under the given policy.

    Args:
      command_line: the fully qualified command line.
      tf_policy_path: path to TF policy direcoty on local disk.
      reward_only: whether only return reward.

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

    working_dir = tempfile.mkdtemp(dir=workdir)

    log_path = os.path.join(working_dir, 'log')
    output_native_path = os.path.join(working_dir, 'native')

    result = {}
    cmdline = []
    if self._launcher_path:
      cmdline.append(self._launcher_path)
    cmdline.extend([self._clang_path] + list(command_line) + [
        '-mllvm', '-regalloc-enable-advisor=development', '-mllvm',
        '-regalloc-training-log=' + log_path, '-o', output_native_path
    ])

    if tf_policy_path:
      cmdline.extend(['-mllvm', '-regalloc-model=' + tf_policy_path])
    compilation_runner.start_cancellable_process(cmdline,
                                                 self._compilation_timeout,
                                                 self._cancellation_manager)

    # TODO(#202)
    log_result = log_reader.read_log_as_sequence_examples(log_path)

    for fct_name, trajectory in log_result.items():
      if not trajectory.HasField('feature_lists'):
        continue
      score = (
          trajectory.feature_lists.feature_list['reward'].feature[-1].float_list
          .value[0])
      if reward_only:
        result[fct_name] = (None, score)
      else:
        del trajectory.feature_lists.feature_list['reward']
        result[fct_name] = (trajectory, score)

    return result
