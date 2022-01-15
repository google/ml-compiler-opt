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

"""Module for collect data of inlining-for-size."""

import os
import subprocess
import tempfile

from typing import Tuple

import tensorflow as tf

from compiler_opt.rl import compilation_runner

# TODO(mtrofin): maybe the deadline is a requirement for plugins (such as the
# inliner) and the data collector expects and uses it to define its own? This
# would serve as an extra hint to the developer of a new plugin to make sure
# their long-running tasks have timeouts.
_DEADLINE_IN_SECONDS = 60


class InliningRunner(compilation_runner.CompilationRunner):
  """Class for collecting data for inlining-for-size.

  Usage:
  inliner = InliningRunner(clang_path, llvm_size_path)
  serialized_sequence_example, default_reward = inliner.collect_data(
      ir_path, tf_policy_path, default_reward)
  """

  def __init__(self, clang_path, llvm_size_path, launcher_path=None):
    """Initialization of InliningRunner class.

    Args:
      clang_path: path to the clang binary.
      llvm_size_path: path to the llvm-size binary.
      launcher_path: path to the launcher binary.
    """
    super().__init__(
        clang_path=clang_path,
        llvm_size_path=llvm_size_path,
        launcher_path=launcher_path)

  def _compile_fn(self, file_paths: Tuple[str, str], tf_policy_path: str,
                  reward_only: bool) -> Tuple[tf.train.SequenceExample, float]:
    """Run inlining for the given IR file under the given policy.

    Args:
      file_paths: path to files needed for inlining, Tuple of (.bc, .cmd).
      tf_policy_path: path to TF policy direcoty on local disk.
      reward_only: whether only return native size.

    Returns:
      A tuple containing:
        sequence_example: A tf.SequenceExample proto describing inlining trace.
        native_size: Native size of the final native code.

    Raises:
      subprocess.CalledProcessError if process fails.
    """
    working_dir = tempfile.mkdtemp()

    log_path = os.path.join(working_dir, 'log')
    output_native_path = os.path.join(working_dir, 'native')

    input_ir_path, cmd_path = file_paths
    with open(cmd_path) as f:
      cmds = f.read().split('\0')

    try:
      command_line = []
      if self._launcher_path:
        command_line.append(self._launcher_path)
      command_line.extend([self._clang_path] + cmds + [
          '-mllvm', '-enable-ml-inliner=development', input_ir_path, '-mllvm',
          '-training-log=' + log_path, '-o', output_native_path
      ])
      if tf_policy_path:
        command_line.extend(
            ['-mllvm', '-ml-inliner-model-under-training=' + tf_policy_path])
      # This is the long-running task. It should have a way to terminate.
      subprocess.check_call(command_line, timeout=_DEADLINE_IN_SECONDS)

      command_line = [self._llvm_size_path, output_native_path]
      output = subprocess.check_output(command_line).decode('utf-8')

      tmp = output.split('\n')
      if len(tmp) != 3:
        raise RuntimeError('Wrong llvm-size output %s' % output)
      tmp = tmp[1].split('\t')
      native_size = int(tmp[0])

      if reward_only:
        return None, native_size

      # Temporarily try and support text protobuf. We don't want to penalize the
      # binary case, so we try it first.
      sequence_example = tf.train.SequenceExample()
      sequence_example.ParseFromString(f.read())

    except (subprocess.CalledProcessError, tf.errors.OpError) as e:
      raise e
    finally:
      tf.io.gfile.rmtree(working_dir)

    return sequence_example, native_size
