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
"""Module for collect data of loop unroll."""

import base64
import io
import os
import tempfile
from typing import Dict, Optional, Tuple

import gin
import tensorflow as tf

from google.protobuf import struct_pb2  # pytype: disable=pyi-error
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus


@gin.configurable(module='runners')
class LoopUnrollRunner(compilation_runner.CompilationRunner):
  """Class for collecting data for loop partial unroll.

  Usage:
  runner = LoopUnrollRunner(
               clang_path, llvm_objcopy_path, parse_reward_script_path,
              moving_average_decay_rate)
  policy_reward = unroll.collect_data(
      ir_path, tf_policy_path, default_reward, moving_average_reward)
  """

  def __init__(self, llvm_objcopy_path: str, parse_reward_script_path: str,
               latency_coefficient: str, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._llvm_objcopy_path = llvm_objcopy_path
    self._parse_reward_script_path = parse_reward_script_path
    self._latency_coefficient = float(latency_coefficient)

  def compile_fn(
      self, command_line: corpus.FullyQualifiedCmdLine, tf_policy_path: str,
      reward_only: bool, cancellation_manager: Optional[
          compilation_runner.WorkerCancellationManager]
  ) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:
    """Run loop unroll for the given IR file under the given policy.

    Args:
      command_line: the fully qualified command line.
      tf_policy_path: path to TF policy directory on local disk.
      reward_only: whether to only return reward (icache pressure and latency)
      cancellation_manager: handler for early termination by killing any running
      processes

    Returns:
      For loop unroll, the result is in module level. IWS and Latency is
      already weighted by the probability to be executed, checkout
      parse_reward.py and code embedded under AsmPrinter.cpp for more detail).

      Since the reward is calculated at late stage in a compiler that is after
      inlining some functions may be inlined and not be found for some loops,
      so we sum all functions into a single float, reward_total.

      The function returns in the format:
      {
        "loop1_key": (loop1_features, reward_total),
        "loop2_key": (loop2_features, reward_total),
        ...,
        "loopN_key": (loopN_features, reward_total)
      }
      - reward_total: sum of IWS and Latency of all functions in this module

    Early return:
      The function early returns when the compiled module doesn't record any
      logs or the log file doesn't record any loop. This happens when
      `LoopUnrollPass` is not triggered or no loop triggered "partial unroll"
      in the pass.
    """
    working_dir = tempfile.mkdtemp()

    # The compiler will log input feature (loop properties) and decision
    # (unroll count) into the specified log path
    log_path = os.path.join(working_dir, 'log')

    # The compilation will generate object files, and our augmentation under
    # AsmPrinter.cpp will create section data `llvm_block_data`.
    object_path = os.path.join(working_dir, 'object')
    # llvm-objcopy extracts the section data from object to data
    data_path = os.path.join(working_dir, 'data')
    # Reward parsing script parses data into parsed_reward
    parsed_reward_path = os.path.join(working_dir, 'parsed_reward')

    try:
      # Construct command to execute clang
      command_line = []

      # parameters for MLGO unroll
      command_line.extend([self._clang_path] + list(command_line) + [
          '-mllvm', '-mlgo-unroll-mode=training', '-mllvm',
          '-mlgo-unroll-training-log=' +
          log_path, '-mllvm', '-calc-reward', '-o', object_path
      ])

      # Under `training mode`...
      # If model path is provided, compiler will use ModelUnderTrainingRunner
      # Otherwise, compiler will use NoInferenceModelRunner
      if tf_policy_path:
        command_line.extend(
            ['-mllvm', 'mlgo-unroll-train-model=' + tf_policy_path])

      print('Command to execute clang: ', command_line)

      # run clang
      compilation_runner.start_cancellable_process(command_line,
                                                   self._compilation_timeout,
                                                   cancellation_manager)

      # A module may not generate a log if none of the loops go into the
      # LoopUnroll decision. Early return here if log_path cannot be found.
      if not os.path.exists(log_path):
        print('Early return, log file not found.')
        return {}

      # A log file may not have anything inside when none of the loops goes
      # into PartialUnroll decision. Early return a log file is created but
      # nothing inside.
      if os.path.getsize(log_path) == 0:
        print('Early return, log file contains nothing.')
        return {}

      # Run llvm-objcopy to get section data
      command_line = [
          self._llvm_objcopy_path,
          '--dump-section=.llvm_block_data.=' + data_path, object_path
      ]
      print('Command to get section data: ', command_line)
      compilation_runner.start_cancellable_process(command_line,
                                                   self._compilation_timeout,
                                                   cancellation_manager)

      # Run parse_reward.py to get reward
      command_line = [
          self._parse_reward_script_path, data_path, parsed_reward_path
      ]
      print('Command to parse reward: ', command_line)
      compilation_runner.start_cancellable_process(command_line,
                                                   self._compilation_timeout,
                                                   cancellation_manager)

      # Sum rewards of all functions into a single float
      reward_total = 0
      with io.open(parsed_reward_path, 'r', encoding='utf-8') as reward_f:
        for line in reward_f.readlines():
          line = line[:-1]  # strip end-line
          items = line.split(',')
          assert len(items) == 3
          # function_name = items[0] (commented out because currently unused)
          iws = float(items[1])
          latency = float(items[2])
          reward_total = reward_total + (
              iws + latency * self._latency_coefficient)

      if reward_only:
        return {'default': (None, reward_total)}

      result = {}

      # Read training log, fill them in to result.
      sequence_examples = struct_pb2.Struct()
      with io.open(log_path, 'rb') as log_f:
        sequence_examples.ParseFromString(log_f.read())

      for key, value in sequence_examples.fields.items():
        entry = tf.train.SequenceExample()
        entry.ParseFromString(base64.b64decode(value.string_value))

        if not entry.HasField('feature_lists'):
          continue

        result[key] = (entry, reward_total)

    finally:
      tf.io.gfile.rmtree(working_dir)

    return result
