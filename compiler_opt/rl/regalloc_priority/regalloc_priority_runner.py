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
"""Module for collect data of regalloc priority prediction."""

import gin
import tensorflow as tf

import os
import tempfile
from typing import Dict, Optional, Tuple

from compiler_opt.rl import compilation_runner
from compiler_opt.rl import log_reader


@gin.configurable(module='runners')
class RegAllocPriorityRunner(compilation_runner.CompilationRunner):
  """Class for collecting data for regalloc-priority-prediction."""

  def _compile_fn(
      self, file_paths: Tuple[str, ...], tf_policy_path: str, reward_only: bool,
      workdir: str, cancellation_manager: Optional[
          compilation_runner.WorkerCancellationManager]
  ) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:

    file_paths = file_paths[0].replace('.bc', '')
    working_dir = tempfile.mkdtemp(dir=workdir)

    log_path = os.path.join(working_dir, 'log')
    output_native_path = os.path.join(working_dir, 'native')

    result = {}
    command_line = []
    if self._launcher_path:
      command_line.append(self._launcher_path)
    command_line.extend([self._clang_path] + [
        '-c', file_paths, '-O3', '-mllvm', '-regalloc-priority-training-log=' +
        log_path, '-mllvm', '-regalloc-enable-priority-advisor=development',
        '-o', output_native_path
    ])

    if tf_policy_path:
      command_line.extend(
          ['-mllvm', '-regalloc-priority-model=' + tf_policy_path])
    compilation_runner.start_cancellable_process(command_line,
                                                 self._compilation_timeout,
                                                 cancellation_manager)

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
