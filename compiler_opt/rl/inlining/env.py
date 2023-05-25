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
"""Implementation of the inlining for size environment."""

import subprocess

import gin
import os

from compiler_opt.rl import env
from compiler_opt.rl.inlining import config

from typing import Dict, List, Optional

_COMPILED_MODULE_NAME = 'compiled_module'


@gin.configurable
class InliningForSizeTask(env.MLGOTask):
  """Implementation of the inlining-for-size MLGOTask."""

  def __init__(self, llvm_size_path: str):
    super().__init__()
    self._llvm_size_path = llvm_size_path

  def get_cmdline(self, clang_path: str, base_args: List[str],
                  interactive_base_path: Optional[str],
                  working_dir: str) -> List[str]:
    if interactive_base_path:
      interactive_args = [
          '-mllvm',
          '-enable-ml-inliner=release',
          '-mllvm',
          f'-inliner-interactive-channel-base={interactive_base_path}',
          #'-mllvm',
          #'-inliner-interactive-include-default',
      ]
    else:
      interactive_args = []
    compiled_module_path = os.path.join(working_dir, _COMPILED_MODULE_NAME)
    return [clang_path
           ] + base_args + interactive_args + ['-o', compiled_module_path]

  def get_module_scores(self, working_dir: str) -> Dict[str, float]:
    compiled_module_path = os.path.join(working_dir, _COMPILED_MODULE_NAME)
    cmdline = [self._llvm_size_path, compiled_module_path]
    completed_proc = subprocess.run(cmdline, capture_output=True, check=True)
    if not completed_proc.stdout:
      raise RuntimeError(f'Empty llvm-size output: {" ".join(cmdline)}')
    output = completed_proc.stdout.decode('utf-8')
    tmp = output.split('\n')
    if len(tmp) != 3:
      raise RuntimeError(f'Wrong llvm-size output {output}')
    tmp = tmp[1].split('\t')
    native_size = int(tmp[0])
    return {'default': native_size}


@gin.configurable
def get_inlining_env(clang_path: str) -> env.MLGOEnvironmentBase:
  time_step_spec, action_spec = config.get_inlining_signature_spec()

  return env.MLGOEnvironmentBase(
      clang_path=clang_path,
      task_type=InliningForSizeTask,
      obs_spec=time_step_spec.observation,
      action_spec=action_spec,
  )
