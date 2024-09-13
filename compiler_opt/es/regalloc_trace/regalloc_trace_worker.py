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
"""Worker for collecting rewards for regalloc using a trace."""

from typing import List, Optional
import tempfile
import concurrent.futures

import gin

from compiler_opt.distributed import worker
from compiler_opt.rl import policy_saver
from compiler_opt.rl import corpus


@gin.configurable
class RegallocTraceWorker(worker.Worker):
  """A worker that produces rewards for a given regalloc policy."""

  def __init__(self, clang_path: str, basic_block_trace_model_path: str,
               thread_count: int):
    self._clang_path = clang_path
    self._basic_block_trace_model_path = basic_block_trace_model_path
    self._thread_count = thread_count

  def compile_module(self, module_to_compile: corpus.LoadedModuleSpec,
                     output_directory: str, tflite_policy_path: Optional[str]):
    cmdline_flags = list(module_to_compile.build_command_line(output_directory))

    if tflite_policy_path:
      cmdline_flags.extend([
          '-mllvm', '-regalloc-enable-advisor=development',
          f'-regalloc-model={tflite_policy_path}'
      ])
    pass

  def build_corpus(self, modules: List[corpus.LoadedModuleSpec],
                   output_directory: str,
                   tflite_policy: Optional[policy_saver.Policy]):
    with tempfile.TemporaryDirectory() as tflite_policy_dir:
      if tflite_policy:
        tflite_policy.to_filesystem(tflite_policy_dir)
      else:
        tflite_policy_dir = None

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._thread_count) as thread_pool:
      for module in modules:
        thread_pool.submit(self.compile_module, module, output_directory,
                           tflite_policy_dir)

  def evaluate_corpus(self):
    pass

  def compile(self, params: list[float],
              corpus_modules: List[corpus.LoadedModuleSpec],
              tflite_policy: Optional[policy_saver.Policy]) -> float:
    pass
