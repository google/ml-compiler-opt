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
"""Worker for regalloc using trace-based cost modeling."""

from typing import Optional, Collection
import os
import pathlib
import subprocess
import json
import concurrent.futures
import tempfile

import gin

from compiler_opt.rl import corpus
from compiler_opt.distributed import worker
from compiler_opt.rl import policy_saver


@gin.configurable
class RegallocTraceWorker(worker.Worker):
  """A worker that produces rewards for a given regalloc policy."""

  def __init__(self, clang_path: str, basic_block_trace_model_path: str,
               thread_count: int, corpus_path: str):
    self._clang_path = clang_path
    self._basic_block_trace_model_path = basic_block_trace_model_path
    self._thread_count = thread_count
    self._corpus_path = corpus_path

  def _compile_module(self, module_to_compile: corpus.ModuleSpec,
                      output_directory: str, tflite_policy_path: Optional[str]):
    command_vector = [self._clang_path]
    context = corpus.Corpus.ReplaceContext(
        os.path.join(self._corpus_path, module_to_compile.name) + ".bc",
        # We add the additional ThinLTO index unconditionallyas if we are not
        # using ThinLTO, we will just never end up replacing anything.
        os.path.join(self._corpus_path, module_to_compile.name) + ".thinlto.bc")
    command_vector.extend([
        option.format(context=context)
        for option in module_to_compile.command_line
    ])

    if tflite_policy_path is not None:
      command_vector.extend([
          "-mllvm", "-regalloc-enable-advisor=development", "-mllvm",
          f"-regalloc-model={tflite_policy_path}"
      ])
    else:
      # Force the default advisor if we aren't explicitly using a new policy
      # to prevent enabling the release advisor if it was specified in the
      # corpus.
      command_vector.extend(["-mllvm", "-regalloc-enable-advisor=default"])

    module_output_path = os.path.join(output_directory,
                                      module_to_compile.name + ".bc.o")
    pathlib.Path(os.path.dirname(module_output_path)).mkdir(
        parents=True, exist_ok=True)
    command_vector.extend(["-o", module_output_path])

    subprocess.run(
        command_vector,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

  def _build_corpus(self, modules: Collection[corpus.ModuleSpec],
                    output_directory: str,
                    tflite_policy: Optional[policy_saver.Policy]):
    with tempfile.TemporaryDirectory() as tflite_policy_dir:
      if tflite_policy:
        tflite_policy.to_filesystem(tflite_policy_dir)
      else:
        tflite_policy_dir = None

    compile_futures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._thread_count) as thread_pool:
      for module in modules:
        compile_futures.append(
            thread_pool.submit(self._compile_module, module, output_directory,
                               tflite_policy_dir))

      for future in compile_futures:
        if future.exception() is not None:
          raise future.exception()

      # Write out a corpus description for basic_block_trace_model.
      corpus_description_path = os.path.join(output_directory,
                                             "corpus_description.json")
      corpus_description = {
          "modules": [module_spec.name for module_spec in modules]
      }

      with open(
          corpus_description_path, "w",
          encoding="utf-8") as corpus_description_file:
        json.dump(corpus_description, corpus_description_file)

  def _evaluate_corpus(self, module_directory: str, function_index_path: str,
                       bb_trace_path: str):
    corpus_description_path = os.path.join(module_directory,
                                           "corpus_description.json")

    command_vector = [
        self._basic_block_trace_model_path,
        f"--corpus_path={corpus_description_path}",
        f"--function_index_path={function_index_path}",
        f"--thread_count={self._thread_count}",
        f"--bb_trace_path={bb_trace_path}", "--model_type=mca"
    ]

    output = subprocess.run(
        command_vector,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True)

    segment_costs = []
    for line in output.stdout.decode("utf-8").split("\n"):
      try:
        value = float(line)
        segment_costs.append(value)
      except ValueError:
        continue

    if len(segment_costs) < 1:
      raise ValueError("Did not find any valid segment costs.")

    return segment_costs

  def compile_corpus_and_evaluate(
      self, modules: Collection[corpus.ModuleSpec], function_index_path: str,
      bb_trace_path: str,
      tflite_policy: Optional[policy_saver.Policy]) -> float:
    with tempfile.TemporaryDirectory() as compilation_dir:
      self._build_corpus(modules, compilation_dir, tflite_policy)

      segment_costs = self._evaluate_corpus(compilation_dir,
                                            function_index_path, bb_trace_path)
      return sum(segment_costs)
