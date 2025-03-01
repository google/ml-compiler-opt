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
"""Worker for regalloc using trace-based cost modeling.

This worker is designed specifically for a trace based cost modelling
methodology. It compiles an entire corpus in parallel with a thread pool, and
then passes all those modules to basic_block_trace_model along with traces and
other relevant data to produce an overall cost for the model being evaluated.
"""

from collections.abc import Collection
import os
import pathlib
import subprocess
import json
import concurrent.futures
import tempfile
import shutil

import gin

from compiler_opt.rl import corpus
from compiler_opt.distributed import worker
from compiler_opt.rl import policy_saver
from compiler_opt.es import policy_utils


@gin.configurable
class RegallocTraceWorker(worker.Worker):
  """A worker that produces rewards for a given regalloc policy.

  RegallocTraceWorker exposes a compile_corpus_and_evaluate function, which
  compiles a set of modules in parallel locally, evaluates them with
  basic_block_trace_model, and then returns the total cost of the evaluated
  segments.
  """

  def _setup_base_policy(self):
    self._tf_base_temp_dir = tempfile.mkdtemp()
    policy = policy_utils.create_actor_policy()
    saver = policy_saver.PolicySaver({"policy": policy})
    saver.save(self._tf_base_temp_dir)
    self._tf_base_policy_path = os.path.join(self._tf_base_temp_dir, "policy")

  def __init__(self, *, gin_config: str, clang_path: str,
               basic_block_trace_model_path: str, thread_count: int,
               corpus_path: str):
    """Initializes the RegallocTraceWorker class.

    Args:
      clang_path: The path to the clang binary to use for compiling the corpus.
      basic_block_trace_model_path: The path to the basic_block_trace_model
        binary to use for trace-based modelling. basic_block_trace_model takes
        in a set of modules, a trace, and auxiliary information for
        interpreting the trace, simulates the trace against the code in the
        passed-in modules, returning estimated cycle counts.
      thread_count: The number of threads to use for concurrent compilation
        and modelling.
      corpus_path: The path to the corpus that modules will be compiled from.
    """
    self._clang_path = clang_path
    self._basic_block_trace_model_path = basic_block_trace_model_path
    self._thread_count = thread_count
    self._corpus_path = corpus_path

    gin.parse_config(gin_config)
    self._setup_base_policy()

  def __del__(self):
    shutil.rmtree(self._tf_base_temp_dir)

  def _compile_module(self, module_to_compile: corpus.ModuleSpec,
                      output_directory: str, tflite_policy_path: str | None):
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

    subprocess.run(command_vector, check=True, capture_output=True)

  def _build_corpus(self, modules: Collection[corpus.ModuleSpec],
                    output_directory: str, tflite_policy_path: str | None):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._thread_count) as thread_pool:
      compile_futures = [
          thread_pool.submit(self._compile_module, module, output_directory,
                             tflite_policy_path) for module in modules
      ]

    for future in compile_futures:
      if future.exception() is not None:
        raise future.exception()

    # Write out a corpus description. basic_block_trace_model uses a corpus
    # description JSON to know which object files to load, so we need to emit
    # one before performing evaluation.
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

    output = subprocess.run(command_vector, capture_output=True, check=True)

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

  def compile_corpus_and_evaluate(self, modules: Collection[corpus.ModuleSpec],
                                  function_index_path: str, bb_trace_path: str,
                                  policy_as_bytes: bytes | None) -> float:
    with tempfile.TemporaryDirectory() as compilation_dir:
      tflite_policy_path = None
      if policy_as_bytes is not None:
        tflite_policy_path = policy_utils.convert_to_tflite(
            policy_as_bytes, compilation_dir, self._tf_base_policy_path)

      self._build_corpus(modules, compilation_dir, tflite_policy_path)

      segment_costs = self._evaluate_corpus(compilation_dir,
                                            function_index_path, bb_trace_path)
      return sum(segment_costs)
