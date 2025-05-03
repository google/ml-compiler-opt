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
"""Worker for inlining for size.
"""

from collections.abc import Collection
import logging
from compiler_opt.rl import compilation_runner
import os
import pathlib
import subprocess
import json
import concurrent.futures
import tempfile
import shutil

import gin
from absl import flags
from compiler_opt.rl import corpus
from compiler_opt.distributed import worker
from compiler_opt.rl import policy_saver
from compiler_opt.es import policy_utils


@gin.configurable
class InliningWorker(worker.Worker):
  """A worker that produces rewards for a given Inlining policy.

  InliningWorker exposes a compile function, which
  compiles a set of modules in parallel remotely, evaluates them with
  llvm-size, and then computes the rewards based on the baseline size.
  """

  def _setup_base_policy(self):
    self._tf_base_temp_dir = tempfile.mkdtemp()
    policy = policy_utils.create_actor_policy()
    saver = policy_saver.PolicySaver({"policy": policy})
    saver.save(self._tf_base_temp_dir)
    self._tf_base_policy_path = os.path.join(self._tf_base_temp_dir, "policy")

  def __init__(self,
               *,
               gin_config: str,
               clang_path: str,
               llvm_size_path: str,
               ir2vec_vocab_path: str | None = None,
               ir2vec_avg: bool = False,
               thread_count: int,
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
    self._thread_count = thread_count
    self._corpus_path = corpus_path
    self._llvm_size_path = llvm_size_path
    self._ir2vec_vocab_path = ir2vec_vocab_path
    self._ir2vec_avg = ir2vec_avg
    self._compilation_timeout = compilation_runner.COMPILATION_TIMEOUT.value
    self._cancellation_manager = compilation_runner.WorkerCancellationManager()

    gin.parse_config(gin_config)
    self._setup_base_policy()

  # Deletion here is best effort as it occurs at GC time. If the shutdown is
  # forced, cleanup might not happen as expected. This does not matter too
  # much though as resource leakage will be small, and any cloud setups will
  # have tempdirs wiped periodically.
  def __del__(self):
    shutil.rmtree(self._tf_base_temp_dir, ignore_errors=True)

  def _compile_module_and_get_size(self,
                                   loaded_module_spec: corpus.LoadedModuleSpec,
                                   output_directory: str,
                                   tflite_policy_path: str | None) -> float:
    """Compiles a single LoadedModuleSpec and returns its native code size."""
    working_dir = tempfile.mkdtemp(dir=output_directory)
    log_path = os.path.join(working_dir, 'log')
    output_native_path = os.path.join(working_dir, 'native.o')

    # Build the final command line using LoadedModuleSpec
    original_cmd_line = loaded_module_spec.build_command_line(working_dir)

    cmdline = []
    cmdline.extend([self._clang_path] + list(original_cmd_line))

    # Add ML Inliner flags
    cmdline.extend(['-mllvm', '-enable-ml-inliner=development'])
    if self._ir2vec_vocab_path is not None:
      cmdline.extend([
          '-mllvm', '-ml-inliner-ir2vec-vocab-file=' + self._ir2vec_vocab_path,
          '-mllvm', '-ml-inliner-ir2vec-avg=' + str(self._ir2vec_avg)
      ])
    if tflite_policy_path:
      cmdline.extend(
          ['-mllvm', f'-ml-inliner-model-under-training={tflite_policy_path}'])
    # Add other necessary flags (e.g., ir2vec, -mllvm -training-log=...)

    cmdline.extend(
        ['-mllvm', '-training-log=' + log_path, '-o', output_native_path])

    # Run Clang Compilation using cancellable process
    compilation_runner.start_cancellable_process(
        cmdline,
        timeout=self._compilation_timeout,
        cancellation_manager=self._cancellation_manager)

    # Run llvm-size
    size_cmd = [self._llvm_size_path, output_native_path]
    output_bytes = compilation_runner.start_cancellable_process(
        size_cmd,
        timeout=self._compilation_timeout,
        cancellation_manager=self._cancellation_manager,
        want_output=True)

    if not output_bytes:
      raise RuntimeError(f'Empty llvm-size output: {" ".join(size_cmd)}')

    # Parse llvm-size output (adjust parsing as needed)
    output = output_bytes.decode('utf-8')
    tmp = output.split('\n')
    if len(tmp) != 3:
      raise RuntimeError(f'Wrong llvm-size output {output}')
    tmp = tmp[1].split('\t')
    native_size = int(tmp[0])

    return native_size

  def compile(self, policy: bytes | None,
              modules: list[corpus.LoadedModuleSpec]) -> float:
    with tempfile.TemporaryDirectory() as compilation_dir:
      tflite_policy_path = None
      if policy is not None:
        tflite_policy_path = policy_utils.convert_to_tflite(
            policy, compilation_dir, self._tf_base_policy_path)

      with concurrent.futures.ThreadPoolExecutor(
          max_workers=self._thread_count) as thread_pool:
        compile_futures = {
            thread_pool.submit(self._compile_module_and_get_size, module,
                               compilation_dir, tflite_policy_path):
                module for module in modules
        }

        # Recheck this logic
        total_size = 0
        for future in concurrent.futures.as_completed(compile_futures):
          module = compile_futures[future]
          try:
            size = future.result()
            # Check for failure indicator from the compile function
            if size == float('inf'):
              logging.warning(
                  f"Module {module.name} failed compilation/size measurement.")
            total_size += size
          except Exception as exc:
            # Catch unexpected errors during future processing
            logging.error(
                f'Module {module.name} generated an exception during future processing: {exc}'
            )
            total_size = float('inf')

      return total_size
