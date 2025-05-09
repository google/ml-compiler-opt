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
"""Worker for inlining for size."""

import logging
import os
import concurrent.futures
import subprocess
import tempfile
import shutil
import gin

from compiler_opt.rl.inlining import inlining_runner
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
  # TODO: Same method is defined in RegallocTraceWorker. Needs to be
  # refactored.
  def _setup_base_policy(self):
    self._tf_base_temp_dir = tempfile.mkdtemp()
    policy = policy_utils.create_actor_policy()
    saver = policy_saver.PolicySaver({"policy": policy})
    saver.save(self._tf_base_temp_dir)
    self._tf_base_policy_path = os.path.join(self._tf_base_temp_dir, "policy")

  def __init__(
      self,
      *,
      gin_config: str,
      clang_path: str,
      llvm_size_path: str,
      thread_count: int = 128,
  ):
    """Initializes the InliningWorker class."""
    gin.parse_config(gin_config)
    self._setup_base_policy()
    self._inliner = inlining_runner.InliningRunner(
        clang_path=clang_path,
        llvm_size_path=llvm_size_path,
    )
    self._thread_count = thread_count

  # Deletion here is best effort as it occurs at GC time. If the shutdown is
  # forced, cleanup might not happen as expected. This does not matter too
  # much though as resource leakage will be small, and any cloud setups will
  # have tempdirs wiped periodically.
  def __del__(self):
    shutil.rmtree(self._tf_base_temp_dir, ignore_errors=True)

  def _compile_module_and_get_size(
      self,
      loaded_module_spec: corpus.LoadedModuleSpec,
      output_directory: str,
      tflite_policy_path: str | None,
  ) -> float:
    """Compiles a single LoadedModuleSpec and returns its native code size."""
    working_dir = tempfile.mkdtemp(dir=output_directory)

    # Build the final command line using LoadedModuleSpec
    original_cmd_line = loaded_module_spec.build_command_line(working_dir)

    result = self._inliner.compile_fn(original_cmd_line, tflite_policy_path,
                                      True, working_dir)

    return result[inlining_runner.DEFAULT_IDENTIFIER][1]

  def compile(self, policy: bytes | None,
              modules: list[corpus.LoadedModuleSpec]) -> float | None:
    with tempfile.TemporaryDirectory() as compilation_dir:
      tflite_policy_path = None
      if policy is not None:
        tflite_policy_path = policy_utils.convert_to_tflite(
            policy, compilation_dir, self._tf_base_policy_path)

      with concurrent.futures.ThreadPoolExecutor(
          max_workers=self._thread_count) as thread_pool:
        compile_futures = {
            thread_pool.submit(
                self._compile_module_and_get_size,
                module,
                compilation_dir,
                tflite_policy_path,
            ):
                module for module in modules
        }

        total_size = 0
        for future in concurrent.futures.as_completed(compile_futures):
          e = future.exception()
          if e is not None:
            # Even if one of the compilations fail, currently we return None which
            # will be skipped later by get_rewards method.
            logging.error(
                "Module generated an exception during future"
                " processing: %s", str(e))
            return None

          size = future.result()
          # Check for failure indicator from the compile function
          if size == float("inf"):
            raise ValueError(
                "Size obtained is infinity. This is not expected.")
          total_size += size
      return total_size
