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
from typing import Any

import gin
import tensorflow as tf

from compiler_opt.rl import corpus
from compiler_opt.distributed import worker
from compiler_opt.rl import policy_saver
from compiler_opt.es import policy_utils


def _make_dirs_and_copy(old_file_path: str, new_file_path: str):
  tf.io.gfile.makedirs(os.path.dirname(new_file_path))
  tf.io.gfile.copy(old_file_path, new_file_path)


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

  # TODO(issues/471): aux_file_replacement_flags should be refactored out of
  # regalloc_trace_worker as it will need to be used in other places
  # eventually.
  def _copy_corpus(self, corpus_path: str, copy_corpus_locally_path: str | None,
                   aux_file_replacement_flags: dict[str, str]) -> None:
    """Makes a local copy of the corpus if requested.

    This function makes a local copy of the corpus by copying the remote
    corpus to a user-specified directory.

    Args:
      corpus_path: The path to the remote corpus.
      copy_corpus_locally: The local path to copy the corpus to.
      aux_file_replacement_flags: Additional files to copy over that are
        passed in through flags, like profiles.
    """
    # We use the tensorflow APIs below rather than the standard Python file
    # APIs for compatibility with more filesystems.

    if tf.io.gfile.exists(copy_corpus_locally_path):
      return

    with tf.io.gfile.GFile(
        os.path.join(corpus_path, "corpus_description.json"),
        "r") as corpus_description_file:
      corpus_description: dict[str, Any] = json.load(corpus_description_file)

    file_extensions_to_copy = [".bc", ".cmd"]
    if corpus_description["has_thinlto"]:
      file_extensions_to_copy.append(".thinlto.bc")

    copy_futures = []
    with concurrent.futures.ThreadPoolExecutor(self._thread_count *
                                               5) as copy_thread_pool:
      for module in corpus_description["modules"]:
        for extension in file_extensions_to_copy:
          current_path = os.path.join(corpus_path, module + extension)
          new_path = os.path.join(copy_corpus_locally_path, module + extension)
          copy_futures.append(
              copy_thread_pool.submit(_make_dirs_and_copy, current_path,
                                      new_path))

      if aux_file_replacement_flags is not None:
        for flag_name in aux_file_replacement_flags:
          aux_replacement_file = aux_file_replacement_flags[flag_name]
          new_path = os.path.join(copy_corpus_locally_path,
                                  os.path.basename(aux_replacement_file))
          copy_futures.append(
              copy_thread_pool.submit(_make_dirs_and_copy, aux_replacement_file,
                                      new_path))

    for copy_future in copy_futures:
      if copy_future.exception() is not None:
        raise copy_future.exception()

  def __init__(
      self,
      *,
      gin_config: str,
      clang_path: str,
      basic_block_trace_model_path: str,
      thread_count: int,
      corpus_path: str,
      copy_corpus_locally_path: str | None = None,
      aux_file_replacement_flags: dict[str, str] | None = None,
      extra_bb_trace_model_flags: list[str] | None = None,
  ):
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
      copy_corpus_locally_path: If set, specifies the path that the corpus
        should be copied to before utilizing the modules for evaluation.
        Setting this to None signifies that no copying is desired.
      aux_file_replacement_flags: A dictionary mapping sentinel values intended
        to be set using the corpus replace_flags feature to actual file paths
        local to the worker. This is intended to be used in distributed
        training setups where training corpora and auxiliary files need to be
        copied locally before being compiled.
      extra_bb_trace_model_flags: Extra flags to pass to the
        basic_block_trace_model invocation.
    """
    self._clang_path = clang_path
    self._basic_block_trace_model_path = basic_block_trace_model_path
    self._thread_count = thread_count
    self._extra_bb_trace_model_flags = ([] if not extra_bb_trace_model_flags
                                        else extra_bb_trace_model_flags)

    self._has_local_corpus = False
    self._corpus_path = corpus_path
    if copy_corpus_locally_path is not None:
      self._copy_corpus(corpus_path, copy_corpus_locally_path,
                        aux_file_replacement_flags)
      self._corpus_path = copy_corpus_locally_path
      self._has_local_corpus = True

    if (copy_corpus_locally_path is None and
        aux_file_replacement_flags is not None):
      raise ValueError(
          "additional_replacement_flags is incompatible with fully local "
          "corpus setups. Please directly replace the flag with the correct "
          "value.")
    self._aux_file_replacement_flags = aux_file_replacement_flags
    self._aux_file_replacement_context = {}
    if aux_file_replacement_flags is not None:
      for flag_name in self._aux_file_replacement_flags:
        self._aux_file_replacement_context[flag_name] = os.path.join(
            self._corpus_path,
            os.path.basename(self._aux_file_replacement_flags[flag_name]),
        )

    gin.parse_config(gin_config)
    self._setup_base_policy()

  # Deletion here is best effort as it occurs at GC time. If the shutdown is
  # forced, cleanup might not happen as expected. This does not matter too
  # much though as resource leakage will be small, and any cloud setups will
  # have tempdirs wiped periodically.
  def __del__(self):
    shutil.rmtree(self._tf_base_temp_dir)
    if self._has_local_corpus:
      shutil.rmtree(self._corpus_path)

  def _compile_module(self, module_to_compile: corpus.ModuleSpec,
                      output_directory: str, tflite_policy_path: str | None,
                      compiled_module_suffix: str):
    command_vector = [self._clang_path]
    context = corpus.Corpus.ReplaceContext(
        os.path.join(self._corpus_path, module_to_compile.name) + ".bc",
        # We add the additional ThinLTO index unconditionallyas if we are not
        # using ThinLTO, we will just never end up replacing anything.
        os.path.join(self._corpus_path, module_to_compile.name) + ".thinlto.bc")
    command_vector.extend([
        option.format(context=context, **self._aux_file_replacement_context)
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

    module_output_path = os.path.join(
        output_directory, module_to_compile.name + compiled_module_suffix)
    pathlib.Path(os.path.dirname(module_output_path)).mkdir(
        parents=True, exist_ok=True)
    command_vector.extend(["-o", module_output_path])

    subprocess.run(command_vector, check=True, capture_output=True)

  def _build_corpus(self,
                    modules: Collection[corpus.ModuleSpec],
                    output_directory: str,
                    tflite_policy_path: str | None,
                    compiled_module_suffix=".bc.o"):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._thread_count) as thread_pool:
      compile_futures = [
          thread_pool.submit(self._compile_module, module, output_directory,
                             tflite_policy_path, compiled_module_suffix)
          for module in modules
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
    command_vector.extend(self._extra_bb_trace_model_flags)

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
