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
"""Test facilities for Blackbox classes."""

from collections.abc import Collection

import gin

from compiler_opt.distributed import worker
from compiler_opt.rl import corpus
from compiler_opt.rl import policy_saver


@gin.configurable
class ESWorker(worker.Worker):
  """Temporary placeholder worker.
  Each time a worker is called, the function value
  it will return increases."""

  def __init__(self, *, delta=1.0, initial_value=0.0):
    self.function_value = initial_value
    self._delta = delta

  def compile(self, policy: policy_saver.Policy,
              modules: list[corpus.LoadedModuleSpec]) -> float:
    if policy and modules:
      self.function_value += self._delta
      return self.function_value
    else:
      return 0.0


class SizeReturningESWorker(worker.Worker):
  """A mock worker that returns the size of the first module."""

  def compile(self, policy: bytes | None,
              modules: list[corpus.LoadedModuleSpec]) -> int:
    del policy  # Unused.
    if not modules:
      return 0
    return len(modules[0].loaded_ir)


class ESTraceWorker(worker.Worker):
  """Temporary placeholder worker.

  This is a test worker for TraceBlackboxEvaluator that expects a slightly
  different interface than other workers.
  """

  def __init__(self):
    self._function_value = 0.0

  def compile_corpus_and_evaluate(self, modules: Collection[corpus.ModuleSpec],
                                  function_index_path: str, bb_trace_path: str,
                                  policy_as_bytes: bytes | None) -> float:
    if modules and function_index_path and bb_trace_path and policy_as_bytes:
      self._function_value += 1
      return self._function_value
    else:
      return 10
