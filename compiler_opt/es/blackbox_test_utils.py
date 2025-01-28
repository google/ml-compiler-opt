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
"""Test facilities for Blackbox classes."""

from typing import List, Collection, Optional

import gin

from compiler_opt.distributed import worker
from compiler_opt.rl import corpus
from compiler_opt.rl import policy_saver


@gin.configurable
class ESWorker(worker.Worker):
  """Temporary placeholder worker.
  Each time a worker is called, the function value
  it will return increases."""

  def __init__(self, arg, *, kwarg):
    self._arg = arg
    self._kwarg = kwarg
    self.function_value = 0.0

  def compile(self, policy: policy_saver.Policy,
              samples: List[corpus.ModuleSpec]) -> float:
    if policy and samples:
      self.function_value += 1.0
      return self.function_value
    else:
      return 0.0


class ESTraceWorker(worker.Worker):
  """Temporary placeholder worker.
  
  This is a test worker for TraceBlackboxEvaluator that expects a slightly
  different interface than other workers.
  """

  def __init__(self, arg, *, kwarg):
    del arg  # Unused.
    del kwarg  # Unused.
    self._function_value = 0.0

  def compile_corpus_and_evaluate(
      self, modules: Collection[corpus.ModuleSpec], function_index_path: str,
      bb_trace_path: str,
      tflite_policy: Optional[policy_saver.Policy]) -> float:
    if modules and function_index_path and bb_trace_path and tflite_policy:
      self._function_value += 1
      return self._function_value
    else:
      return 10
