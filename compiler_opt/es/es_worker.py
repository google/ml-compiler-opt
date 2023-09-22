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
"""Worker for ES Training."""

import gin
from typing import List
from compiler_opt.distributed import worker
from compiler_opt.rl import corpus


@gin.configurable
class ESWorker(worker.Worker):
  """Temporary placeholder worker.
  Each time a worker is called, the function value
  it will return increases."""

  def __init__(self, arg, *, kwarg):
    self._arg = arg
    self._kwarg = kwarg
    self.function_value = 0.0

  def temp_compile(self, policy: bytes,
                   samples: List[corpus.ModuleSpec]) -> float:
    if policy and samples:
      self.function_value += 1.0
      return self.function_value
    else:
      return 0.0
