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
"""The interface for WorkerManager."""

import abc
from collections.abc import Callable
from contextlib import AbstractContextManager
import pickle
from typing import Any

from compiler_opt.distributed import worker


class WorkerManager(AbstractContextManager, metaclass=abc.ABCMeta):
  """An interface that implementations should derive from."""

  @abc.abstractmethod
  def __init__(self,
               worker_class: type[worker.Worker],
               pickle_func: Callable[[Any], bytes] = pickle.dumps,
               *,
               count: int | None,
               worker_args: tuple = (),
               worker_kwargs: dict | None = None):
    raise ValueError("Not Implemented")

  @abc.abstractmethod
  def __enter__(self) -> worker.FixedWorkerPool:
    raise ValueError("Not Implemented")

  @abc.abstractmethod
  def __exit__(self, *args):
    raise ValueError("Not Implemented")
