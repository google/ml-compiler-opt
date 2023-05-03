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
"""Common abstraction for a worker contract."""

import abc
import sys
from typing import Any, List, Iterable, Optional, Protocol, TypeVar

import gin


class Worker(Protocol):

  @classmethod
  def is_priority_method(cls, method_name: str) -> bool:
    _ = method_name
    return False


T = TypeVar('T')


class WorkerPool(metaclass=abc.ABCMeta):
  """Abstraction of a pool of workers that may be refreshed."""

  # Issue #155 would strongly-type the return type.
  @abc.abstractmethod
  def get_currently_active(self) -> List[Any]:
    raise NotImplementedError()

  @abc.abstractmethod
  def get_worker_concurrency(self) -> int:
    raise NotImplementedError()


class FixedWorkerPool(WorkerPool):
  """A WorkerPool built from a fixed list of workers."""

  # Issue #155 would strongly-type `workers`
  def __init__(self, workers: List[Any], worker_concurrency: int = 2):
    self._workers = workers
    self._worker_concurrency = worker_concurrency

  def get_currently_active(self):
    return self._workers

  def get_worker_concurrency(self):
    return self._worker_concurrency


# Dask's Futures are limited. This captures that.
class WorkerFuture(Protocol[T]):

  def result(self) -> T:
    raise NotImplementedError()

  def done(self) -> bool:
    raise NotImplementedError()

  def add_done_callback(self, fn) -> None:
    raise NotImplementedError


def wait_for(futures: Iterable[WorkerFuture]):
  """Dask futures don't support more than result() and done()."""
  for f in futures:
    try:
      _ = f.result()
    except:  # pylint: disable=bare-except
      pass


def get_exception(worker_future: WorkerFuture) -> Optional[Exception]:
  assert worker_future.done()
  try:
    _ = worker_future.result()
    return None
  except Exception as e:  # pylint: disable=broad-except
    return e


def get_full_worker_args(worker_class: 'type[Worker]', **current_kwargs):
  """Get the union of given kwargs and gin config.

  This allows the worker hosting process be set up differently from the training
  process - e.g. no need to initialize gin variables there, for example.
  """
  gin_config = {}
  try:
    gin_config = gin.get_bindings(worker_class)
  except ValueError:
    # we don't have a way to check if `worker_class` is even known to gin, and
    # it's not a requirement that it were. Tests, for instance, don't use gin.
    pass
  # Issue #38
  if sys.version_info.minor >= 9:
    return current_kwargs | gin_config
  else:
    return {**current_kwargs, **gin_config}
