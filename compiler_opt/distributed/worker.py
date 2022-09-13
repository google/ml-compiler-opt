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

from typing import Iterable, Optional, Protocol, TypeVar


class Worker(Protocol):

  @classmethod
  def is_priority_method(cls, method_name: str) -> bool:
    _ = method_name
    return False


T = TypeVar('T')


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
