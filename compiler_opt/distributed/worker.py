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

import concurrent.futures
import time
from typing import Iterable, Optional, TypeVar


class Worker:

  @classmethod
  def is_priority_method(cls, method_name: str) -> bool:
    _ = method_name
    return False


T = TypeVar('T')

WorkerFuture = concurrent.futures.Future


def wait_for(futures: Iterable[WorkerFuture]):
  """Dask futures don't support more than result() and done()."""
  for f in futures:
    while not f.done():
      time.sleep(0.1)


def get_exception(worker_future: WorkerFuture) -> Optional[Exception]:
  assert worker_future.done()
  try:
    _ = worker_future.result()
    return None
  except Exception as e:  # pylint: disable=broad-except
    return e
