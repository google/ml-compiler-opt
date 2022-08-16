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
"""An optimal push-pull-based load balancer which attempts to maintain at least
`buffer` tasks assigned to each worker.
"""

import concurrent.futures
import threading

from typing import List, Callable, TypeVar

from compiler_opt.distributed import worker


class DeadlockDetected(Exception):

  def __init__(self):
    Exception.__init__(self)


def raise_deadlock(*args, **kwargs):
  # This is raised if attempting to call .result() on _NEVER_FUTURE
  raise DeadlockDetected()


_NEVER_FUTURE = concurrent.futures.Future()  # A Future that never completes.
_NEVER_FUTURE.result = raise_deadlock
T = TypeVar('T')


def schedule(work: List[Callable[[T], worker.WorkerFuture]],
             workers: List[T],
             buffer=2) -> List[worker.WorkerFuture]:
  """
  Assigns work to workers once previous work of the worker are
  completed.
  Args:
    work: Function to call with a worker.
    workers: List of workers that are the singular argument to callable.
    buffer: Number of work to maintain on each worker.
  Returns:
    A list of Futures that should be stored AS IS. Treat as a reference-to-list.
  """
  # Create "fake" futures to be returned first, these futures will be replaced
  # with "real" futures later on.
  results: List[worker.WorkerFuture] = [_NEVER_FUTURE] * len(work)
  idx = -1
  idx_lock = threading.Lock()

  # Simple atomic increment and get.
  # Used to iterate over work like a thread-safe queue.
  def fetch_idx():
    nonlocal idx
    with idx_lock:
      idx += 1
      return idx

  def chain_work(wkr):
    if (i := fetch_idx()) < len(work):
      results[i] = work[i](wkr)

      # It is necessary for the done callback to execute on a separate thread,
      # which is on an anonymous/fresh callstack. This is to prevent a potential
      # deadlock arising from the current thread holding resources (e.g. a lock)
      # which is required to successfully completed the callback.
      # See: test_deadlock
      results[i].add_done_callback(
          lambda _: threading.Thread(target=chain_work, args=(wkr,)).start())

  for _ in range(buffer):
    for w in workers:
      chain_work(w)

  return results
