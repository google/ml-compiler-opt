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
    A list of Futures.
  """
  # Create futures to be returned first, these futures aren't bound to
  # anything now, but they will be later.
  results = [concurrent.futures.Future() for _ in range(len(work))]
  idx = -1
  idx_lock = threading.Lock()

  # Simple atomic increment and get.
  # Used to iterate over `work` like a thread-safe queue without making a copy.
  def fetch_idx():
    nonlocal idx
    with idx_lock:
      idx += 1
      return idx

  def make_result_handler(wkr: T, result_future: concurrent.futures.Future):

    def handler(worker_future: concurrent.futures.Future):
      if (e := worker_future.exception()) is not None:
        result_future.set_exception(e)
      else:
        result_future.set_result(worker_future.result())
      chain_work(wkr)

    return handler

  def chain_work(wkr: T):
    if (i := fetch_idx()) < len(work):
      # This potentially causes a deadlock if chain_work is called via a
      # future.set_result() context which holds a resource that is also required
      # to complete the call work[i](wkr) call below. For an example, see:
      # https://gist.github.com/Northbadge/a57f2d4e0a71e8f3934bdb47e59e343e
      # A fix/workaround would be using threading below, but that introduces
      # overhead of creating a new thread.
      work[i](wkr).add_done_callback(make_result_handler(wkr, results[i]))

  # Use min() in case buffer is huge for some reason.
  for _ in range(min(buffer, (len(work) // len(workers)) + 1)):
    for w in workers:
      chain_work(w)

  return results
