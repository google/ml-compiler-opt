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
`buffer` tasks assigned to each assignee.
"""

import concurrent.futures
import threading

from typing import List, Callable, TypeVar


class DeadlockDetected(Exception):

  def __init__(self):
    Exception.__init__(self)


def raise_deadlock(*args, **kwargs):
  # This is raised if attempting to call .result() on _NEVER_FUTURE
  raise DeadlockDetected()


_NEVER_FUTURE = concurrent.futures.Future()  # A Future that never completes.
_NEVER_FUTURE.result = raise_deadlock
T = TypeVar('T')


def schedule(assignments: List[Callable[[T], concurrent.futures.Future]],
             assignees: List[T],
             buffer=2) -> List[concurrent.futures.Future]:
  """
  Assigns assignments to assignees once previous assignments of the assignee are
  completed.
  Args:
    assignments: Function to call with an assignee.
    assignees: List of assignees that are the singular argument to callable.
    buffer: Number of assignments to maintain on each assignee.
  Returns:
    A list of Futures that should be stored AS IS. Treat as a reference-to-list.
  """
  # Create "fake" futures to be returned first, these futures will be replaced
  # with "real" futures later on.
  results = [_NEVER_FUTURE] * len(assignments)
  idx = -1
  idx_lock = threading.Lock()

  # Simple atomic increment and get.
  def fetch_idx():
    nonlocal idx
    with idx_lock:
      idx += 1
      return idx

  def chain_assignments(assignee):
    if (i := fetch_idx()) < len(assignments):
      results[i] = assignments[i](assignee)

      # It is necessary for the done callback to execute on a separate thread,
      # which is on an anonymous/fresh callstack. This is to prevent a potential
      # deadlock arising from the current thread holding resources (e.g. a lock)
      # which is required to successfully completed the callback.
      results[i].add_done_callback(lambda _: threading.Thread(
          target=chain_assignments, args=(assignee,)).start())
      # results[i].add_done_callback(lambda _: chain_assignments(assignee))

  for _ in range(buffer):
    for a in assignees:
      chain_assignments(a)

  return results
