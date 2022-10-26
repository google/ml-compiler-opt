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
"""A worker hosting multiple workers."""

from typing import Optional

import threading
from compiler_opt.distributed import worker
from contextlib import AbstractContextManager

WorkerID = int


class NestingWorker(worker.Worker):
  """A worker which hosts a number of other worker objects."""

  def __init__(self):
    self._lock = threading.Lock()
    self._workers = {}
    self._next_id = 0

  def create(self, cls: 'type[worker.Worker]', *args, **kwargs) -> WorkerID:
    worker_instance = cls(*args, **kwargs)
    with self._lock:
      worker_id = self._next_id
      self._workers[worker_id] = worker_instance
      self._next_id += 1
      return worker_id

  def release(self, worker_id: WorkerID):
    with self._lock:
      self._workers.pop(worker_id)

  def call(self, worker_id: WorkerID, method: str, *args, **kwargs):
    with self._lock:
      worker_instance = self._workers[worker_id]
    return getattr(worker_instance, method)(*args, **kwargs)

  def _get_registered_workers(self):
    return self._workers


def create_nested_worker_manager(underlying_pool: worker.WorkerPool):
  """Create a worker manager class on an underlying pool of NestingWorkers."""
  class _Stub:
    """Stub to worker hosted by a NestingWorker"""
    def __init__(self, nesting_worker: NestingWorker, worker_id: WorkerID):
      self._nesting_worker = nesting_worker
      self._id = worker_id

    def __getattr__(self, method: str):

      def func(*args, **kwargs):
        return self._nesting_worker.call(self._id, method, *args, **kwargs)

      return func

    def release(self):
      self._nesting_worker.release(self._id)

  class _Nester(AbstractContextManager):
    """The worker manager class."""
    def __init__(self, worker_class: 'type[worker.Worker]',
                 count: Optional[int], *args, **kwargs):
      self._underlying_pool = underlying_pool
      self._pool = []
      current_workers = self._underlying_pool.get_currently_active()
      for i in range(count):
        nesting_worker = current_workers[i % len(current_workers)]
        self._pool.append(
            _Stub(
                nesting_worker=nesting_worker,
                worker_id=nesting_worker.create(worker_class, *args,
                                                **kwargs).result()))

    def __enter__(self):
      return worker.FixedWorkerPool(
          workers=self._pool,
          worker_concurrency=self._underlying_pool.get_worker_concurrency())

    def __exit__(self, *args):
      for w in self._pool:
        w.release()

  return _Nester
