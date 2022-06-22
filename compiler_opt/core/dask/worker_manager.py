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
"""Dask - based middleware implementation."""
from absl import logging
import gin
import dask.config
import dask.utils
import multiprocessing
import tempfile

from compiler_opt.core.abstract_worker import AbstractWorker
from concurrent.futures import ThreadPoolExecutor
from dask.distributed import Client, Worker, LocalCluster
from typing import Callable, Optional, Tuple


class MTWorker(Worker):
  """Multi-threaded worker."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.executors['actor']:
      self.executors['actor'].shutdown()
    self.executors['actor'] = ThreadPoolExecutor(
        max_workers=None, thread_name_prefix='Dask-Actor-MT')


class LocalManager:
  """Local, dask-based worker manager."""

  def __init__(self):
    self._tmpdir = tempfile.TemporaryDirectory()
    dask.config.set({
        'temporary-directory': self._tmpdir.name,
        'distributed.worker.daemon': False,
        'work-stealing': False
    })

    self._client = Client(
        dashboard_address=None,
        processes=True,
        n_workers=1,
        worker_class=MTWorker)
    print(self._client)

  def shutdown(self):
    self._client.close()
    self._tmpdir.cleanup()

  def get_client(self):
    return self._client


def get_local_compilation_jobs(ctor: Callable[[], AbstractWorker],
                               count: Optional[int]) -> Tuple[Callable, list]:

  class DaskStubWrapper(AbstractWorker):

    def __init__(self, stub: AbstractWorker):
      self._stub = stub

    def cancel_all_work(self):
      return self._stub.cancel_all_work(separate_thread=False)

    def __getattr__(self, name):
      return self._stub.__getattr__(name)

  if not count:
    count = multiprocessing.cpu_count()
  instance = LocalManager()
  workers = [
      instance.get_client().submit(ctor, actor=True) for _ in range(count)
  ]
  return instance.shutdown, [
      DaskStubWrapper(worker.result()) for worker in workers
  ]
