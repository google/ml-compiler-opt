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
"""Local Process Pool - based middleware implementation.

This is a simple implementation of a worker pool, running on the local machine.
Each worker object is hosted by a separate process. Each worker object may
handle a number of concurrent requests. The client is given a stub object that
exposes the same methods as the worker, just that they return Futures.

There is a pair of queues between a stub and its corresponding process/worker.
One queue is used to place tasks (method calls), the other to receive results.
Tasks and results are correlated by a monotonically incrementing counter
maintained by the stub.

The worker process dequeues tasks promptly and either re-enqueues them to a
local thread pool, or, if the task is 'urgent', it executes it promptly.
"""
import concurrent.futures
import dataclasses
import functools
import multiprocessing
import multiprocessing.connection
import queue
import threading

# pylint: disable=unused-import
from compiler_opt.distributed.worker import Worker

from contextlib import AbstractContextManager
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class Task:
  msgid: int
  func_name: str
  args: list
  kwargs: dict
  is_urgent: bool


@dataclasses.dataclass(frozen=True)
class TaskResult:
  msgid: int
  success: bool
  value: Any


def _run(in_q: 'queue.Queue[Task]', out_q: 'queue.Queue[TaskResult]',
         worker_class: 'type[Worker]', *args, **kwargs):
  """Worker process entrypoint."""

  pool = concurrent.futures.ThreadPoolExecutor()
  obj = worker_class(*args, **kwargs)

  def make_ondone(msgid):

    def on_done(f: concurrent.futures.Future):
      if f.exception():
        out_q.put(TaskResult(msgid=msgid, success=False, value=f.exception()))
      else:
        out_q.put(TaskResult(msgid=msgid, success=True, value=f.result()))

    return on_done

  # Run forever. The stub will just kill the runner when done.
  while True:
    task = in_q.get()
    the_func = getattr(obj, task.func_name)
    application = functools.partial(the_func, *task.args, **task.kwargs)
    if task.is_urgent:
      try:
        res = application()
        out_q.put(TaskResult(msgid=task.msgid, success=True, value=res))
      except Exception as e:  # pylint: disable=broad-except
        out_q.put(TaskResult(msgid=task.msgid, success=False, value=e))
    else:
      pool.submit(application).add_done_callback(make_ondone(task.msgid))


def _make_stub(cls: 'type[Worker]', *args, **kwargs):

  class _Stub():
    """Client stub to a worker hosted by a process."""

    def __init__(self):
      self._send: 'queue.Queue[Task]' = multiprocessing.get_context().Queue()
      self._receive: 'queue.Queue[TaskResult]' = multiprocessing.get_context(
      ).Queue()

      # this is the process hosting one worker instance.
      self._process = multiprocessing.Process(
          target=functools.partial(
              _run,
              worker_class=cls,
              in_q=self._send,
              out_q=self._receive,
              *args,
              **kwargs))
      # lock for the msgid -> reply future map
      self._lock = threading.Lock()
      self._map: dict[int, concurrent.futures.Future] = {}
      # thread drainig the receive queue
      self._pump = threading.Thread(target=self._msg_pump)

      # event set by _pump when it exits
      self._done = threading.Event()
      # event set by the stub to tell the message pump to exit.
      self._stop_pump = threading.Event()

      # atomic control to _msgid
      self._msgidlock = threading.Lock()
      self._msgid = 0

      # start the worker and the message pump
      self._process.start()
      self._pump.start()

    def _msg_pump(self):
      while not self._stop_pump.is_set():
        try:
          # avoid blocking so we may notice if _stop_pump was set. We aren't
          # very concerned with delay between shutdown request and the message
          # pump noticing it, this happens only at the very end of training.
          task_result = self._receive.get(timeout=1.0)
          with self._lock:
            future = self._map[task_result.msgid]
            del self._map[task_result.msgid]
            if task_result.success:
              future.set_result(task_result.value)
            else:
              future.set_exception(task_result.value)
        except queue.Empty:
          continue
      self._done.set()

    def __getattr__(self, name):
      with self._msgidlock:
        msgid = self._msgid
        self._msgid += 1

      def remote_call(*args, **kwargs):
        self._send.put(
            Task(
                msgid=msgid,
                func_name=name,
                args=args,
                kwargs=kwargs,
                is_urgent=cls.is_priority_method(name)))
        result_future = concurrent.futures.Future()
        with self._lock:
          self._map[msgid] = result_future
        return result_future

      return remote_call

    def shutdown(self):
      try:
        self._process.kill()
      except:  # pylint: disable=bare-except
        pass
      self._stop_pump.set()
      self._done.wait()

    def __dir__(self):
      return [n for n in dir(cls) if not n.startswith('_')]

  return _Stub()


class LocalWorkerPool(AbstractContextManager):
  """A pool of workers hosted on the local machines, each in its own process."""

  def __init__(self, worker_class: 'type[Worker]', count: Optional[int], *args,
               **kwargs):
    if not count:
      count = multiprocessing.cpu_count()
    self._stubs = [
        _make_stub(worker_class, *args, **kwargs) for _ in range(count)
    ]

  def __enter__(self):
    return self._stubs

  def __exit__(self, *args, **kwargs):
    # each shutdown may take ~1 second because of the timeout in the message
    # pump, so let's parallelize shutting everything down.
    with concurrent.futures.ThreadPoolExecutor() as tpe:
      concurrent.futures.wait([tpe.submit(s.shutdown) for s in self._stubs])
