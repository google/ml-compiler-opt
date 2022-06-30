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
import queue  # pylint: disable=unused-import
import threading

from absl import logging
# pylint: disable=unused-import
from compiler_opt.distributed.worker import Worker

from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, Optional


@dataclasses.dataclass(frozen=True)
class Task:
  msgid: int
  func_name: str
  args: tuple
  kwargs: dict
  is_urgent: bool


@dataclasses.dataclass(frozen=True)
class TaskResult:
  msgid: int
  success: bool
  value: Any


def _run_impl(in_q: 'queue.Queue[Task]', out_q: 'queue.Queue[TaskResult]',
              worker_class: 'type[Worker]', *args, **kwargs):
  """Worker process entrypoint."""
  # Note: the out_q is typed as taking only TaskResult objects, not
  # Optional[TaskResult], despite that being the type it is used on the Stub
  # side. This is because the `None` value is only injected by the Stub itself.
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
      except BaseException as e:  # pylint: disable=broad-except
        out_q.put(TaskResult(msgid=task.msgid, success=False, value=e))
    else:
      pool.submit(application).add_done_callback(make_ondone(task.msgid))


def _run(*args, **kwargs):
  try:
    _run_impl(*args, **kwargs)
  except BaseException as e:
    logging.error(e)
    raise e


def _make_stub(cls: 'type[Worker]', *args, **kwargs):

  class _Stub():
    """Client stub to a worker hosted by a process."""

    def __init__(self):
      self._send: 'queue.Queue[Task]' = multiprocessing.get_context().Queue()
      self._receive: 'queue.Queue[Optional[TaskResult]]' = \
         multiprocessing.get_context().Queue()

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
      self._map: Dict[int, concurrent.futures.Future] = {}
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

    def _is_cancelled_or_broken(self):
      return self._stop_pump.is_set() or not self._process.is_alive()

    def _msg_pump(self):
      while not self._is_cancelled_or_broken():
        task_result = self._receive.get()
        if not task_result:
          # we're shutting down. Let the while loop condition exit.
          continue
        with self._lock:
          future = self._map[task_result.msgid]
          del self._map[task_result.msgid]
          if task_result.success:
            future.set_result(task_result.value)
          else:
            future.set_exception(task_result.value)
      self._done.set()
      with self._lock:
        for _, v in self._map.items():
          v.set_exception(concurrent.futures.CancelledError())
        self._map = None

    def __getattr__(self, name) -> Callable[[Any], concurrent.futures.Future]:
      result_future = concurrent.futures.Future()
      if self._is_cancelled_or_broken():
        result_future.set_exception(concurrent.futures.CancelledError())
        return lambda *_, **__: result_future

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
      # Unblock the msg pump
      self._receive.put(None)

    def wait_for_msg_pump_exit(self):
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
    # first, trigger killing the worker process and exiting of the msg pump,
    # which will also clear out any pending futures.
    for s in self._stubs:
      s.shutdown()
    # now wait for the message pumps to indicate they exit.
    for s in self._stubs:
      s.wait_for_msg_pump_exit()
