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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent.futures
import functools
import multiprocessing
import multiprocessing.connection
import queue
import threading

from compiler_opt.core.abstract_worker import AbstractWorker
from typing import Callable, Optional, Tuple

def worker(ctor, in_q:queue.Queue, out_q:queue.Queue):
  pool = ThreadPoolExecutor()
  obj = ctor()

  def make_ondone(msgid):

    def on_done(f: concurrent.futures.Future):
      if f.exception():
        out_q.put((msgid, False, f.exception()))
      else:
        out_q.put((msgid, True, f.result()))
    return on_done

  while True:
    msgid, fname, args, kwargs, urgent = in_q.get()
    the_func = getattr(obj, fname)
    if urgent:
      try:
        res = the_func(*args, **kwargs)
        out_q.put((msgid, True, res))
      except Exception as e:
        out_q.put((msgid, False, e))
    else:
      pool.submit(the_func, *args,
                  **kwargs).add_done_callback(make_ondone(msgid))


class Stub:
  def __init__(self, ctor):
    self._send = multiprocessing.get_context().Queue()
    self._receive = multiprocessing.get_context().Queue()

    self._process = multiprocessing.Process(
        target=functools.partial(worker, ctor, self._send, self._receive))
    self._lock = threading.Lock()
    self._map:dict[object, concurrent.futures.Future] = {}
    self._pump = threading.Thread(target=self._msg_pump)
    self._done = threading.Event()
    self._msgidlock = threading.Lock()
    self._msgid = 0
    self._process.start()
    self._pump.start()

  def _msg_pump(self):
    while not self._done.is_set():
      msgid, succeeded, value = self._receive.get()
      with self._lock:
        future = self._map[msgid]
        del self._map[msgid]
        if succeeded:
          future.set_result(value)
        else:
          future.set_exception(value)

  def __getattr__(self, name):
    with self._msgidlock:
      msgid = self._msgid
      self._msgid += 1
    def remote_call(*args, **kwargs):
      self._send.put(
          (msgid, name, args, kwargs, name == 'cancel_all_work'))
      future = concurrent.futures.Future()
      with self._lock:
        self._map[msgid] = future
      return future

    return remote_call
  
  def kill(self):
    try:
      self._process.kill()
    except:
      pass
    self._done.set()

def get_compilation_jobs(ctor: Callable[[], AbstractWorker],
                         count: Optional[int]) -> Tuple[Callable, list]:

  if not count:
    count = multiprocessing.cpu_count()
  stubs = [Stub(ctor) for _ in range(count)]
  def shutdown():
    for s in stubs:
      s.kill()
  return shutdown, [Stub(ctor) for _ in range(count)]
