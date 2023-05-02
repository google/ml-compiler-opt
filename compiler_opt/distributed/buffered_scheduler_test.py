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
"""Test for buffered_scheduler."""

import concurrent.futures
import threading
import time

from absl.testing import absltest
from compiler_opt.distributed import worker
from compiler_opt.distributed import buffered_scheduler
from compiler_opt.distributed.local import local_worker_manager


class BufferedSchedulerTest(absltest.TestCase):

  def test_canonical_futures(self):

    class WaitingWorker(worker.Worker):

      def wait_seconds(self, n: int):
        time.sleep(n)
        return n + 1

    with local_worker_manager.LocalWorkerPoolManager(WaitingWorker, 2) as pool:
      _, futures = buffered_scheduler.schedule_on_worker_pool(
          lambda w, v: w.wait_seconds(v), range(4), pool)
      not_done = futures
      entered_count = 0
      while not_done:
        _, not_done = concurrent.futures.wait(not_done, timeout=0.5)
        entered_count += 1
      self.assertGreater(entered_count, 1)
      self.assertListEqual([r.result() for r in futures], list(range(1, 5)))

  def test_simple_scheduling(self):

    class TheWorker(worker.Worker):

      def square(self, the_value, extra_factor=1):
        return the_value * the_value * extra_factor

    with local_worker_manager.LocalWorkerPoolManager(TheWorker, 2) as pool:
      workers, futures = buffered_scheduler.schedule_on_worker_pool(
          lambda w, v: w.square(v), range(10), pool)
      self.assertLen(workers, 2)
      self.assertLen(futures, 10)
      worker.wait_for(futures)
      self.assertListEqual([f.result() for f in futures],
                           [x * x for x in range(10)])

      _, futures = buffered_scheduler.schedule_on_worker_pool(
          lambda w, v: w.square(**v), [dict(the_value=v) for v in range(10)],
          pool)
      worker.wait_for(futures)
      self.assertListEqual([f.result() for f in futures],
                           [x * x for x in range(10)])

      # same idea, but mix some kwargs
      _, futures = buffered_scheduler.schedule_on_worker_pool(
          lambda w, v: w.square(v[0], **v[1]),
          [(v, dict(extra_factor=10)) for v in range(10)], pool)
      worker.wait_for(futures)
      self.assertListEqual([f.result() for f in futures],
                           [x * x * 10 for x in range(10)])

  def test_schedules(self):
    call_count = [0] * 4
    locks = [threading.Lock() for _ in range(4)]

    def wkr_factory(i):

      def wkr():
        with locks[i]:
          call_count[i] += 1

      return wkr

    wkrs = [wkr_factory(i) for i in range(4)]

    def job(wkr):
      future = concurrent.futures.Future()

      def task():
        wkr()
        future.set_result(0)

      threading.Timer(interval=0.10, function=task).start()
      return future

    work = [job] * 20

    worker.wait_for(buffered_scheduler.schedule(work, wkrs))
    self.assertEqual(sum(call_count), 20)

  def test_balances(self):
    call_count = [0] * 4
    locks = [threading.Lock() for _ in range(4)]

    def wkr_factory(i):

      def wkr():
        with locks[i]:
          call_count[i] += 1

      return wkr

    def slow_wkr():
      with locks[0]:
        call_count[0] += 1
      time.sleep(1)

    wkrs = [slow_wkr] + [wkr_factory(i) for i in range(1, 4)]

    def job(wkr):
      future = concurrent.futures.Future()

      def task():
        wkr()
        future.set_result(0)

      threading.Timer(interval=0.10, function=task).start()
      return future

    work = [job] * 20

    worker.wait_for(buffered_scheduler.schedule(work, wkrs, buffer=2))
    self.assertEqual(sum(call_count), 20)
    # since buffer=2, 2 tasks get assigned to the slow wkr, the rest
    # should've been assigned elsewhere if load balancing works.
    self.assertEqual(call_count[0], 2)


if __name__ == '__main__':
  absltest.main()
