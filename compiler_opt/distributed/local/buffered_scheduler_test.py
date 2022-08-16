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
from compiler_opt.distributed.local import buffered_scheduler


class BufferedSchedulerTest(absltest.TestCase):

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
