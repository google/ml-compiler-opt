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
"""Test for local worker manager."""

import concurrent.futures
import time

from absl.testing import absltest
from compiler_opt.distributed.worker import Worker
from compiler_opt.distributed.local import local_worker_manager
from tf_agents.system import system_multiprocessing as multiprocessing


class JobNormal(Worker):
  """Test worker."""

  def __init__(self):
    self._token = 0

  @classmethod
  def is_priority_method(cls, method_name: str) -> bool:
    return method_name == 'priority_method'

  def priority_method(self):
    return f'priority {self._token}'

  def get_token(self):
    return self._token

  def set_token(self, value):
    self._token = value


class JobFail(Worker):

  def __init__(self, wont_be_passed):
    self._arg = wont_be_passed

  def method(self):
    return self._arg


class JobSlow(Worker):

  def method(self):
    time.sleep(3600)


class JobCounter(Worker):
  """Test worker."""

  def __init__(self):
    self.times = []

  @classmethod
  def is_priority_method(cls, method_name: str) -> bool:
    return method_name == 'get_times'

  def start(self):
    while True:
      self.times.append(time.time())
      time.sleep(0.05)

  def get_times(self):
    return self.times


class LocalWorkerManagerTest(absltest.TestCase):

  def test_pool(self):

    with local_worker_manager.LocalWorkerPool(JobNormal, 2) as pool:
      p1 = pool[0]
      p2 = pool[1]
      set_futures = [p1.set_token(1), p2.set_token(2)]
      done, not_done = concurrent.futures.wait(set_futures)
      self.assertLen(done, 2)
      self.assertEmpty(not_done)
      self.assertLen([f for f in done if not f.exception()], 2)
      self.assertEqual(p1.get_token().result(), 1)
      self.assertEqual(p2.get_token().result(), 2)
      self.assertEqual(p1.priority_method().result(), 'priority 1')
      self.assertEqual(p2.priority_method().result(), 'priority 2')
      # wait - to make sure the pump doesn't panic if there's no new messages
      time.sleep(3)
      # everything still works
      self.assertEqual(p2.get_token().result(), 2)

  def test_failure(self):

    with local_worker_manager.LocalWorkerPool(JobFail, 2) as pool:
      with self.assertRaises(concurrent.futures.CancelledError):
        # this will fail because we didn't pass the arg to the ctor, so the
        # worker hosting process will crash.
        pool[0].method().result()

  def test_worker_crash_while_waiting(self):

    with local_worker_manager.LocalWorkerPool(JobSlow, 2) as pool:
      p = pool[0]
      f = p.method()
      self.assertFalse(f.done())
      try:
        p._process.kill()  # pylint: disable=protected-access
      finally:
        with self.assertRaises(concurrent.futures.CancelledError):
          _ = f.result()

  def test_pause_resume(self):

    with local_worker_manager.LocalWorkerPool(JobCounter, 1) as pool:
      p = pool[0]

      # Fill the q for 1 second
      p.start()
      time.sleep(1)

      # Then pause the process for 1 second
      p.pause()
      time.sleep(1)

      # Then resume the process and wait 1 more second
      p.resume()
      time.sleep(1)

      times = p.get_times().result()

      # If pause/resume worked, there should be a gap of at least 0.5 seconds.
      # Otherwise, this will throw an exception.
      self.assertNotEmpty(times)
      last_time = times[0]
      for cur_time in times:
        if cur_time - last_time > 0.5:
          return
      raise ValueError('Failed to find a 2 second gap in times.')


if __name__ == '__main__':
  multiprocessing.handle_test_main(absltest.main)
