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

  def __init__(self, arg, *, kwarg):
    self._token = 0
    self._arg = arg
    self._kwarg = kwarg

  @classmethod
  def is_priority_method(cls, method_name: str) -> bool:
    return method_name == 'priority_method'

  def priority_method(self):
    return f'priority {self._token}'

  def get_token(self):
    return self._token

  def set_token(self, value):
    self._token = value

  def get_arg(self):
    return self._arg

  def get_kwarg(self):
    return self._kwarg


class JobFail(Worker):

  def __init__(self, wont_be_passed):
    self._arg = wont_be_passed

  def method(self):
    return self._arg


class JobSlow(Worker):

  def method(self):
    time.sleep(3600)


class LocalWorkerManagerTest(absltest.TestCase):

  def test_pool(self):

    arg = 'foo'
    kwarg = 'bar'

    with local_worker_manager.LocalWorkerPoolManager(
        JobNormal, 2, arg, kwarg=kwarg) as pool:
      p1 = pool.get_currently_active()[0]
      p2 = pool.get_currently_active()[1]
      set_futures = [p1.set_token(1), p2.set_token(2)]
      done, not_done = concurrent.futures.wait(set_futures)
      self.assertLen(done, 2)
      self.assertEmpty(not_done)
      self.assertLen([f for f in done if not f.exception()], 2)
      self.assertEqual(p1.get_token().result(), 1)
      self.assertEqual(p2.get_token().result(), 2)
      self.assertEqual(p1.priority_method().result(), 'priority 1')
      self.assertEqual(p2.priority_method().result(), 'priority 2')
      self.assertEqual(p1.get_arg().result(), 'foo')
      self.assertEqual(p2.get_arg().result(), 'foo')
      self.assertEqual(p2.get_kwarg().result(), 'bar')
      self.assertEqual(p2.get_kwarg().result(), 'bar')
      # wait - to make sure the pump doesn't panic if there's no new messages
      time.sleep(3)
      # everything still works
      self.assertEqual(p2.get_token().result(), 2)

  def test_failure(self):

    with local_worker_manager.LocalWorkerPoolManager(JobFail, 2) as pool:
      with self.assertRaises(concurrent.futures.CancelledError):
        # this will fail because we didn't pass the arg to the ctor, so the
        # worker hosting process will crash.
        pool.get_currently_active()[0].method().result()

  def test_worker_crash_while_waiting(self):

    with local_worker_manager.LocalWorkerPoolManager(JobSlow, 2) as pool:
      p = pool.get_currently_active()[0]
      f = p.method()
      self.assertFalse(f.done())
      try:
        p._process.kill()  # pylint: disable=protected-access
      finally:
        with self.assertRaises(concurrent.futures.CancelledError):
          _ = f.result()


if __name__ == '__main__':
  multiprocessing.handle_test_main(absltest.main)
