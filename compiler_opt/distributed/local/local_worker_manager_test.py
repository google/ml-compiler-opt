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


class LocalWorkerManagerTest(absltest.TestCase):

  def test_pool(self):

    class Job(Worker):
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

    with local_worker_manager.LocalWorkerPool(Job, 2) as pool:
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


if __name__ == '__main__':
  multiprocessing.handle_test_main(absltest.main)
