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

# pylint: disable=protected-access
from absl.testing import absltest
from compiler_opt.distributed import nesting_worker
from compiler_opt.distributed import worker
from compiler_opt.distributed.local import local_worker_manager


class ToyWorker(worker.Worker):

  def say_hi(self):
    return 'hi!'


class NestingWorkerTest(absltest.TestCase):

  def test_setup(self):
    with local_worker_manager.LocalWorkerPoolManager(
        worker_class=nesting_worker.NestingWorker, count=2) as lwpm:
      nesting_manager = nesting_worker.create_nested_worker_manager(lwpm)
      lwpm_active = lwpm.get_currently_active()
      self.assertLen(lwpm_active, 2)
      with nesting_manager(worker_class=ToyWorker, count=10) as mgr:
        self.assertLen(mgr.get_currently_active(), 10)
        for nw in lwpm_active:
          self.assertLen(nw._get_registered_workers().result(), 5)
        for wkr in mgr.get_currently_active():
          self.assertEqual(wkr.say_hi().result(), 'hi!')
      for nw in lwpm_active:
        self.assertLen(nw._get_registered_workers().result(), 0)


if __name__ == '__main__':
  absltest.main()
