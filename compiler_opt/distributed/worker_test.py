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
"""Test for worker."""

from absl.testing import absltest
import concurrent.futures

from compiler_opt.distributed import worker


class LiftFuturesThroughListTest(absltest.TestCase):

  def test_normal_path(self):
    expected_list = [1, True, [2.0, False]]
    future_list = concurrent.futures.Future()
    list_future = worker.lift_futures_through_list(future_list,
                                                   len(expected_list))
    future_list.set_result(expected_list)
    worker.wait_for(list_future)

    self.assertEqual([f.result() for f in list_future], expected_list)

  def test_with_exceptions_in_list(self):
    expected_list = [1, ValueError('error')]
    future_list = concurrent.futures.Future()
    list_future = worker.lift_futures_through_list(future_list,
                                                   len(expected_list))
    future_list.set_result(expected_list)
    worker.wait_for(list_future)

    self.assertEqual(list_future[0].result(), expected_list[0])
    self.assertTrue(
        isinstance(worker.get_exception(list_future[1]), ValueError))

  def test_list_is_exception(self):
    expected_size = 42
    future_list = concurrent.futures.Future()
    list_future = worker.lift_futures_through_list(future_list, expected_size)
    future_list.set_exception(ValueError('error'))

    worker.wait_for(list_future)
    self.assertEqual(len(list_future), expected_size)
    for f in list_future:
      self.assertTrue(isinstance(worker.get_exception(f), ValueError))


if __name__ == '__main__':
  absltest.main()
