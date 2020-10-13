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

"""Tests for compiler_opt.rl.local_data_collector."""

from unittest import mock

import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl import inline_runner
from compiler_opt.rl import local_data_collector


class LocalDataCollectorTest(tf.test.TestCase):

  def test_local_data_collector(self):
    mock_inliner = mock.create_autospec(inline_runner.InlineRunner)

    def mock_collect_data(ir_path, tf_policy_dir, default_policy_size):
      assert ir_path == 'a'
      assert tf_policy_dir == 'policy'
      assert default_policy_size is None or default_policy_size == 1
      if default_policy_size is None:
        return 1, 1
      else:
        return 2, 1

    mock_inliner.collect_data = mock_collect_data

    def create_test_iterator_fn():
      def _test_iterator_fn(data_list):
        assert data_list in ([1] * 10, [2] * 10)
        if data_list == [1] * 10:
          return iter(tf.data.Dataset.from_tensor_slices([1, 2, 3]))
        else:
          return iter(tf.data.Dataset.from_tensor_slices([4, 5, 6]))

      return _test_iterator_fn

    collector = local_data_collector.LocalDataCollector(
        ir_files=['a'] * 100,
        num_workers=4,
        num_modules=10,
        runner=mock_inliner.collect_data,
        parser=create_test_iterator_fn())

    data_iterator = collector.collect_data(policy_path='policy')
    data = list(data_iterator)
    self.assertEqual([1, 2, 3], data)

    data_iterator = collector.collect_data(policy_path='policy')
    data = list(data_iterator)
    self.assertEqual([4, 5, 6], data)


if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
