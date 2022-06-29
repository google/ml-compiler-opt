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
"""Tests for data_collector."""

# pylint: disable=protected-access
import sys
from unittest import mock

from absl.testing import absltest

from compiler_opt.rl import data_collector


class DataCollectorTest(absltest.TestCase):

  def test_build_distribution_monitor(self):
    data = [3, 2, 1]
    monitor_dict = data_collector.build_distribution_monitor(data)
    reference_dict = {'mean': 2, 'p_0.1': 1}
    # Issue #38
    if sys.version_info.minor >= 9:
      self.assertEqual(monitor_dict, monitor_dict | reference_dict)
    else:
      self.assertEqual(monitor_dict, {**monitor_dict, **reference_dict})

  @mock.patch('time.time')
  def test_early_exit(self, mock_time):
    mock_time.return_value = 0
    early_exit = data_collector.EarlyExitChecker(
        num_modules=10, deadline=10, thresholds=((.9, 0), (.5, .5), (0, 1)))

    # We've waited no time, so have to hit 90% to early exit
    self.assertFalse(early_exit._should_exit(0))
    self.assertFalse(early_exit._should_exit(5))
    self.assertTrue(early_exit._should_exit(9))
    self.assertEqual(early_exit.waited_time(), 0)

    # We've waited 50% of the time, so only need to hit 50% to exit
    mock_time.return_value = 5
    self.assertFalse(early_exit._should_exit(0))
    self.assertTrue(early_exit._should_exit(5))
    self.assertEqual(early_exit.waited_time(), 5)

    # We've waited 100% of the time, exit no matter what
    mock_time.return_value = 10
    self.assertTrue(early_exit._should_exit(0))
    self.assertEqual(early_exit.waited_time(), 10)


if __name__ == '__main__':
  absltest.main()
