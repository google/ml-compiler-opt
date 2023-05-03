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

import gin

from absl.testing import absltest
from compiler_opt.distributed import worker


@gin.configurable(module='_test')
class SomeType:

  def __init__(self, argument):
    pass


class WorkerTest(absltest.TestCase):

  def test_gin_args(self):
    with gin.unlock_config():
      gin.bind_parameter('_test.SomeType.argument', 42)
    real_args = worker.get_full_worker_args(
        SomeType, more_args=2, even_more_args='hi')
    self.assertDictEqual(real_args,
                         dict(argument=42, more_args=2, even_more_args='hi'))


if __name__ == '__main__':
  absltest.main()
