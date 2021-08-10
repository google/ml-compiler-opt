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

"""Tests for compiler_opt.rl.feature_ops."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from compiler_opt.rl import constant
from compiler_opt.rl import feature_ops

_WITH_Z_SCORE_SQRT_PRODUCT_VALUES = [('with_sqrt_with_z_score', True, True),
                                     ('with_sqrt_without_z_score', True, False),
                                     ('without_sqrt_with_z_score', False, True),
                                     ('without_sqrt_without_z_score', False,
                                      False)]


class FeatureUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self._quantile_file_dir = os.path.join(constant.BASE_DIR, 'testdata')
    super(FeatureUtilsTest, self).setUp()

  def test_build_quantile_map(self):
    quantile_map = feature_ops.build_quantile_map(self._quantile_file_dir)

    self.assertLen(quantile_map, 1)

    self.assertIn('edge_count', quantile_map)

    quantile, mean, std = quantile_map['edge_count']

    # quantile
    self.assertLen(quantile, 9)
    self.assertEqual(2, quantile[0])
    self.assertEqual(8, quantile[6])

    # mean
    self.assertAllClose(10.333333, mean)

    # std
    self.assertAllClose(14.988885, std)

  def test_discard_fn(self):
    # obs in shape of [2, 1].
    obs = tf.constant(value=[[2.0], [8.0]])
    output = feature_ops.discard_fn(obs)

    self.assertAllEqual([2, 1, 0], output.shape)

  @parameterized.named_parameters(*_WITH_Z_SCORE_SQRT_PRODUCT_VALUES)
  def test_normalize_fn(self, with_sqrt, with_z_score):
    quantile_map = feature_ops.build_quantile_map(self._quantile_file_dir)
    quantile, mean, std = quantile_map['edge_count']
    normalize_fn = feature_ops.get_normalize_fn(quantile, mean, std, with_sqrt,
                                                with_z_score)

    obs = tf.constant(value=[[2.0], [8.0]])
    output = normalize_fn(obs)

    expected_shape = [2, 1, 2]
    expected = np.array([[[0.333333, 0.111111]], [[0.777778, 0.604938]]])

    if with_sqrt:
      expected_shape[2] += 1
      expected = np.concatenate([expected, [[[0.57735]], [[0.881917]]]],
                                axis=-1)

    if with_z_score:
      expected_shape[2] += 1
      expected = np.concatenate([expected, [[[-0.555968]], [[-0.155671]]]],
                                axis=-1)

    self.assertAllEqual(expected_shape, output.shape)
    self.assertAllClose(expected, output)


if __name__ == '__main__':
  tf.test.main()
