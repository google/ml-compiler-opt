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

  def test_build_quantile_map_from_config(self):
    quantile_map = feature_ops._build_quantile_map(self._quantile_file_dir)

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

  @parameterized.named_parameters(*_WITH_Z_SCORE_SQRT_PRODUCT_VALUES)
  def test_create_observation_processing_layer(self, with_z_score, with_sqrt):
    observation_processing_layer = (
        feature_ops.get_observation_processing_layer_creator(
            self._quantile_file_dir, with_sqrt, with_z_score))

    obs_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='edge_count')
    processing_layer = observation_processing_layer(obs_spec)

    inputs = tf.constant(value=[[2.0], [8.0]])
    outputs = processing_layer(inputs)

    outputs = self.evaluate(outputs)

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

    self.assertAllEqual(expected_shape, outputs.shape)
    self.assertAllClose(expected.tolist(), outputs)

  @parameterized.named_parameters(*_WITH_Z_SCORE_SQRT_PRODUCT_VALUES)
  def test_create_observation_processing_layer_for_dummy_features(
      self, with_z_score, with_sqrt):
    observation_processing_layer = (
        feature_ops.get_observation_processing_layer_creator(
            self._quantile_file_dir, with_sqrt, with_z_score))

    obs_spec = tf.TensorSpec(dtype=tf.float32, shape=(), name='dummy_feature')
    processing_layer = observation_processing_layer(obs_spec)

    inputs = tf.constant(value=[[2.0], [8.0]])
    outputs = processing_layer(inputs)

    outputs = self.evaluate(outputs)

    self.assertAllEqual([2, 1, 1], outputs.shape)
    self.assertAllClose([[[0]], [[0]]], outputs)


if __name__ == '__main__':
  tf.test.main()
