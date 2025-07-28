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

import json
import os
import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from compiler_opt.rl import constant
from compiler_opt.rl import feature_ops


def _get_sqrt_z_score_preprocessing_fn_cross_product():
  testcases = []
  for sqrt in [True, False]:
    for z_score in [True, False]:
      for preprocessing_fn in [None, lambda x: x * x]:
        # pylint: disable=line-too-long
        test_name = (
            f'sqrt_{sqrt}_zscore_{z_score}_preprocessfn_{preprocessing_fn}')
        testcases.append((test_name, sqrt, z_score, preprocessing_fn))
  return testcases


class FeatureUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self._quantile_file_dir = os.path.join(constant.BASE_DIR, 'testdata')
    super().setUp()

  def _create_temp_vocab_file(self, vocab_data):
    """Helper method to create a temporary vocab file with given data."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False) as f:
      if isinstance(vocab_data, str):
        f.write(vocab_data)  # For invalid JSON
      else:
        json.dump(vocab_data, f)
      return f.name

  def _test_vocab_file_dimensions(self, vocab_data, expected_dimensions):
    """Helper method to test vocab file dimension reading."""
    temp_file = self._create_temp_vocab_file(vocab_data)
    try:
      dimensions = feature_ops.get_ir2vec_dimensions_from_vocab_file(temp_file)
      self.assertEqual(expected_dimensions, dimensions)
    finally:
      os.unlink(temp_file)

  def test_build_quantile_map(self):
    quantile_map = feature_ops.build_quantile_map(self._quantile_file_dir)

    self.assertLen(quantile_map, 1)

    self.assertIn('edge_count', quantile_map)

    quantile = quantile_map['edge_count']

    # quantile
    self.assertLen(quantile, 9)
    self.assertEqual(2, quantile[0])
    self.assertEqual(8, quantile[6])

  def test_discard_fn(self):
    # obs in shape of [2, 1].
    obs = tf.constant(value=[[2.0], [8.0]])
    output = feature_ops.discard_fn(obs)

    self.assertAllEqual([2, 1, 1], output.shape)

  def test_identity_fn(self):
    # obs in shape of [2, 1].
    obs = tf.constant(value=[[2.0], [8.0]])
    output = feature_ops.identity_fn(obs)

    expected = np.array([[[2.0]], [[8.0]]])

    self.assertAllEqual([2, 1, 1], output.shape)
    self.assertAllClose(expected.tolist(), output)

  def test_identity_fn_without_expand_dims(self):
    # obs in shape of [2, 1].
    obs = tf.constant(value=[[2.0], [8.0]])
    output = feature_ops.identity_fn(obs, expand_dims=False)
    self.assertAllEqual(obs.shape, output.shape)
    self.assertAllClose(obs, output)

  @parameterized.named_parameters(
      *_get_sqrt_z_score_preprocessing_fn_cross_product())
  def test_normalize_fn_sqrt_z_normalization(self, with_sqrt, with_z_score,
                                             preprocessing_fn):
    quantile_map = feature_ops.build_quantile_map(self._quantile_file_dir)
    quantile = quantile_map['edge_count']
    normalize_fn = feature_ops.get_normalize_fn(
        quantile, with_sqrt, with_z_score, preprocessing_fn=preprocessing_fn)

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
      if preprocessing_fn:
        expected = np.concatenate([expected, [[[-0.406244]], [[-0.33180502]]]],
                                  axis=-1)
      else:
        expected = np.concatenate([expected, [[[-0.555968]], [[-0.155671]]]],
                                  axis=-1)

    self.assertAllEqual(expected_shape, output.shape)
    self.assertAllClose(expected, output)

  def test_get_ir2vec_normalize_fn(self):
    normalize_fn = feature_ops.get_ir2vec_normalize_fn()
    obs = tf.constant(value=[[2.0], [8.0]])
    output = normalize_fn(obs)
    self.assertAllEqual(obs.shape, output.shape)
    self.assertAllClose(obs, output)

    normalize_fn = feature_ops.get_ir2vec_normalize_fn(
        with_standardization=True)
    output = normalize_fn(obs)
    expected = np.array([[-1], [1]])
    self.assertAllEqual(obs.shape, output.shape)
    self.assertAllClose(expected, output)

  def test_get_ir2vec_dimensions_from_vocab_file_valid_file(self):
    vocab_data = {
        'section1': {
            'instruction1': [1.0, 2.0, 3.0, 4.0, 5.0],  # 5 dimensions
            'instruction2': [2.0, 3.0, 4.0, 5.0, 6.0]
        },
        'section2': {
            'instruction3': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    }
    self._test_vocab_file_dimensions(vocab_data, 5)

  @parameterized.named_parameters(
      ('empty_embedding', {
          'section1': {
              'instruction1': []
          }
      }, 0),
      ('invalid_json', 'invalid json content {', 0),
      ('wrong_structure_list', [], 0),
      ('section_not_dict', {
          'section1': 'not_a_dict'
      }, 0),
      ('embedding_not_list', {
          'section1': {
              'instruction1': 'not_a_list'
          }
      }, 0),
      ('empty_sections', {}, 0),
      ('section_with_no_embeddings', {
          'section1': {}
      }, 0),
  )
  def test_get_ir2vec_dimensions_from_vocab_file_error_cases(
      self, vocab_data, expected_dimensions):
    """Test various error cases that should return 0 dimensions."""
    self._test_vocab_file_dimensions(vocab_data, expected_dimensions)

  def test_get_ir2vec_dimensions_from_vocab_file_nonexistent_file(self):
    """Test handling of non-existent file."""
    dimensions = feature_ops.get_ir2vec_dimensions_from_vocab_file(
        '/nonexistent/path/file.json')
    self.assertEqual(0, dimensions)


if __name__ == '__main__':
  tf.test.main()
