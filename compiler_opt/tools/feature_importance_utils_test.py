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
"""Tests for the feature_importance_utils.py module"""

from absl.testing import absltest

from compiler_opt.tools import feature_importance_utils

import tensorflow as tf
import numpy

from types import SimpleNamespace

test_tensor_spec = {
    'test1': tf.constant([1, 2], dtype=tf.int64),
    'test2': tf.constant([3, 4], dtype=tf.int64)
}


class FeatureImportanceTest(absltest.TestCase):

  def test_get_input_signature(self):
    input_signature = feature_importance_utils.get_input_signature(
        test_tensor_spec)
    self.assertListEqual(list(input_signature.keys()), ['test1', 'test2'])
    self.assertListEqual(input_signature['test1'][0].tolist(), [2])
    self.assertEqual(input_signature['test1'][1], tf.int64)

  def test_get_signature_total_size(self):
    input_signature = feature_importance_utils.get_input_signature(
        test_tensor_spec)
    total_size = feature_importance_utils.get_signature_total_size(
        input_signature)
    self.assertEqual(total_size, 4)

  def test_packing_flattening(self):
    input_signature = feature_importance_utils.get_input_signature(
        test_tensor_spec)
    total_size = feature_importance_utils.get_signature_total_size(
        input_signature)
    flattened_input = feature_importance_utils.flatten_input(
        test_tensor_spec, total_size)
    packed_input = feature_importance_utils.pack_flat_array_into_input(
        flattened_input, input_signature)
    self.assertListEqual(list(packed_input.keys()), ['test1', 'test2'])
    self.assertTrue(
        numpy.array_equal(test_tensor_spec['test1'].numpy(),
                          packed_input['test1'].numpy()))
    self.assertTrue(
        numpy.array_equal(test_tensor_spec['test2'].numpy(),
                          packed_input['test2'].numpy()))

  def test_trajectory_processing(self):
    batched_test_tensor_spec = {
        'test1': tf.constant([[1, 2]]),
        'test2': tf.constant([[3, 4]])
    }
    pre_trajectory = {
        'observation': batched_test_tensor_spec,
        'step_type': tf.constant([[1]]),
        'reward': tf.constant([[2]]),
        'discount': tf.constant([[3]])
    }
    trajectory = SimpleNamespace(**pre_trajectory)
    processed_trajectory = feature_importance_utils.process_raw_trajectory(
        trajectory)
    self.assertListEqual(
        list(processed_trajectory.keys()),
        ['test1', 'test2', 'step_type', 'reward', 'discount'])
    # make sure values are correct
    self.assertTrue(
        numpy.array_equal(processed_trajectory['test1'].numpy(), [1, 2]))
    self.assertTrue(
        numpy.array_equal(processed_trajectory['test2'].numpy(), [3, 4]))
    # make sure the tensors got squeezed
    self.assertTrue(
        numpy.array_equal(tf.shape(processed_trajectory['test1']).numpy(), [2]))
    self.assertTrue(
        numpy.array_equal(
            tf.shape(processed_trajectory['discount']).numpy(), [1]))

  def test_get_max_part_size(self):
    ragged_nested_tensor_spec = {
        'test1': tf.constant([1, 2, 3, 4]),
        'test2': tf.constant([1, 2])
    }
    input_signature = feature_importance_utils.get_input_signature(
        ragged_nested_tensor_spec)
    max_part_size = feature_importance_utils.get_max_part_size(input_signature)
    self.assertEqual(max_part_size, 4)
