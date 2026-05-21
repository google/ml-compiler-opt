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
"""Tests for perturbations."""

from absl.testing import absltest
import numpy as np

from compiler_opt.es import perturbations


class PerturbationsTest(absltest.TestCase):

  def test_get_perturbations(self):
    scale_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dimension = 3
    precision_parameter = 1.5
    total_num_perturbations = 5

    # Test non-antithetic
    perts_obj = perturbations.Perturbations(
        dimension=dimension,
        precision_parameter=precision_parameter,
        perturbation_scale_vector=scale_vector,
        total_num_perturbations=total_num_perturbations,
        is_antithetic=False,
        seed=42)

    next_perts = perts_obj.get_next_perturbations()
    self.assertLen(next_perts, total_num_perturbations)
    for p in next_perts:
      self.assertEqual(p.shape, (dimension,))

    # Verify deterministic generation
    perts_obj_2 = perturbations.Perturbations(
        dimension=dimension,
        precision_parameter=precision_parameter,
        perturbation_scale_vector=scale_vector,
        total_num_perturbations=total_num_perturbations,
        is_antithetic=False,
        seed=42)
    next_perts_2 = perts_obj_2.get_next_perturbations()
    for p1, p2 in zip(next_perts, next_perts_2):
      np.testing.assert_array_almost_equal(p1, p2)

  def test_get_perturbations_antithetic(self):
    scale_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dimension = 3
    precision_parameter = 1.5
    total_num_perturbations = 5

    perts_obj = perturbations.Perturbations(
        dimension=dimension,
        precision_parameter=precision_parameter,
        perturbation_scale_vector=scale_vector,
        total_num_perturbations=total_num_perturbations,
        is_antithetic=True,
        seed=42)

    next_perts = perts_obj.get_next_perturbations()
    # For antithetic, each generated perturbation b is expanded to b and -b.
    # So length should be 2 * total_num_perturbations.
    self.assertLen(next_perts, 2 * total_num_perturbations)

    # Verify that odd indices are the negative of even indices.
    for i in range(0, len(next_perts), 2):
      np.testing.assert_array_almost_equal(next_perts[i], -next_perts[i + 1])

  def test_prune_skipped_perturbations(self):
    scale_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    perts_obj = perturbations.Perturbations(
        dimension=3,
        precision_parameter=1.0,
        perturbation_scale_vector=scale_vector,
        total_num_perturbations=5,
        is_antithetic=False)

    dummy_perturbations = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9]),
    ]
    rewards = [1.5, None, -0.5]

    pruned_p, pruned_r = perts_obj.prune_skipped_perturbations(
        dummy_perturbations, rewards)

    self.assertLen(pruned_p, 2)
    self.assertLen(pruned_r, 2)
    np.testing.assert_array_equal(pruned_p[0], dummy_perturbations[0])
    np.testing.assert_array_equal(pruned_p[1], dummy_perturbations[2])
    self.assertEqual(pruned_r, [1.5, -0.5])

  def test_prune_skipped_perturbations_antithetic(self):
    scale_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    perts_obj = perturbations.Perturbations(
        dimension=3,
        precision_parameter=1.0,
        perturbation_scale_vector=scale_vector,
        total_num_perturbations=6,
        is_antithetic=True)

    dummy_perturbations = [
        np.array([1, 2, 3]),
        np.array([-1, -2, -3]),
        np.array([4, 5, 6]),
        np.array([-4, -5, -6]),
        np.array([7, 8, 9]),
        np.array([-7, -8, -9]),
    ]

    # Case 1: Second element of first pair is pruned -> both elements of the
    # first pair are pruned.
    rewards = [1.0, None, 2.0, 3.0, 4.0, 5.0]
    pruned_p, pruned_r = perts_obj.prune_skipped_perturbations(
        dummy_perturbations, rewards)
    self.assertLen(pruned_p, 4)
    self.assertLen(pruned_r, 4)
    np.testing.assert_array_equal(pruned_p[0], dummy_perturbations[2])
    np.testing.assert_array_equal(pruned_p[1], dummy_perturbations[3])
    np.testing.assert_array_equal(pruned_p[2], dummy_perturbations[4])
    np.testing.assert_array_equal(pruned_p[3], dummy_perturbations[5])
    self.assertEqual(pruned_r, [2.0, 3.0, 4.0, 5.0])

    # Case 2: First element of second pair is pruned -> both elements of the
    # second pair are pruned.
    rewards = [1.0, 2.0, None, 3.0, 4.0, 5.0]
    pruned_p, pruned_r = perts_obj.prune_skipped_perturbations(
        dummy_perturbations, rewards)
    self.assertLen(pruned_p, 4)
    self.assertLen(pruned_r, 4)
    np.testing.assert_array_equal(pruned_p[0], dummy_perturbations[0])
    np.testing.assert_array_equal(pruned_p[1], dummy_perturbations[1])
    np.testing.assert_array_equal(pruned_p[2], dummy_perturbations[4])
    np.testing.assert_array_equal(pruned_p[3], dummy_perturbations[5])
    self.assertEqual(pruned_r, [1.0, 2.0, 4.0, 5.0])


if __name__ == '__main__':
  absltest.main()
