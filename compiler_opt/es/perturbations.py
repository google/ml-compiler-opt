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
"""Class for generating and processing perturbations."""

import numpy as np
import numpy.typing as npt


class Perturbations:
  """Wraps perturbations and logic about them."""

  def __init__(self,
               dimension: int,
               precision_parameter: float,
               perturbation_scale_vector: npt.NDArray[np.float32],
               total_num_perturbations: int,
               is_antithetic: bool,
               seed: int | None = None):
    """Construct a Perturbations object.

    Args:
      dimension: Dimension of each perturbation vector.
      precision_parameter: Standard deviation scale factor of perturbations.
      perturbation_scale_vector: Scale vector for perturbations.
      total_num_perturbations: How many independent perturbations to attempt.
      is_antithetic: Whether to use antithetic (positive-negative pair)
        exploration.
      seed: Seed for the random number generator.
    """
    self._dimension = dimension
    self._precision_parameter = precision_parameter
    self._perturbation_scale_vector = perturbation_scale_vector
    self._total_num_perturbations = total_num_perturbations
    self._is_antithetic = is_antithetic
    self._rng = np.random.default_rng(seed=seed)

  def get_next_perturbations(self) -> list[npt.NDArray[np.float32]]:
    """Get the next list of perturbations."""
    perturbations = [
        self._rng.normal(size=self._dimension) * self._precision_parameter *
        self._perturbation_scale_vector
        for _ in range(self._total_num_perturbations)
    ]

    if self._is_antithetic:
      perturbations = [p for b in perturbations for p in (b, -b)]

    return perturbations

  def prune_skipped_perturbations(
      self, perturbations: list[npt.NDArray[np.float32]],
      rewards: list[float | None]
  ) -> tuple[list[npt.NDArray[np.float32]], list[float]]:
    """Remove perturbations that were skipped during the training step.

    Perturbations may be skipped due to an early exit condition or a server
    error (clang timeout, malformed training example, etc).

    If is_antithetic is True, if either the positive or negative perturbation
    of a pair fails, we prune both of them to maintain pairing.

    Args:
      perturbations: The model perturbations used for the ES training step.
      rewards: The rewards for each perturbation.

    Returns:
      A tuple (pruned_perturbations, pruned_rewards) of the pruned lists.
    """
    if self._is_antithetic:
      pairs = list(
          zip(perturbations[0::2], perturbations[1::2], rewards[0::2],
              rewards[1::2]))
      valid_pairs = [(p_pos, p_neg, r_pos, r_neg)
                     for p_pos, p_neg, r_pos, r_neg in pairs
                     if r_pos is not None and r_neg is not None]
      pruned_perturbations = [
          p for p_pos, p_neg, _, _ in valid_pairs for p in (p_pos, p_neg)
      ]
      pruned_rewards = [
          r for _, _, r_pos, r_neg in valid_pairs for r in (r_pos, r_neg)
      ]
    else:
      valid = [(p, r) for p, r in zip(perturbations, rewards) if r is not None]
      pruned_perturbations = [p for p, _ in valid]
      pruned_rewards = [r for _, r in valid]

    return pruned_perturbations, pruned_rewards
