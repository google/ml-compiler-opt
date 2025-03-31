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
"""Class for coordinating blackbox optimization."""

from absl import logging
import dataclasses
import gin
import math
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from typing import Protocol

from compiler_opt.distributed.worker import FixedWorkerPool
from compiler_opt.es import blackbox_optimizers
from compiler_opt.rl import corpus
from compiler_opt.es import blackbox_evaluator  # pylint: disable=unused-import

# Pytype cannot pick up the pyi file for tensorflow.summary. Disable the error
# here as these errors are false positives.
# pytype: disable=pyi-error

# If less than 40% of requests succeed, skip the step.
_SKIP_STEP_SUCCESS_RATIO = 0.4

# The percentiles to report as individual values in Tensorboard.
_PERCENTILES_TO_REPORT = [25, 50, 75]


@gin.configurable
@dataclasses.dataclass(frozen=True)
class BlackboxLearnerConfig:
  """Hyperparameter configuration for BlackboxLearner."""

  # Total steps to train for
  total_steps: int

  # Name of the blackbox optimization algorithm
  blackbox_optimizer: blackbox_optimizers.Algorithm

  # What kind of ES training?
  #   - antithetic: for each perturbtation, try an antiperturbation
  #   - forward_fd: try total_num_perturbations independent perturbations
  estimator_type: blackbox_optimizers.EstimatorType

  # Should the rewards for blackbox optimization in a single step be normalized?
  fvalues_normalization: bool

  # How to update optimizer hyperparameters
  hyperparameters_update_method: blackbox_optimizers.UpdateMethod

  # Number of top performing perturbations to select in the optimizer
  # 0 means all
  num_top_directions: int

  # The type of evaluator to use.
  evaluator: 'type[blackbox_evaluator.BlackboxEvaluator]'

  # How many perturbations to attempt at each perturbation
  total_num_perturbations: int

  # How much to scale the stdev of the perturbations
  precision_parameter: float

  # Learning rate
  step_size: float

  # Whether or not to save a policy if it has the greatest reward seen so far.
  save_best_policy: bool


def _prune_skipped_perturbations(perturbations: list[npt.NDArray[np.float32]],
                                 rewards: list[float | None]):
  """Remove perturbations that were skipped during the training step.

  Perturbations may be skipped due to an early exit condition or a server error
  (clang timeout, malformed training example, etc). The blackbox optimizer
  assumes that each perturbations has a valid reward, so we must remove any of
  these "skipped" perturbations.

  Pruning occurs in-place.

  Args:
    perturbations: the model perturbations used for the ES training step.
    rewards: the rewards for each perturbation.

  Returns:
    The number of perturbations that were pruned.
  """
  indices_to_prune = []
  for i, reward in enumerate(rewards):
    if reward is None:
      indices_to_prune.append(i)

  # Iterate in reverse so that the indices remain valid
  for i in reversed(indices_to_prune):
    del perturbations[i]
    del rewards[i]

  return len(indices_to_prune)


class PolicySaverCallableType(Protocol):
  """Protocol for the policy saver function.
  A Protocol is required to type annotate
  the function with keyword arguments"""

  def __call__(self, parameters: npt.NDArray[np.float32],
               policy_name: str) -> None:
    ...


class BlackboxLearner:
  """Implementation of blackbox learning."""

  def __init__(self,
               blackbox_opt: blackbox_optimizers.BlackboxOptimizer,
               train_corpus: corpus.Corpus,
               output_dir: str,
               policy_saver_fn: PolicySaverCallableType,
               model_weights: npt.NDArray[np.float32],
               config: BlackboxLearnerConfig,
               initial_step: int = 0,
               deadline: float = 30.0,
               seed: int | None = None):
    """Construct a BlackboxLeaner.

    Args:
      blackbox_opt: the blackbox optimizer to use
      train_corpus: the training corpus to utiilize
      output_dir: the directory to write all outputs
      policy_saver_fn: function to save a policy to cns
      model_weights: the weights of the current model
      config: configuration for blackbox optimization.
      initial_step: the initial step for learning.
      deadline: the deadline in seconds for requests to the inlining server.
    """
    self._blackbox_opt = blackbox_opt
    self._train_corpus = train_corpus
    self._output_dir = output_dir
    self._policy_saver_fn = policy_saver_fn
    self._model_weights = model_weights
    self._config = config
    self._step = initial_step
    self._deadline = deadline
    self._seed = seed
    self._global_max_reward = 0.0

    self._summary_writer = tf.summary.create_file_writer(output_dir)

    self._evaluator = self._config.evaluator(self._train_corpus,
                                             self._config.estimator_type)

  def _get_perturbations(self) -> list[npt.NDArray[np.float32]]:
    """Get perturbations for the model weights."""
    rng = np.random.default_rng(seed=self._seed)
    return [
        rng.normal(size=len(self._model_weights)) *
        self._config.precision_parameter
        for _ in range(self._config.total_num_perturbations)
    ]

  def _update_model(self, perturbations: list[npt.NDArray[np.float32]],
                    rewards: list[float]) -> None:
    """Update the model given a list of perturbations and rewards."""
    self._model_weights = self._blackbox_opt.run_step(
        perturbations=np.array(perturbations),
        function_values=np.array(rewards),
        current_input=self._model_weights,
        current_value=np.mean(rewards))

  def _log_rewards(self, rewards: list[float]) -> None:
    """Log reward to console."""
    logging.info('Train reward: [%f]', np.mean(rewards))

  def _log_tf_summary(self, rewards: list[float]) -> None:
    """Log tensorboard data."""
    with self._summary_writer.as_default():
      tf.summary.scalar(
          'reward/average_reward_train', np.mean(rewards), step=self._step)

      tf.summary.scalar(
          'reward/maximum_reward_train', np.max(rewards), step=self._step)

      for percentile_to_report in _PERCENTILES_TO_REPORT:
        percentile_value = np.percentile(rewards, percentile_to_report)
        tf.summary.scalar(
            f'reward/{percentile_to_report}_percentile',
            percentile_value,
            step=self._step)

      tf.summary.histogram('reward/reward_train', rewards, step=self._step)

      train_regressions = [reward for reward in rewards if reward < 0]
      tf.summary.scalar(
          'reward/regression_probability_train',
          len(train_regressions) / len(rewards),
          step=self._step)

      tf.summary.scalar(
          'reward/regression_avg_train',
          np.mean(train_regressions) if len(train_regressions) > 0 else 0,
          step=self._step)

      # The "max regression" is the min value, as the regressions are negative.
      tf.summary.scalar(
          'reward/regression_max_train',
          min(train_regressions, default=0),
          step=self._step)

      train_wins = [reward for reward in rewards if reward > 0]
      tf.summary.scalar(
          'reward/win_probability_train',
          len(train_wins) / len(rewards),
          step=self._step)

  def _save_model(self) -> None:
    """Save the model."""
    logging.info('Saving the model.')
    self._policy_saver_fn(
        parameters=self._model_weights, policy_name=f'iteration{self._step}')

  def get_model_weights(self) -> npt.NDArray[np.float32]:
    return self._model_weights

  def set_baseline(self, pool: FixedWorkerPool) -> None:
    self._evaluator.set_baseline(pool)

  def run_step(self, pool: FixedWorkerPool) -> None:
    """Run a single step of blackbox learning.
    This does not instantaneously return due to several I/O
    and executions running while this waits for the responses"""
    logging.info('-' * 80)
    logging.info('Step [%d]', self._step)

    initial_perturbations = self._get_perturbations()
    # positive-negative pairs
    if (self._config.estimator_type ==
        blackbox_optimizers.EstimatorType.ANTITHETIC):
      initial_perturbations = [
          p for p in initial_perturbations for p in (p, -p)
      ]

    perturbations_as_bytes = [
        (self._model_weights + perturbation).astype(np.float32).tobytes()
        for perturbation in initial_perturbations
    ]

    results = self._evaluator.get_results(pool, perturbations_as_bytes)
    rewards = self._evaluator.get_rewards(results)

    num_pruned = _prune_skipped_perturbations(initial_perturbations, rewards)
    logging.info('Pruned [%d]', num_pruned)
    min_num_rewards = math.ceil(_SKIP_STEP_SUCCESS_RATIO * len(results))
    if len(rewards) < min_num_rewards:
      logging.warning(
          'Skipping the step, too many requests failed: %d of %d '
          'train requests succeeded (required: %d)', len(rewards), len(results),
          min_num_rewards)
      return

    self._update_model(initial_perturbations, rewards)
    self._log_rewards(rewards)
    self._log_tf_summary(rewards)

    if self._config.save_best_policy and np.max(
        rewards) > self._global_max_reward:
      self._global_max = np.max(rewards)
      logging.info('Found new best model with reward %f at step '
                   '%d, saving.', self._global_max, self._step)
      max_index = np.argmax(rewards)
      perturbation = initial_perturbations[max_index]
      self._policy_saver_fn(
          parameters=self._model_weights + perturbation,
          policy_name=f'best_policy_{self._global_max}_step_{self._step}',
      )

    self._save_model()

    self._step += 1
