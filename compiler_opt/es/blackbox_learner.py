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
"""Class for coordinating blackbox optimization."""

import os
from absl import logging
import concurrent.futures
import dataclasses
import gin
import math
import numpy as np
import numpy.typing as npt
import tempfile
import tensorflow as tf
from typing import List, Optional, Protocol

from compiler_opt.distributed import buffered_scheduler
from compiler_opt.distributed.worker import FixedWorkerPool
from compiler_opt.es import blackbox_optimizers
from compiler_opt.es import policy_utils
from compiler_opt.rl import corpus
from compiler_opt.rl import policy_saver

# If less than 40% of requests succeed, skip the step.
_SKIP_STEP_SUCCESS_RATIO = 0.4


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
  est_type: blackbox_optimizers.EstimatorType

  # Should the rewards for blackbox optimization in a single step be normalized?
  fvalues_normalization: bool

  # How to update optimizer hyperparameters
  hyperparameters_update_method: blackbox_optimizers.UpdateMethod

  # Number of top performing perturbations to select in the optimizer
  # 0 means all
  num_top_directions: int

  # How many IR files to try a single perturbation on?
  num_ir_repeats_within_worker: int

  # How many times should we reuse IR to test different policies?
  num_ir_repeats_across_worker: int

  # How many IR files to sample from the test corpus at each iteration
  num_exact_evals: int

  # How many perturbations to attempt at each perturbation
  total_num_perturbations: int

  # How much to scale the stdev of the perturbations
  precision_parameter: float

  # Learning rate
  step_size: float


def _prune_skipped_perturbations(perturbations: List[npt.NDArray[np.float32]],
                                 rewards: List[Optional[float]]):
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
               sampler: corpus.Corpus,
               tf_policy_path: str,
               output_dir: str,
               policy_saver_fn: PolicySaverCallableType,
               model_weights: npt.NDArray[np.float32],
               config: BlackboxLearnerConfig,
               initial_step: int = 0,
               deadline: float = 30.0,
               seed: Optional[int] = None):
    """Construct a BlackboxLeaner.

    Args:
      blackbox_opt: the blackbox optimizer to use
      train_sampler: corpus_sampler for training data.
      tf_policy_path: where to write the tf policy
      output_dir: the directory to write all outputs
      policy_saver_fn: function to save a policy to cns
      model_weights: the weights of the current model
      config: configuration for blackbox optimization.
      stubs: grpc stubs to inlining/regalloc servers
      initial_step: the initial step for learning.
      deadline: the deadline in seconds for requests to the inlining server.
    """
    self._blackbox_opt = blackbox_opt
    self._sampler = sampler
    self._tf_policy_path = tf_policy_path
    self._output_dir = output_dir
    self._policy_saver_fn = policy_saver_fn
    self._model_weights = model_weights
    self._config = config
    self._step = initial_step
    self._deadline = deadline
    self._seed = seed

    # While we're waiting for the ES requests, we can
    # collect samples for the next round of training.
    self._samples = []

    self._summary_writer = tf.summary.create_file_writer(output_dir)

  def _get_perturbations(self) -> List[npt.NDArray[np.float32]]:
    """Get perturbations for the model weights."""
    perturbations = []
    rng = np.random.default_rng(seed=self._seed)
    for _ in range(self._config.total_num_perturbations):
      perturbations.append(
          rng.normal(size=(len(self._model_weights))) *
          self._config.precision_parameter)
    return perturbations

  def _get_rewards(
      self, results: List[concurrent.futures.Future]) -> List[Optional[float]]:
    """Convert ES results to reward numbers."""
    rewards = [None] * len(results)

    for i in range(len(results)):
      if not results[i].exception():
        rewards[i] = results[i].result()
      else:
        logging.info('Error retrieving result from future: %s',
                     str(results[i].exception()))

    return rewards

  def _update_model(self, perturbations: List[npt.NDArray[np.float32]],
                    rewards: List[float]) -> None:
    """Update the model given a list of perturbations and rewards."""
    self._model_weights = self._blackbox_opt.run_step(
        perturbations=np.array(perturbations),
        function_values=np.array(rewards),
        current_input=self._model_weights,
        current_value=np.mean(rewards))

  def _log_rewards(self, rewards: List[float]) -> None:
    """Log reward to console."""
    logging.info('Train reward: [%f]', np.mean(rewards))

  def _log_tf_summary(self, rewards: List[float]) -> None:
    """Log tensorboard data."""
    with self._summary_writer.as_default():
      tf.summary.scalar(
          'reward/average_reward_train', np.mean(rewards), step=self._step)

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

  def _get_results(
      self, pool: FixedWorkerPool,
      perturbations: List[bytes]) -> List[concurrent.futures.Future]:
    if not self._samples:
      for _ in range(self._config.total_num_perturbations):
        sample = self._sampler.sample(self._config.num_ir_repeats_within_worker)
        self._samples.append(sample)
        # add copy of sample for antithetic perturbation pair
        if self._config.est_type == (
            blackbox_optimizers.EstimatorType.ANTITHETIC):
          self._samples.append(sample)

    compile_args = zip(perturbations, self._samples)

    _, futures = buffered_scheduler.schedule_on_worker_pool(
        action=lambda w, v: w.compile(v[0], v[1]),
        jobs=compile_args,
        worker_pool=pool)

    not_done = futures
    # wait for all futures to finish
    while not_done:
      # update lists as work gets done
      _, not_done = concurrent.futures.wait(
          not_done, return_when=concurrent.futures.FIRST_COMPLETED)

    return futures

  def _get_policy_as_bytes(self,
                           perturbation: npt.NDArray[np.float32]) -> bytes:
    sm = tf.saved_model.load(self._tf_policy_path)
    # devectorize the perturbation
    policy_utils.set_vectorized_parameters_for_policy(sm, perturbation)

    with tempfile.TemporaryDirectory() as tmpdir:
      sm_dir = os.path.join(tmpdir, 'sm')
      tf.saved_model.save(sm, sm_dir, signatures=sm.signatures)
      src = os.path.join(self._tf_policy_path, policy_saver.OUTPUT_SIGNATURE)
      dst = os.path.join(sm_dir, policy_saver.OUTPUT_SIGNATURE)
      tf.io.gfile.copy(src, dst)

      # convert to tflite
      tfl_dir = os.path.join(tmpdir, 'tfl')
      policy_saver.convert_mlgo_model(sm_dir, tfl_dir)

      # create and return policy
      policy_obj = policy_saver.Policy.from_filesystem(tfl_dir)
      return policy_obj.policy

  def run_step(self, pool: FixedWorkerPool) -> None:
    """Run a single step of blackbox learning.
    This does not instantaneously return due to several I/O
    and executions running while this waits for the responses"""
    logging.info('-' * 80)
    logging.info('Step [%d]', self._step)

    initial_perturbations = self._get_perturbations()
    # positive-negative pairs
    if self._config.est_type == blackbox_optimizers.EstimatorType.ANTITHETIC:
      initial_perturbations = [
          p for p in initial_perturbations for p in (p, -p)
      ]

    # convert to bytes for compile job
    # TODO: current conversion is inefficient.
    # consider doing this on the worker side
    perturbations_as_bytes = []
    for perturbation in initial_perturbations:
      perturbations_as_bytes.append(self._get_policy_as_bytes(perturbation))

    results = self._get_results(pool, perturbations_as_bytes)
    rewards = self._get_rewards(results)

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

    self._save_model()

    self._step += 1
