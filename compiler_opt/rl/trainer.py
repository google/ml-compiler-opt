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

"""LLVM Policy Trainer."""

import time

from absl import logging

import gin
import tensorflow as tf

from tf_agents.environments import trajectory_replay
from tf_agents.utils import common as common_utils

_INLINING_DEFAULT_KEY = 'inlining_default'


@gin.configurable
class Trainer(object):
  """Object that trains LLVM policy.

  After initialization, the function 'train' can be called multiple times to
  train on different datasets. An example usage:

  ```python
  trainer = Trainer(root_dir, agent)
  trainer.train(data_iter_1, num_iterations_1)
  trainer.train(data_iter_2, num_iterations_2)
  ```
  """

  def __init__(
      self,
      root_dir,
      agent,
      # Params for summaries and logging
      checkpoint_interval=10000,
      log_interval=100,
      summary_interval=1000,
      summaries_flush_secs=10):
    """Initialize the Trainer object.

    Args:
      root_dir: str, the root directory to host all required sub-directories.
      agent: a tf_agents.agents.TFAgent object.
      checkpoint_interval: int, the training step interval for saving
        checkpoint.
      log_interval: int, the training step interval for logging.
      summary_interval: int, the training step interval for exporting to
        tensorboard.
      summaries_flush_secs: int, the seconds for flushing to tensorboard.
    """
    self._root_dir = root_dir
    self._agent = agent
    self._checkpoint_interval = checkpoint_interval
    self._log_interval = log_interval
    self._summary_interval = summary_interval

    self._summary_writer = tf.summary.create_file_writer(
        self._root_dir, flush_millis=summaries_flush_secs * 1000)
    self._summary_writer.set_as_default()

    self._global_step = tf.compat.v1.train.get_or_create_global_step()

    # Initialize agent and trajectory replay.
    # Wrap training and trajectory replay in a tf.function to make it much
    # faster.
    self._agent.initialize()
    self._trajectory_replay = trajectory_replay.TrajectoryReplay(
        policy=self._agent.collect_policy)
    self._agent.train = common_utils.function(self._agent.train)
    self._trajectory_replay.run = common_utils.function(
        self._trajectory_replay.run)

    self._initialize_metrics()

    self._checkpointer = common_utils.Checkpointer(
        ckpt_dir=self._root_dir,
        agent=self._agent,
        global_step=self._global_step)
    self._checkpointer.initialize_or_restore()

    self._start_time = time.time()
    self._last_checkpoint_step = 0
    self._last_log_step = 0

  def _initialize_metrics(self):
    """Initializes metrics."""
    # Measures whether tf_agent.policy makes the same decisions as actions
    # recorded in training data.
    self._train_adherence = tf.keras.metrics.Accuracy()
    self._train_adherence_0 = tf.keras.metrics.Accuracy()
    self._train_adherence_1 = tf.keras.metrics.Accuracy()

    # How often the policy makes the decision to inline.
    self._data_action_mean = tf.keras.metrics.Mean()
    self._policy_action_mean = tf.keras.metrics.Mean()

    # Average reward of each step in training data.
    # An improved policy will have a higher mean.
    self._data_reward_mean_ignoring_early_termination = tf.keras.metrics.Mean()
    self._total_early_termination = tf.keras.metrics.Sum()
    self._total_modules = tf.keras.metrics.Sum()

  def _update_metrics(self, experience, replay_action):
    """Updates metrics and exports to Tensorboard."""
    is_action = ~experience.is_boundary()

    self._train_adherence.update_state(
        experience.observation[_INLINING_DEFAULT_KEY],
        replay_action,
        sample_weight=is_action)
    self._train_adherence_0.update_state(
        experience.observation[_INLINING_DEFAULT_KEY],
        replay_action,
        sample_weight=tf.logical_and(
            is_action, experience.observation[_INLINING_DEFAULT_KEY] == 0))
    self._train_adherence_1.update_state(
        experience.observation[_INLINING_DEFAULT_KEY],
        replay_action,
        sample_weight=tf.logical_and(
            is_action, experience.observation[_INLINING_DEFAULT_KEY] == 1))

    self._data_action_mean.update_state(
        experience.action, sample_weight=is_action)
    self._policy_action_mean.update_state(
        replay_action, sample_weight=is_action)

    self._data_reward_mean_ignoring_early_termination.update_state(
        experience.reward,
        sample_weight=tf.logical_and(is_action, experience.reward >= -9999))
    self._total_early_termination.update_state(
        experience.reward < -9999, sample_weight=is_action)
    self._total_modules.update_state(experience.is_first())

    with tf.name_scope('Monitor/'):
      tf.summary.scalar(
          name='train_adherence',
          data=self._train_adherence.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='train_adherence_to_default_not_inline',
          data=self._train_adherence_0.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='train_adherence_to_default_inline',
          data=self._train_adherence_1.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='data_action_mean',
          data=self._data_action_mean.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='policy_action_mean',
          data=self._policy_action_mean.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='data_reward_mean_ignoring_early_termination',
          data=self._data_reward_mean_ignoring_early_termination.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='total_early_termination',
          data=self._total_early_termination.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='total_modules',
          data=self._total_modules.result(),
          step=self._global_step)

    tf.summary.histogram(
        name='reward', data=experience.reward, step=self._global_step)

  def _reset_metrics(self):
    """Reset all metrics."""
    self._train_adherence.reset_states()
    self._train_adherence_0.reset_states()
    self._train_adherence_1.reset_states()
    self._data_action_mean.reset_states()
    self._policy_action_mean.reset_states()
    self._data_reward_mean_ignoring_early_termination.reset_states()
    self._total_early_termination.reset_states()
    self._total_modules.reset_states()

  def _log_experiment(self, loss):
    """Log training info."""
    global_step_val = self._global_step.numpy()
    if global_step_val - self._last_log_step >= self._log_interval:
      logging.info('step = %d, loss = %g, train_adherence = %g',
                   global_step_val, loss, self._train_adherence.result())
      time_acc = time.time() - self._start_time
      steps_per_sec = (global_step_val - self._last_log_step) / time_acc
      logging.info('%.3f steps/sec', steps_per_sec)
      self._last_log_step = global_step_val
      self._start_time = time.time()

  def _save_checkpoint(self):
    if (self._global_step.numpy() - self._last_checkpoint_step >=
        self._checkpoint_interval):
      self._checkpointer.save(global_step=self._global_step)
      self._last_checkpoint_step = self._global_step.numpy()

  def train(self, dataset_iter, num_iterations):
    """Trains policy with data from dataset_iter for num_iterations steps."""
    self._reset_metrics()
    with tf.summary.record_if(
        lambda: tf.math.equal(self._global_step % self._summary_interval, 0)):
      for _ in range(num_iterations):
        experience = next(dataset_iter)
        replay_action, _, _ = self._trajectory_replay.run(experience)
        loss = self._agent.train(experience)

        self._update_metrics(experience, replay_action)
        self._log_experiment(loss.loss)
        self._save_checkpoint()
