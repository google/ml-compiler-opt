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
from compiler_opt.rl import random_net_distillation
from tf_agents.agents import tf_agent
from tf_agents.policies import policy_loader

from tf_agents.utils import common as common_utils
from typing import Optional


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
      root_dir: str,
      agent: tf_agent.TFAgent,
      random_network_distillation: Optional[
          random_net_distillation.RandomNetworkDistillation] = None,
      warmstart_policy_dir: Optional[str] = None,
      # Params for summaries and logging
      checkpoint_interval=10000,
      log_interval=100,
      summary_log_interval=100,
      summary_export_interval=1000,
      summaries_flush_secs=10):
    """Initialize the Trainer object.

    Args:
      root_dir: str, the root directory to host all required sub-directories.
      agent: a tf_agents.agents.TFAgent object.
      random_network_distillation: a random_net_distillation.RND object.
      warmstart_policy_dir: the directory to warmstart the policy if given.
      checkpoint_interval: int, the training step interval for saving
        checkpoint.
      log_interval: int, the training step interval for logging.
      summary_log_interval: the number of steps in between logging metrics
        to tensorboard.
      summary_export_interval: int, the training step interval for exporting
        to tensorboard.
      summaries_flush_secs: int, the seconds for flushing to tensorboard.
    """
    self._root_dir = root_dir
    self._agent = agent
    self._random_network_distillation = random_network_distillation
    self._checkpoint_interval = checkpoint_interval
    self._log_interval = log_interval
    self._summary_log_interval = summary_log_interval
    self._summary_export_interval = summary_export_interval

    self._summary_writer = tf.summary.create_file_writer(
        self._root_dir, flush_millis=summaries_flush_secs * 1000)
    self._summary_writer.set_as_default()

    self._global_step = tf.compat.v1.train.get_or_create_global_step()

    # Initialize agent and trajectory replay.
    # Wrap training and trajectory replay in a tf.function to make it much
    # faster.
    self._agent.initialize()
    self._agent.train = common_utils.function(self._agent.train)
    if self._random_network_distillation:
      self._random_network_distillation.train = common_utils.function(
          self._random_network_distillation.train)

    self._initialize_metrics()

    # Load warmstart policy before restoring from checkpoint.
    if warmstart_policy_dir:
      warmstart_policy = policy_loader.load(warmstart_policy_dir)
      self._agent.policy.update(
          policy=warmstart_policy,
          tau=1.0,
          tau_non_trainable=None,
          sort_variables_by_name=False)

    self._checkpointer = common_utils.Checkpointer(
        ckpt_dir=self._root_dir,
        agent=self._agent,
        global_step=self._global_step)
    self._checkpointer.initialize_or_restore()

    self._start_time = time.time()

  def _initialize_metrics(self):
    """Initializes metrics."""
    self._data_action_mean = tf.keras.metrics.Mean()
    self._data_reward_mean = tf.keras.metrics.Mean()
    self._num_trajectories = tf.keras.metrics.Sum()

  def _update_metrics(self, experience, monitor_dict):
    """Updates metrics and exports to Tensorboard."""
    if tf.math.equal(self._global_step % self._summary_log_interval, 0):
      is_action = ~experience.is_boundary()

      self._data_action_mean.update_state(
          experience.action, sample_weight=is_action)
      self._data_reward_mean.update_state(
          experience.reward, sample_weight=is_action)
      self._num_trajectories.update_state(experience.is_first())

    # Check earlier rather than later if we should record summaries.
    # TF also checks it, but much later. Needed to avoid looping through
    # the dict so gave the if a bigger scope
    if tf.summary.should_record_summaries():
      with tf.name_scope('default/'):
        tf.summary.scalar(
            name='data_action_mean',
            data=self._data_action_mean.result(),
            step=self._global_step)
        tf.summary.scalar(
            name='data_reward_mean',
            data=self._data_reward_mean.result(),
            step=self._global_step)
        tf.summary.scalar(
            name='num_trajectories',
            data=self._num_trajectories.result(),
            step=self._global_step)

      for name_scope, d in monitor_dict.items():
        with tf.name_scope(name_scope + '/'):
          for key, value in d.items():
            tf.summary.scalar(name=key, data=value, step=self._global_step)

      tf.summary.histogram(
          name='reward', data=experience.reward, step=self._global_step)

  def _reset_metrics(self):
    """Reset num_trajectories."""
    self._num_trajectories.reset_states()

  def _log_experiment(self, loss):
    """Log training info."""
    if tf.math.equal(self._global_step % self._log_interval, 0):
      global_step_val = self._global_step.numpy()
      logging.info('step = %d, loss = %g', global_step_val, loss)
      time_acc = time.time() - self._start_time
      steps_per_sec = self._log_interval / time_acc
      logging.info('%.3f steps/sec', steps_per_sec)
      self._start_time = time.time()

  def _save_checkpoint(self):
    if tf.math.equal(self._global_step % self._checkpoint_interval, 0):
      self._checkpointer.save(global_step=self._global_step)

  def global_step_numpy(self):
    return self._global_step.numpy()

  def train(self, dataset_iter, monitor_dict, num_iterations: int):
    """Trains policy with data from dataset_iter for num_iterations steps."""
    self._reset_metrics()
    # context management is implemented in decorator
    # pylint: disable=not-context-manager
    with tf.summary.record_if(lambda: tf.math.equal(
        self._global_step % self._summary_export_interval, 0)):
      for _ in range(num_iterations):
        # When the data is not enough to fill in a batch, next(dataset_iter)
        # will throw StopIteration exception, logging a warning message instead
        # of killing the training when it happens.
        try:
          experience = next(dataset_iter)
        except StopIteration:
          logging.warning(
              ('Warning: skip training because do not have enough data to fill '
               'in a batch, consider increase data or reduce batch size.'))
          break

        # random network distillation for intrinsic reward generation
        if self._random_network_distillation:
          experience = self._random_network_distillation.train(experience)

        loss = self._agent.train(experience)

        self._update_metrics(experience, monitor_dict)
        self._log_experiment(loss.loss)
        self._save_checkpoint()
