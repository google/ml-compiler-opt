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

"""Module for running compilation and collect training data."""

import subprocess
from typing import Tuple, Optional

import tensorflow as tf


def _postprocessing_sequence_example(
    sequence_example: tf.train.SequenceExample, default_reward: float,
    policy_reward: float) -> tf.train.SequenceExample:
  """Post-processing of the trace (sequence_example).

  It computes the reward ratio change of the TF policy compared with the
  default policy, and uses this ratio as the whole trajectory reward to
  overwrite the original reward after each action.

  Args:
    sequence_example: A tf.SequenceExample proto describing compilation trace.
    default_reward: The reward under default policy.
    policy_reward: The reward under the TF policy.

  Returns:
    The tf.SequenceExample proto after post-processing.
  """
  reward = 1 - policy_reward / default_reward

  sequence_length = len(
      next(iter(sequence_example.feature_lists.feature_list.values())).feature)

  reward_list = sequence_example.feature_lists.feature_list['reward']
  for _ in range(sequence_length):
    added_feature = reward_list.feature.add()
    added_feature.float_list.value.append(reward)

  return sequence_example


class CompilationRunner:
  """Base class for collecting compilation data."""

  def __init__(self,
               clang_path: str,
               llvm_size_path: str,
               launcher_path: Optional[str] = None,
               moving_average_decay_rate: float = 1):
    """Initialization of CompilationRunner class.

    Args:
      clang_path: path to the clang binary.
      llvm_size_path: path to the llvm-size binary.
      launcher_path: path to the launcher binary.
      moving_average_decay_rate: moving average decay rate during training.
    """
    self._clang_path = clang_path
    self._llvm_size_path = llvm_size_path
    self._launcher_path = launcher_path
    self._moving_average_decay_rate = moving_average_decay_rate

  def collect_data(
      self, file_paths: Tuple[str, ...], tf_policy_path: str,
      default_reward: Optional[float], moving_average_reward: Optional[float]
  ) -> Tuple[str, float, float, float]:
    """Collect data for the given IR file and policy.

    Args:
      file_paths: path to files needed for inlining, Tuple of (.bc, .cmd).
      tf_policy_path: path to the tensorflow policy.
      default_reward: reward under default policy, None if unknown.
      moving_average_reward: moving average reward during training, None if
                             unknown.

    Returns:
      A tuple containing:
        sequence_example: A serialized tf.SequenceExample proto.
        default_reward: reward under default policy.
        moving_average_reward: updated moving average reward.
        policy_reward: reward under the ml policy.

    Raises:
      subprocess.CalledProcessError if process fails.
    """
    try:
      if default_reward is None:
        default_sequence_example, default_reward = self._compile_fn(
            file_paths, tf_policy_path='', reward_only=bool(tf_policy_path))
        moving_average_reward = default_reward

      # Return empty example if the default policy size is 0 since it is a data
      # only module and we can do nothing about it.
      if default_reward == 0:
        return '', 0, 0, 0

      if tf_policy_path:
        sequence_example, policy_reward = self._compile_fn(
            file_paths, tf_policy_path, reward_only=False)
      else:
        (sequence_example, policy_reward) = (default_sequence_example,
                                             default_reward)

    except subprocess.CalledProcessError as e:
      raise e

    if not sequence_example.HasField('feature_lists'):
      return '', 0, 0, 0

    sequence_example = _postprocessing_sequence_example(sequence_example,
                                                        moving_average_reward,
                                                        policy_reward)
    moving_average_reward = (
        moving_average_reward * self._moving_average_decay_rate +
        policy_reward * (1 - self._moving_average_decay_rate))

    return (sequence_example.SerializeToString(), default_reward,
            moving_average_reward, policy_reward)

  def _compile_fn(self, file_paths: Tuple[str, ...], tf_policy_path: str,
                  reward_only: bool) -> Tuple[tf.train.SequenceExample, float]:
    """Compiles for the given IR file under the given policy.

    Args:
      file_paths: path to files needed for compilation.
      tf_policy_path: path to TF policy directory on local disk.
      reward_only: whether only return reward.

    Returns:
      A tuple containing:
        sequence_example: A tf.SequenceExample proto describing compilation
        trace.
        reward: reward under the policy.

    Raises:
      subprocess.CalledProcessError if process fails.
    """
    raise NotImplementedError('Not implemented compile fn.')
