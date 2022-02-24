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

import dataclasses
import json
import subprocess
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from compiler_opt.rl import constant


@dataclasses.dataclass
class RewardStat:
  default_reward: float
  moving_average_reward: float


class DataClassJSONEncoder(json.JSONEncoder):

  def default(self, o):
    if dataclasses.is_dataclass(o):
      return dataclasses.asdict(o)
    return super().default(o)


def _postprocessing_sequence_example(
    sequence_example: tf.train.SequenceExample, moving_average_reward: float,
    policy_reward: float) -> tf.train.SequenceExample:
  """Post-processing of the trace (sequence_example).

  It computes the reward ratio change of the TF policy compared with the
  moving average reward, and uses this ratio as the whole trajectory reward to
  overwrite the original reward after each action.

  Args:
    sequence_example: A tf.SequenceExample proto describing compilation trace.
    moving_average_reward: The moving average reward.
    policy_reward: The reward under the TF policy.

  Returns:
    The tf.SequenceExample proto after post-processing.
  """
  reward = 1 - policy_reward / moving_average_reward

  sequence_length = len(
      next(iter(sequence_example.feature_lists.feature_list.values())).feature)

  reward_list = sequence_example.feature_lists.feature_list['reward']
  for _ in range(sequence_length):
    added_feature = reward_list.feature.add()
    added_feature.float_list.value.append(reward)

  return sequence_example


def get_command_line_for_bundle(cmd_file: str,
                                ir_file: str,
                                thinlto: Optional[str] = None) -> List[str]:
  with open(cmd_file) as f:
    return f.read().split('\0') + ['-x', 'ir'] + [ir_file] + (
        ['-fthinlto-index=' + thinlto] if thinlto else [])


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
      reward_stat: Optional[Dict[str, RewardStat]]
  ) -> Tuple[List[str], Dict[str, RewardStat], List[float]]:
    """Collect data for the given IR file and policy.

    Args:
      file_paths: path to files needed for inlining, Tuple of (.bc, .cmd).
      tf_policy_path: path to the tensorflow policy.
      reward_stat: reward stat of this module, None if unknown.

    Returns:
      A tuple containing:
        sequence_example: A list of serialized tf.SequenceExample proto.
        reward_stat: Updated reward stat of this module.
        rewards: rewards under the current ml policy.

    Raises:
      subprocess.CalledProcessError if process fails.
      ValueError if example under default policy and ml policy does not match.
    """
    try:
      if reward_stat is None:
        default_result = self._compile_fn(
            file_paths, tf_policy_path='', reward_only=bool(tf_policy_path))
        reward_stat = {
            k: RewardStat(v[1], v[1]) for (k, v) in default_result.items()
        }

      if tf_policy_path:
        policy_result = self._compile_fn(
            file_paths, tf_policy_path, reward_only=False)
      else:
        policy_result = default_result

    except subprocess.CalledProcessError as e:
      raise e

    sequence_example_list = []
    rewards = []
    for k, v in policy_result.items():
      sequence_example = v[0]
      policy_reward = v[1]
      if k not in reward_stat:
        raise ValueError(
            'Example %s does not exist under default policy for module %s' %
            (k, file_paths[0]))
      default_reward = reward_stat[k].default_reward
      moving_average_reward = reward_stat[k].moving_average_reward
      sequence_example = _postprocessing_sequence_example(
          sequence_example, moving_average_reward, policy_reward)
      sequence_example_list.append(sequence_example.SerializeToString())
      reward_stat[k].moving_average_reward = (
          moving_average_reward * self._moving_average_decay_rate +
          policy_reward * (1 - self._moving_average_decay_rate))
      rewards.append(1 - (policy_reward + constant.DELTA) /
                     (default_reward + constant.DELTA))

    return (sequence_example_list, reward_stat, rewards)

  def _compile_fn(
      self, file_paths: Tuple[str, ...], tf_policy_path: str,
      reward_only: bool) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:
    """Compiles for the given IR file under the given policy.

    Args:
      file_paths: path to files needed for compilation.
      tf_policy_path: path to TF policy directory on local disk.
      reward_only: whether only return reward.

    Returns:
      A dict mapping from example identifier to tuple containing:
        sequence_example: A tf.SequenceExample proto describing compilation
        trace, None if reward_only == True.
        reward: reward under the policy.

    Raises:
      subprocess.CalledProcessError if process fails.
    """
    raise NotImplementedError('Not implemented compile fn.')
