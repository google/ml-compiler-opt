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
"""Module for running compilation and collect data for behavior cloning."""

from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step


class ExplorationWithPolicy:
  """Policy which selects states for exploration.

  Exploration is fascilitated in the following way. First the policy plays
  all actions from the replay_prefix. At the following state the policy computes
  a gap which is difference between the most likely action and the second most
  likely action according to the randomized exploration policy (distr).
  If the current gap is smaller than previously maintained gap, the gap is
  updated and the exploration state is set to the current state.
  The trajectory is completed by following following the policy from the
  constructor.

  Attributes:
    replay_prefix: a replay buffer of actions
    policy: policy to follow after exhausting the replay buffer
    explore_policy: randomized policy which is used to compute the gap
    curr_step: current step of the trajectory
    explore_step: current candidate for exploration step
    gap: current difference at explore step between probability of most likely
      action according to explore_policy and second most likely action
    explore_on_features: dict of feature names and functions which specify
      when to explore on the respective feature
  """

  def __init__(
      self,
      replay_prefix: List[np.ndarray],
      policy: Callable[[time_step.TimeStep], np.ndarray],
      explore_policy: Callable[[time_step.TimeStep], policy_step.PolicyStep],
      explore_on_features: Optional[Dict[str, Callable[[tf.Tensor],
                                                       bool]]] = None,
  ):
    self.replay_prefix = replay_prefix
    self.policy = policy
    self.explore_policy = explore_policy
    self.curr_step = 0
    self.explore_step = 0
    self.gap = np.inf
    self.explore_on_features = explore_on_features
    self._stop_exploration = False

  def advice(self, state: time_step.TimeStep) -> np.ndarray:
    """Action function for the policy.

    Args:
      state: current state in the trajectory

    Returns:
      policy_deca: action to take at the current state.

    """
    if self.curr_step < len(self.replay_prefix):
      self.curr_step += 1
      return np.array(self.replay_prefix[self.curr_step - 1])
    policy_deca = self.policy(state)
    distr = tf.nn.softmax(self.explore_policy(state).action.logits).numpy()[0]
    if not self._stop_exploration and distr.shape[0] > 1 and self.gap > np.abs(
        distr[0] - distr[1]):
      self.gap = np.abs(distr[0] - distr[1])
      self.explore_step = self.curr_step
    if not self._stop_exploration and self.explore_on_features is not None:
      for feature_name, explore_on_feature in self.explore_on_features.items():
        if explore_on_feature(state.observation[feature_name]):
          self.explore_step = self.curr_step
          self._stop_exploration = True
          break
    self.curr_step += 1
    return policy_deca
