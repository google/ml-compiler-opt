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
"""Module for storing and processing best trajectories."""

import dataclasses
import json
from typing import Dict, List

import tensorflow as tf

from compiler_opt.rl import constant


@dataclasses.dataclass(frozen=True)
class BestTrajectory:
  reward: float
  action_list: List[int]


class BestTrajectoryRepo:
  """Class for storing and processing best trajectory related operations."""

  def __init__(self, action_name: str):
    """Constructor.

    Args:
      action_name: action name of the trajectory, used for extracting action
        list from tensorflow.SequenceExample.
    """
    # {module_name: {identifier: best trajectory}}
    self._best_trajectories: Dict[str, Dict[str, BestTrajectory]] = {}
    self._action_name: str = action_name

  @property
  def best_trajectories(self) -> Dict[str, Dict[str, BestTrajectory]]:
    return self._best_trajectories.copy()

  def sink_to_json_file(self, path: str):
    with tf.io.gfile.GFile(path, 'w') as f:
      json.dump(self._best_trajectories, f, cls=constant.DataClassJSONEncoder)

  def load_from_json_file(self, path: str):
    with tf.io.gfile.GFile(path, 'r') as f:
      data = json.load(f)
    for k, v in data.items():
      if v:
        self._best_trajectories[k] = {
            sub_k: BestTrajectory(**sub_v) for sub_k, sub_v in v.items()
        }

  def sink_to_csv_file(self, path: str):
    """sink to csv file format consumable by compiler."""
    with tf.io.gfile.GFile(path, 'w') as f:
      for k, v in self._best_trajectories.items():
        for sub_k, sub_v in v.items():
          f.write(','.join([k, sub_k] + [str(x) for x in sub_v.action_list]) +
                  '\n')

  def combine_with_other_repo(self, other: 'BestTrajectoryRepo'):
    """combine and update with other best trajectory repo."""
    for k, v in other.best_trajectories.items():
      if k not in self._best_trajectories:
        self._best_trajectories[k] = v
        continue
      for sub_k, sub_v in v.items():
        if sub_v.reward < self._best_trajectories[k][sub_k].reward:
          self._best_trajectories[k][sub_k] = sub_v

  def update_if_better_trajectory(self, module_name: str, identifier: str,
                                  reward: float, trajectory: bytes):
    """update with incoming trajectory if the reward is lower.

    Args:
      module_name: module name of the trajectory.
      identifier: identifier of the trajectory within module.
      reward: reward of the trajectory.
      trajectory: trajectory in the format of serialized SequenceExample.
    """
    if module_name not in self._best_trajectories:
      self._best_trajectories[module_name] = {}
    if (identifier not in self._best_trajectories[module_name] or
        self._best_trajectories[module_name][identifier].reward > reward):
      example = tf.train.SequenceExample.FromString(trajectory)
      action_list = [
          x.int64_list.value[0]
          for x in example.feature_lists.feature_list[self._action_name].feature
      ]
      self._best_trajectories[module_name][identifier] = BestTrajectory(
          reward=reward, action_list=action_list)
