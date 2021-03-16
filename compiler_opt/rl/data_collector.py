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

"""Data collection module."""

import abc
from typing import Iterator, Tuple, Dict
from tf_agents.trajectories import trajectory


class DataCollector(metaclass=abc.ABCMeta):
  """Abstract class for data collection."""

  @abc.abstractmethod
  def collect_data(
      self, policy_path: str
  ) -> Tuple[Iterator[trajectory.Trajectory], Dict[str, float]]:
    """Collect data for a given policy.

    Args:
      policy_path: the path to the policy directory to collect data with.

    Returns:
      An iterator of batched trajectory.Trajectory that are ready to be fed to
        training.
      A dict of extra monitoring information, e.g., how many modules succeeded.
      They will be reported using `tf.scalar.summary` by the trainer so these
      information is viewable in Tensorboard.
    """

  @abc.abstractmethod
  def on_dataset_consumed(self,
                          dataset_iterator: Iterator[trajectory.Trajectory]):
    """Clean up after the data has been consumed.

    Args:
      dataset_iterator: the dataset_iterator that has been consumed.
    """
