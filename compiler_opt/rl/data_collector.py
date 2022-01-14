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
import time

from typing import Iterator, Tuple, Dict
from tf_agents.trajectories import trajectory

# Deadline for data collection.
DEADLINE_IN_SECONDS = 30

# We don't wait for all data collection to finish --- it continues if either of
# the wait_termination_conditions is met.
# (0.8, 0.5) means it won't wait for more data collection to finish if 80% of
# the data collection have finished and it has waited 50% of
# _DEADLINE_IN_SECONDS time.
WAIT_TERMINATION = ((0.9, 0), (0.8, 0.5), (0, 1))


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
      information is viewable in TensorBoard.
    """

  @abc.abstractmethod
  def on_dataset_consumed(self,
                          dataset_iterator: Iterator[trajectory.Trajectory]):
    """Clean up after the data has been consumed.

    Args:
      dataset_iterator: the dataset_iterator that has been consumed.
    """


class EarlyExitChecker:
  """Class which checks if it is ok to early-exit from data collection."""

  def __init__(self,
               num_modules: int,
               deadline: float = DEADLINE_IN_SECONDS,
               thresholds: Tuple[Tuple[float, float], ...] = WAIT_TERMINATION):
    """Initializes the early exit checker.

    Args:
      num_modules: How many total modules we are waiting for.
      deadline: The deadline for data collection, in seconds.
      thresholds: Early exit thresholds, e.g. [(0.8, 0.5)] means early exit is
        allowable if 80% of data has been collected and we've waited 50% of the
        maximum waiting time.
    """
    self._num_modules = num_modules
    self._deadline = deadline
    self._thresholds = thresholds
    self._start_time = time.time()
    self._waited_time = 0

  def _should_exit(self, collected: int):
    """Checks whether we should exit early.

    If collected is negative, _should_exit will always return false.

    Args:
      collected: The amount data we have collected.

    Returns:
      True if we should exit, otherwise False.
    """
    if collected < 0:
      return False

    self._waited_time = round(time.time() - self._start_time)
    for (data_threshold, deadline_threshold) in self._thresholds:
      if ((collected >= self._num_modules * data_threshold) and
          (self._waited_time >= self._deadline * deadline_threshold)):
        return True
    return False

  def wait(self, get_num_finished_work):
    """Waits until the deadline has expired or an early exit is possible.

    Args:
      get_num_finished_work: a callable object which returns the amount of
      finished work.

    Returns:
      The amount of time waited.
    """
    while not self._should_exit(get_num_finished_work()):
      time.sleep(1)
    return self.waited_time()

  def waited_time(self):
    return self._waited_time
