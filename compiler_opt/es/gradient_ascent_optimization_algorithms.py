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

###############################################################################
#
#
# This is a port of the code by Krzysztof Choromanski, Deepali Jain and Vikas
# Sindhwani, based on the portfolio of Blackbox optimization algorithms listed
# below:
#
# "On Blackbox Backpropagation and Jacobian Sensing"; K. Choromanski,
#  V. Sindhwani, NeurIPS 2017
# "Optimizing Simulations with Noise-Tolerant Structured Exploration"; K.
#  Choromanski, A. Iscen, V. Sindhwani, J. Tan, E. Coumans, ICRA 2018
# "Structured Evolution with Compact Architectures for Scalable Policy
#  Optimization"; K. Choromanski, M. Rowland, V. Sindhwani, R. Turner, A.
#  Weller, ICML 2018, https://arxiv.org/abs/1804.02395
#  "From Complexity to Simplicity: Adaptive ES-Active Subspaces for Blackbox
#   Optimization";  K. Choromanski, A. Pacchiano, J. Parker-Holder, Y. Tang, V.
#   Sindhwani, NeurIPS 2019
# "i-Sim2Real: Reinforcement Learning on Robotic Policies in Tight Human-Robot
#  Interaction Loops"; L. Graesser, D. D'Ambrosio, A. Singh, A. Bewley, D. Jain,
#  K. Choromanski, P. Sanketi , CoRL 2022, https://arxiv.org/abs/2207.06572
# "Agile Catching with Whole-Body MPC and Blackbox Policy Learning"; S.
#  Abeyruwan, A. Bewley, N. Boffi, K. Choromanski, D. D'Ambrosio, D. Jain, P.
#  Sanketi, A. Shankar, V. Sindhwani, S. Singh, J. Slotine, S. Tu, L4DC,
#  https://arxiv.org/abs/2306.08205
# "Robotic Table Tennis: A Case Study into a High Speed Learning System"; A.
#  Bewley, A. Shankar, A. Iscen, A. Singh, C. Lynch, D. D'Ambrosio, D. Jain,
#  E. Coumans, G. Versom, G. Kouretas, J. Abelian, J. Boyd, K. Oslund,
#  K. Reymann, K. Choromanski, L. Graesser, M. Ahn, N. Jaitly, N. Lazic,
#  P. Sanketi, P. Xu, P. Sermanet, R. Mahjourian, S. Abeyruwan, S. Kataoka,
#  S. Moore, T. Nguyen, T. Ding, V. Sindhwani, V. Vanhoucke, W. Gao, Y. Kuang,
#  to be presented at RSS 2023
###############################################################################
r"""Library of gradient ascent algorithms.

Library of stateful gradient ascent algorithms taking as input the gradient and
current parameters, and output the new parameters.
"""

import abc

import numpy as np
import numpy.typing as npt
from typing import List, Optional


class GradientAscentOptimizer(metaclass=abc.ABCMeta):
  """Abstract class for general gradient ascent optimizers.

  Class is responsible for encoding different gradient ascent optimization
  techniques.
  """

  @abc.abstractmethod
  def run_step(self, current_input: npt.NDArray[np.float32],
               gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Conducts a single step of gradient ascent optimization.

    Conduct a single step of gradient ascent optimization procedure, given the
    current parameters and the raw gradient.

    Args:
      current_input: the current parameters.
      gradient: the raw gradient.

    Returns:
      New parameters by conducting a single step of gradient ascent.
    """
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def get_state(self) -> List[float]:
    """Returns the state of the optimizer.

    Returns the state of the optimizer.

    Args:

    Returns:
      The state of the optimizer.
    """
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def set_state(self, state: npt.NDArray[np.float32]) -> None:
    """Sets up the internal state of the optimizer.

    Sets up the internal state of the optimizer.

    Args:
      state: state to be set up

    Returns:
    """
    raise NotImplementedError("Abstract method")


class MomentumOptimizer(GradientAscentOptimizer):
  """Class implementing momentum gradient ascent optimizer.

  Setting momentum coefficient to zero is equivalent to vanilla gradient
  ascent.

  the state is the moving average as a list
  """

  def __init__(self, step_size: float, momentum: float):
    self.step_size = step_size
    self.momentum = momentum

    self.moving_average = np.asarray([], dtype=np.float32)
    super().__init__()

  def run_step(self, current_input: npt.NDArray[np.float32],
               gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if self.moving_average.size == 0:
      # Initialize the moving average
      self.moving_average = np.zeros(len(current_input), dtype=np.float32)
    elif len(self.moving_average) != len(current_input):
      raise ValueError(
          "Dimensions of the parameters and moving average do not match")

    self.moving_average = self.momentum * self.moving_average + (
        1 - self.momentum) * gradient
    step = self.step_size * self.moving_average

    return current_input + step

  def get_state(self) -> List[float]:
    return self.moving_average.tolist()

  def set_state(self, state: npt.NDArray[np.float32]) -> None:
    self.moving_average = np.asarray(state, dtype=np.float32)


class AdamOptimizer(GradientAscentOptimizer):
  """Class implementing ADAM gradient ascent optimizer.

  The state is the first moment moving average, the second
  moment moving average, and t (current step number)
  combined in that order into one list
  """

  def __init__(self,
               step_size: float,
               beta1: Optional[float] = 0.9,
               beta2: Optional[float] = 0.999,
               epsilon: Optional[float] = 1e-07):
    self.step_size = step_size
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

    self.first_moment_moving_average = np.asarray([], dtype=np.float32)
    self.second_moment_moving_average = np.asarray([], dtype=np.float32)
    self.t = 0
    super().__init__()

  def run_step(self, current_input: npt.NDArray[np.float32],
               gradient: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if self.first_moment_moving_average.size == 0:
      # Initialize the moving averages
      self.first_moment_moving_average = np.zeros(
          len(current_input), dtype=np.float32)
      self.second_moment_moving_average = np.zeros(
          len(current_input), dtype=np.float32)
      # Initialize the step counter
      self.t = 0
    elif len(self.first_moment_moving_average) != len(current_input):
      raise ValueError(
          "Dimensions of the parameters and moving averages do not match")

    self.first_moment_moving_average = (
        self.beta1 * self.first_moment_moving_average +
        (1 - self.beta1) * gradient)
    self.second_moment_moving_average = (
        self.beta2 * self.second_moment_moving_average + (1 - self.beta2) *
        (gradient * gradient))

    self.t += 1
    scale = np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

    step = self.step_size * scale * self.first_moment_moving_average / (
        np.sqrt(self.second_moment_moving_average) + self.epsilon)

    return current_input + step

  def get_state(self) -> List[float]:
    return (self.first_moment_moving_average.tolist() +
            self.second_moment_moving_average.tolist() + [self.t])

  def set_state(self, state: npt.NDArray[np.float32]) -> None:
    total_len = len(state)
    if total_len % 2 != 1:
      raise ValueError("The dimension of the state should be odd")
    dim = total_len // 2

    self.first_moment_moving_average = np.asarray(state[:dim], dtype=np.float32)
    self.second_moment_moving_average = np.asarray(
        state[dim:2 * dim], dtype=np.float32)
    self.t = int(state[-1])
    if self.t < 0:
      raise ValueError("The step counter should be non-negative")
