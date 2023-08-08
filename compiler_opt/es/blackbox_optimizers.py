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
r"""Library of blackbox optimization algorithms.

Library of stateful blackbox optimization algorithms taking as input the values
of the blackbox function in the neighborhood of a given point and outputting new
point obtained after conducting one optimization step.
"""

import abc
import enum
import gin
import math

import numpy as np
import numpy.typing as npt
import scipy.optimize as sp_opt
from sklearn import linear_model
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from compiler_opt.es import gradient_ascent_optimization_algorithms

FloatArray = npt.NDArray[np.float32]

# should specifically be a 2d numpy array of floats
# but numpy.typing does not allow for that indication
FloatArray2D = Sequence[FloatArray]

SequenceOfFloats = Union[Sequence[float], FloatArray]

LinearModel = Union[linear_model.Ridge, linear_model.Lasso,
                    linear_model.LinearRegression]


class CurrentPointEstimate(enum.Enum):
  CURRENT = 1
  AVERAGE = 2


@gin.constants_from_enum(module='blackbox_optimizers')
class Algorithm(enum.Enum):
  MONTE_CARLO = 1
  TRUST_REGION = 2
  SKLEARN_REGRESSION = 3


@gin.constants_from_enum(module='blackbox_optimizers')
class EstimatorType(enum.Enum):
  FORWARD_FD = 1
  ANTITHETIC = 2


@gin.constants_from_enum(module='blackbox_optimizers')
class GradientType(enum.Enum):
  MC = 1
  REGRESSION = 2


@gin.constants_from_enum(module='blackbox_optimizers')
class RegressionType(enum.Enum):
  LASSO = 1
  RIDGE = 2
  LINEAR = 3


@gin.constants_from_enum(module='blackbox_optimizers')
class UpdateMethod(enum.Enum):
  STATE_NORMALIZATION = 1
  NO_METHOD = 2


DEFAULT_ARMIJO = 1e-4


def filter_top_directions(
    perturbations: FloatArray2D, function_values: FloatArray,
    est_type: EstimatorType,
    num_top_directions: int) -> Tuple[FloatArray, FloatArray]:
  """Select the subset of top-performing perturbations.

  Args:
    perturbations: np array of perturbations
                   For antithetic, it is assumed that the input puts the pair of
                   p, -p in the even/odd entries, so the directions p_1,...,p_n
                   will be ordered (p_1, -p_1, p_2, -p_2,...)
    function_values: np array of reward values (maximization)
    est_type: (forward_fd | antithetic)
    num_top_directions: the number of top directions to include
                        For antithetic, the total number of perturbations will
                        be 2* this number, because we count p, -p as a single
                        direction
  Returns:
    A pair (perturbations, function_values) consisting of the top perturbations.
    function_values[i] is the reward of perturbations[i]
    For antithetic, the perturbations will be reordered so that we have
    (p_1,...,p_n, -p_1,...,-p_n).
  """
  if not num_top_directions > 0:
    return (perturbations, function_values)
  if est_type == EstimatorType.FORWARD_FD:
    top_index = np.argsort(-function_values)
  elif est_type == EstimatorType.ANTITHETIC:
    top_index = np.argsort(-np.abs(function_values[0::2] -
                                   function_values[1::2]))
  top_index = top_index[:num_top_directions]
  if est_type == EstimatorType.FORWARD_FD:
    perturbations = perturbations[top_index]
    function_values = function_values[top_index]
  elif est_type == EstimatorType.ANTITHETIC:
    perturbations = np.concatenate(
        (perturbations[2 * top_index], perturbations[2 * top_index + 1]),
        axis=0)
    function_values = np.concatenate(
        (function_values[2 * top_index], function_values[2 * top_index + 1]),
        axis=0)
  return (perturbations, function_values)


class BlackboxOptimizer(metaclass=abc.ABCMeta):
  """Abstract class for general blackbox optimization.

  Class is responsible for encoding different blackbox optimization techniques.
  """

  @abc.abstractmethod
  def run_step(self, perturbations: FloatArray2D, function_values: FloatArray,
               current_input: FloatArray, current_value: float) -> FloatArray:
    """Conducts a single step of blackbox optimization procedure.

    Conducts a single step of blackbox optimization procedure, given values of
    the blackox function in the neighborhood of the current input.

    Args:
      perturbations: perturbation directions encoded as 1D numpy arrays
      function_values: corresponding function values
      current_input: current input to the blackbox function
      current_value: value of the blackbox function for the current input

    Returns:
      New input obtained by conducting a single step of the blackbox
      optimization procedure.
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def get_hyperparameters(self) -> List[float]:
    """Returns the list of hyperparameters for blackbox function runs.

    Returns the list of hyperparameters for blackbox function runs that can be
    updated on the fly.

    Args:

    Returns:
      The set of hyperparameters for blackbox function runs.
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def get_state(self) -> List[float]:
    """Returns the state of the optimizer.

    Returns the state of the optimizer.

    Args:

    Returns:
      The state of the optimizer.
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def update_state(self, evaluation_stats: SequenceOfFloats) -> None:
    """Updates the state for blackbox function runs.

    Updates the state of the optimizer for blackbox function runs.

    Args:
      evaluation_stats: stats from evaluation used to update hyperparameters

    Returns:
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def set_state(self, state: SequenceOfFloats) -> None:
    """Sets up the internal state of the optimizer.

    Sets up the internal state of the optimizer.

    Args:
      state: state to be set up

    Returns:
    """
    raise NotImplementedError('Abstract method')


class StatefulOptimizer(BlackboxOptimizer):
  """Intermediate parent class for stateful optimizers.

  Class contains common methods for handling the state.
  """

  def __init__(self, est_type: EstimatorType, normalize_fvalues: bool,
               hyperparameters_update_method: UpdateMethod,
               extra_params: Optional[Sequence[int]]):

    self.est_type = est_type
    self.normalize_fvalues = normalize_fvalues
    self.hyperparameters_update_method = hyperparameters_update_method
    if hyperparameters_update_method == UpdateMethod.STATE_NORMALIZATION:
      self.state_dim = extra_params[0]
      self.nb_steps = 0
      self.sum_state_vector = [0.0] * self.state_dim
      self.squares_state_vector = [0.0] * self.state_dim
      self.mean_state_vector = [0.0] * self.state_dim
      self.std_state_vector = [1.0] * self.state_dim
    super().__init__()

  def get_hyperparameters(self) -> List[float]:
    if self.hyperparameters_update_method == UpdateMethod.STATE_NORMALIZATION:
      return self.mean_state_vector + self.std_state_vector
    else:
      return []

  def get_state(self) -> List[float]:
    if self.hyperparameters_update_method == UpdateMethod.STATE_NORMALIZATION:
      current_state = [self.nb_steps]
      current_state += self.sum_state_vector
      current_state += self.squares_state_vector
      current_state += self.mean_state_vector + self.std_state_vector
      return current_state
    else:
      return []

  def update_state(self, evaluation_stats: SequenceOfFloats) -> None:
    if self.hyperparameters_update_method != UpdateMethod.STATE_NORMALIZATION:
      return
    self.nb_steps += evaluation_stats[0]
    evaluation_stats = evaluation_stats[1:]
    first_half = evaluation_stats[:self.state_dim]
    second_half = evaluation_stats[self.state_dim:]
    self.sum_state_vector = [
        sum(x) for x in zip(self.sum_state_vector, first_half)
    ]
    self.squares_state_vector = [
        sum(x) for x in zip(self.squares_state_vector, second_half)
    ]
    self.mean_state_vector = [
        x / float(self.nb_steps) for x in self.sum_state_vector
    ]
    mean_squares_state_vector = [
        x / float(self.nb_steps) for x in self.squares_state_vector
    ]

    self.std_state_vector = [
        math.sqrt(max(a - b * b, 0.0))
        for a, b in zip(mean_squares_state_vector, self.mean_state_vector)
    ]

  def set_state(self, state: SequenceOfFloats) -> None:
    if self.hyperparameters_update_method != UpdateMethod.STATE_NORMALIZATION:
      return
    self.nb_steps = state[0]
    state = state[1:]
    self.sum_state_vector = state[:self.state_dim]
    state = state[self.state_dim:]
    self.squares_state_vector = state[:self.state_dim]
    state = state[self.state_dim:]
    self.mean_state_vector = state[:self.state_dim]
    state = state[self.state_dim:]
    self.std_state_vector = state[:self.state_dim]


class MonteCarloBlackboxOptimizer(StatefulOptimizer):
  """Class implementing GD optimizer with MC estimation of the gradient."""

  def __init__(self,
               precision_parameter: float,
               est_type: EstimatorType,
               normalize_fvalues: bool,
               hyperparameters_update_method: UpdateMethod,
               extra_params: Optional[Sequence[int]],
               step_size: Optional[float] = None,
               num_top_directions: int = 0,
               gradient_ascent_optimizer: Optional[
                   gradient_ascent_optimization_algorithms
                   .GradientAscentOptimizer] = None):
    # Check step_size and gradient_ascent_optimizer
    if (step_size is None) == (gradient_ascent_optimizer is None):
      raise ValueError('Exactly one of step_size and \
            gradient_ascent_optimizer should be provided')
    if step_size:
      gradient_ascent_optimizer = (
          gradient_ascent_optimization_algorithms.MomentumOptimizer(
              step_size=step_size, momentum=0.0))

    self.precision_parameter = precision_parameter
    self.num_top_directions = num_top_directions
    self.gradient_ascent_optimizer = gradient_ascent_optimizer
    super().__init__(est_type, normalize_fvalues, hyperparameters_update_method,
                     extra_params)

  # TODO: Issue #285
  def run_step(self, perturbations: FloatArray2D, function_values: FloatArray,
               current_input: FloatArray, current_value: float) -> FloatArray:
    dim = len(current_input)
    if self.normalize_fvalues:
      values = function_values.tolist()
      values.append(current_value)
      mean = sum(values) / float(len(values))
      stdev = np.std(np.array(values))
      normalized_values = [(x - mean) / stdev for x in values]
      function_values = np.array(normalized_values[:-1])
      current_value = normalized_values[-1]
    top_ps, top_fs = filter_top_directions(perturbations, function_values,
                                           self.est_type,
                                           self.num_top_directions)
    gradient = np.zeros(dim)
    for i, perturbation in enumerate(top_ps):
      function_value = top_fs[i]
      if self.est_type == EstimatorType.FORWARD_FD:
        gradient_sample = (function_value - current_value) * perturbation
      elif self.est_type == EstimatorType.ANTITHETIC:
        gradient_sample = function_value * perturbation
      gradient_sample /= self.precision_parameter**2
      gradient += gradient_sample
    gradient /= len(top_ps)
    # this next line is for compatibility with the Blackbox used for Toaster.
    # in that code, the denominator for antithetic was num_top_directions.
    # we maintain compatibility for now so that the same hyperparameters
    # currently used in Toaster will have the same effect
    if self.est_type == EstimatorType.ANTITHETIC and \
    len(top_ps) < len(perturbations):
      gradient *= 2
    # Use the gradient ascent optimizer to compute the next parameters with the
    # gradients
    return self.gradient_ascent_optimizer.run_step(current_input, gradient)

  def get_state(self) -> List[float]:
    gradient_ascent_state = self.gradient_ascent_optimizer.get_state()
    return super().get_state() + gradient_ascent_state

  def set_state(self, state: SequenceOfFloats) -> None:
    super().set_state(state)
    self.gradient_ascent_optimizer.set_state(state)


class SklearnRegressionBlackboxOptimizer(StatefulOptimizer):
  """Class implementing GD optimizer with regression for grad. estimation."""

  def __init__(self,
               regression_method: RegressionType,
               regularizer: float,
               est_type: EstimatorType,
               normalize_fvalues: bool,
               hyperparameters_update_method: UpdateMethod,
               extra_params: Optional[Sequence[int]],
               step_size: Optional[float] = None,
               gradient_ascent_optimizer: Optional[
                   gradient_ascent_optimization_algorithms
                   .GradientAscentOptimizer] = None):
    # Check step_size and gradient_ascent_optimizer
    if (step_size is None) == (gradient_ascent_optimizer is None):
      raise ValueError('Exactly one of step_size and gradient_ascent_optimizer \
            should be provided')
    if step_size:
      gradient_ascent_optimizer = (
          gradient_ascent_optimization_algorithms.MomentumOptimizer(
              step_size=step_size, momentum=0.0))

    if regression_method == RegressionType.LASSO:
      self.clf = linear_model.Lasso(alpha=regularizer)
    elif regression_method == RegressionType.RIDGE:
      self.clf = linear_model.Ridge(alpha=regularizer)
    elif regression_method == RegressionType.LINEAR:
      self.clf = linear_model.LinearRegression()
    else:
      raise ValueError('Optimization procedure option not available')
    self.gradient_ascent_optimizer = gradient_ascent_optimizer
    super().__init__(est_type, normalize_fvalues, hyperparameters_update_method,
                     extra_params)

  def run_step(self, perturbations: FloatArray2D, function_values: FloatArray,
               current_input: FloatArray, current_value: float) -> FloatArray:
    dim = len(current_input)
    if self.normalize_fvalues:
      values = function_values.tolist()
      values.append(current_value)
      mean = sum(values) / float(len(values))
      stdev = np.std(np.array(values))
      normalized_values = [(x - mean) / stdev for x in values]
      function_values = np.array(normalized_values[:-1])
      current_value = normalized_values[-1]

    matrix = None
    b_vector = None
    if self.est_type == EstimatorType.FORWARD_FD:
      matrix = np.array(perturbations)
      b_vector = (
          function_values - np.array([current_value] * len(function_values)))
    elif self.est_type == EstimatorType.ANTITHETIC:
      matrix = np.array(perturbations[::2])
      function_even_values = np.array(function_values.tolist()[::2])
      function_odd_values = np.array(function_values.tolist()[1::2])
      b_vector = (function_even_values - function_odd_values) / 2.0
    else:
      raise ValueError('FD method not available.')

    self.clf.fit(matrix, b_vector)
    gradient = self.clf.coef_[0:dim]
    # Use the gradient ascent optimizer to compute the next parameters with the
    # gradients
    return self.gradient_ascent_optimizer.run_step(current_input, gradient)

  def get_state(self) -> List[float]:
    gradient_ascent_state = self.gradient_ascent_optimizer.get_state()
    return super().get_state + gradient_ascent_state

  def set_state(self, state: SequenceOfFloats) -> None:
    super().set_state(state)
    self.gradient_ascent_optimizer.set_state(state)


# pylint: disable=pointless-string-statement
"""Secondorder optimizers.

Experimental optimizers based on blackbox ES.

See class descriptions for more detailed notes on each algorithm.
"""
"""Gradient estimators.
The blackbox pipeline has two steps:
estimate gradient/Hessian --> optimizer --> next weight
which are decoupled.

Implemented: monte_carlo_gradient
             sklearn_regression_gradient
"""


def normalize_function_values(
    function_values: FloatArray,
    current_value: float) -> Tuple[FloatArray, List[float]]:
  values = function_values.tolist()
  values.append(current_value)
  mean = sum(values) / float(len(values))
  stdev = np.std(np.array(values))
  normalized_values = [(x - mean) / stdev for x in values]
  return (np.array(normalized_values[:-1]), normalized_values[-1])


def monte_carlo_gradient(precision_parameter: float,
                         est_type: EstimatorType,
                         perturbations: FloatArray2D,
                         function_values: FloatArray,
                         current_value: float,
                         energy: Optional[float] = 0) -> FloatArray:
  """Calculates Monte Carlo gradient.

  There are several ways of estimating the gradient. This is specified by the
  attribute self.est_type. Currently, forward finite difference (FFD) and
  antithetic are supported.

  Args:
    precision_parameter: sd of Gaussian perturbations
    est_type: 'forward_fd' (FFD) or 'antithetic'
    perturbations: the simulated perturbations
    function_values: reward from perturbations (possibly normalized)
    current_value: estimated reward at current point (possibly normalized)
    energy: optional, for softmax weighting of the average (default = 0)
  Returns:
    The Monte Carlo gradient estimate.
  """
  dim = len(perturbations[0])
  b_vector = None
  if est_type == EstimatorType.FORWARD_FD:
    b_vector = (function_values -
                np.array([current_value] * len(function_values))) / (
                    precision_parameter * precision_parameter)
  elif est_type == EstimatorType.ANTITHETIC:
    b_vector = function_values / (2.0 * precision_parameter *
                                  precision_parameter)
  else:
    raise ValueError('FD method not available.')

  # the average is given by softmax weights
  # when the energy is 0 (default), it's the arithmetic mean
  adj_function_values = energy * (function_values - max(function_values))
  softmax_weights = np.divide(
      np.exp(adj_function_values), sum(np.exp(adj_function_values)))
  gradient = np.zeros(dim)
  for i in range(len(perturbations)):
    gradient += softmax_weights[i] * b_vector[i] * perturbations[i]
  return gradient


def sklearn_regression_gradient(clf: LinearModel, est_type: EstimatorType,
                                perturbations: FloatArray2D,
                                function_values: FloatArray,
                                current_value: float) -> FloatArray:
  """Calculates gradient by function difference regression.

  Args:
    clf: an object (SkLearn linear model) which fits Ax = b
    est_type: 'forward_fd' (FFD) or 'antithetic'
    perturbations: the simulated perturbations
    function_values: reward from perturbations (possibly normalized)
    current_value: estimated reward at current point (possibly normalized)
  Returns:
    The regression estimate of the gradient.
  """
  matrix = None
  b_vector = None
  dim = perturbations[0].size
  if est_type == EstimatorType.FORWARD_FD:
    matrix = np.array(perturbations)
    b_vector = (
        function_values - np.array([current_value] * len(function_values)))
  elif est_type == EstimatorType.ANTITHETIC:
    matrix = np.array(perturbations[::2])
    function_even_values = np.array(function_values.tolist()[::2])
    function_odd_values = np.array(function_values.tolist()[1::2])
    b_vector = (function_even_values - function_odd_values) / 2.0
  else:
    raise ValueError('FD method not available.')

  clf.fit(matrix, b_vector)
  return clf.coef_[0:dim]


r"""Auxiliary objects and solvers.

1) Trust region subproblem solver.
     Solves the trust-region subproblem
     min_x Q(x) s.t. \|x\|_2 \leq R
     where Q(x) is a quadratic function. This can be solved to global optimality
     by first-order methods (see PGD below).

     The outer trust region loop is in blackbox_optimization_algorithms
     so as to use the BlackBoxOptimizer class, to better integrate with
     existing frameworks for running the experiments.

  2) Projected Gradient Descent
    A first-order method for solving
    min_x f(x) s.t. x in C
    given access to a function P(x) which projects x onto the set C.

    Since we anticipate this will typically be used to solve the trust-region
    subproblem, the true function and gradient evaluations are cheap to compute.
    This makes it feasible to use a line search algorithm instead of fixed
    step size. The line search is borrowed from scipy.optimize and tests the
    Armijo condition.
    If the line search fails (usually because of very badly condition problems
    and/or numerical issues), a small number (usually 10) steps are conducted
    with a fixed step size.
"""


class QuadraticModel(object):
  """A class for quadratic functions.

  Presents an interface for evaluating functions of the form
  f(x) = 1/2x^TAx + b^Tx + c
  """

  # pylint: disable=invalid-name
  # argument Av should be capitalized as such for mathematical convention
  def __init__(self,
               Av: Callable[[FloatArray], FloatArray],
               b: FloatArray,
               c: Optional[float] = 0):
    """Initialize quadratic function.

    Args:
      Av: a function of one argument which returns the matrix-vector product
          A @ v.
      b: the vector b
      c (optional): a constant which is added when evaluating f.
    """
    self.quad_v = Av
    self.b = b
    self.c = c

  # pylint: enable=invalid-name
  def f(self, x: FloatArray) -> float:
    """Evaluate the quadratic function.

    Args:
      x: numpy vector
    Returns:
      Scalar f(x)
    """
    return 0.5 * np.dot(x, self.quad_v(x)) + np.dot(x, self.b) + self.c

  def grad(self, x: FloatArray) -> FloatArray:
    """Evaluate the gradient of the quadratic, Ax + b.

    Args:
      x: input vector
    Returns:
      A vector of the same dimension as x, the gradient of the quadratic at x.
    """
    return self.quad_v(x) + self.b


class ProjectedGradientOptimizer(object):
  r"""A class implementing the projected gradient algorithm.

   The update is given by
   x^+ = P(x - \eta f'(x))
   where f is the function to minimize and P is the projection
   operator onto a closed set.

   The functions in this class are non-destructive,
   i.e. the input variables will not be mutated.
  """

  def __init__(self, function_object: QuadraticModel,
               projection_operator: Callable[[FloatArray], FloatArray],
               pgd_params: Mapping[str, Any], x_init: FloatArray):
    self.f = function_object
    self.proj = projection_operator
    if pgd_params is not None:
      self.params = pgd_params
    else:
      self.params = {}
    self.x = np.copy(x_init)
    self.k = 0  # iteration counter
    self.x_diff_norm = np.Inf  # L2 norm of x^+ - x

  def run_step(self) -> None:
    """Take a single step of projected gradient descent.

    Algorithm description:
    1) Compute search direction (negative gradient)
    2) Armijo line search to get step size
    3) Project resulting point onto constraints
    """
    fval = getattr(self.f, 'f', None)
    grad = getattr(self.f, 'grad', None)
    if fval is None or grad is None:
      raise ValueError('Function/gradient not supplied')

    # Line search for a step size
    c1 = self.params.get('c1', DEFAULT_ARMIJO)
    c2 = self.params.get('c2', -np.Inf)
    # since we have negative curvature, ignore Wolfe condition
    search_direction = -grad(self.x)
    ls_result = sp_opt.line_search(
        fval, grad, self.x, search_direction, c1=c1, c2=c2)
    if ls_result[0] is not None:
      step_size = ls_result[0]
    else:
      step_size = self.params['const_step_size']  # take a fixed step

    # project step onto feasible set
    x_next = self.proj(self.x + step_size * search_direction)
    # record the size of x^+ - x
    self.x_diff_norm = np.linalg.norm(x_next - self.x, 2)
    self.x = x_next
    self.k += 1

  def get_solution(self) -> FloatArray:
    return self.x

  def get_x_diff_norm(self) -> float:
    return self.x_diff_norm

  def get_iterations(self) -> int:
    return self.k


def make_projector(radius: float) -> Callable[[FloatArray], FloatArray]:
  """Makes an L2 projector function centered at origin.

  Args:
    radius: the radius to project on
  Returns:
    A function of one argument that projects onto L2 ball.
  """

  def projector(w: FloatArray) -> FloatArray:
    w_norm = np.linalg.norm(w, 2)
    if w_norm > radius:
      return radius / w_norm * w
    else:
      return w

  return projector


class TrustRegionSubproblemOptimizer(object):
  r"""Solves the trust region subproblem over the L2 ball.

   min_x f(x) s.t. \|x - p\| \leq R
   where f is a quadratic model.

   It is known that first-order methods can attain global optimality if
   initialized from 0 [1]. See Beck and Vaisbourd, "Globally Solving the Trust
   Region Subproblem Using Simple First-Order Methods", SIAM J. Optim.

   The current termination criterion is when the difference between
   successive iterates is sufficiently small.

   [1] Except for one `bad' case which does not occur almost surely.
  """

  def __init__(self,
               model_function: QuadraticModel,
               trust_region_params: Dict[str, Any],
               x_init: Optional[FloatArray] = None):
    self.mf = model_function
    self.params = trust_region_params
    self.center = x_init
    self.radius = self.params['radius']
    self.dim = self.params['problem_dim']
    self.params['const_step_size'] = 0.1 * self.radius
    if x_init is not None:
      self.x = np.copy(x_init)
    else:
      self.x = np.zeros(self.dim)

  def solve_trust_region_subproblem(self) -> None:
    """Solves the trust region subproblem.

    The currently implemented solver is Projected Gradient Descent.
    """
    sub_optimizer = self.params['subproblem_solver']
    if sub_optimizer == 'PGD':
      if (not hasattr(self.mf, 'f')) or (not hasattr(self.mf, 'grad')):
        raise ValueError('Quadratic model function not supplied')

      projection_operator = make_projector(self.radius)
      pgd_solver = ProjectedGradientOptimizer(self.mf, projection_operator,
                                              self.params, self.x)

      inner_loop_termination = False
      while not inner_loop_termination:
        pgd_solver.run_step()
        if (pgd_solver.get_x_diff_norm() < self.params['sub_terminate_stable']
            or pgd_solver.get_iterations() > self.params['subproblem_maxiter']):
          inner_loop_termination = True

      self.x = pgd_solver.get_solution()

  def get_solution(self) -> FloatArray:
    return self.x


"""Blackbox Optimization Algorithms
1. Second Order Trust Region
"""


class TrustRegionOptimizer(StatefulOptimizer):
  r"""A second-order trust region method for solving the ES problem.

  ####################
  Overview
  The trust region (TR) problem is to solve:
  min_s 1/2 s^TAs + b^Ts
        \|s\| \leq R
  where A = \nabla^2 F(x) and b = \nabla F(x). That is, the TR problem is to
  minimize the quadratic model of the objective function F within a radius R.
  The radius is then adjusted based on the quality of the resulting solution.
  Let m(s) be the value of the quadratic model.
    trust region ratio: (F(x) - F(x+s)) / (m(0) - m(s))
                               = (F(x) - F(x+s)) / ( -m(s))
                               because we drop constant terms in the model
    absolute ratio: |F(x) - F(x+s)| / |F(y)|
                     where y = x + s if x + s improved the objective
                     and y = x otherwise
  The classic trust region uses only the trust region ratio (TRR), and grows the
  radius if TRR is close to 1, and shrinks the radius if TRR is close to 0.
  In stochastic optimization, especially for RL, this is excessively
  conservative and we instead will only shrink the radius when the objective
  gets worse, and the absolute ratio is large enough (signifying a relatively
  large worsening). If the step is sufficiently bad, then we reject it,
  meaning that we return to the previous point without taking a step, and also
  reduce the radius.
  ####################
  Observations and Examples
  Some careful tuning of the hyperparameters is required to get good performance
  from Trust Region. In terms of overall sample complexity, we have not
  observed consistent gains over using ARS, and the hyperparameter tuning is
  somewhat involved, because the radius adjustment mechanism corresponds to
  tuning an entire step size schedule at once.

  HalfCheetah example:
  Affine policy, MC forward_fd estimator, 500 perturbations, precision 0.1
  State normalization, no function normalization
  Use 'current' for the current point estimate
  Trust Region settings:
    tr_init_radius = 1
    tr_grow_threshold = 1e-5
    tr_grow_factor = 1.1
    tr_shrink_neg_threshold = 0.5
    tr_shrink_factor = 0.9
    tr_reject_threshold = 1
    tr_reject_factor = 0.75
    tr_dense_hessian = 1
    tr_sub_termination = 1e-3
    tr_subproblem_maxiter = 10
    tr_minimum_radius = 0.1

  These settings work reasonably well. The initial radius of 1 is arbitrary and
  somewhat large, and you will observe that TR spends the first ~40 iterations
  to shrink the radius down to an effective size. Unfortunately, these settings
  are highly problem-dependent and we have not found a universal value which
  works broadly across even all Mujoco locomotion tasks, and across different
  parameterizations of the policies (e.g affine vs two-layer Toeplitz networks
  will be different).
  ####################
  Hyperparameters
  The parameters of the algorithm are to be passed as a dictionary.
  Keys:
    tr_init_radius: The starting radius of trust region.
    tr_grow_threshold: Radius increases if the trust region ratio is
                       greater than this threshold.
    tr_grow_factor: Factor by which the radius increases when it grows.
    tr_shrink_neg_threshold: Radius shrinks if the objective gets worse and the
                             absolute ratio is larger than threshold.
    tr_shrink_factor: Radius shrinks by this factor.
    tr_reject_threshold: The step is rejected if the objective gets worse and
                         absolute ratio is larger than threshold.
    tr_reject_factor: Radius shrinks by this factor when the step is rejected.
    tr_dense_hessian: Bool, True if storing a dense matrix for Hessian.
    tr_sub_termination: Gradient norm termination for the TR subproblem solver.
    tr_subproblem_maxiter: Maximum number of iterations for the TR subproblem.
    tr_minimum_radius: If the radius shrinks to below this size, reset to min.
    tr_grad_type: string, ( 'MC' | 'regression' )
                  Type of gradient estimator.
    tr_grad_reg: string, ('lasso' | 'ridge' )
                 Type of regularizer when using Regression gradient.
    tr_grad_reg_alpha: Regularizer weight when using Regression.
  ####################
  Improvements/Extensions
    1. Currently the gradient and dense Hessian, when updated, average the new
       samples with the previous samples when steps are rejected (to use all
       information available about that point). This averaging causes
       exponential decay of the information from old samples. This maybe should
       be changed to a proper arithmetic average.
    2. The minimum radius should probably shrink over time, but this is another
       schedule that would have to be tuned.
  """

  def __init__(self, precision_parameter: float, est_type: EstimatorType,
               normalize_fvalues: bool,
               hyperparameters_update_method: UpdateMethod,
               extra_params: Optional[Sequence[int]], tr_params: Mapping[str,
                                                                         Any]):
    self.precision_parameter = precision_parameter
    super().__init__(est_type, normalize_fvalues, hyperparameters_update_method,
                     extra_params)

    self.accepted_quadratic_model = None
    self.accepted_function_value = None
    self.accepted_weights = None
    self.saved_hessian = None
    self.saved_perturbations = None
    self.saved_function_values = None
    self.saved_gradient = None
    self.normalized_current_value = None

    self.params = tr_params
    self.radius = self.params['init_radius']
    self.current_point_estimate = CurrentPointEstimate.CURRENT

    if self.params['grad_type'] == GradientType.REGRESSION:
      if self.params['grad_reg_type'] == RegressionType.RIDGE:
        self.clf = linear_model.Ridge(alpha=self.params['grad_reg_alpha'])
      elif self.params['grad_reg_type'] == RegressionType.LASSO:
        self.clf = linear_model.Lasso(alpha=self.params['grad_reg_alpha'])
    self.is_returned_step = False

  def trust_region_test(self, current_input: FloatArray,
                        current_value: float) -> bool:
    """Test the next step to determine how to update the trust region.

    The possible outcomes:
      0) If the previous step was rejected and the current point was previously
         accepted, then we can skip the test and immediately return TRUE.
      1) REJECT step
         Reduce the radius.
         Set is_returned_step TRUE since the next step goes back to prev.
         Return FALSE.
      2) ACCEPT step
         Update radius.
         Return TRUE.

    Args:
      current_input: the weights of current candidate point
      current_value: the reward of the current point
    Returns:
      TRUE if the step is accepted
      FALSE is the step is rejected
    """
    prev_mf = getattr(self.accepted_quadratic_model, 'f', None)
    if (not self.is_returned_step) and prev_mf is not None:
      is_ascent = current_value > self.accepted_function_value
      if is_ascent:
        absolute_max_reward = abs(current_value)
      else:
        absolute_max_reward = abs(self.accepted_function_value)
      absolute_max_reward = max(absolute_max_reward, 1e-08)
      abs_ratio = (
          abs(current_value - self.accepted_function_value) /
          absolute_max_reward)
      # pylint: disable=not-callable
      # this lint warning is incorrect. prev_mf is callable
      tr_imp_ratio = ((current_value - self.accepted_function_value) /
                      (-prev_mf(current_input)))
      # pylint: enable=not-callable

      # test criteria for rejecting step/updating radius
      should_reject = (not is_ascent and
                       abs_ratio > self.params['reject_threshold'])
      should_shrink = (not is_ascent and
                       abs_ratio > self.params['shrink_neg_threshold'])
      should_grow = ((is_ascent and
                      tr_imp_ratio > self.params['grow_threshold']))
      log_message = (' fval pct change: ' + str(abs_ratio) + ' tr_ratio: ' +
                     str(tr_imp_ratio))
      if should_reject:
        self.radius *= self.params['reject_factor']
        if self.radius < self.params['minimum_radius']:
          self.radius = self.params['minimum_radius']
        self.is_returned_step = True
        print('Step rejected. Shrink: ' + str(self.radius) + log_message)
        return False
      else:  # accept step
        if should_shrink:
          self.radius *= self.params['shrink_factor']
          if self.radius < self.params['minimum_radius']:
            self.radius = self.params['minimum_radius']
          print('Shrink: ' + str(self.radius) + log_message)
        elif should_grow:
          self.radius *= self.params['grow_factor']
          print('Grow: ' + str(self.radius) + log_message)
        else:
          print('Unchanged: ' + str(self.radius) + log_message)
    return True

  def update_hessian_part(self, perturbations: FloatArray2D,
                          function_values: FloatArray, current_value: float,
                          is_update: bool) -> None:
    """Updates the internal state which stores Hessian information.

    Recall that the Hessian is given by
    1/s^2 ( EX[f(x+sg)gg^T] - gs(x) I )
    Note that each perturbation passed in is actually s*g, so we have to scale
    down by 1/s^2 /twice/ on the perturbation. This function performs scaling
    by 1/s^2 /once/, and a second scaling is done in create_hessv_function.

    If the parameter 'dense_hessian' is true, then we'll explicitly store
    the matrix.
    Otherwise, we simply extend the list of perturbations with the new ones,
    and the list of function values.

    See run_step() for a description of arguments.

    Args:
      perturbations:
      function_values: (possibly normalized) function values
      current_value: (possibly normalized) current value, used as the
                      Gaussian smoothing estimate if current_point_estimate
                      is set to 'current'
      is_update: whether the new perturbations should be added, or overwrite
    """
    dim = perturbations[0].size
    if self.current_point_estimate == CurrentPointEstimate.CURRENT:
      current_point_estimate = current_value
    elif self.current_point_estimate == CurrentPointEstimate.AVERAGE:
      current_point_estimate = np.mean(function_values)
    if self.params['dense_hessian']:
      new_hessian = np.zeros((dim, dim))
      for i, perturbation in enumerate(perturbations):
        new_hessian += ((function_values[i] - current_point_estimate) *
                        np.outer(perturbation, perturbation) /
                        np.power(self.precision_parameter, 2))
      # We subtract current_point_estimate since
      # E[(f(x+sg) - gs(x))gg^T] = E[f(x+sg)gg^T] - gs(x)I,
      # where gs(x) is the Gaussian smoothing at x. We don't have access
      # to the exact value of gs(), but we can use either the MC approximation
      # of the Gaussian smoothing or use the actual function value as an
      # estimate.
      # This is an unbiased estimator of the Hessian. We call this the
      # 'sensing-subspace Hessian'.
      # An alternative would be to calculate
      # E[f(x+sg)gg^T] and then subtract off the scaled identity gs(x)I, but
      # depending on how many perturbations are used, that version either
      # forces large/small steps in unexplored directions based on the sign of
      # gs(x).
      new_hessian /= float(len(perturbations))
      if not is_update:
        self.saved_hessian = new_hessian
      else:
        self.saved_hessian = 0.5 * self.saved_hessian + 0.5 * new_hessian
    else:
      if not is_update:
        self.saved_perturbations = perturbations
        self.saved_function_values = function_values
      else:
        self.saved_perturbations = np.append(
            self.saved_perturbations, perturbations, axis=0)
        self.saved_function_values = np.append(self.saved_function_values,
                                               function_values)

  def create_hessv_function(self) -> Callable[[FloatArray], FloatArray]:
    """Returns a function of one argument that evaluates Hessian-vector product.
    """
    if self.params['dense_hessian']:

      def hessv_func(x: FloatArray) -> FloatArray:
        """Calculates Hessian-vector product from dense Hessian.

        Args:
          x: the direction to evaluate the product, i.e Hx
        Returns:
          Hessian-vector product.
        """
        hessv = np.matmul(self.saved_hessian, x)
        # Reminder:
        # If not using sensing-subspace Hessian, also subract diagonal gs(x)*I
        hessv /= np.power(self.precision_parameter, 2)
        hessv *= -1
        return hessv
    else:

      def hessv_func(x: FloatArray) -> FloatArray:
        """Calculates Hessian-vector product from perturbation/value pairs.

        Args:
          x: the direction to evaluate the product, i.e Hx
        Returns:
          Hessian-vector product.
        """
        dim = self.saved_perturbations[0].size
        if self.current_point_estimate == CurrentPointEstimate.CURRENT:
          current_point_estimate = self.normalized_current_value
        elif self.current_point_estimate == CurrentPointEstimate.AVERAGE:
          current_point_estimate = np.mean(self.saved_function_values)
        hessv = np.zeros(dim)
        for i, perturbation in enumerate(self.saved_perturbations):
          hessv += ((self.saved_function_values[i] - current_point_estimate) *
                    np.dot(perturbation, x) * perturbation /
                    np.power(self.precision_parameter, 2))
        hessv /= float(len(self.saved_perturbations))
        # Reminder:
        # If not using sensing-subspace Hessian, also subract diagonal gs(x)*I
        hessv /= np.power(self.precision_parameter, 2)
        hessv *= -1
        return hessv

    return hessv_func

  def update_quadratic_model(self, perturbations: FloatArray2D,
                             function_values: FloatArray, current_value: float,
                             is_update: bool) -> QuadraticModel:
    """Updates the internal state of the optimizer with new perturbations.

    The gradient here is the standard Monte Carlo gradient, but could
    be replaced by any other method of gradient estimation.

    When performing an update on the same point, we just average the
    new and previous estimates for the gradient and function value.

    Args:
      perturbations: sensing directions
      function_values: unnormalized rewards of the perturbations
      current_value: unnormalized reward of the current policy
      is_update: whether the previous step was rejected and this is the same
                 as the last accepted policy
    Returns:
      A QuadraticModel object with the local quadratic model after the updates.
    """
    if not is_update:
      self.accepted_function_value = current_value
    else:
      self.accepted_function_value = (0.5 * current_value +
                                      0.5 * self.accepted_function_value)
    if self.normalize_fvalues:
      normalized_values = normalize_function_values(function_values,
                                                    current_value)
      function_values = normalized_values[0]
      current_value = normalized_values[1]
      self.normalized_current_value = current_value
    if self.params['grad_type'] == GradientType.REGRESSION:
      new_gradient = sklearn_regression_gradient(self.clf, self.est_type,
                                                 perturbations, function_values,
                                                 current_value)
    else:
      new_gradient = monte_carlo_gradient(self.precision_parameter,
                                          self.est_type, perturbations,
                                          function_values, current_value)
    new_gradient *= -1  # TR subproblem solver performs minimization
    if not is_update:
      self.saved_gradient = new_gradient
    else:
      self.saved_gradient = 0.5 * new_gradient + 0.5 * self.saved_gradient

    self.update_hessian_part(perturbations, function_values, current_value,
                             is_update)
    return QuadraticModel(self.create_hessv_function(), self.saved_gradient)

  def run_step(self, perturbations: FloatArray2D, function_values: FloatArray,
               current_input: FloatArray, current_value: float) -> FloatArray:
    """Run a single step of trust region optimizer.

    Args:
      perturbations: list of numpy vectors, the perturbations
      function_values: list of scalars, reward corresponding to perturbation
      current_input: numpy vector, current model weights
      current_value: scalar, reward of current model
    Returns:
      updated model weights
    """
    dim = len(current_input)
    # we first check whether the current point should be accepted
    is_accepted = self.trust_region_test(current_input, current_value)
    if not is_accepted:
      return self.accepted_weights
      # If the step is rejected, then we return the last accepted weights.
      # Note that the radius has already been updated inside the
      # trust_region_test method.
    else:
      # If this step is a new point that was accepted, we build its local
      # second-order model.
      # Note that the only difference between the following calls to
      # update_quadratic_model is whether the final is_update bool is set to
      # T or F.
      if not self.is_returned_step:
        mf = self.update_quadratic_model(perturbations, function_values,
                                         current_value, False)
      elif self.is_returned_step:
        mf = self.update_quadratic_model(perturbations, function_values,
                                         current_value, True)
        self.is_returned_step = False
        # This point was just returned to after rejecting a step.
        # We update the model by averaging the previous gradient/Hessian
        # with the current perturbations. Then we set is_returned_step to False
        # in preparation for taking the next step after re-solving the
        # trust region at this point again, with smaller radius. """
      self.accepted_quadratic_model = mf
      self.accepted_weights = current_input
      # This step has been accepted, so store the most recent quadratic model
      # and the current weights.
    """
    The trust region problem at this point is defined by
    min_s 1/2 s^T Hess s + grad^T s
    where the next step will be given by x^+ = x + s.
    Note that the generic optimizer routines perform minimization,
    so we pass in the grad/hessian of the negative reward.
    """
    subproblem_params = {
        'subproblem_solver': 'PGD',
        'problem_dim': dim,
        'radius': self.radius,
        'sub_terminate_stable': self.params['sub_termination'],
        'subproblem_maxiter': self.params['subproblem_maxiter']
    }
    subproblem_init_point = np.zeros(dim)
    tr_sub_optimizer = TrustRegionSubproblemOptimizer(mf, subproblem_params,
                                                      subproblem_init_point)
    tr_sub_optimizer.solve_trust_region_subproblem()
    x_update = tr_sub_optimizer.get_solution()
    return current_input + x_update
