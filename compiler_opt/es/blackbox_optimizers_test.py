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
r"""Tests for blackbox_optimization_algorithms."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from compiler_opt.es import blackbox_optimizers
from compiler_opt.es.blackbox_optimizers import EstimatorType, UpdateMethod, RegressionType
from compiler_opt.es import gradient_ascent_optimization_algorithms

perturbation_array = np.array([[0, 1], [2, -1], [4, 2],
                               [-2, -2], [0, 3], [0, -3], [0, 4], [0, -4],
                               [-1, 5], [1, -5], [2, 6], [8, -6]])
function_value_array = np.array(
    [-1, 1, 10, -10, -2, 2, -0.5, 0.5, 4, -4, -8, 8])


class BlackboxOptimizationAlgorithmsTest(parameterized.TestCase):

  @parameterized.parameters(
      (perturbation_array, function_value_array, EstimatorType.ANTITHETIC, 3,
       np.array([[4, 2], [2, 6], [-1, 5], [-2, -2], [8, -6], [1, -5]
                ]), np.array([10, -8, 4, -10, 8, -4])),
      (perturbation_array, function_value_array, EstimatorType.FORWARD_FD, 5,
       np.array([[4, 2], [8, -6], [-1, 5], [0, -3], [2, -1]
                ]), np.array([10, 8, 4, 2, 1])))
  def test_filtering(self, perturbations, function_values, est_type,
                     num_top_directions, expected_ps, expected_fs):
    top_ps, top_fs = blackbox_optimizers.filter_top_directions(
        perturbations, function_values, est_type, num_top_directions)
    np.testing.assert_array_equal(expected_ps, top_ps)
    np.testing.assert_array_equal(expected_fs, top_fs)

  @parameterized.parameters(
      (perturbation_array, function_value_array, EstimatorType.ANTITHETIC, 3,
       np.array([100, -16])), (perturbation_array, function_value_array,
                               EstimatorType.FORWARD_FD, 5, np.array([76, -9])),
      (perturbation_array, function_value_array, EstimatorType.ANTITHETIC, 0,
       np.array([102, -34])),
      (perturbation_array, function_value_array, EstimatorType.FORWARD_FD, 0,
       np.array([74, -34])))
  def test_monte_carlo_gradient(self, perturbations, function_values, est_type,
                                num_top_directions, expected_gradient):
    precision_parameter = 0.1
    step_size = 0.01
    current_value = 2
    blackbox_object = blackbox_optimizers.MonteCarloBlackboxOptimizer(
        precision_parameter, est_type, False, UpdateMethod.NO_METHOD, None,
        step_size, num_top_directions)
    current_input = np.zeros(2)
    step = blackbox_object.run_step(perturbations, function_values,
                                    current_input, current_value)
    gradient = step * (precision_parameter**2) / step_size
    if num_top_directions == 0:
      gradient *= len(perturbations)
    else:
      gradient *= num_top_directions

    np.testing.assert_array_almost_equal(expected_gradient, gradient)

  @parameterized.parameters(
      (perturbation_array, function_value_array, EstimatorType.ANTITHETIC, 3,
       np.array([100, -16])), (perturbation_array, function_value_array,
                               EstimatorType.FORWARD_FD, 5, np.array([76, -9])),
      (perturbation_array, function_value_array, EstimatorType.ANTITHETIC, 0,
       np.array([102, -34])),
      (perturbation_array, function_value_array, EstimatorType.FORWARD_FD, 0,
       np.array([74, -34])))
  def test_monte_carlo_gradient_with_gradient_ascent_optimizer(
      self, perturbations, function_values, est_type, num_top_directions,
      expected_gradient):
    precision_parameter = 0.1
    step_size = 0.01
    current_value = 2
    gradient_ascent_optimizer = (
        gradient_ascent_optimization_algorithms.MomentumOptimizer(
            step_size, 0.0))
    blackbox_object = (
        blackbox_optimizers.MonteCarloBlackboxOptimizer(
            precision_parameter, est_type, False, UpdateMethod.NO_METHOD, None,
            None, num_top_directions, gradient_ascent_optimizer))
    current_input = np.zeros(2)
    step = blackbox_object.run_step(perturbations, function_values,
                                    current_input, current_value)
    gradient = step * (precision_parameter**2) / step_size
    if num_top_directions == 0:
      gradient *= len(perturbations)
    else:
      gradient *= num_top_directions

    np.testing.assert_array_almost_equal(expected_gradient, gradient)

  @parameterized.parameters(
      (perturbation_array, function_value_array, EstimatorType.ANTITHETIC, 3,
       np.array([0.00483, -0.007534])),
      (perturbation_array, function_value_array, EstimatorType.FORWARD_FD, 5,
       np.array([0.012585, 0.000748])),
      (perturbation_array, function_value_array, EstimatorType.ANTITHETIC, 0,
       np.array([0.019319, -0.030134])),
      (perturbation_array, function_value_array, EstimatorType.FORWARD_FD, 0,
       np.array([0.030203, 0.001796])))
  def test_sklearn_gradient(self, perturbations, function_values, est_type,
                            num_top_directions, expected_gradient):
    precision_parameter = 0.1
    step_size = 0.01
    current_value = 2
    regularizer = 0.01
    gradient_ascent_optimizer = (
        gradient_ascent_optimization_algorithms.MomentumOptimizer(
            step_size, 0.0))
    blackbox_object = blackbox_optimizers.SklearnRegressionBlackboxOptimizer(
        RegressionType.RIDGE, regularizer, est_type, True,
        UpdateMethod.NO_METHOD, [], None, gradient_ascent_optimizer)
    current_input = np.zeros(2)
    step = blackbox_object.run_step(perturbations, function_values,
                                    current_input, current_value)
    gradient = step * (precision_parameter**2) / step_size
    if num_top_directions == 0:
      gradient *= len(perturbations)
    else:
      gradient *= num_top_directions

    np.testing.assert_array_almost_equal(expected_gradient, gradient)


class SecondorderBlackboxOptimizersTest(absltest.TestCase):

  def setUp(self):
    """Create common data matrices for tests.

    We will use a particular quadratic problem
    f(x) = 1/2x^TAx + b^Tx + c
    in several of the tests and initialize it here.
    The matrix A is indefinite and has eigs
    [2.15, 0.53, -2.67]
    """
    super().setUp()
    # pylint: disable=bad-whitespace,invalid-name
    self.A = np.array([[1, -1, 0], [-1, 0, 2], [0, 2, -1]])
    self.b = np.array([1, 0, 1])
    self.c = 2.5
    self.multiply_Av = lambda v: np.matmul(self.A, v)
    # pylint: enable=bad-whitespace,invalid-name

  def testQuadraticModelFunctionValue(self):
    quad_model = blackbox_optimizers.QuadraticModel(self.multiply_Av, self.b,
                                                    self.c)
    x = np.array([1, 0, -2])
    self.assertEqual(0.0, quad_model.f(x))

  def testQuadraticModelGradient(self):
    quad_model = blackbox_optimizers.QuadraticModel(self.multiply_Av, self.b,
                                                    self.c)
    x = np.array([1, 0, -2])
    gradient = quad_model.grad(x)
    self.assertEqual(gradient[0], 2.0)
    self.assertEqual(gradient[1], -5.0)
    self.assertEqual(gradient[2], 3.0)

  def testMakeProjector(self):
    radius = 3.0
    x = np.array([3.0, 3.0])
    projector = blackbox_optimizers.make_projector(radius)
    projected_point = projector(x)
    self.assertTrue(np.isclose(3.0 / np.sqrt(2), projected_point[0]))
    self.assertTrue(np.isclose(3.0 / np.sqrt(2), projected_point[0]))

  def testProjectedGradientOptimizer(self):
    """Simple test for correctness of PGD.

    Minimize the function
    (x + 1)^2 + (y - 1)^2
    over the nonnegative orthant.
    The exact solution is (0,1).
    """

    class GenericFunction(blackbox_optimizers.QuadraticModel):

      def __init__(self):  # pylint: disable=super-init-not-called
        pass

      def f(self, x):
        return (x[0] + 1)**2 + (x[1] - 1)**2

      def grad(self, x):
        return np.array([2 * (x[0] + 1), 2 * (x[1] - 1)])

    def projector(x):
      return np.maximum(0, x)

    objective_function = GenericFunction()
    pgd_params = {'const_step_size': 0.5}
    pgd_optimizer = blackbox_optimizers.ProjectedGradientOptimizer(
        objective_function, projector, pgd_params, np.array([1, 1]))
    while (pgd_optimizer.get_iterations() <= 10 and
           pgd_optimizer.get_x_diff_norm() > 1e-9):
      pgd_optimizer.run_step()
    solution = pgd_optimizer.get_solution()
    self.assertLess(abs(solution[0]), 1e-8)
    self.assertLess(abs(solution[1] - 1.0), 1e-8)

  def testTrustRegionSubproblem(self):
    """Simple test for trust region subproblem solver with indefinite quadratic.

    Test the Trust Region Subproblem by solving the quadratic
    1/2x^TAx + b^Tx over the ball of radius 2, where A, b are created
    in setUp.
    """
    quad_model = blackbox_optimizers.QuadraticModel(self.multiply_Av, self.b,
                                                    self.c)
    tr_params = {
        'radius': 2,
        'problem_dim': 3,
        'subproblem_solver': 'PGD',
        'subproblem_maxiter': 100,
        'sub_terminate_stable': 1e-9
    }
    tr_subproblem_solver = blackbox_optimizers.TrustRegionSubproblemOptimizer(
        quad_model, tr_params)
    tr_subproblem_solver.solve_trust_region_subproblem()
    solution = tr_subproblem_solver.get_solution()
    self.assertLess(abs(solution[0] - 0.03157944), 1e-8)
    self.assertLess(abs(solution[1] - 1.12525306), 1e-8)
    self.assertLess(abs(solution[2] + 1.65312076), 1e-8)


if __name__ == '__main__':
  absltest.main()
