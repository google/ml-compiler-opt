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
# This is a port of the work by: Krzysztof Choromanski, Mark Rowland,
# Vikas Sindhwani, Richard E. Turner, Adrian Weller:  "Structured Evolution
# with Compact Architectures for Scalable Policy Optimization",
# https://arxiv.org/abs/1804.02395
#
###############################################################################
r"""Tests for gradient_ascent_optimization_algorithms."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from compiler_opt.es import gradient_ascent_optimization_algorithms


class GradientAscentOptimizationAlgorithmsTest(parameterized.TestCase):

  @parameterized.parameters((np.asarray([1., 2., 3.], dtype=np.float32),),
                            (np.asarray([1.1, 2.2, 3.3], dtype=np.float32),))
  def test_momentum_set_state(self, state):
    optimizer = gradient_ascent_optimization_algorithms.MomentumOptimizer(
        0.1, 0.9)
    optimizer.set_state(state)
    recovered_state = optimizer.get_state()
    np.testing.assert_array_almost_equal(state, recovered_state)

  @parameterized.parameters(
      (np.asarray([1., 2., 3., 4., 5., 6., 7], dtype=np.float32),),
      (np.asarray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7], dtype=np.float32),))
  def test_adam_set_state(self, state):
    optimizer = gradient_ascent_optimization_algorithms.AdamOptimizer(0.1)
    optimizer.set_state(state)
    recovered_state = optimizer.get_state()
    np.testing.assert_array_almost_equal(state, recovered_state)

  @parameterized.parameters(
      (0.1, 0.9, np.asarray([1.1, 0.0], dtype=np.float32),
       np.asarray([1.0, 1.0],
                  dtype=np.float32), np.asarray([1.0, 1.0], dtype=np.float32),
       np.asarray([1.129, 0.029], dtype=np.float32)))
  def test_momentum_step(self, step_size, momentum, ini_parameter, gradient1,
                         gradient2, final_parameter):
    optimizer = gradient_ascent_optimization_algorithms.MomentumOptimizer(
        step_size, momentum)
    parameter = ini_parameter
    parameter = optimizer.run_step(parameter, gradient1)
    parameter = optimizer.run_step(parameter, gradient2)
    np.testing.assert_array_almost_equal(parameter, final_parameter)

  @parameterized.parameters(
      (0.1, 0.2, 0.5, np.asarray([1.1, 0.0], dtype=np.float32),
       np.asarray([1.0, 1.0],
                  dtype=np.float32), np.asarray([1.0, 1.0], dtype=np.float32),
       np.asarray([1.3, 0.2], dtype=np.float32)))
  def test_adam_step(self, step_size, beta1, beta2, ini_parameter, gradient1,
                     gradient2, final_parameter):
    optimizer = gradient_ascent_optimization_algorithms.AdamOptimizer(
        step_size, beta1, beta2)
    parameter = ini_parameter
    parameter = optimizer.run_step(parameter, gradient1)
    parameter = optimizer.run_step(parameter, gradient2)
    np.testing.assert_array_almost_equal(parameter, final_parameter)


if __name__ == '__main__':
  absltest.main()
