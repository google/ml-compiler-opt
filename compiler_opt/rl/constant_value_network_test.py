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
"""Tests for compiler_opt.rl.constant_value_network."""

import tensorflow as tf

from tf_agents.specs import tensor_spec

from compiler_opt.rl import constant_value_network


class ConstantValueNetworkTest(tf.test.TestCase):

  def testBuilds(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    observation = tensor_spec.sample_spec_nest(
        observation_spec, outer_dims=(1,))

    net = constant_value_network.ConstantValueNetwork(
        input_tensor_spec=observation_spec, constant_output_val=0)

    value, _ = net(observation)
    self.assertAllEqual([0.], value)

  def testHandlesExtraOuterDims(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    observation = tensor_spec.sample_spec_nest(
        observation_spec, outer_dims=(2, 2))

    net = constant_value_network.ConstantValueNetwork(
        input_tensor_spec=observation_spec, constant_output_val=1)

    value, _ = net(observation)
    self.assertAllEqual([[1., 1.], [1., 1.]], value)


if __name__ == '__main__':
  tf.test.main()
