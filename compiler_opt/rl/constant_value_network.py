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
"""Constant Value Network that always predicts a constant value."""

import gin
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.utils import nest_utils


@gin.configurable
class ConstantValueNetwork(network.Network):
  """Constant value network that predicts zero per batch item."""

  def __init__(self, input_tensor_spec, constant_output_val=0, name=None):
    """Creates an instance of `ConstantValueNetwork`.

    Network supports calls with shape outer_rank + observation_spec.shape. Note
    outer_rank must be at least 1.

    Args:
      input_tensor_spec: A `tensor_spec.TensorSpec` or a tuple of specs
        representing the input observations.
      constant_output_val: A constant scalar value the network will output.
      name: A string representing name of the network.

    Raises:
      ValueError: If input_tensor_spec is not an instance of network.InputSpec.
    """
    super().__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    self._constant_output_val = constant_output_val

  def call(self, inputs, step_type=None, network_state=(), training=False):
    _ = (step_type, training)
    shape = nest_utils.get_outer_shape(inputs, self._input_tensor_spec)
    constant = tf.constant(self._constant_output_val, tf.float32)
    return tf.fill(shape, constant), network_state
