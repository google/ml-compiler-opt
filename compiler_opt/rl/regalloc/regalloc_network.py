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
"""Actor network for Register Allocation."""

from typing import Any
from collections.abc import Sequence, Callable

import gin
import tensorflow as tf
from tf_agents.networks import categorical_projection_network
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class RegAllocEncodingNetwork(encoding_network.EncodingNetwork):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # remove the first layer (Flatten) in postprocessing_layers cause this will
    # flatten the B x T x 33 x dim to B x T x (33 x dim).
    self._postprocessing_layers = self._postprocessing_layers[1:]


class RegAllocProbProjectionNetwork(
    categorical_projection_network.CategoricalProjectionNetwork):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # shape after projection_layer: B x T x 33 x 1; then gets re-shaped to
    # B x T x 33.
    self._projection_layer = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=kwargs['logits_init_output_factor']),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='logits')


@gin.configurable
class RegAllocRNDEncodingNetwork(RegAllocEncodingNetwork):

  def __init__(self, **kwargs):
    pooling_layer = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last')
    super().__init__(**kwargs)
    # add a pooling layer at the end to to convert B x T x 33 x dim to
    # B x T x dim.
    self._postprocessing_layers.append(pooling_layer)


@gin.configurable
class RegAllocNetwork(network.DistributionNetwork):
  """Creates the actor network for register allocation policy training."""

  def __init__(
      self,
      input_tensor_spec: types.NestedTensorSpec,
      output_tensor_spec: types.NestedTensorSpec,
      preprocessing_layers: types.NestedLayer | None = None,
      preprocessing_combiner: tf.keras.layers.Layer | None = None,
      conv_layer_params: Sequence[Any] | None = None,
      fc_layer_params: Sequence[int] | None = (200, 100),
      dropout_layer_params: Sequence[float] | None = None,
      activation_fn: Callable[[types.Tensor],
                              types.Tensor] = tf.keras.activations.relu,
      kernel_initializer: tf.keras.initializers.Initializer | None = None,
      batch_squash: bool = True,
      context_fc_params: Sequence[int] | None = None,
      dtype: tf.DType = tf.float32,
      name: str = 'RegAllocNetwork'):
    """Creates an instance of `RegAllocNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent`, if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform.
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      context_fc_params: Optional fully connected layers applied per register
        after concatenating its embedding with cross-register context: masked
        max/mean pools, the candidate (last) register embedding, the register
        position and a prefix sum over earlier registers. None keeps the
        registers independent.
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation.
    """

    if not kernel_initializer:
      kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

    # input: B x T x obs_spec
    # output: B x T x 33 x dim
    encoder = RegAllocEncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=batch_squash,
        dtype=dtype)

    projection_network = RegAllocProbProjectionNetwork(
        sample_spec=output_tensor_spec, logits_init_output_factor=0.1)
    output_spec = projection_network.output_spec

    super().__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._encoder = encoder
    self._projection_network = projection_network
    self._output_tensor_spec = output_tensor_spec
    self._context_layers = None
    if context_fc_params:
      self._context_gate = tf.keras.layers.Dense(
          8, activation='sigmoid', kernel_initializer=kernel_initializer)
      self._context_layers = [
          tf.keras.layers.Dense(
              n, activation=activation_fn, kernel_initializer=kernel_initializer)
          for n in context_fc_params
      ]

  def _add_context(self, state, mask):
    # state: [..., R, D], mask: [..., R].
    m = tf.expand_dims(tf.cast(mask, state.dtype), -1)
    num_regs = tf.shape(state)[-2]
    big = tf.constant(1e9, state.dtype)
    max_pool = tf.reduce_max(state * m - big * (1 - m), axis=-2, keepdims=True)
    mean_pool = tf.reduce_sum(
        state * m, axis=-2, keepdims=True) / tf.maximum(
            tf.reduce_sum(m, axis=-2, keepdims=True), 1)
    candidate = state[..., -1:, :]
    tile = tf.ones_like(state[..., :1])
    pos = tf.reshape(
        tf.linspace(tf.constant(0, state.dtype), 1, num_regs), [-1, 1]) * tile
    prefix = tf.math.tanh(
        tf.math.cumsum(self._context_gate(state) * m, axis=-2, exclusive=True))
    x = tf.concat(
        [state, max_pool * tile, mean_pool * tile, candidate * tile, pos, prefix],
        axis=-1)
    for layer in self._context_layers:
      x = layer(x)
    return x

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self,
           observations: types.NestedTensor,
           step_type: types.NestedTensor,
           network_state=(),
           training: bool = False,
           mask=None):
    _ = mask
    state, network_state = self._encoder(
        observations,
        step_type=step_type,
        network_state=network_state,
        training=training)
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    if self._context_layers:
      state = self._add_context(state, observations['mask'])

    # mask un-evictable registers.
    distribution, _ = self._projection_network(
        state, outer_rank, training=training, mask=observations['mask'])

    return distribution, network_state
