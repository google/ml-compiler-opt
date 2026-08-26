"""Propeller-specific Behavioral Cloning agent configuration."""

import os
from typing import Any

import gin
import tensorflow as tf
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec

from tf_agents.agents import tf_agent
from tf_agents.typing import types
from ..agent_config import AgentConfig


@gin.configurable(module='agents')
class PropellerRegressionCloningNetwork(network.Network):
  """A tf_agents Network that processes features and outputs regression score."""

  def __init__(
      self,
      input_tensor_spec,
      output_tensor_spec,
      preprocessing_layers=None,
      preprocessing_combiner=None,
      fc_layer_params=(64, 32),
      dropout_rate=0.2,
      name='PropellerRegressionNetwork',
      **kwargs
  ):
    super().__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name, **kwargs
    )
    self._output_dim = (
        output_tensor_spec.shape[0] if len(output_tensor_spec.shape) > 0 else 1
    )

    if preprocessing_layers is None:
      self._flat_preprocessing_layers = None
    else:
      self._flat_preprocessing_layers = [
          layer for layer in tf.nest.flatten(preprocessing_layers)
      ]
    self._preprocessing_nest = tf.nest.map_structure(
        lambda l: None, preprocessing_layers
    )
    self._preprocessing_combiner = preprocessing_combiner

    self._dense_layers = []
    self._dropout_layers = []

    for num_units in fc_layer_params:
      self._dense_layers.append(
          tf.keras.layers.Dense(num_units, activation='relu')
      )
      if dropout_rate > 0.0:
        self._dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))

    self._score_layer = tf.keras.layers.Dense(
        self._output_dim, activation='sigmoid'
    )

  def call(
      self, observations, step_type=None, network_state=(), training=False
  ):

    if self._flat_preprocessing_layers is not None:
      processed = []
      for obs, layer in zip(
          nest.flatten_up_to(self._preprocessing_nest, observations),
          self._flat_preprocessing_layers,
      ):
        res = layer(obs, training=training) if layer is not None else obs
        if (
            len(obs.shape) > 1
            and obs.shape[-1] == 1
            and len(res.shape) > len(obs.shape)
        ):
          res = tf.squeeze(res, axis=-2)
        processed.append(res)
      if self._preprocessing_combiner is not None:
        processed = self._preprocessing_combiner(processed)
      else:
        processed = nest.pack_sequence_as(self._preprocessing_nest, processed)
    else:
      processed = observations

    # Clean up features not used by scoring
    processed.pop('mask', None)
    processed.pop('is_chosen', None)
    processed.pop('merge_order', None)
    processed.pop('score_gain', None)
    processed.pop('decision_id', None)

    feature_list = []
    for feat in tf.nest.flatten(processed):
      feature_list.append(feat)

    x = tf.concat(feature_list, axis=-1)

    flat_x = x
    if len(flat_x.shape) > 2:
      orig_shape = tf.shape(flat_x)
      last_dim = flat_x.shape[-1]
      flat_x = tf.reshape(flat_x, [-1, last_dim])
    else:
      orig_shape = None

    for i in range(len(self._dense_layers)):
      flat_x = self._dense_layers[i](flat_x)
      if self._dropout_layers:
        flat_x = self._dropout_layers[i](flat_x, training=training)

    flat_scores = self._score_layer(flat_x)

    if orig_shape is not None:
      new_shape = tf.concat([orig_shape[:-1], [self._output_dim]], axis=0)
      raw_scores = tf.reshape(flat_scores, new_shape)
    else:
      raw_scores = flat_scores

    return raw_scores, network_state


@gin.configurable(module='agents')
class PropellerBCAgentConfig(AgentConfig):
  """Behavioral Cloning agent configuration for Propeller regression."""

  is_regression = True

  def create_agent(
      self,
      preprocessing_layers: tf.keras.layers.Layer,
      policy_network: types.Network,
  ) -> tf_agent.TFAgent:
    """Creates a behavioral_cloning_agent."""

    network = policy_network(
        self.time_step_spec.observation,
        self.action_spec,
        preprocessing_layers=preprocessing_layers,
        name='RegressionNetwork',
    )

    def custom_bc_loss(experience, training=False):
      batch_size = (
          tf.compat.dimension_value(experience.step_type.shape[0])
          or tf.shape(experience.step_type)[0]
      )
      network_state = network.get_initial_state(batch_size)
      bc_predictions, _ = network(
          experience.observation,
          step_type=experience.step_type,
          training=training,
          network_state=network_state,
      )

      if (
          isinstance(preprocessing_layers, dict)
          and 'score_gain' in preprocessing_layers
      ):
        layer = preprocessing_layers['score_gain']
        true_processed = layer(experience.observation['score_gain'])
      elif (
          hasattr(preprocessing_layers, 'get')
          and 'score_gain' in preprocessing_layers
      ):
        layer = preprocessing_layers.get('score_gain')
        true_processed = layer(experience.observation['score_gain'])
      else:
        flat_layers = [l for l in tf.nest.flatten(preprocessing_layers)]
        flat_obs = [
            o
            for o in tf.nest.flatten_up_to(
                preprocessing_layers, experience.observation
            )
        ]
        processed_dict = tf.nest.pack_sequence_as(
            preprocessing_layers,
            [
                l(o) if l is not None else o
                for o, l in zip(flat_obs, flat_layers)
            ],
        )
        true_processed = processed_dict['score_gain']

      if len(true_processed.shape) > len(bc_predictions.shape):
        true_processed = tf.squeeze(true_processed, axis=-2)

      # The 1st element (index 0) is the bucketized percentile (quantile).
      # We predict the quantile instead of predicting the raw score gain.
      target_1d = true_processed[..., 0:1]

      # Unweighted Mean Squared Error (MSE) Loss
      squared_error = tf.square(target_1d - bc_predictions)
      losses = tf.squeeze(squared_error, axis=-1)

      return losses

    return behavioral_cloning_agent.BehavioralCloningAgent(
        self.time_step_spec,
        self.action_spec,
        cloning_network=network,
        num_outer_dims=2,
        loss_fn=custom_bc_loss,
    )
