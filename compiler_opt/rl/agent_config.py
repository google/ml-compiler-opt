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
"""util function to create a tf_agent."""

from typing import Any, Callable, Dict

import abc
import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.specs import tensor_spec
from tf_agents.typing import types

from compiler_opt.rl import constant_value_network
from compiler_opt.rl.distributed import agent as distributed_ppo_agent


class AgentConfig(metaclass=abc.ABCMeta):
  """Agent creation and data processing hook-ups."""

  def __init__(self, *, time_step_spec: types.NestedTensorSpec,
               action_spec: types.NestedTensorSpec):
    self._time_step_spec = time_step_spec
    self._action_spec = action_spec

  @property
  def time_step_spec(self):
    return self._time_step_spec

  @property
  def action_spec(self):
    return self._action_spec

  @abc.abstractmethod
  def create_agent(self, preprocessing_layers: tf.keras.layers.Layer,
                   policy_network: types.Network) -> tf_agent.TFAgent:
    """Specific agent configs must implement this."""
    raise NotImplementedError()

  def get_policy_info_parsing_dict(
      self) -> Dict[str, tf.io.FixedLenSequenceFeature]:
    """Return the parsing dict for the policy info."""
    return {}

  # pylint: disable=unused-argument
  def process_parsed_sequence_and_get_policy_info(
      self, parsed_sequence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Function to process parsed_sequence and to return policy_info.

    Args:
      parsed_sequence: A dict from feature_name to feature_value parsed from TF
        SequenceExample.

    Returns:
      A nested policy_info for given agent.
    """
    return {}


@gin.configurable
def create_agent(agent_config: AgentConfig,
                 preprocessing_layer_creator: Callable[[types.TensorSpec],
                                                       tf.keras.layers.Layer],
                 policy_network: types.Network):
  """Gin configurable wrapper of AgentConfig.create_agent.
  Works around the fact that class members aren't gin-configurable."""
  preprocessing_layers = tf.nest.map_structure(
      preprocessing_layer_creator, agent_config.time_step_spec.observation)
  return agent_config.create_agent(preprocessing_layers, policy_network)


@gin.configurable(module='agents')
class BCAgentConfig(AgentConfig):
  """Behavioral Cloning agent configuration."""

  def create_agent(self, preprocessing_layers: tf.keras.layers.Layer,
                   policy_network: types.Network) -> tf_agent.TFAgent:
    """Creates a behavioral_cloning_agent."""

    network = policy_network(
        self.time_step_spec.observation,
        self.action_spec,
        preprocessing_layers=preprocessing_layers,
        name='QNetwork')

    return behavioral_cloning_agent.BehavioralCloningAgent(
        self.time_step_spec,
        self.action_spec,
        cloning_network=network,
        num_outer_dims=2)


@gin.configurable(module='agents')
class DQNAgentConfig(AgentConfig):
  """DQN agent configuration."""

  def create_agent(self, preprocessing_layers: tf.keras.layers.Layer,
                   policy_network: types.Network) -> tf_agent.TFAgent:
    """Creates a dqn_agent."""
    network = policy_network(
        self.time_step_spec.observation,
        self.action_spec,
        preprocessing_layers=preprocessing_layers,
        name='QNetwork')

    return dqn_agent.DqnAgent(
        self.time_step_spec, self.action_spec, q_network=network)


@gin.configurable(module='agents')
class PPOAgentConfig(AgentConfig):
  """PPO/Reinforce agent configuration."""

  def create_agent(self, preprocessing_layers: tf.keras.layers.Layer,
                   policy_network: types.Network) -> tf_agent.TFAgent:
    """Creates a ppo_agent."""

    actor_network = policy_network(
        self.time_step_spec.observation,
        self.action_spec,
        preprocessing_layers=preprocessing_layers,
        name='ActorDistributionNetwork')

    critic_network = constant_value_network.ConstantValueNetwork(
        self.time_step_spec.observation, name='ConstantValueNetwork')

    return ppo_agent.PPOAgent(
        self.time_step_spec,
        self.action_spec,
        actor_net=actor_network,
        value_net=critic_network)

  def get_policy_info_parsing_dict(
      self) -> Dict[str, tf.io.FixedLenSequenceFeature]:
    if tensor_spec.is_discrete(self._action_spec):
      return {
          'CategoricalProjectionNetwork_logits':
              tf.io.FixedLenSequenceFeature(
                  shape=(self._action_spec.maximum - self._action_spec.minimum +
                         1),
                  dtype=tf.float32)
      }
    else:
      return {
          'NormalProjectionNetwork_scale':
              tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32),
          'NormalProjectionNetwork_loc':
              tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32)
      }

  def process_parsed_sequence_and_get_policy_info(
      self, parsed_sequence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if tensor_spec.is_discrete(self._action_spec):
      policy_info = {
          'dist_params': {
              'logits': parsed_sequence['CategoricalProjectionNetwork_logits']
          }
      }
      del parsed_sequence['CategoricalProjectionNetwork_logits']
    else:
      policy_info = {
          'dist_params': {
              'scale': parsed_sequence['NormalProjectionNetwork_scale'],
              'loc': parsed_sequence['NormalProjectionNetwork_loc']
          }
      }
      del parsed_sequence['NormalProjectionNetwork_scale']
      del parsed_sequence['NormalProjectionNetwork_loc']
    return policy_info


@gin.configurable(module='agents')
class DistributedPPOAgentConfig(PPOAgentConfig):
  """Distributed PPO/Reinforce agent configuration."""

  def _create_agent_implt(self, preprocessing_layers: tf.keras.layers.Layer,
                          policy_network: types.Network) -> tf_agent.TFAgent:
    """Creates a ppo_distributed agent."""
    actor_network = policy_network(
        self.time_step_spec.observation,
        self.action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Concatenate(),
        name='ActorDistributionNetwork')

    critic_network = constant_value_network.ConstantValueNetwork(
        self.time_step_spec.observation, name='ConstantValueNetwork')

    return distributed_ppo_agent.MLGOPPOAgent(
        self.time_step_spec,
        self.action_spec,
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4, epsilon=1e-5),
        actor_net=actor_network,
        value_net=critic_network,
        value_pred_loss_coef=0.0,
        entropy_regularization=0.01,
        importance_ratio_clipping=0.2,
        discount_factor=1.0,
        gradient_clipping=1.0,
        debug_summaries=False,
        value_clipping=None,
        aggregate_losses_across_replicas=True,
        loss_scaling_factor=1.0)
