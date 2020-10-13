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

import gin
import tensorflow as tf

from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent

from compiler_opt.rl import actor_behavioral_cloning_agent
from compiler_opt.rl import constant_value_network
from compiler_opt.rl import feature_ops


def _create_actor_behavioral_cloning_agent(time_step_spec, action_spec,
                                           policy_network):
  """Creates a actor behavioral_cloning_agent."""
  layers = tf.nest.map_structure(
      feature_ops.get_observation_processing_layer_creator(),
      time_step_spec.observation)

  network = policy_network(
      time_step_spec.observation,
      action_spec,
      preprocessing_layers=layers,
      name='ActorDistributionNetwork')

  return actor_behavioral_cloning_agent.ActorBCAgent(
      time_step_spec, action_spec, cloning_network=network, num_outer_dims=2)


def _create_behavioral_cloning_agent(time_step_spec, action_spec,
                                     policy_network):
  """Creates a behavioral_cloning_agent."""
  layers = tf.nest.map_structure(
      feature_ops.get_observation_processing_layer_creator(),
      time_step_spec.observation)

  network = policy_network(
      time_step_spec.observation,
      action_spec,
      preprocessing_layers=layers,
      name='QNetwork')

  return behavioral_cloning_agent.BehavioralCloningAgent(
      time_step_spec, action_spec, cloning_network=network, num_outer_dims=2)


def _create_dqn_agent(time_step_spec, action_spec, policy_network):
  """Creates a dqn_agent."""
  layers = tf.nest.map_structure(
      feature_ops.get_observation_processing_layer_creator(),
      time_step_spec.observation)

  network = policy_network(
      time_step_spec.observation,
      action_spec,
      preprocessing_layers=layers,
      name='QNetwork')

  return dqn_agent.DqnAgent(time_step_spec, action_spec, q_network=network)


def _create_ppo_agent(time_step_spec, action_spec, policy_network):
  """Creates a ppo_agent."""
  layers = tf.nest.map_structure(
      feature_ops.get_observation_processing_layer_creator(),
      time_step_spec.observation)

  actor_network = policy_network(
      time_step_spec.observation,
      action_spec,
      preprocessing_layers=layers,
      name='ActorDistributionNetwork')

  critic_network = constant_value_network.ConstantValueNetwork(
      time_step_spec.observation, name='ConstantValueNetwork')

  return ppo_agent.PPOAgent(
      time_step_spec,
      action_spec,
      actor_net=actor_network,
      value_net=critic_network)


@gin.configurable
def create_agent(agent_name=None,
                 time_step_spec=None,
                 action_spec=None,
                 policy_network=None):
  """Creates a tf_agents.agents.TFAgent object.

  Args:
    agent_name: str, name of the agent to create.
    time_step_spec: A `TimeStep` spec of the expected time_steps.
    action_spec: A nest of BoundedTensorSpec representing the actions.
    policy_network: A tf_agents.networks.Network class.

  Returns:
    tf_agent: A tf_agents.agents.TFAgent object.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert policy_network is not None
  assert agent_name is not None
  if agent_name == 'behavioral_cloning':
    return _create_behavioral_cloning_agent(time_step_spec, action_spec,
                                            policy_network)
  elif agent_name == 'actor_behavioral_cloning':
    return _create_actor_behavioral_cloning_agent(time_step_spec, action_spec,
                                                  policy_network)
  elif agent_name == 'dqn':
    return _create_dqn_agent(time_step_spec, action_spec, policy_network)
  elif agent_name == 'ppo':
    return _create_ppo_agent(time_step_spec, action_spec, policy_network)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))
