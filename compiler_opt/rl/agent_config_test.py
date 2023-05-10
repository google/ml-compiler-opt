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
"""Tests for compiler_opt.rl.agent_config."""

import gin
import tensorflow as tf
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

from compiler_opt.rl import agent_config


def _observation_processing_layer(obs_spec):
  del obs_spec

  def identity(obs):
    return tf.expand_dims(tf.cast(obs, tf.float32), -1)

  return tf.keras.layers.Lambda(identity)


class AgentCreatorsTest(tf.test.TestCase):

  def setUp(self):
    observation_spec = tf.TensorSpec(
        dtype=tf.int64, shape=(), name='callee_users')
    self._time_step_spec = time_step.time_step_spec(observation_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int64,
        shape=(),
        minimum=0,
        maximum=1,
        name='inlining_decision')
    super().setUp()

  def test_create_behavioral_cloning_agent(self):
    gin.bind_parameter('create_agent.policy_network', q_network.QNetwork)
    gin.bind_parameter('BehavioralCloningAgent.optimizer',
                       tf.compat.v1.train.AdamOptimizer())
    tf_agent = agent_config.create_agent(
        agent_config.BCAgentConfig(
            time_step_spec=self._time_step_spec, action_spec=self._action_spec),
        preprocessing_layer_creator=_observation_processing_layer)
    self.assertIsInstance(tf_agent,
                          behavioral_cloning_agent.BehavioralCloningAgent)

  def test_create_dqn_agent(self):
    gin.bind_parameter('create_agent.policy_network', q_network.QNetwork)
    gin.bind_parameter('DqnAgent.optimizer', tf.compat.v1.train.AdamOptimizer())
    tf_agent = agent_config.create_agent(
        agent_config.DQNAgentConfig(
            time_step_spec=self._time_step_spec, action_spec=self._action_spec),
        preprocessing_layer_creator=_observation_processing_layer)
    self.assertIsInstance(tf_agent, dqn_agent.DqnAgent)

  def test_create_ppo_agent(self):
    gin.bind_parameter('create_agent.policy_network',
                       actor_distribution_network.ActorDistributionNetwork)
    gin.bind_parameter('PPOAgent.optimizer', tf.compat.v1.train.AdamOptimizer())
    tf_agent = agent_config.create_agent(
        agent_config.PPOAgentConfig(
            time_step_spec=self._time_step_spec, action_spec=self._action_spec),
        preprocessing_layer_creator=_observation_processing_layer)
    self.assertIsInstance(tf_agent, ppo_agent.PPOAgent)


if __name__ == '__main__':
  tf.test.main()
