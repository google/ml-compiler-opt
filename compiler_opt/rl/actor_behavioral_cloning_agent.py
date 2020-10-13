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

"""BehavioralCloningAgent for Actor policies/networks."""

import gin
import tensorflow as tf
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.policies import actor_policy


@gin.configurable
class ActorBCAgent(behavioral_cloning_agent.BehavioralCloningAgent):
  """BehavioralCloningAgent for Actor policies/networks."""

  def _get_policies(self, time_step_spec, action_spec, cloning_network):
    policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=cloning_network,
        clip=True)

    return policy, policy


@gin.configurable
def loss_fn(dist, action, alpha, loss_function):
  prob = dist.probs_parameter()
  onehot_target = tf.one_hot(action, depth=2)
  target = alpha * onehot_target + (1. - alpha) * (1. - onehot_target)
  loss = loss_function(target, prob)
  return loss
