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
"""Util functions to create and edit a tf_agent policy."""

import gin
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from typing import Union

from tf_agents.networks import network
from tf_agents.policies import actor_policy, greedy_policy, tf_policy
from compiler_opt.rl import policy_saver, registry


@gin.configurable(module='policy_utils')
def create_actor_policy(actor_network_ctor: network.DistributionNetwork,
                        greedy: bool = False) -> tf_policy.TFPolicy:
  """Creates an actor policy."""
  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()
  layers = tf.nest.map_structure(
      problem_config.get_preprocessing_layer_creator(),
      time_step_spec.observation)

  actor_network = actor_network_ctor(
      input_tensor_spec=time_step_spec.observation,
      output_tensor_spec=action_spec,
      preprocessing_layers=layers)

  policy = actor_policy.ActorPolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=actor_network)

  if greedy:
    policy = greedy_policy.GreedyPolicy(policy)

  return policy


def get_vectorized_parameters_from_policy(
    policy: Union[tf_policy.TFPolicy, tf.Module]) -> npt.NDArray[np.float32]:
  if isinstance(policy, tf_policy.TFPolicy):
    variables = policy.variables()
  elif policy.model_variables:
    variables = policy.model_variables

  parameters = [var.numpy().flatten() for var in variables]
  parameters = np.concatenate(parameters, axis=0)
  return parameters


def set_vectorized_parameters_for_policy(
    policy: Union[tf_policy.TFPolicy,
                  tf.Module], parameters: npt.NDArray[np.float32]) -> None:
  if isinstance(policy, tf_policy.TFPolicy):
    variables = policy.variables()
  else:
    try:
      getattr(policy, 'model_variables')
    except AttributeError as e:
      raise TypeError('policy must be a TFPolicy or a loaded SavedModel') from e
    variables = policy.model_variables

  param_pos = 0
  for variable in variables:
    shape = tf.shape(variable).numpy()
    num_ele = np.prod(shape)
    param = np.reshape(parameters[param_pos:param_pos + num_ele], shape)
    variable.assign(param)
    param_pos += num_ele
  if param_pos != len(parameters):
    raise ValueError(
        f'Parameter dimensions are not matched! Expected {len(parameters)} '
        'but only found {param_pos}.')


def save_policy(policy: tf_policy.TFPolicy, parameters: npt.NDArray[np.float32],
                save_folder: str, policy_name: str) -> None:
  set_vectorized_parameters_for_policy(policy, parameters)
  saver = policy_saver.PolicySaver({policy_name: policy})
  saver.save(save_folder)
