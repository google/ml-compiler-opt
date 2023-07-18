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
"""Tests for policy_utils."""

from absl.testing import absltest
import numpy as np
import os
import tensorflow as tf
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy

from compiler_opt.es import policy_utils
from compiler_opt.rl import policy_saver, registry
from compiler_opt.rl.inlining import InliningConfig
from compiler_opt.rl.inlining import config as inlining_config
from compiler_opt.rl.regalloc import config as regalloc_config
from compiler_opt.rl.regalloc import RegallocEvictionConfig, regalloc_network


class ConfigTest(absltest.TestCase):

  def test_inlining_config(self):
    problem_config = registry.get_configuration(implementation=InliningConfig)
    time_step_spec, action_spec = problem_config.get_signature_spec()
    creator = inlining_config.get_observation_processing_layer_creator(
        quantile_file_dir='compiler_opt/rl/inlining/vocab/',
        with_sqrt=False,
        with_z_score_normalization=False)
    layers = tf.nest.map_structure(creator, time_step_spec.observation)

    actor_network = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=time_step_spec.observation,
        output_tensor_spec=action_spec,
        preprocessing_layers=layers,
        preprocessing_combiner=tf.keras.layers.Concatenate(),
        fc_layer_params=(64, 64, 64, 64),
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu)

    policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_network)

    self.assertIsNotNone(policy)
    self.assertIsInstance(
        policy._actor_network,  # pylint: disable=protected-access
        actor_distribution_network.ActorDistributionNetwork)

  def test_regalloc_config(self):
    problem_config = registry.get_configuration(
        implementation=RegallocEvictionConfig)
    time_step_spec, action_spec = problem_config.get_signature_spec()
    creator = regalloc_config.get_observation_processing_layer_creator(
        quantile_file_dir='compiler_opt/rl/regalloc/vocab',
        with_sqrt=False,
        with_z_score_normalization=False)
    layers = tf.nest.map_structure(creator, time_step_spec.observation)

    actor_network = regalloc_network.RegAllocNetwork(
        input_tensor_spec=time_step_spec.observation,
        output_tensor_spec=action_spec,
        preprocessing_layers=layers,
        preprocessing_combiner=tf.keras.layers.Concatenate(),
        fc_layer_params=(64, 64, 64, 64),
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu)

    policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_network)

    self.assertIsNotNone(policy)
    self.assertIsInstance(
        policy._actor_network,  # pylint: disable=protected-access
        regalloc_network.RegAllocNetwork)


class VectorTest(absltest.TestCase):

  def test_set_vectorized_parameters_for_policy(self):
    # create a policy
    problem_config = registry.get_configuration(implementation=InliningConfig)
    time_step_spec, action_spec = problem_config.get_signature_spec()
    creator = inlining_config.get_observation_processing_layer_creator(
        quantile_file_dir='compiler_opt/rl/inlining/vocab/',
        with_sqrt=False,
        with_z_score_normalization=False)
    layers = tf.nest.map_structure(creator, time_step_spec.observation)

    actor_network = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=time_step_spec.observation,
        output_tensor_spec=action_spec,
        preprocessing_layers=layers,
        preprocessing_combiner=tf.keras.layers.Concatenate(),
        fc_layer_params=(64, 64, 64, 64),
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu)

    policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_network)
    saver = policy_saver.PolicySaver({'policy': policy})

    # save the policy
    testing_path = self.create_tempdir()
    policy_save_path = os.path.join(testing_path, 'temp_output/policy')
    saver.save(policy_save_path)

    # set the values of the policy variables
    length_of_a_perturbation = 17218
    params = np.arange(length_of_a_perturbation, dtype=np.float32)
    policy_utils.set_vectorized_parameters_for_policy(policy, params)
    # iterate through variables and check their values
    idx = 0
    for variable in policy.variables():  # pylint: disable=not-callable
      nums = variable.numpy().flatten()
      for num in nums:
        if idx != num:
          raise AssertionError(f'values at index {idx} do not match')
        idx += 1

  def test_get_vectorized_parameters_from_policy(self):
    # create a policy
    problem_config = registry.get_configuration(implementation=InliningConfig)
    time_step_spec, action_spec = problem_config.get_signature_spec()
    creator = inlining_config.get_observation_processing_layer_creator(
        quantile_file_dir='compiler_opt/rl/inlining/vocab/',
        with_sqrt=False,
        with_z_score_normalization=False)
    layers = tf.nest.map_structure(creator, time_step_spec.observation)

    actor_network = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=time_step_spec.observation,
        output_tensor_spec=action_spec,
        preprocessing_layers=layers,
        preprocessing_combiner=tf.keras.layers.Concatenate(),
        fc_layer_params=(64, 64, 64, 64),
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu)

    policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_network)
    saver = policy_saver.PolicySaver({'policy': policy})

    # save the policy
    testing_path = self.create_tempdir()
    policy_save_path = os.path.join(testing_path, 'temp_output/policy')
    saver.save(policy_save_path)

    length_of_a_perturbation = 17218
    params = np.arange(length_of_a_perturbation, dtype=np.float32)
    # functionality verified in previous test
    policy_utils.set_vectorized_parameters_for_policy(policy, params)
    # vectorize and check if the outcome is the same as the start
    output = policy_utils.get_vectorized_parameters_from_policy(policy)
    np.testing.assert_array_almost_equal(output, params)


if __name__ == '__main__':
  absltest.main()
