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
"""Tests for policy_utils."""

from absl.testing import absltest
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.trackable import autotrackable
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy, tf_policy

from compiler_opt.es import policy_utils
from compiler_opt.rl import policy_saver, registry
from compiler_opt.rl.inlining import config as inlining_config
from compiler_opt.rl.inlining import InliningConfig
from compiler_opt.rl.regalloc import config as regalloc_config
from compiler_opt.rl.regalloc import RegallocEvictionConfig, regalloc_network


# TODO(abenalaast): Issue #280
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

    # get saved model to test a loaded policy
    sm = tf.saved_model.load(policy_save_path + '/policy')
    self.assertIsInstance(sm, autotrackable.AutoTrackable)
    self.assertNotIsInstance(sm, tf_policy.TFPolicy)
    params = params[::-1]
    policy_utils.set_vectorized_parameters_for_policy(sm, params)
    val = length_of_a_perturbation - 1
    for variable in sm.model_variables:
      nums = variable.numpy().flatten()
      for num in nums:
        if val != num:
          raise AssertionError(
              f'values at index {length_of_a_perturbation - val} do not match')
        val -= 1

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

    # get saved model to test a loaded policy
    sm = tf.saved_model.load(policy_save_path + '/policy')
    self.assertIsInstance(sm, autotrackable.AutoTrackable)
    self.assertNotIsInstance(sm, tf_policy.TFPolicy)
    params = params[::-1]
    policy_utils.set_vectorized_parameters_for_policy(sm, params)
    # vectorize and check if the outcome is the same as the start
    output = policy_utils.get_vectorized_parameters_from_policy(sm)
    np.testing.assert_array_almost_equal(output, params)


if __name__ == '__main__':
  absltest.main()
