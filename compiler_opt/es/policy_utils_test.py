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

import os

from absl.testing import absltest
import numpy as np
import tensorflow as tf
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy
from tf_agents.policies import tf_policy

from compiler_opt.es import policy_utils
from compiler_opt.rl import inlining
from compiler_opt.rl import policy_saver
from compiler_opt.rl import regalloc
from compiler_opt.rl import registry
from compiler_opt.rl.inlining import config as inlining_config
from compiler_opt.rl.regalloc import config as regalloc_config
from compiler_opt.rl.regalloc import regalloc_network


class ConfigTest(absltest.TestCase):

  # TODO(abenalaast): Issue #280
  def test_inlining_config(self):
    problem_config = registry.get_configuration(
        implementation=inlining.InliningConfig)
    time_step_spec, action_spec = problem_config.get_signature_spec()
    quantile_file_dir = os.path.join('compiler_opt', 'rl', 'inlining', 'vocab')
    creator = inlining_config.get_observation_processing_layer_creator(
        quantile_file_dir=quantile_file_dir,
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

  # TODO(abenalaast): Issue #280
  def test_regalloc_config(self):
    problem_config = registry.get_configuration(
        implementation=regalloc.RegallocEvictionConfig)
    time_step_spec, action_spec = problem_config.get_signature_spec()
    quantile_file_dir = os.path.join('compiler_opt', 'rl', 'regalloc', 'vocab')
    creator = regalloc_config.get_observation_processing_layer_creator(
        quantile_file_dir=quantile_file_dir,
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

  expected_variable_shapes = [(71, 64), (64), (64, 64), (64), (64, 64), (64),
                              (64, 64), (64), (64, 2), (2)]
  expected_length_of_a_perturbation = sum(
      np.prod(shape) for shape in expected_variable_shapes)
  params = np.arange(expected_length_of_a_perturbation, dtype=np.float32)
  POLICY_NAME = 'test_policy_name'

  # TODO(abenalaast): Issue #280
  def test_set_vectorized_parameters_for_policy(self):
    # create a policy
    problem_config = registry.get_configuration(
        implementation=inlining.InliningConfig)
    time_step_spec, action_spec = problem_config.get_signature_spec()
    quantile_file_dir = os.path.join('compiler_opt', 'rl', 'inlining', 'vocab')
    creator = inlining_config.get_observation_processing_layer_creator(
        quantile_file_dir=quantile_file_dir,
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

    # save the policy
    saver = policy_saver.PolicySaver({VectorTest.POLICY_NAME: policy})
    testing_path = self.create_tempdir()
    policy_save_path = os.path.join(testing_path, 'temp_output', 'policy')
    saver.save(policy_save_path)

    # set the values of the policy variables
    policy_utils.set_vectorized_parameters_for_policy(policy, VectorTest.params)
    # iterate through variables and check their shapes and values
    # deep copy params in order to destructively iterate over values
    expected_values = [*VectorTest.params]
    for i, variable in enumerate(policy.variables()):  # pylint: disable=not-callable
      self.assertEqual(variable.shape, VectorTest.expected_variable_shapes[i])
      variable_values = variable.numpy().flatten()
      np.testing.assert_array_almost_equal(
          expected_values[:len(variable_values)], variable_values)
      expected_values = expected_values[len(variable_values):]
    # all values in the copy should have been removed at this point
    self.assertEmpty(expected_values)

    # get saved model to test a loaded policy
    load_path = os.path.join(policy_save_path, VectorTest.POLICY_NAME)
    sm = tf.saved_model.load(load_path)
    self.assertNotIsInstance(sm, tf_policy.TFPolicy)
    policy_utils.set_vectorized_parameters_for_policy(sm, VectorTest.params)
    # deep copy params in order to destructively iterate over values
    expected_values = [*VectorTest.params]
    for i, variable in enumerate(sm.model_variables):
      self.assertEqual(variable.shape, VectorTest.expected_variable_shapes[i])
      variable_values = variable.numpy().flatten()
      np.testing.assert_array_almost_equal(
          expected_values[:len(variable_values)], variable_values)
      expected_values = expected_values[len(variable_values):]
    # all values in the copy should have been removed at this point
    self.assertEmpty(expected_values)

  # TODO(abenalaast): Issue #280
  def test_get_vectorized_parameters_from_policy(self):
    # create a policy
    problem_config = registry.get_configuration(
        implementation=inlining.InliningConfig)
    time_step_spec, action_spec = problem_config.get_signature_spec()
    quantile_file_dir = os.path.join('compiler_opt', 'rl', 'inlining', 'vocab')
    creator = inlining_config.get_observation_processing_layer_creator(
        quantile_file_dir=quantile_file_dir,
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

    # save the policy
    saver = policy_saver.PolicySaver({VectorTest.POLICY_NAME: policy})
    testing_path = self.create_tempdir()
    policy_save_path = os.path.join(testing_path, 'temp_output', 'policy')
    saver.save(policy_save_path)

    # functionality verified in previous test
    policy_utils.set_vectorized_parameters_for_policy(policy, VectorTest.params)
    # vectorize and check if the outcome is the same as the start
    output = policy_utils.get_vectorized_parameters_from_policy(policy)
    np.testing.assert_array_almost_equal(output, VectorTest.params)

    # get saved model to test a loaded policy
    load_path = os.path.join(policy_save_path, VectorTest.POLICY_NAME)
    sm = tf.saved_model.load(load_path)
    self.assertNotIsInstance(sm, tf_policy.TFPolicy)
    policy_utils.set_vectorized_parameters_for_policy(sm, VectorTest.params)
    # vectorize and check if the outcome is the same as the start
    output = policy_utils.get_vectorized_parameters_from_policy(sm)
    np.testing.assert_array_almost_equal(output, VectorTest.params)

  # TODO(abenalaast): Issue #280
  def test_tfpolicy_and_loaded_policy_produce_same_variable_order(self):
    # create a policy
    problem_config = registry.get_configuration(
        implementation=inlining.InliningConfig)
    time_step_spec, action_spec = problem_config.get_signature_spec()
    quantile_file_dir = os.path.join('compiler_opt', 'rl', 'inlining', 'vocab')
    creator = inlining_config.get_observation_processing_layer_creator(
        quantile_file_dir=quantile_file_dir,
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

    # save the policy
    saver = policy_saver.PolicySaver({VectorTest.POLICY_NAME: policy})
    testing_path = self.create_tempdir()
    policy_save_path = os.path.join(testing_path, 'temp_output', 'policy')
    saver.save(policy_save_path)

    # set the values of the variables
    policy_utils.set_vectorized_parameters_for_policy(policy, VectorTest.params)
    # save the changes
    saver.save(policy_save_path)
    # vectorize the tfpolicy
    tf_params = policy_utils.get_vectorized_parameters_from_policy(policy)

    # get loaded policy
    load_path = os.path.join(policy_save_path, VectorTest.POLICY_NAME)
    sm = tf.saved_model.load(load_path)
    # vectorize the loaded policy
    loaded_params = policy_utils.get_vectorized_parameters_from_policy(sm)

    # assert that they result in the same order of values
    np.testing.assert_array_almost_equal(tf_params, loaded_params)


if __name__ == '__main__':
  absltest.main()
