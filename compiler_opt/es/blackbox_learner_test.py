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
"""Tests for blackbox_learner"""

import os
from absl.testing import absltest
import cloudpickle
import gin
import numpy as np
import numpy.typing as npt
import pathlib
import tensorflow as tf
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy

# Pytype cannot pick up the pyi file for tensorflow.summary. Disable the error
# here as these errors are false positives.
# pytype: disable=pyi-error

from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.es import blackbox_learner
from compiler_opt.es import policy_utils
from compiler_opt.es import blackbox_optimizers
from compiler_opt.rl import corpus
from compiler_opt.rl import inlining
from compiler_opt.rl import policy_saver
from compiler_opt.rl import registry
from compiler_opt.rl.inlining import config as inlining_config
from compiler_opt.es import blackbox_evaluator
from compiler_opt.es import blackbox_test_utils
from compiler_opt.es.perturbations import Perturbations


class BlackboxLearnerTests(absltest.TestCase):
  """Tests for blackbox_learner"""

  def tearDown(self):
    gin.clear_config()

  def setUp(self):
    super().setUp()

    gin.bind_parameter('SamplingBlackboxEvaluator.total_num_perturbations', 3)
    gin.bind_parameter('SamplingBlackboxEvaluator.num_ir_repeats_within_worker',
                       1)

    self._learner_config = blackbox_learner.BlackboxLearnerConfig(
        total_steps=1,
        blackbox_optimizer=blackbox_optimizers.Algorithm.MONTE_CARLO,
        estimator_type=blackbox_optimizers.EstimatorType.ANTITHETIC,
        fvalues_normalization=True,
        hyperparameters_update_method=blackbox_optimizers.UpdateMethod
        .NO_METHOD,
        num_top_directions=0,
        evaluator=blackbox_evaluator.SamplingBlackboxEvaluator,
        total_num_perturbations=3,
        precision_parameter=1,
        step_size=1.0,
        save_best_policy=True)

    self._cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir().full_path,
        elements=[
            corpus.ModuleSpec(name='smth', size=1, command_line=('-cc1',))
        ],
        additional_flags=(),
        delete_flags=())

    output_dir = self.create_tempdir()
    policy_name = 'policy_name'

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

    # make the policy all zeros to be deterministic
    expected_policy_length = 17154
    policy_utils.set_vectorized_parameters_for_policy(policy, [0.0] *
                                                      expected_policy_length)
    init_params = policy_utils.get_vectorized_parameters_from_policy(policy)

    # save the policy
    saver = policy_saver.MLGOPolicySaver({policy_name: policy})
    policy_save_path = os.path.join(output_dir.full_path, 'temp_output',
                                    'policy')
    saver.save(policy_save_path)

    self._iteration_policies_path = os.path.join(output_dir.full_path,
                                                 'policies')
    # The directory should be unique per test and thus should not exist
    # before we create it. Raise an error otherwise.
    if os.path.exists(self._iteration_policies_path):
      raise ValueError('Test directory already exists.')
    os.mkdir(self._iteration_policies_path)

    def _policy_saver_fn(parameters: npt.NDArray[np.float32],
                         policy_name: str) -> None:
      if parameters is not None and policy_name:
        pathlib.Path(os.path.join(self._iteration_policies_path,
                                  policy_name)).touch()
        return None
      return None

    self._learner = blackbox_learner.BlackboxLearner(
        blackbox_opt=blackbox_optimizers.MonteCarloBlackboxOptimizer(
            precision_parameter=1.0,
            estimator_type=blackbox_optimizers.EstimatorType.FORWARD_FD,
            normalize_fvalues=True,
            hyperparameters_update_method=blackbox_optimizers.UpdateMethod
            .NO_METHOD,
            extra_params=None,
            step_size=1),
        train_corpus=self._cps,
        output_dir=output_dir.full_path,
        policy_saver_fn=_policy_saver_fn,
        model_weights=init_params,
        config=self._learner_config,
        perturbation_scale_vector=np.ones_like(init_params, dtype=np.float32),
        seed=17)

  def test_get_perturbations(self):
    # test values generated with seed=17
    perturbations = (
        self._learner._perturbations_generator.get_next_perturbations())  # pylint: disable=protected-access
    rng = np.random.default_rng(seed=17)
    for i in range(0, len(perturbations), 2):
      p_raw = rng.normal(size=len(self._learner.get_model_weights()))
      np.testing.assert_array_almost_equal(perturbations[i], p_raw)
      np.testing.assert_array_almost_equal(perturbations[i + 1], -p_raw)

  def test_prune_skipped_perturbations(self):
    perturbations = [1, 2, 3, 4, 5]
    rewards = [1, None, 1, None, 1]
    # Use a dummy helper with is_antithetic=False
    helper = Perturbations(
        dimension=3,
        precision_parameter=1.0,
        perturbation_scale_vector=np.ones(3),
        total_num_perturbations=5,
        is_antithetic=False)
    pruned_p, pruned_r = helper.prune_skipped_perturbations(
        perturbations, rewards)
    self.assertListEqual(pruned_p, [1, 3, 5])
    self.assertListEqual(pruned_r, [1, 1, 1])

  def test_prune_skipped_perturbations_antithetic(self):
    # Antithetic pairs are structured as [p0, -p0, p1, -p1, p2, -p2]
    # Total 6 elements.
    perturbations = [1, 2, 3, 4, 5, 6]

    # Case 1: Prune second element of the first pair (index 1) -> both index 0
    # and 1 should be pruned.
    rewards = [1.0, None, 1.0, 1.0, 1.0, 1.0]
    # pylint: disable=protected-access
    pruned_p, pruned_r = (
        self._learner._perturbations_generator.prune_skipped_perturbations(
            perturbations, rewards))
    self.assertListEqual(pruned_p, [3, 4, 5, 6])
    self.assertListEqual(pruned_r, [1.0, 1.0, 1.0, 1.0])

    # Case 2: Prune first element of the second pair (index 2) -> index 2 and 3
    # (first pair is index 0 and 1, second is 2 and 3).
    perturbations = [1, 2, 3, 4, 5, 6]
    rewards = [1.0, 1.0, None, 1.0, 1.0, 1.0]
    pruned_p, pruned_r = (
        self._learner._perturbations_generator.prune_skipped_perturbations(
            perturbations, rewards))
    # pylint: enable=protected-access
    self.assertListEqual(pruned_p, [1, 2, 5, 6])
    self.assertListEqual(pruned_r, [1.0, 1.0, 1.0, 1.0])

  def test_run_step(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESWorker,
        count=3,
        pickle_func=cloudpickle.dumps,
        worker_args=(),
        worker_kwargs={}) as pool:
      self._learner.set_baseline(pool)
      self._learner.run_step(pool)  # pylint: disable=protected-access
      # expected length calculated from expected shapes of variables
      self.assertEqual(len(self._learner.get_model_weights()), 17154)
      # check that first 5 weights are not all zero
      # this will indicate general validity of all the values
      for value in self._learner.get_model_weights()[:5]:
        self.assertNotAlmostEqual(value, 0.0)

      # Normally the models would be saved asynchronously while
      # blackbox_learner waits for compilation results. Flush them explicitly
      # here so we can see the model.
      self._learner.flush_models()
      self.assertIn('iteration0', os.listdir(self._iteration_policies_path))

  def test_save_best_model(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESWorker,
        count=1,
        pickle_func=cloudpickle.dumps,
        worker_args=(),
        worker_kwargs={
            'delta': -1.0,
            'initial_value': 5
        }) as pool:
      self._learner.set_baseline(pool)
      self._learner.run_step(pool)
      self._learner.run_step(pool)
      # Check the policy from step zero since it will be flushed in step one.
      self.assertIn('best_policy_1.01_step_0',
                    os.listdir(self._iteration_policies_path))
      # Manually flush the model since we are not going to run another step.
      self._learner.flush_models()
      self.assertIn('best_policy_1.07_step_1',
                    os.listdir(self._iteration_policies_path))

  def test_save_best_model_only_saves_best(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESWorker,
        count=1,
        pickle_func=cloudpickle.dumps,
        worker_args=(),
        worker_kwargs={
            'delta': 1.0,
            'initial_value': 5
        }) as pool:
      self._learner.set_baseline(pool)
      self._learner.run_step(pool)

      self._learner.run_step(pool)
      # CHeck the policy from step zero since it will be flushed in step one.
      self.assertIn('best_policy_0.94_step_0',
                    os.listdir(self._iteration_policies_path))
      # Check that the within the next step we only get a new iteration
      # policy and do not save any new best.
      current_policies_count = len(os.listdir(self._iteration_policies_path))
      # Flush the policies since we are not going to run another step.
      self._learner.flush_models()
      self.assertLen(
          os.listdir(self._iteration_policies_path), current_policies_count + 1)
