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
"""Tests for blackbox_learner"""

import os
from absl.testing import absltest
import gin
import tempfile
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy

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


class BlackboxLearnerTests(absltest.TestCase):
  """Tests for blackbox_learner"""

  def tearDown(self):
    gin.clear_config()

  def setUp(self):
    super().setUp()

    gin.bind_parameter('SamplingBlackboxEvaluator.total_num_perturbations', 5)
    gin.bind_parameter('SamplingBlackboxEvaluator.num_ir_repeats_within_worker',
                       5)

    self._learner_config = blackbox_learner.BlackboxLearnerConfig(
        total_steps=1,
        blackbox_optimizer=blackbox_optimizers.Algorithm.MONTE_CARLO,
        est_type=blackbox_optimizers.EstimatorType.ANTITHETIC,
        fvalues_normalization=True,
        hyperparameters_update_method=blackbox_optimizers.UpdateMethod
        .NO_METHOD,
        num_top_directions=0,
        evaluator=blackbox_evaluator.SamplingBlackboxEvaluator,
        total_num_perturbations=3,
        precision_parameter=1,
        step_size=1.0)

    self._cps = corpus.create_corpus_for_testing(
        location=tempfile.gettempdir(),
        elements=[corpus.ModuleSpec(name='smth', size=1)],
        additional_flags=(),
        delete_flags=())

    output_dir = tempfile.gettempdir()
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
    saver = policy_saver.PolicySaver({policy_name: policy})
    policy_save_path = os.path.join(output_dir, 'temp_output', 'policy')
    saver.save(policy_save_path)

    def _policy_saver_fn(parameters: npt.NDArray[np.float32],
                         policy_name: str) -> None:
      if parameters is not None and policy_name:
        return None
      return None

    self._learner = blackbox_learner.BlackboxLearner(
        blackbox_opt=blackbox_optimizers.MonteCarloBlackboxOptimizer(
            precision_parameter=1.0,
            est_type=blackbox_optimizers.EstimatorType.ANTITHETIC,
            normalize_fvalues=True,
            hyperparameters_update_method=blackbox_optimizers.UpdateMethod
            .NO_METHOD,
            extra_params=None,
            step_size=1),
        train_corpus=self._cps,
        tf_policy_path=os.path.join(policy_save_path, policy_name),
        output_dir=output_dir,
        policy_saver_fn=_policy_saver_fn,
        model_weights=init_params,
        config=self._learner_config,
        seed=17)

  def test_get_perturbations(self):
    # test values generated with seed=17
    perturbations = self._learner._get_perturbations()  # pylint: disable=protected-access
    rng = np.random.default_rng(seed=17)
    for perturbation in perturbations:
      for value in perturbation:
        self.assertAlmostEqual(value, rng.normal())

  def test_prune_skipped_perturbations(self):
    perturbations = [1, 2, 3, 4, 5]
    rewards = [1, None, 1, None, 1]
    blackbox_learner._prune_skipped_perturbations(perturbations, rewards)  # pylint: disable=protected-access
    self.assertListEqual(perturbations, [1, 3, 5])

  def test_run_step(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESWorker, count=3, arg='', kwarg='') as pool:
      self._learner.run_step(pool)  # pylint: disable=protected-access
      # expected length calculated from expected shapes of variables
      self.assertEqual(len(self._learner.get_model_weights()), 17154)
      # check that first 5 weights are not all zero
      # this will indicate general validity of all the values
      for value in self._learner.get_model_weights()[:5]:
        self.assertNotAlmostEqual(value, 0.0)
