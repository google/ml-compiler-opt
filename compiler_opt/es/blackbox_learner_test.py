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

import tempfile
from absl.testing import absltest
import concurrent.futures

from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.es import blackbox_learner, blackbox_optimizers, es_worker
from compiler_opt.rl import corpus


class BlackboxLearnerTests(absltest.TestCase):
  """Tests for blackbox_learner"""

  learner_config = blackbox_learner.BlackboxLearnerConfig(
      total_steps=1,
      blackbox_optimizer='',
      est_type=blackbox_optimizers.EstimatorType.ANTITHETIC,
      fvalues_normalization=True,
      hyperparameters_update_method=blackbox_optimizers.UpdateMethod.NO_METHOD,
      num_top_directions=0,
      num_ir_repeats_within_worker=1,
      num_ir_repeats_across_worker=0,
      num_exact_evals=1,
      total_num_perturbations=3,
      precision_parameter=1,
      step_size=1.0)
  cps = corpus.create_corpus_for_testing(
      location=tempfile.gettempdir(),
      elements=[corpus.ModuleSpec(name='smth', size=1)],
      additional_flags=(),
      delete_flags=())
  learner = blackbox_learner.BlackboxLearner(
      blackbox_opt=blackbox_optimizers.MonteCarloBlackboxOptimizer(
          precision_parameter=1.0,
          est_type=blackbox_optimizers.EstimatorType.ANTITHETIC,
          normalize_fvalues=True,
          hyperparameters_update_method=blackbox_optimizers.UpdateMethod
          .NO_METHOD,
          extra_params=None,
          step_size=1),
      sampler=cps,
      tf_policy_path='',
      output_dir='',
      policy_saver_fn=lambda: 1,
      model_weights=[0, 0, 0],
      config=learner_config)

  def test_get_results(self):
    with local_worker_manager.LocalWorkerPoolManager(
        es_worker.ESWorker, count=3, arg='', kwarg='') as pool:
      self._samples = [[corpus.ModuleSpec(name='name1', size=1)],
                       [corpus.ModuleSpec(name='name2', size=1)],
                       [corpus.ModuleSpec(name='name3', size=1)]]
      perturbations = [b'00', b'01', b'10']
      results = BlackboxLearnerTests.learner._get_results(pool, perturbations)  # pylint: disable=protected-access
      self.assertListEqual([result.result() for result in results], [1, 1, 1])

  def test_get_rewards(self):
    f1 = concurrent.futures.Future()
    f1.set_exception(None)
    f2 = concurrent.futures.Future()
    f2.set_result(2)
    results = [f1, f2]
    rewards = BlackboxLearnerTests.learner._get_rewards(results)  # pylint: disable=protected-access
    self.assertEqual(rewards, [None, 2])

  def test_prune_skipped_perturbations(self):
    perturbations = [1, 2, 3, 4, 5]
    rewards = [1, None, 1, None, 1]
    blackbox_learner._prune_skipped_perturbations(perturbations, rewards)  # pylint: disable=protected-access
    self.assertListEqual(perturbations, [1, 3, 5])
