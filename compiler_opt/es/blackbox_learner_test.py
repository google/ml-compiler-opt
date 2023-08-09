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

from absl.testing import absltest
import concurrent.futures
import gin
import tempfile
from typing import List
import numpy as np
import numpy.typing as npt

from compiler_opt.distributed import worker
from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.es import blackbox_learner
from compiler_opt.es import blackbox_optimizers
from compiler_opt.rl import corpus


@gin.configurable
class ESWorker(worker.Worker):
  """Temporary placeholder worker.
  Each time a worker is called, the function value
  it will return increases."""

  def __init__(self, arg, *, kwarg):
    self._arg = arg
    self._kwarg = kwarg
    self.function_value = 0.0

  def compile(self, policy: bytes, samples: List[corpus.ModuleSpec]) -> float:
    if policy and samples:
      self.function_value += 1.0
      return self.function_value
    else:
      return 0.0


class BlackboxLearnerTests(absltest.TestCase):
  """Tests for blackbox_learner"""

  def setUp(self):
    super().setUp()

    self._learner_config = blackbox_learner.BlackboxLearnerConfig(
        total_steps=1,
        blackbox_optimizer='',
        est_type=blackbox_optimizers.EstimatorType.ANTITHETIC,
        fvalues_normalization=True,
        hyperparameters_update_method=blackbox_optimizers.UpdateMethod
        .NO_METHOD,
        num_top_directions=0,
        num_ir_repeats_within_worker=1,
        num_ir_repeats_across_worker=0,
        num_exact_evals=1,
        total_num_perturbations=3,
        precision_parameter=1,
        step_size=1.0)

    self._cps = corpus.create_corpus_for_testing(
        location=tempfile.gettempdir(),
        elements=[corpus.ModuleSpec(name='smth', size=1)],
        additional_flags=(),
        delete_flags=())

    def _policy_saver_fn(parameters: npt.NDArray[np.float32],
                         policy_name: str) -> None:
      if parameters and policy_name:
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
        sampler=self._cps,
        tf_policy_path='',
        output_dir='',
        policy_saver_fn=_policy_saver_fn,
        model_weights=[0, 0, 0],
        config=self._learner_config)

  def test_get_perturbations(self):
    # values generated with seed=17
    expected_perturbations = [
        [1.101262453505847, 0.3384312766461778, -0.5399715152535035],
        [-1.2602418568524327, -1.8946212698392553, 0.018638290983285614],
        [-0.8105670995116028, -0.8721559599345132, -0.22196950708389104]
    ]
    actual_perturbations = self._learner._get_perturbations(seed=17)  # pylint: disable=protected-access
    for i in range(len(expected_perturbations)):
      self.assertSequenceAlmostEqual(expected_perturbations[i],
                                     actual_perturbations[i])

  def test_get_results(self):
    with local_worker_manager.LocalWorkerPoolManager(
        ESWorker, count=3, arg='', kwarg='') as pool:
      self._samples = [[corpus.ModuleSpec(name='name1', size=1)],
                       [corpus.ModuleSpec(name='name2', size=1)],
                       [corpus.ModuleSpec(name='name3', size=1)]]
      perturbations = [b'00', b'01', b'10']
      results = self._learner._get_results(pool, perturbations)  # pylint: disable=protected-access
      self.assertListEqual([result.result() for result in results], [1, 1, 1])

  def test_get_rewards(self):
    f1 = concurrent.futures.Future()
    f1.set_exception(None)
    f2 = concurrent.futures.Future()
    f2.set_result(2)
    results = [f1, f2]
    rewards = self._learner._get_rewards(results)  # pylint: disable=protected-access
    self.assertEqual(rewards, [None, 2])

  def test_prune_skipped_perturbations(self):
    perturbations = [1, 2, 3, 4, 5]
    rewards = [1, None, 1, None, 1]
    blackbox_learner._prune_skipped_perturbations(perturbations, rewards)  # pylint: disable=protected-access
    self.assertListEqual(perturbations, [1, 3, 5])
