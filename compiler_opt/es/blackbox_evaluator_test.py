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
"""Tests for BlackboxEvaluator."""

import concurrent.futures

from absl.testing import absltest

from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.rl import compilation_runner, corpus
from compiler_opt.es import blackbox_test_utils
from compiler_opt.es import blackbox_evaluator
from compiler_opt.es import blackbox_optimizers


class BlackboxEvaluatorTests(absltest.TestCase):
  """Tests for BlackboxEvaluator."""

  def test_sampling_get_results(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESWorker, count=3, worker_args=(),
        worker_kwargs={}) as pool:
      perturbations = [b'00', b'01', b'10']
      evaluator = blackbox_evaluator.SamplingBlackboxEvaluator(
          None, blackbox_optimizers.EstimatorType.FORWARD_FD, 5, None)
      # pylint: disable=protected-access
      evaluator._samples = [[corpus.ModuleSpec(name='name1', size=1)],
                            [corpus.ModuleSpec(name='name2', size=1)],
                            [corpus.ModuleSpec(name='name3', size=1)]]
      # pylint: enable=protected-access
      results = evaluator.get_results(pool, perturbations)
      self.assertSequenceAlmostEqual([result.result() for result in results],
                                     [1.0, 1.0, 1.0])

  def test_sampling_set_baseline(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESWorker, count=1, worker_args=(),
        worker_kwargs={}) as pool:
      test_corpus = corpus.create_corpus_for_testing(
          location=self.create_tempdir().full_path,
          elements=[corpus.ModuleSpec(name='name1', size=1)])
      evaluator = blackbox_evaluator.SamplingBlackboxEvaluator(
          test_corpus, blackbox_optimizers.EstimatorType.FORWARD_FD, 1, 1)

      evaluator.set_baseline(pool)
      # pylint: disable=protected-access
      self.assertAlmostEqual(evaluator._baselines, [0])

  def test_sampling_get_rewards_without_baseline(self):
    evaluator = blackbox_evaluator.SamplingBlackboxEvaluator(
        None, blackbox_optimizers.EstimatorType.FORWARD_FD, 5, None)
    self.assertRaises(RuntimeError, evaluator.get_rewards, None)

  def test_sampling_get_rewards_with_baseline(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESWorker, count=1, worker_args=(),
        worker_kwargs={}) as pool:
      test_corpus = corpus.create_corpus_for_testing(
          location=self.create_tempdir().full_path,
          elements=[corpus.ModuleSpec(name='name1', size=1)])
      evaluator = blackbox_evaluator.SamplingBlackboxEvaluator(
          test_corpus, blackbox_optimizers.EstimatorType.FORWARD_FD, 2, 1)

      evaluator.set_baseline(pool)

      f_policy1 = concurrent.futures.Future()
      f_policy1.set_result(1.5)
      f_policy2 = concurrent.futures.Future()
      f_policy2.set_result(0.5)
      policy_results = [f_policy1, f_policy2]

      rewards = evaluator.get_rewards(policy_results)
      expected_rewards = [
          compilation_runner.calculate_reward(1.5, 0.0),
          compilation_runner.calculate_reward(0.5, 0.0)
      ]
      self.assertSequenceAlmostEqual(rewards, expected_rewards)

  def test_trace_get_results(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESTraceWorker,
        count=3,
        worker_args=(),
        worker_kwargs={}) as pool:
      perturbations = [b'00', b'01', b'10']
      test_corpus = corpus.create_corpus_for_testing(
          location=self.create_tempdir().full_path,
          elements=[corpus.ModuleSpec(name='name1', size=1)])
      evaluator = blackbox_evaluator.TraceBlackboxEvaluator(
          test_corpus, blackbox_optimizers.EstimatorType.FORWARD_FD,
          'fake_bb_trace_path', 'fake_function_index_path')
      results = evaluator.get_results(pool, perturbations)
      self.assertSequenceAlmostEqual([result.result() for result in results],
                                     [1.0, 1.0, 1.0])

  def test_trace_set_baseline(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESTraceWorker,
        count=1,
        worker_args=(),
        worker_kwargs={}) as pool:
      test_corpus = corpus.create_corpus_for_testing(
          location=self.create_tempdir().full_path,
          elements=[corpus.ModuleSpec(name='name1', size=1)])
      evaluator = blackbox_evaluator.TraceBlackboxEvaluator(
          test_corpus, blackbox_optimizers.EstimatorType.FORWARD_FD,
          'fake_bb_trace_path', 'fake_function_index_path')
      evaluator.set_baseline(pool)
      # pylint: disable=protected-access
      self.assertAlmostEqual(evaluator._baseline, 10)

  def test_trace_get_rewards(self):
    f1 = concurrent.futures.Future()
    f1.set_result(2)
    f2 = concurrent.futures.Future()
    f2.set_result(3)
    results = [f1, f2]
    test_corpus = corpus.create_corpus_for_testing(
        location=self.create_tempdir().full_path,
        elements=[corpus.ModuleSpec(name='name1', size=1)])
    evaluator = blackbox_evaluator.TraceBlackboxEvaluator(
        test_corpus, blackbox_optimizers.EstimatorType.FORWARD_FD,
        'fake_bb_trace_path', 'fake_function_index_path')

    # pylint: disable=protected-access
    evaluator._baseline = 2
    rewards = evaluator.get_rewards(results)

    # Only check for two decimal places as the reward calculation uses a
    # reasonably large delta (0.01) when calculating the difference to
    # prevent division by zero.
    self.assertSequenceAlmostEqual(rewards, [0, -0.5], 2)
