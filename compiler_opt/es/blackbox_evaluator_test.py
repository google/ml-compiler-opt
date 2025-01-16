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
"""Tests for BlackboxEvaluator."""

import concurrent.futures

from absl.testing import absltest

from compiler_opt.distributed.local import local_worker_manager
from compiler_opt.rl import corpus
from compiler_opt.es import blackbox_test_utils
from compiler_opt.es import blackbox_evaluator


class BlackboxEvaluatorTests(absltest.TestCase):
  """Tests for BlackboxEvaluator."""

  def test_sampling_get_results(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESWorker, count=3, arg='', kwarg='') as pool:
      perturbations = [b'00', b'01', b'10']
      evaluator = blackbox_evaluator.SamplingBlackboxEvaluator(None, 5, 5, None)
      # pylint: disable=protected-access
      evaluator._samples = [[corpus.ModuleSpec(name='name1', size=1)],
                            [corpus.ModuleSpec(name='name2', size=1)],
                            [corpus.ModuleSpec(name='name3', size=1)]]
      # pylint: enable=protected-access
      results = evaluator.get_results(pool, perturbations)
      self.assertSequenceAlmostEqual([result.result() for result in results],
                                     [1.0, 1.0, 1.0])

  def test_get_rewards(self):
    f1 = concurrent.futures.Future()
    f1.set_exception(None)
    f2 = concurrent.futures.Future()
    f2.set_result(2)
    results = [f1, f2]
    evaluator = blackbox_evaluator.SamplingBlackboxEvaluator(None, 5, 5, None)
    rewards = evaluator.get_rewards(results)
    self.assertEqual(rewards, [None, 2])

  def test_trace_get_results(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESTraceWorker, count=3, arg='', kwarg='') as pool:
      perturbations = [b'00', b'01', b'10']
      test_corpus = corpus.create_corpus_for_testing(
          location=self.create_tempdir(),
          elements=[corpus.ModuleSpec(name='name1', size=1)])
      evaluator = blackbox_evaluator.TraceBlackboxEvaluator(
          test_corpus, 5, 'fake_bb_trace_path', 'fake_function_index_path')
      results = evaluator.get_results(pool, perturbations)
      self.assertSequenceAlmostEqual([result.result() for result in results],
                                     [1.0, 1.0, 1.0])

  def test_trace_set_baseline(self):
    with local_worker_manager.LocalWorkerPoolManager(
        blackbox_test_utils.ESTraceWorker, count=1, arg='', kwarg='') as pool:
      test_corpus = corpus.create_corpus_for_testing(
          location=self.create_tempdir(),
          elements=[corpus.ModuleSpec(name='name1', size=1)])
      evaluator = blackbox_evaluator.TraceBlackboxEvaluator(
          test_corpus, 5, 'fake_bb_trace_path', 'fake_function_index_path')
      evaluator.set_baseline(pool)
      # pylint: disable=protected-access
      self.assertAlmostEqual(evaluator._baseline, 10)
