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
"""Tooling computing rewards in blackbox learner."""

import abc
import concurrent.futures
import os
import random

from absl import logging
import gin
import tensorflow as tf

from compiler_opt.distributed.worker import FixedWorkerPool
from compiler_opt.rl import corpus
from compiler_opt.es import blackbox_optimizers
from compiler_opt.distributed import buffered_scheduler
from compiler_opt.rl import compilation_runner


class BlackboxEvaluator(metaclass=abc.ABCMeta):
  """Blockbox evaluator abstraction."""

  @abc.abstractmethod
  def __init__(self, train_corpus: corpus.Corpus):
    pass

  @abc.abstractmethod
  def get_results(
      self, pool: FixedWorkerPool,
      perturbations: list[bytes]) -> list[concurrent.futures.Future]:
    raise NotImplementedError()

  @abc.abstractmethod
  def set_baseline(self, pool: FixedWorkerPool) -> None:
    raise NotImplementedError()

  def get_rewards(
      self, results: list[concurrent.futures.Future]) -> list[float | None]:
    rewards = [None] * len(results)

    for i in range(len(results)):
      if not results[i].exception():
        rewards[i] = results[i].result()
      else:
        logging.info('Error retrieving result from future: %s',
                     str(results[i].exception()))

    return rewards


@gin.configurable
class SamplingBlackboxEvaluator(BlackboxEvaluator):
  """A blackbox evaluator that samples from a corpus to collect reward."""

  def __init__(self, train_corpus: corpus.Corpus,
               estimator_type: blackbox_optimizers.EstimatorType,
               total_num_perturbations: int, num_ir_repeats_within_worker: int):
    self._samples: list[list[corpus.LoadedModuleSpec]] = []
    self._train_corpus = train_corpus
    self._total_num_perturbations = total_num_perturbations
    self._num_ir_repeats_within_worker = num_ir_repeats_within_worker
    self._estimator_type = estimator_type
    self._baselines: list[float | None] | None = None

    super().__init__(train_corpus)

  def _load_samples(self) -> None:
    """Samples and loads modules if not already done.

    Ensures self._samples contains the expected number of loaded samples.

    Raises:
      RuntimeError if samples are already loaded or if
        loading fails and counts don't match.
    """
    if self._samples:
      raise RuntimeError('Samples have already been loaded.')
    for _ in range(self._total_num_perturbations):
      samples = self._train_corpus.sample(self._num_ir_repeats_within_worker)
      loaded_samples = [
          self._train_corpus.load_module_spec(sample) for sample in samples
      ]
      self._samples.append(loaded_samples)

      # add copy of sample for antithetic perturbation pair
      if self._estimator_type == (blackbox_optimizers.EstimatorType.ANTITHETIC):
        self._samples.append(loaded_samples)

    logging.info('Done sampling and loading modules for evaluator.')
    expected_count = (2 * self._total_num_perturbations if self._estimator_type
                      == (blackbox_optimizers.EstimatorType.ANTITHETIC) else
                      self._total_num_perturbations)

    if len(self._samples) != expected_count:
      raise RuntimeError('Some samples could not be loaded correctly.')

  def _launch_compilation_workers(self,
                                  pool: FixedWorkerPool,
                                  perturbations: list[bytes] | None = None
                                 ) -> list[concurrent.futures.Future]:
    if self._samples is None:
      raise RuntimeError('Loaded samples are not available.')
    if perturbations is None:
      perturbations = [None] * len(self._samples)
    compile_args = zip(perturbations, self._samples)
    _, futures = buffered_scheduler.schedule_on_worker_pool(
        action=lambda w, args: w.compile(policy=args[0], modules=args[1]),
        jobs=compile_args,
        worker_pool=pool)

    not_done = futures
    # wait for all futures to finish
    while not_done:
      # update lists as work gets done
      _, not_done = concurrent.futures.wait(
          not_done, return_when=concurrent.futures.FIRST_COMPLETED)
    return futures

  def get_results(
      self, pool: FixedWorkerPool,
      perturbations: list[bytes]) -> list[concurrent.futures.Future]:
    # We should have _samples by now.
    if not self._samples:
      raise RuntimeError('Loaded samples are not available.')
    return self._launch_compilation_workers(pool, perturbations)

  def set_baseline(self, pool: FixedWorkerPool) -> None:
    if self._baselines is not None:
      raise RuntimeError('The baseline has already been set.')
    self._load_samples()
    results = self._launch_compilation_workers(pool)
    self._baselines = super().get_rewards(results)

  def get_rewards(
      self, results: list[concurrent.futures.Future]) -> list[float | None]:
    if self._baselines is None:
      raise RuntimeError('The baseline has not been set.')

    if len(results) != len(self._baselines):
      raise RuntimeError(
          'The number of results does not match the number of baselines.')

    policy_results = super().get_rewards(results)

    rewards = []
    for policy_result, baseline in zip(
        policy_results, self._baselines, strict=True):
      if policy_result is None or baseline is None:
        rewards.append(None)
      else:
        rewards.append(
            compilation_runner.calculate_reward(policy_result, baseline))
    return rewards


@gin.configurable
class TraceBlackboxEvaluator(BlackboxEvaluator):
  """A blackbox evaluator that utilizes trace based cost modelling."""

  def __init__(self, train_corpus: corpus.Corpus,
               estimator_type: blackbox_optimizers.EstimatorType,
               bb_trace_path: str, function_index_path: str):
    self._train_corpus = train_corpus
    self._estimator_type = estimator_type
    self._bb_trace_paths = []
    if tf.io.gfile.isdir(bb_trace_path):
      self._bb_trace_paths.extend([
          os.path.join(bb_trace_path, bb_trace)
          for bb_trace in tf.io.gfile.listdir(bb_trace_path)
      ])
    else:
      self._bb_trace_paths.append(bb_trace_path)
    self._function_index_path = function_index_path

    self._baselines: list[float] | None = None

  def get_results(
      self, pool: FixedWorkerPool,
      perturbations: list[bytes]) -> list[concurrent.futures.Future]:
    job_args = []
    self._current_baselines = []
    for perturbation in perturbations:
      bb_trace_path_index = random.randrange(len(self._bb_trace_paths))
      bb_trace_path = self._bb_trace_paths[bb_trace_path_index]
      self._current_baselines.append(self._baselines[bb_trace_path_index])
      job_args.append({
          'modules': self._train_corpus.module_specs,
          'function_index_path': self._function_index_path,
          'bb_trace_path': bb_trace_path,
          'policy_as_bytes': perturbation
      })

    _, futures = buffered_scheduler.schedule_on_worker_pool(
        action=lambda w, args: w.compile_corpus_and_evaluate(**args),
        jobs=job_args,
        worker_pool=pool)
    concurrent.futures.wait(
        futures, return_when=concurrent.futures.ALL_COMPLETED)
    return futures

  def set_baseline(self, pool: FixedWorkerPool) -> None:
    if self._baselines is not None:
      raise RuntimeError('The baseline has already been set.')

    job_args = [{
        'modules': self._train_corpus.module_specs,
        'function_index_path': self._function_index_path,
        'bb_trace_path': bb_trace_path,
        'policy_as_bytes': None,
    } for bb_trace_path in self._bb_trace_paths]

    _, futures = buffered_scheduler.schedule_on_worker_pool(
        action=lambda w, args: w.compile_corpus_and_evaluate(**args),
        jobs=job_args,
        worker_pool=pool)

    concurrent.futures.wait(
        futures, return_when=concurrent.futures.ALL_COMPLETED)
    if len(futures) != len(self._bb_trace_paths):
      raise ValueError(
          f'Expected to have {len(self._bb_trace_paths)} results for setting,'
          f'the baseline, got {len(futures)}.')

    self._baselines = [future.result() for future in futures]

  def get_rewards(
      self, results: list[concurrent.futures.Future]) -> list[float | None]:
    rewards = []

    for result, baseline in zip(results, self._current_baselines):
      if result.exception() is not None:
        raise result.exception()

      rewards.append(
          compilation_runner.calculate_reward(result.result(), baseline))

    return rewards
