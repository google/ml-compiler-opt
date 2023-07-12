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
"""Collect experiences for PPO training"""

import collections
import os
from typing import List, Optional
import tempfile
import functools

from absl import logging

import gin
import reverb

import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.replay_buffers import reverb_utils

from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.trajectories import trajectory

from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import local_data_collector
from compiler_opt.rl import corpus
from compiler_opt.rl import data_reader
from compiler_opt.rl import policy_saver
from compiler_opt.rl import registry
from compiler_opt.rl import agent_config
from compiler_opt.rl import compilation_runner


def _get_policy_bytes(agent):
  """Recover the collect_policy bytes from a TF agent"""
  policy_key = 'collect'
  with tempfile.TemporaryDirectory() as tmpdirname:
    saver = policy_saver.PolicySaver(policy_dict={
        policy_key: agent.collect_policy,
    })
    saver.save(tmpdirname)
    return policy_saver.Policy.from_filesystem(
        os.path.join(tmpdirname, policy_key))


def _add_value_prediction(traj: trajectory.Trajectory):
  traj.policy_info['value_prediction'] = tf.constant(0.0, dtype=tf.float32)


class ReverbCompilationObserver(compilation_runner.CompilationResultObserver):
  """Observer which sends compilation results to reverb"""

  def __init__(self,
               agent_cfg,
               replay_buffer_server_address: str,
               sequence_length: int,
               initial_priority: float = 0.0):
    self._observer = reverb_utils.ReverbTrajectorySequenceObserver(
        reverb.Client(replay_buffer_server_address),
        table_name=[
            'training_table',
        ],
        sequence_length=sequence_length,
        stride_length=sequence_length,
        priority=initial_priority)

    self._parser = data_reader.create_flat_sequence_example_dataset_fn(
        agent_cfg=agent_cfg)

  def _is_actionable_result(
      self, result: compilation_runner.CompilationResult) -> bool:
    """Predicate which checks if the result contains valid experiences"""
    return bool(result.keys)

  def observe(self, result: compilation_runner.CompilationResult) -> None:
    """Observe the compilation result.

    Specifically, parses the serialized sequence examples and sends each
    experience to reverb.
    """
    assert result.model_id is not None
    if self._is_actionable_result(result):
      # pylint: disable=protected-access
      self._observer._priority = result.model_id
      parsed = self._parser(result.serialized_sequence_examples)
      for experience in parsed:
        _add_value_prediction(experience)
        self._observer(experience)


def collect(corpus_path: str, replay_buffer_server_address: str,
            variable_container_server_address: str, num_workers: Optional[int],
            worker_manager_class, sequence_length: int) -> None:
  """Collects experience using a policy updated after every episode.

  Args:
    corpus_path: path to the corpus to collect from.
    replay_buffer_server_address: address of the replay buffer in reverb.
    variable_container_server_address: address of the variable container in
      reverb.
    num_workers: number of workers in the pool to collect from.
    worker_manager_class: type of the worker manager to spawn the pool with.
    sequence_length: sequence length of the examples.
  """

  # Set up the problem & initialize the agent
  logging.info('Initializing the distributed PPO agent')
  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()
  agent_cfg = agent_config.DistributedPPOAgentConfig(
      time_step_spec=time_step_spec, action_spec=action_spec)
  agent = agent_config.create_agent(
      agent_cfg,
      preprocessing_layer_creator=problem_config
      .get_preprocessing_layer_creator())

  # Initialize the reverb variable container
  logging.info('Connecting to the reverb server')
  train_step = tf_v1.train.get_or_create_global_step()
  model_id = common.create_variable('model_id')
  variables = {
      reverb_variable_container.POLICY_KEY: agent.collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])
  variable_container.update(variables)

  create_observer_fns = [
      functools.partial(
          ReverbCompilationObserver,
          agent_config=agent_cfg,
          replay_buffer_server_address=replay_buffer_server_address,
          sequence_length=sequence_length)
  ]

  # Setup the corpus
  logging.info('Constructing tf.data pipeline and module corpus')
  dataset_fn = data_reader.create_flat_sequence_example_dataset_fn(
      agent_cfg=agent_cfg)

  def sequence_example_iterator_fn(seq_ex: List[str]):
    return iter(dataset_fn(seq_ex).prefetch(tf.data.AUTOTUNE))

  cps = corpus.Corpus(
      data_path=corpus_path,
      additional_flags=problem_config.flags_to_add(),
      delete_flags=problem_config.flags_to_delete(),
      replace_flags=problem_config.flags_to_replace())

  # Run the experience collection loop.
  logging.info('Constructing the worker pool')
  with worker_manager_class(
      worker_class=problem_config.get_runner_type(),
      count=num_workers,
      moving_average_decay_rate=1,
      create_observer_fns=create_observer_fns) as worker_pool:

    data_collector = local_data_collector.LocalDataCollector(
        cps=cps,
        num_modules=num_workers * 128 if num_workers else 512,
        worker_pool=worker_pool,
        parser=sequence_example_iterator_fn,
        reward_stat_map=collections.defaultdict(lambda: None),
        best_trajectory_repo=None)

    logging.info('Starting collection loop')
    while True:
      logging.info('Collecting with model_id: %d at step %d',
                   model_id.read_value(), train_step.numpy())
      # Collect experiences which will be sent to reverb by each worker
      dataset_iter, monitor_dict = data_collector.collect_data(
          policy=_get_policy_bytes(agent), model_id=model_id.numpy())
      del dataset_iter
      del monitor_dict
      # Fetch updated policy variables
      variable_container.update(variables)


@gin.configurable
def run_collect(root_dir: str, corpus_path: str,
                replay_buffer_server_address: str,
                variable_container_server_address: str,
                num_workers: Optional[int], worker_manager_class,
                sequence_length: int):
  """Collects experience using a policy updated after every episode.

  Waits for a policy to be saved in root_dir before beginning collection.

  Args:
    root_dir: Directory where the trainer will save policies.
    corpus_path: path to the corpus to collect from.
    replay_buffer_server_address: address of the replay buffer in reverb.
    variable_container_server_address: address of the variable container in
      reverb.
    num_workers: number of workers in the pool to collect from.
    worker_manager_class: type of the worker manager to spawn the pool with.
    sequence_length: sequence length of the examples.
  """
  # Wait for the collect policy to become available, as this signifies the
  # trainer has come online
  collect_policy_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR,
                                    learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  collect_policy = train_utils.wait_for_policy(
      collect_policy_dir, load_specs_from_pbtxt=True)

  # But we don't need it, because we will load parameters from the reverb server
  del collect_policy

  collect(
      corpus_path=corpus_path,
      replay_buffer_server_address=replay_buffer_server_address,
      variable_container_server_address=variable_container_server_address,
      num_workers=num_workers,
      worker_manager_class=worker_manager_class,
      sequence_length=sequence_length)
