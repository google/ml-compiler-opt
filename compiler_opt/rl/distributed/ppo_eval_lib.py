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
"""Evaluate the current policy stored in reverb"""

import tempfile
import collections
import os
import time
from typing import List, Optional

from absl import logging

import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from compiler_opt.rl import data_reader
from compiler_opt.rl import local_data_collector
from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import corpus
from compiler_opt.rl import agent_config
from compiler_opt.rl import registry
from compiler_opt.rl import policy_saver
from compiler_opt.rl import data_collector


def evaluate(root_dir: str, corpus_path: str,
             variable_container_server_address: str, num_workers: Optional[int],
             worker_manager_class):
  """Evaluate a given policy on the given corpus.

  Args:
    root_dir: path to write tensorboard summaries.
    corpus_path: path to the corpus to collect from.
    variable_container_server_address: address of the variable container in
      reverb.
    num_workers: number of workers in the pool to collect from.
    worker_manager_class: type of the worker manager to spawn the pool with.
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

  #policy = greedy_policy.GreedyPolicy(agent.policy)
  policy = agent.collect_policy
  policy_dict = {'policy': policy}

  # Initialize the reverb variable container
  logging.info('Connecting to the reverb server')
  train_step = tf_v1.train.get_or_create_global_step()
  model_id = common.create_variable('model_id')
  variables = {
      reverb_variable_container.POLICY_KEY: policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])
  variable_container.update(variables)

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

  summary_writer = tf.summary.create_file_writer(
      root_dir, flush_millis=5 * 1000)
  summary_writer.set_as_default()

  data_action_mean = tf.keras.metrics.Mean()
  data_reward_mean = tf.keras.metrics.Mean()

  # Run the experience collection loop.
  with worker_manager_class(
      worker_class=problem_config.get_runner_type(),
      count=num_workers,
      moving_average_decay_rate=1) as worker_pool:
    logging.info('constructed pool')
    collector = local_data_collector.LocalDataCollector(
        cps=cps,
        num_modules=128,
        worker_pool=worker_pool,
        parser=sequence_example_iterator_fn,
        reward_stat_map=collections.defaultdict(lambda: None),
        best_trajectory_repo=None)
    logging.info('constructed data_collector')
    last_step = -1
    rewards = []
    actions = []
    while True:
      with tempfile.TemporaryDirectory() as tmpdirname:
        saver = policy_saver.PolicySaver(policy_dict=policy_dict)
        saver.save(tmpdirname)
        policy_bytes = policy_saver.Policy.from_filesystem(
            os.path.join(tmpdirname, 'policy'))
      dataset_iter, monitor_dict = collector.collect_data(
          policy=policy_bytes, model_id=model_id.numpy())
      del monitor_dict

      for experience in dataset_iter:
        is_action = ~experience.is_boundary()

        # pylint: disable=not-callable
        data_action_mean.update_state(
            experience.action, sample_weight=is_action)
        # pylint: disable=not-callable
        data_reward_mean.update_state(
            experience.reward, sample_weight=is_action)

        if is_action:
          rewards.append(experience.reward)
          actions.append(experience.action)

      if last_step < train_step.numpy():
        last_step = train_step.numpy()
        with tf.name_scope('default/'):
          tf.summary.scalar(
              name='data_action_mean',
              # pylint: disable=not-callable
              data=data_action_mean.result(),
              step=train_step)
          tf.summary.scalar(
              name='data_reward_mean',
              # pylint: disable=not-callable
              data=data_reward_mean.result(),
              step=train_step)
          tf.summary.scalar(
              name='num_eval_experiences', data=len(rewards), step=train_step)
        tf.summary.histogram(name='reward', data=rewards, step=train_step)
        tf.summary.histogram(name='action', data=actions, step=train_step)
        with tf.name_scope('reward_distribution/'):
          reward_dist = data_collector.build_distribution_monitor(rewards)
          for k, v in reward_dist.items():
            tf.summary.scalar(name=k, data=v, step=train_step)
        data_action_mean.reset_state()
        data_reward_mean.reset_state()
        rewards = []
        actions = []

      variable_container.update(variables)
      logging.info('Evaluating with policy at step: %d', train_step.numpy())
      time.sleep(20)


def run_evaluate(root_dir: str, corpus_path: str,
                 variable_container_server_address: str,
                 num_workers: Optional[int], worker_manager_class):
  """Wait for the collect policy to be ready and run collect job."""
  # Wait for the collect policy to become available, then load it.
  policy_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR,
                            learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  policy = train_utils.wait_for_policy(policy_dir, load_specs_from_pbtxt=True)
  del policy

  evaluate(
      root_dir=root_dir,
      corpus_path=corpus_path,
      variable_container_server_address=variable_container_server_address,
      num_workers=num_workers,
      worker_manager_class=worker_manager_class)
