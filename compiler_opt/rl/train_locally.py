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
r"""Train and Eval LLVM Inliner decision rule with local_data_collector."""

import collections
import functools
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.system import system_multiprocessing as multiprocessing
from typing import List

from compiler_opt.distributed.local import cpu_affinity
from compiler_opt.distributed.local.local_worker_manager import LocalWorkerPool
from compiler_opt.rl import agent_creators
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import constant
from compiler_opt.rl import corpus
from compiler_opt.rl import data_reader
from compiler_opt.rl import local_data_collector
from compiler_opt.rl import local_validation_data_collector
from compiler_opt.rl import policy_saver
from compiler_opt.rl import random_net_distillation
from compiler_opt.rl import registry
from compiler_opt.rl import trainer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('data_path', None,
                    'Path to directory containing the corpus.')
flags.DEFINE_string('validation_data_path', None,
                    'Path to directory containing the validation corpus.')
flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel data collection workers. `None` for max available')
flags.DEFINE_multi_string('gin_files', [],
                          'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(agent_name=constant.AgentName.PPO,
               warmstart_policy_dir=None,
               num_policy_iterations=0,
               num_modules=100,
               num_iterations=100,
               batch_size=64,
               train_sequence_length=1,
               deploy_policy_name='saved_policy',
               use_random_network_distillation=False,
               moving_average_decay_rate=1):
  """Train for LLVM inliner."""
  root_dir = FLAGS.root_dir
  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()
  preprocessing_layer_creator = problem_config.get_preprocessing_layer_creator()

  # Initialize trainer and policy saver.
  agent: tf_agent.TFAgent = agent_creators.create_agent(
      agent_name, time_step_spec, action_spec, preprocessing_layer_creator)
  # create the random network distillation object
  random_network_distillation = None
  if use_random_network_distillation:
    random_network_distillation = (
        random_net_distillation.RandomNetworkDistillation(
            time_step_spec=time_step_spec,
            preprocessing_layer_creator=preprocessing_layer_creator))

  llvm_trainer = trainer.Trainer(
      root_dir=root_dir,
      agent=agent,
      random_network_distillation=random_network_distillation,
      warmstart_policy_dir=warmstart_policy_dir)

  policy_dict = {
      'saved_policy': agent.policy,
      'saved_collect_policy': agent.collect_policy,
  }
  saver = policy_saver.PolicySaver(policy_dict=policy_dict)

  logging.info('Loading module specs from corpus at %s.', FLAGS.data_path)
  cps = corpus.Corpus(FLAGS.data_path, problem_config.flags_to_add(),
                      problem_config.flags_to_delete())
  logging.info('Done loading module specs from corpus.')

  val_cps = None
  if FLAGS.validation_data_path is not None:
    logging.info('Loading module specs from validation corpus at %s.',
                 FLAGS.validation_data_path)
    val_cps = corpus.Corpus(FLAGS.validation_data_path,
                            problem_config.flags_to_add(),
                            problem_config.flags_to_delete())
    logging.info('Done loading module specs from validation corpus.')
  else:
    logging.info('Validation corpus data path not specified.')

  dataset_fn = data_reader.create_sequence_example_dataset_fn(
      agent_name=agent_name,
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      batch_size=batch_size,
      train_sequence_length=train_sequence_length)

  def sequence_example_iterator_fn(seq_ex: List[str]):
    return iter(dataset_fn(seq_ex).repeat().prefetch(tf.data.AUTOTUNE))

  reward_stat_map = collections.defaultdict(lambda: None)
  reward_stat_map_path = os.path.join(root_dir, 'reward_stat_map')

  # Reload reward_stat_map if exists.
  # reward_stat_map of defaultdict(str, {str: RewardStat})
  if tf.io.gfile.exists(reward_stat_map_path):
    with tf.io.gfile.GFile(reward_stat_map_path, 'r') as f:
      data = json.load(f)
    for k, v in data.items():
      if v:
        reward_stat_map[k] = {
            sub_k: compilation_runner.RewardStat(**sub_v)
            for sub_k, sub_v in v.items()
        }
    logging.info('Loaded Reward Stat Map from disk, containing %d modules',
                 len(reward_stat_map))
  val_pool_args = {'worker_class': problem_config.get_runner_type()}
  with LocalWorkerPool(
      worker_class=problem_config.get_runner_type(),
      count=FLAGS.num_workers,
      moving_average_decay_rate=moving_average_decay_rate
  ) as worker_pool, LocalWorkerPool(
      worker_class=local_validation_data_collector.LocalValidationDataCollector,
      count=1,
      cps=val_cps,
      worker_pool_args=val_pool_args,
      reward_stat_map=reward_stat_map,
      max_cpus=FLAGS.num_workers) as validation_collector_pool:

    data_collector = local_data_collector.LocalDataCollector(
        cps=cps,
        num_modules=num_modules,
        worker_pool=worker_pool,
        parser=sequence_example_iterator_fn,
        reward_stat_map=reward_stat_map)

    validation_collector = validation_collector_pool[0]

    if val_cps is not None:
      cpu_affinity.set_and_get(is_main_process=True, max_cpus=FLAGS.num_workers)
    # Repeat for num_policy_iterations iterations.
    t1 = time.time()
    while (llvm_trainer.global_step_numpy() <
           num_policy_iterations * num_iterations):
      t2 = time.time()
      logging.info('Last iteration took: %f', t2 - t1)
      t1 = t2
      with tf.io.gfile.GFile(reward_stat_map_path, 'w') as f:
        json.dump(
            reward_stat_map, f, cls=compilation_runner.DataClassJSONEncoder)

      policy_path = os.path.join(root_dir, 'policy',
                                 str(llvm_trainer.global_step_numpy()))
      # Pausing is done before saving to give the validation collector's
      # children time to receive the stop signal, minimizing the risk of it
      # executing simultaneously with the main data_collector's children.
      if val_cps is not None:
        validation_collector.pause_children()
        time.sleep(15)
      saver.save(policy_path)

      policy_fullpath = os.path.join(policy_path, deploy_policy_name)
      dataset_iter, monitor_dict = data_collector.collect_data(
          policy_path=policy_fullpath)
      if val_cps is not None:
        validation_dict_maybe = validation_collector.collect_data_async(
            policy_path=policy_fullpath,
            step=llvm_trainer.global_step_numpy()).result()
        if validation_dict_maybe is not None:
          llvm_trainer.write_validation_data(validation_dict_maybe)

      llvm_trainer.train(dataset_iter, monitor_dict, num_iterations)

      data_collector.on_dataset_consumed(dataset_iter)

    # Save final policy.
    saver.save(root_dir)
    # Wait for all the workers to finish.
    data_collector.close_pool()


def main(_):
  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=FLAGS.gin_bindings, skip_unknown=False)
  logging.info(gin.config_str())

  train_eval()


if __name__ == '__main__':
  flags.mark_flag_as_required('data_path')
  multiprocessing.handle_main(functools.partial(app.run, main))
