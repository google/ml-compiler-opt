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
from tf_agents.system import system_multiprocessing as multiprocessing
from typing import List

from compiler_opt.rl import agent_creators
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import constant
from compiler_opt.rl import corpus
from compiler_opt.rl import data_reader
from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import local_data_collector
from compiler_opt.rl import policy_saver
from compiler_opt.rl import random_net_distillation
from compiler_opt.rl import registry
from compiler_opt.rl import trainer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('data_path', None,
                    'Path to CNS folder containing IR files.')
flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel data collection workers. `None` for max available')
flags.DEFINE_integer('num_modules', 100,
                     'Number of modules to collect data for each iteration.')
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
               num_iterations=100,
               batch_size=64,
               train_sequence_length=1,
               deploy_policy_name='saved_policy',
               use_random_network_distillation=False,
               moving_average_decay_rate=1,
               additional_compilation_flags=(),
               delete_compilation_flags=()):
  """Train for LLVM inliner."""
  root_dir = FLAGS.root_dir
  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()
  preprocessing_layer_creator = problem_config.get_preprocessing_layer_creator()

  # Initialize trainer and policy saver.
  tf_agent = agent_creators.create_agent(agent_name, time_step_spec,
                                         action_spec,
                                         preprocessing_layer_creator)
  # create the random network distillation object
  random_network_distillation = None
  if use_random_network_distillation:
    random_network_distillation = (
        random_net_distillation.RandomNetworkDistillation(
            time_step_spec=time_step_spec,
            preprocessing_layer_creator=preprocessing_layer_creator))

  llvm_trainer = trainer.Trainer(
      root_dir=root_dir,
      agent=tf_agent,
      random_network_distillation=random_network_distillation,
      warmstart_policy_dir=warmstart_policy_dir)

  policy_dict = {
      'saved_policy': tf_agent.policy,
      'saved_collect_policy': tf_agent.collect_policy,
  }
  saver = policy_saver.PolicySaver(policy_dict=policy_dict)

  logging.info('Loading module specs from corpus')
  module_specs = corpus.read(FLAGS.data_path, additional_compilation_flags,
                             delete_compilation_flags)
  logging.info('Done loading module specs from corpus')

  runner = problem_config.get_runner_type()(
      moving_average_decay_rate=moving_average_decay_rate)

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

  data_collector = local_data_collector.LocalDataCollector(
      module_specs=module_specs,
      num_workers=FLAGS.num_workers,
      num_modules=FLAGS.num_modules,
      runner=runner,
      parser=sequence_example_iterator_fn,
      reward_stat_map=reward_stat_map)

  # Repeat for num_policy_iterations iterations.
  t1 = time.time()
  while (llvm_trainer.global_step_numpy() <
         num_policy_iterations * num_iterations):
    t2 = time.time()
    logging.info('Last iteration took: %f', t2 - t1)
    t1 = t2
    with tf.io.gfile.GFile(reward_stat_map_path, 'w') as f:
      json.dump(reward_stat_map, f, cls=compilation_runner.DataClassJSONEncoder)

    policy_path = os.path.join(root_dir, 'policy',
                               str(llvm_trainer.global_step_numpy()))
    saver.save(policy_path)

    dataset_iter, monitor_dict = data_collector.collect_data(
        policy_path=os.path.join(policy_path, deploy_policy_name))
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
