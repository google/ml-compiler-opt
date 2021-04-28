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

import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin
from tf_agents.policies import policy_loader
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl import agent_creators
from compiler_opt.rl import data_reader
from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import inlining_runner
from compiler_opt.rl import local_data_collector
from compiler_opt.rl import policy_saver
from compiler_opt.rl import trainer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('data_path', None,
                    'Path to CNS folder containing IR files.')
flags.DEFINE_string('clang_path', 'clang', 'Path to clang binary.')
flags.DEFINE_string('llvm_size_path', 'llvm-size', 'Path to llvm_size binary.')
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
def train_eval(get_signature_spec_fn=None,
               agent_name='ppo',
               warmstart_policy_dir=None,
               num_policy_iterations=0,
               num_iterations=100,
               batch_size=64,
               train_sequence_length=1,
               deploy_policy_name='saved_policy',
               use_stale_results=False):
  """Train for LLVM inliner."""
  root_dir = FLAGS.root_dir

  # Initialize trainer and policy saver.
  time_step_spec, action_spec = get_signature_spec_fn()
  tf_agent = agent_creators.create_agent(agent_name, time_step_spec,
                                         action_spec)
  llvm_trainer = trainer.Trainer(root_dir=root_dir, agent=tf_agent)
  policy_dict = {
      'saved_policy': tf_agent.policy,
      'saved_collect_policy': tf_agent.collect_policy,
  }
  saver = policy_saver.PolicySaver(policy_dict=policy_dict)

  if warmstart_policy_dir:
    warmstart_policy = policy_loader.load(warmstart_policy_dir)
    tf_agent.policy.update(
        policy=warmstart_policy,
        tau=1.0,
        tau_non_trainable=None,
        sort_variables_by_name=False)

  with open(os.path.join(FLAGS.data_path, 'module_paths'), 'r') as f:
    module_paths = [
        os.path.join(FLAGS.data_path, name.rstrip('\n')) for name in f
    ]
    file_paths = [(path + '.bc', path + '.cmd') for path in module_paths]

  runner = inlining_runner.InliningRunner(
      clang_path=FLAGS.clang_path, llvm_size_path=FLAGS.llvm_size_path)

  sequence_example_iterator_fn = (
      data_reader.create_sequence_example_iterator_fn(
          agent_name=agent_name,
          time_step_spec=time_step_spec,
          action_spec=action_spec,
          batch_size=batch_size,
          train_sequence_length=train_sequence_length))

  data_collector = local_data_collector.LocalDataCollector(
      file_paths=file_paths,
      num_workers=FLAGS.num_workers,
      num_modules=FLAGS.num_modules,
      runner=runner.collect_data,
      parser=sequence_example_iterator_fn,
      use_stale_results=use_stale_results)

  # Repeat for num_policy_iterations iterations.
  while (llvm_trainer.global_step_numpy() <
         num_policy_iterations * num_iterations):
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
