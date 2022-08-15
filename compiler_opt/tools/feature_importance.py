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
"""A tool for analyzing which features a model uses to make a decision."""

from absl import app
from absl import flags
from absl import logging
import gin

from compiler_opt.rl import data_reader
from compiler_opt.rl import constant
from compiler_opt.rl import registry

import tensorflow as tf

_DATA_PATH = flags.DEFINE_multi_string(
  'data_path', [],
  'Path to TFRecord file(s) containing trace data.')
_MODEL_PATH = flags.DEFINE_string(
  'model_path', '',
  'Path to the model to explain')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()

  tfrecord_dataset_fn = data_reader.create_tfrecord_dataset_fn(
    agent_name=constant.AgentName.BEHAVIORAL_CLONE,
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    batch_size=1,
    train_sequence_length=1)
  
  dataset_iter = iter(tfrecord_dataset_fn(_DATA_PATH.value).repeat())

  observation = next(dataset_iter)

  # observation keys
  # 0 - step_type or next_step_type
  # 1 - actual observation
  # 2 - action
  # 3 - policy_thing (not needed)
  # 4 - step_type or next_step_type
  # 5 - reward
  # 6 - discount

  real_observation = observation[1]
  real_observation.update({
    'step_type': observation[0],
    'reward': observation[5],
    'discount': observation[6]
  })

  for key in real_observation:
    real_observation[key] = tf.squeeze(real_observation[key], axis=0)

  logging.info(real_observation)

  #flattened_observation = tf.nest.flatten(observation)[0].numpy()

  #logging.info(flattened_observation)

  saved_policy = tf.saved_model.load(_MODEL_PATH.value)
  #output = model_to_evaluate(observation).numpy()
  #logging.info(output)

  #logging.info(list(model_to_evaluate.signatures.keys()))

  #evaluator = model_to_evaluate.signatures['action']
  #output = evaluator(observation)

  #get_initial_state_fn = saved_policy.signatures['get_initial_state']
  action_fn = saved_policy.signatures['action']

  #logging.info(action_fn.structured_outputs)

  #logging.info(policy_state_dict)

  output = action_fn(**real_observation)
  logging.info(output)
  tf.print(output)

if __name__ == '__main__':
  app.run(main)