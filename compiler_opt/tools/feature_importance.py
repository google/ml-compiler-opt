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

from email.contentmanager import raw_data_manager
from absl import app
from absl import flags
from absl import logging
import gin

from compiler_opt.rl import data_reader
from compiler_opt.rl import constant
from compiler_opt.rl import registry

import tensorflow as tf
import shap
import numpy

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

def get_input_signature(example_input):
  input_signature = {}
  for input in example_input:
    input_signature[input] = tf.shape(example_input[input]).numpy()
  return input_signature

def get_signature_total_size(input_signature):
  total_size = 0
  for input in input_signature:
    total_size += numpy.prod(input_signature[input])
  return total_size

def pack_flat_array_into_input(flat_array, signature_spec):
  output_input_dict = {}
  current_index = 0
  for needed_input in signature_spec:
    part_size = numpy.prod(signature_spec[needed_input])
    needed_subset = flat_array[current_index:part_size]
    current_index += part_size
    output_input_dict[needed_input] = tf.constant(needed_subset)
  return output_input_dict

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

  raw_trajectory = next(dataset_iter)

  observation = raw_trajectory.observation
  observation.update({
    'step_type': raw_trajectory.step_type,
    'reward': raw_trajectory.reward,
    'discount': raw_trajectory.discount
  })
  
  # remove batch size dimension
  for key in observation:
    observation[key] = tf.squeeze(observation[key], axis=0)

  saved_policy = tf.saved_model.load(_MODEL_PATH.value)
  action_fn = saved_policy.signatures['action']

  #output = action_fn(**observation)
  
  #explainer = shap.DeepExplainer()

  input_sig = get_input_signature(observation)
  logging.info(input_sig)
  logging.info(get_signature_total_size(input_sig))

if __name__ == '__main__':
  app.run(main)