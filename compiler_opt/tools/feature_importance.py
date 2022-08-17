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
_OUTPUT_FILE = flags.DEFINE_string(
  'output_file', '',
  'The path to the output file containing the SHAP values')
_NUM_EXAMPLES = flags.DEFINE_integer(
  'num_examples', '',
  'The number of examples to process from the trace')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

def get_input_signature(example_input):
  input_signature = {}
  for input in example_input:
    input_signature[input] = (tf.shape(example_input[input]).numpy(), example_input[input].dtype)
  return input_signature

def get_signature_total_size(input_signature):
  total_size = 0
  for input in input_signature:
    total_size += numpy.prod(input_signature[input][0])
  return total_size

def pack_flat_array_into_input(flat_array, signature_spec):
  output_input_dict = {}
  current_index = 0
  for needed_input in signature_spec:
    part_size = numpy.prod(signature_spec[needed_input][0])
    needed_subset = flat_array[current_index:current_index + part_size]
    current_index += part_size
    output_input_dict[needed_input] = tf.cast(tf.constant(needed_subset, shape=signature_spec[needed_input][0]), dtype=signature_spec[needed_input][1])
  return output_input_dict

def flatten_input(input, array_size):
  output_array = numpy.empty(array_size)
  input_index = 0
  for input_key in input:
    current_size = tf.size(input[input_key])
    end_index = input_index + current_size
    output_array[input_index:end_index] = input[input_key].numpy().astype(numpy.float32)
    input_index += current_size
  return output_array

def process_raw_trajectory(raw_trajectory):
  observation = raw_trajectory.observation
  observation.update({
    'step_type': raw_trajectory.step_type,
    'reward': raw_trajectory.reward,
    'discount': raw_trajectory.discount
  })
  
  # remove batch size dimension
  for key in observation:
    observation[key] = tf.squeeze(observation[key], axis=0)
  
  return observation

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

  saved_policy = tf.saved_model.load(_MODEL_PATH.value)
  action_fn = saved_policy.signatures['action']

  observation = process_raw_trajectory(raw_trajectory)
  input_sig = get_input_signature(observation)

  def run_model(flat_input_array):
    output = numpy.empty(flat_input_array.shape[0])
    for index, flat_input in enumerate(flat_input_array):
      input_dict = pack_flat_array_into_input(flat_input, input_sig)
      model_output = action_fn(**input_dict).items()
      # get the value of the first item as a numpy array
      output[index] = list(model_output)[0][1].numpy()[0]
    return output
  
  total_size = get_signature_total_size(input_sig)
  flattened_input = flatten_input(observation, total_size)
  flattened_input = numpy.expand_dims(flattened_input, axis=0)
  dataset = numpy.empty((_NUM_EXAMPLES.value, total_size))
  for i in range(0, _NUM_EXAMPLES.value):
    raw_trajectory = next(dataset_iter)
    observation = process_raw_trajectory(raw_trajectory)
    flat_input = flatten_input(observation, total_size)
    dataset[i] = flat_input
  
  explainer = shap.KernelExplainer(run_model, numpy.zeros((1, total_size)))
  shap_values = explainer.shap_values(dataset, nsamples=1000)

  numpy.savetxt(_OUTPUT_FILE.value, shap_values)

if __name__ == '__main__':
  app.run(main)