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
"""A tool for analyzing which features a model uses to make a decision.

This script allows for processing a set of examples generated from a trace
created through generate_default_trace into a set of shap values which
represent how much that specific feature contributes to the final output of
the model. These values can then be imported into an IPython notebook and
graphed with the help of the feature_importance_graphs.py module in the same
folder.

Usage:
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/tools/feature_importance.py \
    --gin_files=compiler_opt/rl/regalloc/gin_configs/common.gin \
    --gin_bindings=config_registry.get_configuration.implementation=\
      @configs.RegallocEvictionConfig \
    --data_path=/default_trace \
    --model_path=/warmstart/saved_policy \
    --num_examples=5 \
    --output_file=./explanation_data.json

The type of trace that is performed (ie if it is just tracing the default
heuristic or if it is a trace of a ML model) doesn't matter as the only data
that matters re the input features. The num_examples flag sets the number of
examples that get processed into shap values. Increasing this value will
potentially allow you to reach better conclusions depending upon how you're
viewing the data, but increasing it will also increase the runtime of this
script quite significantly as the process is not multithreaded.
"""

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
import numpy.typing
import json

from tf_agents.typing import types
from typing import Dict, Tuple

SignatureType = Dict[str, Tuple[numpy.typing.ArrayLike, tf.dtypes.DType]]

# only create flags if running as a script to prevent interference with pytest
if __name__ == '__main__':
  _DATA_PATH = flags.DEFINE_multi_string(
      'data_path', [], 'Path to TFRecord file(s) containing trace data.')
  _MODEL_PATH = flags.DEFINE_string('model_path', '',
                                    'Path to the model to explain')
  _OUTPUT_FILE = flags.DEFINE_string(
      'output_file', '',
      'The path to the output file containing the SHAP values')
  _NUM_EXAMPLES = flags.DEFINE_integer(
      'num_examples', 1, 'The number of examples to process from the trace')
  _GIN_FILES = flags.DEFINE_multi_string(
      'gin_files', [], 'List of paths to gin configuration files.')
  _GIN_BINDINGS = flags.DEFINE_multi_string(
      'gin_bindings', [],
      'Gin bindings to override the values set in the config files.')


def get_input_signature(example_input: types.NestedTensorSpec) -> SignatureType:
  """Gets the signature of an observation

  This function takes in an example input and returns a signature of that
  input containing all of the info needed to restructure a flat array back into
  the original format later on. This function returns a dictionary with the
  same keys as the original input but with the items being tuples where the
  first value is the shape of that feature and the second is its data type.

  Args:
    example_input: a nested tensor spec (dictionary of tensors) that serves
      as an example for generating the signature.
  """
  input_signature = {}
  for input_key in example_input:
    input_signature[input_key] = (tf.shape(example_input[input_key]).numpy(),
                                  example_input[input_key].dtype)
  return input_signature


def get_signature_total_size(input_signature: SignatureType) -> int:
  """Gets the total number of elements in a single problem instance

  Args:
    input_signature: An input signature to calculate the number of elements in
  """
  total_size = 0
  for input_key in input_signature:
    total_size += numpy.prod(input_signature[input_key][0])
  return total_size


def pack_flat_array_into_input(
    flat_array: numpy.typing.ArrayLike,
    signature_spec: SignatureType) -> types.NestedTensorSpec:
  """Packs a flat array into a nested tensor spec to feed into a model

  Args:
    flat_array: The data to be packed back into the specified nested tensor
      specification
    signature_spec: A signature that is used to create the correct structure
      for all of the values in the flat array
  """
  output_input_dict = {}
  current_index = 0
  for needed_input in signature_spec:
    part_size = numpy.prod(signature_spec[needed_input][0])
    needed_subset = flat_array[current_index:current_index + part_size]
    current_index += part_size
    output_input_dict[needed_input] = tf.cast(
        tf.constant(needed_subset, shape=signature_spec[needed_input][0]),
        dtype=signature_spec[needed_input][1])
  return output_input_dict


def flatten_input(to_flatten: types.NestedTensorSpec,
                  array_size: int) -> numpy.typing.ArrayLike:
  """Flattens problem instance data into a flat array for shap

  Args:
    to_flatten: A nested tensor spec of data that needs to be flattend into
      an array
    array_size: An integer representing the size of the output array. Used for
      allocating the flat array to place all the data in.
  """
  output_array = numpy.empty(array_size)
  input_index = 0
  for input_key in to_flatten:
    current_size = tf.size(to_flatten[input_key])
    end_index = input_index + current_size
    output_array[input_index:end_index] = to_flatten[input_key].numpy().astype(
        numpy.float32)
    input_index += current_size
  return output_array


def process_raw_trajectory(
    raw_trajectory: types.ForwardRef) -> types.NestedTensorSpec:
  """Processes the raw example data into a nested tensor spec that can be
  easily fed into a model.

  Args:
    raw_trajectory: Raw data representing an individual problem instance from
      a trace.
  """
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


def collapse_values(
    input_signature: SignatureType,
    shap_values: numpy.typing.ArrayLike) -> numpy.typing.ArrayLike:
  output_shap_values = numpy.empty((_NUM_EXAMPLES.value, len(input_signature)))
  for i in range(0, _NUM_EXAMPLES.value):
    current_index = 0
    current_feature = 0
    for input_key in input_signature:
      part_size = numpy.prod(input_signature[input_key][0])
      output_shap_values[i, current_feature] = numpy.sum(
          shap_values[i, current_index:current_index + part_size])
      current_feature += 1
      current_index += part_size
  return output_shap_values


def get_max_part_size(input_signature: SignatureType) -> int:
  part_sizes = numpy.empty(len(input_signature))
  for index, input_key in enumerate(input_signature):
    part_sizes[index] = numpy.prod(input_signature[input_key][0])
  return numpy.max(part_sizes)


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
  processed_shap_values = collapse_values(input_sig, shap_values)

  # if we have more than one value per feature, just set the dataset to zeros
  # as summing across a dimension produces data that doesn't really mean
  # anything
  if get_max_part_size(input_sig) > 1:
    dataset = numpy.zeros(processed_shap_values.shape)

  feature_names = list(input_sig.keys())

  output_file_data = {
      'expected_values': explainer.expected_value,
      'shap_values': processed_shap_values.tolist(),
      'data': dataset.tolist(),
      'feature_names': feature_names
  }

  with open(_OUTPUT_FILE.value, 'w', encoding='utf-8') as output_file:
    json.dump(output_file_data, output_file)


if __name__ == '__main__':
  app.run(main)
