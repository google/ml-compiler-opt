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
"""Utilities for the feature_importance.py script

Refactored into a separate script so that we can run all of these utilities
through pytest without needing to add any odd conditionals to deal with
duplicate absl flags etc.
"""

import tensorflow as tf
import numpy
import numpy.typing

from tf_agents.typing import types
from typing import Callable, Dict, Tuple

SignatureType = Dict[str, Tuple[numpy.typing.ArrayLike, tf.dtypes.DType]]


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


def collapse_values(input_signature: SignatureType,
                    shap_values: numpy.typing.ArrayLike,
                    num_examples: int) -> numpy.typing.ArrayLike:
  """Collapses shap values so that there is only a single value per feature

  Args:
    input_signature: The signature of the model input. Used to determine what
      (if any) features need to be collapsed.
    shap_values: A numpy array of shap values that need to be processed.
  """
  output_shap_values = numpy.empty((num_examples, len(input_signature)))
  for i in range(0, num_examples):
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
  """Gets the size (as a single scalar) of the largest feature in terms of
  the number of elements.

  Args:
    input_signature: The input signature that we want to find the largest
      feature in.
  """
  part_sizes = numpy.empty(len(input_signature))
  for index, input_key in enumerate(input_signature):
    part_sizes[index] = numpy.prod(input_signature[input_key][0])
  return numpy.max(part_sizes)


def create_run_model_function(action_fn: Callable,
                              input_sig: SignatureType) -> Callable:
  """Returns a function that takes in a flattend input array and returns the
  model output as a scalar.

  Args:
    action_fn: The action function from the tensorflow saved model saved
      through tf_agents
    input_sig: The input signature for the model currently under analysis.
      Used to pack the flat array back into a nested tensor spec.
  """

  def run_model(flat_input_array):
    output = numpy.empty(flat_input_array.shape[0])
    for index, flat_input in enumerate(flat_input_array):
      input_dict = pack_flat_array_into_input(flat_input, input_sig)
      model_output = action_fn(**input_dict).items()
      # get the value of the first item as a numpy array
      output[index] = list(model_output)[0][1].numpy()[0]
    return output

  return run_model
