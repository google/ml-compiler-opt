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

from compiler_opt.rl import agent_config
from compiler_opt.rl import data_reader
from compiler_opt.rl import registry

from compiler_opt.tools import feature_importance_utils

import tensorflow as tf
import shap
import numpy
import numpy.typing
import json

_DATA_PATH = flags.DEFINE_multi_string(
    'data_path', [], 'Path to TFRecord file(s) containing trace data.')
_MODEL_PATH = flags.DEFINE_string('model_path', '',
                                  'Path to the model to explain')
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file', '', 'The path to the output file containing the SHAP values')
_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples', 1, 'The number of examples to process from the trace')
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
  agent_cfg = agent_config.BCAgentConfig(
      time_step_spec=time_step_spec, action_spec=action_spec)
  tfrecord_dataset_fn = data_reader.create_tfrecord_dataset_fn(
      agent_cfg=agent_cfg, batch_size=1, train_sequence_length=1)

  dataset_iter = iter(tfrecord_dataset_fn(_DATA_PATH.value).repeat())

  raw_trajectory = next(dataset_iter)

  saved_policy = tf.saved_model.load(_MODEL_PATH.value)
  action_fn = saved_policy.signatures['action']

  observation = feature_importance_utils.process_raw_trajectory(raw_trajectory)
  input_sig = feature_importance_utils.get_input_signature(observation)

  run_model = feature_importance_utils.create_run_model_function(
      action_fn, input_sig)

  total_size = feature_importance_utils.get_signature_total_size(input_sig)
  flattened_input = feature_importance_utils.flatten_input(
      observation, total_size)
  flattened_input = numpy.expand_dims(flattened_input, axis=0)
  dataset = numpy.empty((_NUM_EXAMPLES.value, total_size))
  for i in range(0, _NUM_EXAMPLES.value):
    raw_trajectory = next(dataset_iter)
    observation = feature_importance_utils.process_raw_trajectory(
        raw_trajectory)
    flat_input = feature_importance_utils.flatten_input(observation, total_size)
    dataset[i] = flat_input

  explainer = shap.KernelExplainer(run_model, numpy.zeros((1, total_size)))
  shap_values = explainer.shap_values(dataset, nsamples=1000)
  processed_shap_values = feature_importance_utils.collapse_values(
      input_sig, shap_values, _NUM_EXAMPLES.value)

  # if we have more than one value per feature, just set the dataset to zeros
  # as summing across a dimension produces data that doesn't really mean
  # anything
  if feature_importance_utils.get_max_part_size(input_sig) > 1:
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
