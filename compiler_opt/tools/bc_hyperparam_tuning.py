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
"""Script for testing different hyperparameter combinations for BC training.

This script will automatically search through a large hyperparaemter space
defined in a gin config file (eg the one for regalloc:
compiler_opt/rl/regalloc/gin_configs/bc_hyperparam_tuning.gin) using a
bayesian optimizer to find the most optimal set of hyperparameters for the
current problem for behavioral cloning.

Usage:
PYTHONPATH=$PYTHONPATH:. python3 compiler_opt/tools/bc_hyperparam_tuning.py \
  --gin_files=compiler_opt/rl/regalloc/gin_configs/bc_hyperparam_tuning.gin \
  --data_path=/default_trace \
  --max_trials=20 \
  --output_dir=./hyperparam_tuning
"""

import os
import statistics
import subprocess
import glob

from absl import app
from absl import flags
from absl import logging
import tempfile
import gin
import json

import tensorflow as tf
import keras_tuner
from tensorboard.plugins.hparams import api as hp

_DATA_PATH = flags.DEFINE_string(
    'data_path', '',
    'Path to TFRecord file(s) containing training data. Skip training and dump '
    'an untrained model with random weights (for testing purpose) if '
    'unspecified.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', '', 'Path to the output data')
_MAX_TRIALS = flags.DEFINE_integer(
    'max_trials', 5,
    'the maximum number of trials that the optimizer can use when traversing '
    'the space of possible hyperparameter combinations')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')


@gin.configurable()
def extend_command_with_gin_files(command, gin_files):
  for gin_file in gin_files:
    command.append(f'--gin_files={gin_file}')
  return command


def run_training(hparams, per_layer_hparams):
  with tempfile.TemporaryDirectory() as temp_dir:
    bc_script_command = [
        'python3', 'compiler_opt/rl/train_bc.py', f'--root_dir={temp_dir}',
        f'--data_path={_DATA_PATH.value}'
    ]
    bc_script_command = extend_command_with_gin_files(bc_script_command)
    for hparam in hparams:
      gin_name, hparam_value = hparam
      bc_script_command.append(f'--gin_bindings={gin_name}={hparam_value}')

    for per_layer_hparam in per_layer_hparams:
      bc_script_command.append(f'--gin_bindings={per_layer_hparam}='
                               f'{tuple(per_layer_hparams[per_layer_hparam])}')

    logging.info(bc_script_command)
    with subprocess.Popen(bc_script_command) as bc_process:
      bc_process.wait()
    event_file_path = glob.glob(os.path.join(temp_dir, 'events*'))[0]
    loss_values = []
    for event in tf.compat.v1.train.summary_iterator(event_file_path):
      for value in event.summary.value:
        if value.tag == 'Losses/loss':
          loss_values.append(tf.make_ndarray(value.tensor).item())
    index_to_start_from = int(len(loss_values) - 0.2 * len(loss_values))
    return statistics.mean(loss_values[index_to_start_from:])


def get_hyperparameter(name, arg_type, arg_min, arg_max, trial_hyperparams):
  if arg_type == 'int':
    return trial_hyperparams.Int(name, min_value=arg_min, max_value=arg_max)
  elif arg_type == 'float':
    return trial_hyperparams.Float(name, min_value=arg_min, max_value=arg_max)
  else:
    logging.fatal('unrecognized type specified')


class BCTuner(keras_tuner.BayesianOptimization):
  """A subclass of the keras tuner bayesian optimizer. Implements the run_trial
  method which runs an individual trial given a set of hyperparameters."""

  def run_trial(self, trial, **kwargs):
    trial_hyperparams = trial.hyperparameters
    arg_config_array = kwargs['hparams_to_configure']
    arg_gin_config_settings = []
    tensorboard_hparams = {}
    for arg_config in arg_config_array:
      gin_name, python_type, arg_min, arg_max = arg_config
      arg_hparam = get_hyperparameter(gin_name, python_type, arg_min, arg_max,
                                      trial_hyperparams)
      arg_gin_config_settings.append((gin_name, arg_hparam))
      tensorboard_hparams[gin_name] = arg_hparam

    min_layers, max_layers = kwargs['layer_count']
    num_layers = trial_hyperparams.Int(
        'layer_count', min_value=min_layers, max_value=max_layers)

    per_layer_arg_config_map = {}
    for per_layer_arg_config in kwargs['per_layer_hparams']:
      gin_name, python_type, arg_min, arg_max = per_layer_arg_config
      per_layer_arg_config_map[gin_name] = []
      for layer_index in range(0, num_layers):
        hparam_name = f'layer_{gin_name}_{layer_index}'
        arg_hparam = get_hyperparameter(hparam_name, python_type, arg_min,
                                        arg_max, trial_hyperparams)
        per_layer_arg_config_map[gin_name].append(arg_hparam)
        tensorboard_hparams[hparam_name] = arg_hparam

    loss_value = run_training(arg_gin_config_settings, per_layer_arg_config_map)

    with tf.summary.create_file_writer(
        os.path.join(_OUTPUT_DIR.value,
                     f'./tensorboard_data/{trial.trial_id}')).as_default():
      hp.hparams(tensorboard_hparams)
      tf.summary.scalar('last_loss', loss_value, step=1)

    return loss_value


@gin.configurable()
def run_tuning(hparams_to_configure, layer_count, per_layer_hparams):
  tuner = BCTuner(
      max_trials=_MAX_TRIALS.value,
      directory=_OUTPUT_DIR.value,
      project_name='bc_hparam_tuning')
  tuner.search(
      hparams_to_configure=hparams_to_configure,
      layer_count=layer_count,
      per_layer_hparams=per_layer_hparams)
  best_hparams_dict = {}
  for hparam in hparams_to_configure:
    gin_name = hparam[0]
    best_hparams_dict[gin_name] = tuner.get_best_hyperparameters()[0].get(
        gin_name)
  return best_hparams_dict


def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)

  best_hparams = run_tuning()
  with open(
      os.path.join(_OUTPUT_DIR.value, 'best_hparams.json'),
      'w',
      encoding='utf-8') as output_file:
    output_file.write(json.dumps(best_hparams))


if __name__ == '__main__':
  app.run(main)
