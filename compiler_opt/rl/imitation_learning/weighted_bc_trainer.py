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
"""Module for training an inlining policy with imitation learning."""

import json

import gin
import tensorflow as tf
from absl import app, flags, logging

from compiler_opt.rl import policy_saver
from compiler_opt.rl.imitation_learning.weighted_bc_trainer_lib import (
  ImitationLearningTrainer,
  TrainingWeights,
  WrapKerasModel,
)
from compiler_opt.rl.inlining import imitation_learning_config as config

_TRAINING_DATA = flags.DEFINE_multi_string(
    'training_data', None, 'Training data for one step of BC-Max')
_PROFILING_DATA = flags.DEFINE_multi_string(
    'profiling_data', None,
    ('Paths to profile files for computing the TrainingWeights'
     'If specified the order for each pair of json files is'
     'comparator.json followed by eval.json and the number of'
     'files should always be even.'))
_SAVE_MODEL_DIR = flags.DEFINE_string(
    'save_model_dir', None, 'Location to save the keras and TFAgents policies.')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')


def train():
  training_weights = None
  if _PROFILING_DATA.value:
    if len(_PROFILING_DATA.value) % 2 != 0:
      raise ValueError('Profiling file paths should always be an even number.')
    training_weights = TrainingWeights()
    for i in range(len(_PROFILING_DATA.value) // 2):
      with open(
          _PROFILING_DATA.value[2 * i], encoding='utf-8') as comp_f, open(
              _PROFILING_DATA.value[2 * i + 1], encoding='utf-8') as eval_f:
        comparator_prof = json.load(comp_f)
        eval_prof = json.load(eval_f)
        training_weights.update_weights(
            comparator_profile=comparator_prof, policy_profile=eval_prof)
  trainer = ImitationLearningTrainer(
      save_model_dir=_SAVE_MODEL_DIR.value, training_weights=training_weights)
  trainer.train(filepaths=_TRAINING_DATA.value)
  if _SAVE_MODEL_DIR.value:
    keras_policy = trainer.get_policy()
    expected_signature, action_spec = config.get_input_signature()
    wrapped_keras_model = WrapKerasModel(
        keras_policy=keras_policy,
        time_step_spec=expected_signature,
        action_spec=action_spec)
    policy_dict = {'tf_agents_policy': wrapped_keras_model}
    saver = policy_saver.PolicySaver(policy_dict=policy_dict)
    saver.save(_SAVE_MODEL_DIR.value)


def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, _GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  tf.compat.v1.enable_eager_execution()

  train()


if __name__ == '__main__':
  app.run(main)
