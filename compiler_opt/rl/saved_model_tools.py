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
"""Tools for manipulating saved models."""

import tensorflow as tf
import os

# greedy_policy is needed during conversion
from tf_agents.policies import greedy_policy  # pylint: disable=unused-import


def convert_saved_model(sm_dir: str, tflite_model_file_path: str):
  """Convert a saved model to tflite.

  Args:
    sm_dir: path to the saved model to convert

    tflite_model_file_path: desired output file path. Directory structure will
    be created by this function, as needed.
  """
  tf.io.gfile.makedirs(os.path.dirname(tflite_model_file_path))
  converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,
  ]
  tfl_model = converter.convert()
  with tf.io.gfile.GFile(tflite_model_file_path, 'wb') as f:
    f.write(tfl_model)


def convert_mlgo_model(mlgo_model_dir: str, tflite_model_dir: str):
  """Convert a mlgo saved model to mlgo tflite.

  Args:
    mlgo_model_dir: path to the mlgo saved model dir. It is expected to contain
    the saved model files (i.e. saved_model.pb, the variables dir) and the
    output_spec.json file

    tflite_model_dir: path to a directory where the tflite model will be placed.
    The model will be named model.tflite. Alongside it will be placed a copy of
    the output_spec.json file.
  """
  tf.io.gfile.makedirs(tflite_model_dir)
  convert_saved_model(mlgo_model_dir,
                      os.path.join(tflite_model_dir, 'model.tflite'))

  json_file = 'output_spec.json'
  src_json = os.path.join(mlgo_model_dir, json_file)
  dest_json = os.path.join(tflite_model_dir, json_file)
  tf.io.gfile.copy(src_json, dest_json)
