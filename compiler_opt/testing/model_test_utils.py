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
"""Utilities for running tests that involve tensorflow model.s"""

import os

import tensorflow as tf


# copied from the llvm regalloc generator
def gen_test_model(outdir: str):
  policy_decision_label = 'index_to_evict'
  policy_output_spec = """
  [
      {
          "logging_name": "index_to_evict",
          "tensor_spec": {
              "name": "StatefulPartitionedCall",
              "port": 0,
              "type": "int64_t",
              "shape": [
                  1
              ]
          }
      }
  ]
  """
  per_register_feature_list = ['mask']
  num_registers = 33

  def get_input_signature():
    """Returns (time_step_spec, action_spec) for LLVM register allocation."""
    inputs = {
        key: tf.TensorSpec(dtype=tf.int64, shape=(num_registers), name=key)
        for key in per_register_feature_list
    }
    return inputs

  module = tf.Module()
  # We have to set this useless variable in order for the TF C API to correctly
  # intake it
  module.var = tf.Variable(0, dtype=tf.int64)

  def action(*inputs):
    result = tf.math.argmax(
        tf.cast(inputs[0]['mask'], tf.int32), axis=-1) + module.var
    return {policy_decision_label: result}

  module.action = tf.function()(action)
  action = {
      'action': module.action.get_concrete_function(get_input_signature())
  }
  tf.saved_model.save(module, outdir, signatures=action)
  output_spec_path = os.path.join(outdir, 'output_spec.json')
  with tf.io.gfile.GFile(output_spec_path, 'w') as f:
    print(f'Writing output spec to {output_spec_path}.')
    f.write(policy_output_spec)
