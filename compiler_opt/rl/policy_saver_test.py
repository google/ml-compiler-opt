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
"""Tests for compiler_opt.rl.policy_saver."""

import filecmp
import json
import os

import tensorflow as tf

from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.networks import q_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

from compiler_opt.rl import policy_saver


# copied from the llvm regalloc generator
def _gen_test_model(outdir: str):
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
    inputs = dict(
        (key, tf.TensorSpec(dtype=tf.int64, shape=(num_registers), name=key))
        for key in per_register_feature_list)
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


class PolicySaverTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    observation_spec = tf.TensorSpec(
        dtype=tf.int64, shape=(), name='callee_users')
    self._time_step_spec = time_step.time_step_spec(observation_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int64,
        shape=(),
        minimum=0,
        maximum=1,
        name='inlining_decision')
    self._network = q_rnn_network.QRnnNetwork(
        input_tensor_spec=self._time_step_spec.observation,
        action_spec=self._action_spec,
        lstm_size=(40,))

  def test_save_policy(self):
    test_agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec, self._action_spec, self._network,
        tf.compat.v1.train.AdamOptimizer())
    policy_dict = {
        'saved_policy': test_agent.policy,
        'saved_collect_policy': test_agent.collect_policy
    }
    test_policy_saver = policy_saver.PolicySaver(policy_dict=policy_dict)

    root_dir = self.get_temp_dir()
    test_policy_saver.save(root_dir)

    sub_dirs = tf.io.gfile.listdir(root_dir)
    self.assertCountEqual(['saved_policy', 'saved_collect_policy'], sub_dirs)

    for sub_dir in ['saved_policy', 'saved_collect_policy']:
      self.assertTrue(
          tf.io.gfile.exists(os.path.join(root_dir, sub_dir, 'saved_model.pb')))
      self.assertTrue(
          tf.io.gfile.exists(os.path.join(root_dir, sub_dir, 'model.tflite')))
      self.assertTrue(
          tf.io.gfile.exists(
              os.path.join(root_dir, sub_dir,
                           'variables/variables.data-00000-of-00001')))
      output_signature_fn = os.path.join(root_dir, sub_dir, 'output_spec.json')
      self.assertTrue(tf.io.gfile.exists(output_signature_fn))
      self.assertEqual([{
          'logging_name': 'inlining_decision',
          'tensor_spec': {
              'name': 'StatefulPartitionedCall',
              'port': 0,
              'type': 'int64_t',
              'shape': [1],
          }
      }], json.loads(tf.io.gfile.GFile(output_signature_fn).read()))

  def test_tflite_conversion(self):
    sm_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    tflite_dir = os.path.join(self.get_temp_dir(), 'tflite_model')
    _gen_test_model(sm_dir)
    policy_saver.convert_mlgo_model(sm_dir, tflite_dir)
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(tflite_dir, policy_saver.TFLITE_MODEL_NAME)))
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(tflite_dir, policy_saver.OUTPUT_SIGNATURE)))

  def test_policy_serialization(self):
    sm_dir = os.path.join(self.get_temp_dir(), 'model')
    orig_dir = os.path.join(self.get_temp_dir(), 'orig_model')
    dest_dir = os.path.join(self.get_temp_dir(), 'dest_model')
    _gen_test_model(sm_dir)
    policy_saver.convert_mlgo_model(sm_dir, orig_dir)

    serialized_policy = policy_saver.Policy.from_filesystem(orig_dir)
    serialized_policy.to_filesystem(dest_dir)

    self.assertTrue(
        filecmp.cmp(
            os.path.join(orig_dir, policy_saver.TFLITE_MODEL_NAME),
            os.path.join(dest_dir, policy_saver.TFLITE_MODEL_NAME),
            shallow=False))
    self.assertTrue(
        filecmp.cmp(
            os.path.join(orig_dir, policy_saver.OUTPUT_SIGNATURE),
            os.path.join(dest_dir, policy_saver.OUTPUT_SIGNATURE),
            shallow=False))


if __name__ == '__main__':
  tf.test.main()
