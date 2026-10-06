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
"""Tests for compiler_opt.tools.strip_tfa_tflite_lib."""

import os

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import interpreter as tflite_interpreter
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools import flatbuffer_utils
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.networks import q_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

from compiler_opt.rl import policy_saver
from compiler_opt.tools import strip_tfa_tflite_lib


class StripTfaTfliteLibTest(tf.test.TestCase):

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

  def _create_tfa_tflite_model(self) -> str:
    test_agent = behavioral_cloning_agent.BehavioralCloningAgent(
        self._time_step_spec, self._action_spec, self._network,
        tf.compat.v1.train.AdamOptimizer())
    saver = policy_saver.MLGOPolicySaver({'saved_policy': test_agent.policy})
    root_dir = self.get_temp_dir()
    saver.save(root_dir)
    return os.path.join(root_dir, 'saved_policy',
                        policy_saver.TFLITE_MODEL_NAME)

  def test_strip_tfa_tflite_file(self):
    input_path = self._create_tfa_tflite_model()
    orig_model = flatbuffer_utils.read_model(input_path)
    self.assertGreater(len(orig_model.subgraphs), 1)
    self.assertGreater(len(orig_model.signatureDefs), 1)

    output_path = os.path.join(self.get_temp_dir(), 'stripped', 'model.tflite')
    strip_tfa_tflite_lib.strip_tfa_tflite_file(input_path, output_path)

    stripped_model = flatbuffer_utils.read_model(output_path)
    self.assertLen(stripped_model.subgraphs, 1)
    self.assertEqual(stripped_model.subgraphs[0].name, b'action')
    self.assertLen(stripped_model.signatureDefs, 1)
    self.assertEqual(stripped_model.signatureDefs[0].signatureKey, b'action')
    self.assertEqual(stripped_model.signatureDefs[0].subgraphIndex, 0)
    self.assertLess(len(stripped_model.buffers), len(orig_model.buffers))

    orig_interp = tflite_interpreter.Interpreter(model_path=input_path)
    orig_interp.allocate_tensors()
    stripped_interp = tflite_interpreter.Interpreter(model_path=output_path)
    stripped_interp.allocate_tensors()

    for orig_in, stripped_in in zip(orig_interp.get_input_details(),
                                    stripped_interp.get_input_details()):
      self.assertEqual(orig_in['name'], stripped_in['name'])
      val = np.ones(orig_in['shape'], dtype=orig_in['dtype'])
      orig_interp.set_tensor(orig_in['index'], val)
      stripped_interp.set_tensor(stripped_in['index'], val)

    orig_interp.invoke()
    stripped_interp.invoke()

    for orig_out, stripped_out in zip(orig_interp.get_output_details(),
                                      stripped_interp.get_output_details()):
      self.assertEqual(orig_out['name'], stripped_out['name'])
      self.assertAllEqual(
          orig_interp.get_tensor(orig_out['index']),
          stripped_interp.get_tensor(stripped_out['index']))

  def test_strip_idempotent_on_already_stripped_model(self):
    input_path = self._create_tfa_tflite_model()
    with tf.io.gfile.GFile(input_path, 'rb') as f:
      raw_bytes = f.read()
    once = strip_tfa_tflite_lib.strip_tfa_functions(raw_bytes)
    twice = strip_tfa_tflite_lib.strip_tfa_functions(once)
    self.assertEqual(once, twice)

  def test_garbage_collects_unused_opcodes(self):
    input_path = self._create_tfa_tflite_model()
    model = flatbuffer_utils.read_model(input_path)
    extra_opcode = schema_fb.OperatorCodeT()
    extra_opcode.builtinCode = schema_fb.BuiltinOperator.COS
    extra_opcode.deprecatedBuiltinCode = schema_fb.BuiltinOperator.COS
    model.operatorCodes.insert(0, extra_opcode)
    for op in model.subgraphs[0].operators:
      op.opcodeIndex += 1
    unused_op = schema_fb.OperatorT()
    unused_op.opcodeIndex = 0
    model.subgraphs[1].operators = [unused_op]

    strip_tfa_tflite_lib.strip_tfa_functions_from_object(model)
    opcode_codes = [oc.builtinCode for oc in model.operatorCodes]
    self.assertNotIn(schema_fb.BuiltinOperator.COS, opcode_codes)

  def test_fails_on_unexpected_subgraph(self):
    input_path = self._create_tfa_tflite_model()
    model = flatbuffer_utils.read_model(input_path)
    extra_sg = schema_fb.SubGraphT()
    extra_sg.name = b'unexpected_fn'
    model.subgraphs.append(extra_sg)
    with self.assertRaisesRegex(ValueError, 'Unexpected subgraph'):
      strip_tfa_tflite_lib.strip_tfa_functions_from_object(model)

  def test_fails_on_unexpected_signature(self):
    input_path = self._create_tfa_tflite_model()
    model = flatbuffer_utils.read_model(input_path)
    extra_sig = schema_fb.SignatureDefT()
    extra_sig.signatureKey = b'serving_default'
    extra_sig.subgraphIndex = 0
    model.signatureDefs.append(extra_sig)
    with self.assertRaisesRegex(ValueError, 'Unexpected signature keys'):
      strip_tfa_tflite_lib.strip_tfa_functions_from_object(model)

  def test_fails_on_missing_action_signature(self):
    input_path = self._create_tfa_tflite_model()
    model = flatbuffer_utils.read_model(input_path)
    model.signatureDefs = [
        s for s in model.signatureDefs if s.signatureKey != b'action'
    ]
    with self.assertRaisesRegex(ValueError, "Expected 1 b'action' signature"):
      strip_tfa_tflite_lib.strip_tfa_functions_from_object(model)


if __name__ == '__main__':
  tf.test.main()
