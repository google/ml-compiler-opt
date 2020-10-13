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

import json
import os

import tensorflow as tf

from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.networks import q_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

from compiler_opt.rl import policy_saver


class PolicySaverTest(tf.test.TestCase):

  def setUp(self):
    super(PolicySaverTest, self).setUp()
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

if __name__ == '__main__':
  tf.test.main()
