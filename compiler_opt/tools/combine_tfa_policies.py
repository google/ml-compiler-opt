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
"""Runs the policy combiner."""
from absl import app

import tensorflow as tf

from compiler_opt.rl import policy_saver
from compiler_opt.tools import combine_tfa_policies_lib as cfa_lib


def main(_):
  expected_signature = cfa_lib.get_input_signature()
  action_spec = cfa_lib.get_action_spec()
  policy1_name = input("First policy name: ")
  policy1_path = input(policy1_name + " path: ")
  policy2_name = input("Second policy name: ")
  policy2_path = input(policy2_name + " path: ")
  policy1 = tf.saved_model.load(policy1_path, tags=None, options=None)
  policy2 = tf.saved_model.load(policy2_path, tags=None, options=None)
  combined_policy = cfa_lib.CombinedTFPolicy(
     tf_policies={policy1_name:policy1, policy2_name:policy2},
     time_step_spec=expected_signature,
     action_spec=action_spec
  )
  combined_policy_path = input("Save combined policy path: ")
  policy_dict = {'combined_policy': combined_policy}
  saver = policy_saver.PolicySaver(policy_dict=policy_dict)
  saver.save(combined_policy_path)

if __name__ == "__main__": 
    app.run(main)

