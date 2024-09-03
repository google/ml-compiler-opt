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

