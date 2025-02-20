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
from absl import flags
from absl import logging

import sys

import gin

import tensorflow as tf

from compiler_opt.rl import policy_saver
from compiler_opt.rl import registry
from compiler_opt.tools import combine_tfa_policies_lib as cfa_lib

_COMBINE_POLICIES_NAMES = flags.DEFINE_multi_string(
    'policies_names', [], 'List in order of policy names for combined policies.'
    'Order must match that of policies_paths.')
_COMBINE_POLICIES_PATHS = flags.DEFINE_multi_string(
    'policies_paths', [], 'List in order of policy paths for combined policies.'
    'Order must match that of policies_names.')
_COMBINED_POLICY_PATH = flags.DEFINE_string(
    'combined_policy_path', '', 'Path to save the combined policy.')
_GIN_FILES = flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files.')
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')


def main(_):
  flags.mark_flag_as_required('policies_names')
  flags.mark_flag_as_required('policies_paths')
  flags.mark_flag_as_required('combined_policy_path')
  if len(_COMBINE_POLICIES_NAMES.value) != len(_COMBINE_POLICIES_PATHS.value):
    logging.error(
        'Length of policies_names: %d must equal length of policies_paths: %d.',
        len(_COMBINE_POLICIES_NAMES.value), len(_COMBINE_POLICIES_PATHS.value))
    sys.exit(1)
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)

  problem_config = registry.get_configuration()
  expected_signature, action_spec = problem_config.get_signature_spec()
  expected_signature.observation.update({
      'model_selector':
          tf.TensorSpec(shape=(2,), dtype=tf.uint64, name='model_selector')
  })
  # TODO(359): We only support combining two policies.Generalize this to handle
  # multiple policies.
  if len(_COMBINE_POLICIES_NAMES.value) != 2:
    logging.error('Policy combiner only supports two policies, %d given.',
                  len(_COMBINE_POLICIES_NAMES.value))
    sys.exit(1)
  policy1_name = _COMBINE_POLICIES_NAMES.value[0]
  policy1_path = _COMBINE_POLICIES_PATHS.value[0]
  policy2_name = _COMBINE_POLICIES_NAMES.value[1]
  policy2_path = _COMBINE_POLICIES_PATHS.value[1]
  policy1 = tf.saved_model.load(policy1_path, tags=None, options=None)
  policy2 = tf.saved_model.load(policy2_path, tags=None, options=None)
  combined_policy = cfa_lib.CombinedTFPolicy(
      tf_policies={
          policy1_name: policy1,
          policy2_name: policy2
      },
      time_step_spec=expected_signature,
      action_spec=action_spec)
  combined_policy_path = _COMBINED_POLICY_PATH.value
  policy_dict = {'combined_policy': combined_policy}
  saver = policy_saver.PolicySaver(policy_dict=policy_dict)
  saver.save(combined_policy_path)


if __name__ == '__main__':
  app.run(main)
