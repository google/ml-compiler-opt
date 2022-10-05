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
"""util function to save the policy and model config file."""

import dataclasses
import json
import os

import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.policies import policy_saver

from typing import Dict, Tuple

OUTPUT_SIGNATURE = 'output_spec.json'
TFLITE_MODEL_NAME = 'model.tflite'

_TYPE_CONVERSION_DICT = {
    tf.float32: 'float',
    tf.float64: 'double',
    tf.int8: 'int8_t',
    tf.uint8: 'uint8_t',
    tf.int16: 'int16_t',
    tf.uint16: 'uint16_t',
    tf.int32: 'int32_t',
    tf.uint32: 'uint32_t',
    tf.int64: 'int64_t',
    tf.uint64: 'uint64_t',
}


def _split_tensor_name(name):
  """Return tuple (op, port) with the op and int port for the tensor name."""
  op_port = name.split(':', 2)
  if len(op_port) == 1:
    return op_port, 0
  else:
    return op_port[0], int(op_port[1])


# TODO(b/156295309): more sophisticated way of finding tensor names.
def _get_non_identity_op(tensor):
  """Get the true output op aliased by Identity `tensor`.

  Output signature tensors are in a Function that refrences the true call
  in the base SavedModel metagraph.  Traverse the function upwards until
  we find this true output op and tensor and return that.

  Args:
    tensor: A tensor from the unstructured output list of a signature.

  Returns:
    The true associated output tensor of the original function in the main
    SavedModel graph.
  """
  while tensor.op.name.startswith('Identity'):
    tensor = tensor.op.inputs[0]
  return tensor


def convert_saved_model(sm_dir: str, tflite_model_path: str):
  """Convert a saved model to tflite.

  Args:
    sm_dir: path to the saved model to convert

    tflite_model_path: desired output file path. Directory structure will
      be created by this function, as needed.
  """
  tf.io.gfile.makedirs(os.path.dirname(tflite_model_path))
  converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,
  ]
  converter.allow_custom_ops = True
  tfl_model = converter.convert()
  with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
    f.write(tfl_model)


def convert_mlgo_model(mlgo_model_dir: str, tflite_model_dir: str):
  """Convert a mlgo saved model to mlgo tflite.

  Args:
    mlgo_model_dir: path to the mlgo saved model dir. It is expected to contain
      the saved model files (i.e. saved_model.pb, the variables dir) and the
      output_spec.json file

    tflite_model_dir: path to a directory where the tflite model will be placed.
      The model will be named model.tflite. Alongside it will be placed a copy
      of the output_spec.json file.
  """
  tf.io.gfile.makedirs(tflite_model_dir)
  convert_saved_model(mlgo_model_dir,
                      os.path.join(tflite_model_dir, TFLITE_MODEL_NAME))

  src_json = os.path.join(mlgo_model_dir, OUTPUT_SIGNATURE)
  dest_json = os.path.join(tflite_model_dir, OUTPUT_SIGNATURE)
  tf.io.gfile.copy(src_json, dest_json)


@dataclasses.dataclass(frozen=True)
class Policy:
  """Serialized mlgo policy, used to pass a policy to workers.

  A policy has 2 components, both being file contents:
    - the content of the output_spec.json file;
    - the content of the tflite policy.

  To construct from a directory accessible by tf.io.gfile:

  policy = Policy.from_filesystem(that_dir)

  To make available to the compiler in a directory:

  policy.to_filesystem(that_dir)
  """

  output_spec: bytes
  policy: bytes

  def to_filesystem(self, location: str):
    os.makedirs(location, exist_ok=True)
    output_sig = os.path.join(location, OUTPUT_SIGNATURE)
    policy_path = os.path.join(location, TFLITE_MODEL_NAME)
    with tf.io.gfile.GFile(output_sig, mode='wb') as f:
      f.write(self.output_spec)
    with tf.io.gfile.GFile(policy_path, mode='wb') as f:
      f.write(self.policy)

  @staticmethod
  def from_filesystem(location: str):
    output_sig = os.path.join(location, OUTPUT_SIGNATURE)
    policy_path = os.path.join(location, TFLITE_MODEL_NAME)
    with tf.io.gfile.GFile(output_sig, mode='rb') as f:
      output_spec = f.read()
    with tf.io.gfile.GFile(policy_path, mode='rb') as f:
      policy = f.read()
    return Policy(output_spec=output_spec, policy=policy)


class PolicySaver(object):
  """Object that saves policy and model config file required by inference.

  ```python
  policy_saver = PolicySaver(policy_dict, config)
  policy_saver.save(root_dir)
  ```
  """

  def __init__(self, policy_dict: Dict[str, tf_policy.TFPolicy]):
    """Initialize the PolicySaver object.

    Args:
      policy_dict: A dict mapping from policy name to policy.
    """
    self._policy_saver_dict: Dict[str, Tuple[
        policy_saver.PolicySaver, tf_policy.TFPolicy]] = {
            policy_name: (policy_saver.PolicySaver(
                policy, batch_size=1, use_nest_path_signatures=False), policy)
            for policy_name, policy in policy_dict.items()
        }

  def _save_policy(self, saver, path):
    """Writes policy, model weights and model_binding.txt to path/."""
    saver.save(path)

  def _write_output_signature(self, saver, path):
    """Writes the output_signature json file into the SavedModel directory."""
    action_signature = saver.policy_step_spec

    # We'll load the actual SavedModel to be able to map signature names to
    # actual tensor names.
    saved_model = tf.saved_model.load(path)

    # Dict mapping spec name to spec in flattened action signature.
    sm_action_signature = (
        tf.nest.flatten(saved_model.signatures['action'].structured_outputs))

    # Map spec name to index in flattened outputs.
    sm_action_indices = dict(
        (k.name.lower(), i) for i, k in enumerate(sm_action_signature))

    # List mapping flattened structured outputs to tensors.
    sm_action_tensors = saved_model.signatures['action'].outputs

    # First entry in output list is the decision (action)
    decision_spec = tf.nest.flatten(action_signature.action)
    if len(decision_spec) != 1:
      raise ValueError(('Expected action decision to have 1 tensor, but '
                        f'saw: {action_signature.action}'))

    # Find the decision's tensor in the flattened output tensor list.
    sm_action_decision = (
        sm_action_tensors[sm_action_indices[decision_spec[0].name.lower()]])

    sm_action_decision = _get_non_identity_op(sm_action_decision)

    # The first entry in the output_signature file corresponds to the decision.
    (tensor_op, tensor_port) = _split_tensor_name(sm_action_decision.name)
    output_list = [{
        'logging_name': decision_spec[0].name,  # used in SequenceExample.
        'tensor_spec': {
            'name': tensor_op,
            'port': tensor_port,
            'type': _TYPE_CONVERSION_DICT[sm_action_decision.dtype],
            'shape': sm_action_decision.shape.as_list(),
        }
    }]
    for info_spec in tf.nest.flatten(action_signature.info):
      sm_action_info = sm_action_tensors[sm_action_indices[
          info_spec.name.lower()]]
      sm_action_info = _get_non_identity_op(sm_action_info)
      (tensor_op, tensor_port) = _split_tensor_name(sm_action_info.name)
      output_list.append({
          'logging_name': info_spec.name,  # used in SequenceExample.
          'tensor_spec': {
              'name': tensor_op,
              'port': tensor_port,
              'type': _TYPE_CONVERSION_DICT[sm_action_info.dtype],
              'shape': sm_action_info.shape.as_list(),
          }
      })

    with tf.io.gfile.GFile(os.path.join(path, OUTPUT_SIGNATURE), 'w') as f:
      f.write(json.dumps(output_list))

  def save(self, root_dir: str):
    """Writes policy and model_binding.txt to root_dir/policy_name/."""
    for policy_name, (saver, _) in self._policy_saver_dict.items():
      saved_model_dir = os.path.join(root_dir, policy_name)
      self._save_policy(saver, saved_model_dir)
      self._write_output_signature(saver, saved_model_dir)
      # This is not quite the most efficient way to do this - we save the model
      # just to load it again and save it as tflite - but it's the minimum,
      # temporary step so we can validate more thoroughly our use of tflite.
      convert_saved_model(saved_model_dir,
                          os.path.join(saved_model_dir, TFLITE_MODEL_NAME))
