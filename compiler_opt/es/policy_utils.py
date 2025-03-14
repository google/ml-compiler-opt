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
"""Util functions to create and edit a tf_agent policy."""

from typing import Protocol
from collections.abc import Sequence
import os

import gin
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import tf_policy

from compiler_opt.rl import policy_saver
from compiler_opt.rl import registry


class HasModelVariables(Protocol):
  model_variables: Sequence[tf.Variable]


# TODO(abenalaast): Issue #280
@gin.configurable(module='policy_utils')
def create_actor_policy(
    actor_network_ctor: type[network.DistributionNetwork],
    greedy: bool = False,
) -> tf_policy.TFPolicy:
  """Creates an actor policy."""
  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()
  layers = tf.nest.map_structure(
      problem_config.get_preprocessing_layer_creator(),
      time_step_spec.observation)

  actor_network = actor_network_ctor(
      input_tensor_spec=time_step_spec.observation,
      output_tensor_spec=action_spec,
      preprocessing_layers=layers)

  policy = actor_policy.ActorPolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=actor_network)

  if greedy:
    policy = greedy_policy.GreedyPolicy(policy)

  return policy


def get_vectorized_parameters_from_policy(
    policy: 'tf_policy.TFPolicy | HasModelVariables'
) -> npt.NDArray[np.float32]:
  """Returns a policy's variable values as a single np array."""
  if isinstance(policy, tf_policy.TFPolicy):
    variables = policy.variables()
  elif hasattr(policy, 'model_variables'):
    variables = policy.model_variables
  else:
    raise ValueError(f'Policy must be a TFPolicy or a loaded SavedModel. '
                     f'Passed policy: {policy}')

  parameters = [var.numpy().flatten() for var in variables]
  parameters = np.concatenate(parameters, axis=0)
  return parameters


def set_vectorized_parameters_for_policy(
    policy: 'tf_policy.TFPolicy | HasModelVariables',
    parameters: npt.NDArray[np.float32]) -> None:
  """Separates values in parameters.

  Packs parameters into the policy's shapes and sets the policy variables to
  those values.
  """
  if isinstance(policy, tf_policy.TFPolicy):
    variables = policy.variables()
  elif hasattr(policy, 'model_variables'):
    variables = policy.model_variables
  else:
    raise ValueError(f'Policy must be a TFPolicy or a loaded SavedModel. '
                     f'Passed policy: {policy}')

  param_pos = 0
  for variable in variables:
    shape = tf.shape(variable).numpy()
    num_elems = np.prod(shape)
    param = np.reshape(parameters[param_pos:param_pos + num_elems], shape)
    variable.assign(param)
    param_pos += num_elems
  if param_pos != len(parameters):
    raise ValueError(
        f'Parameter dimensions are not matched! Expected {len(parameters)} '
        f'but only found {param_pos}.')


def save_policy(policy: 'tf_policy.TFPolicy | HasModelVariables',
                parameters: npt.NDArray[np.float32], save_folder: str,
                policy_name: str) -> None:
  """Assigns a policy a name and writes it to disk.

  Args:
    policy: The policy to save.
    parameters: The model weights for the policy.
    save_folder: The location to save the policy to.
    policy_name: The value to name the policy.
  """
  set_vectorized_parameters_for_policy(policy, parameters)
  saver = policy_saver.PolicySaver({policy_name: policy})
  saver.save(save_folder)


def convert_to_tflite(policy_as_bytes: bytes, scratch_dir: str,
                      base_policy_path: str) -> str:
  """Converts a policy serialized to bytes to TFLite.

  Args:
    policy_as_bytes: An array of model parameters serialized to a byte stream.
    scratch_dir: A temporary directory being used for scratch that the model
      will get saved into.
    base_policy_path: The path to the base TF saved model that is used to
      determine the model architecture.
  """
  perturbation = np.frombuffer(policy_as_bytes, dtype=np.float32)

  saved_model = tf.saved_model.load(base_policy_path)
  set_vectorized_parameters_for_policy(saved_model, perturbation)

  saved_model_dir = os.path.join(scratch_dir, 'saved_model')
  tf.saved_model.save(
      saved_model, saved_model_dir, signatures=saved_model.signatures)
  source = os.path.join(base_policy_path, policy_saver.OUTPUT_SIGNATURE)
  destination = os.path.join(saved_model_dir, policy_saver.OUTPUT_SIGNATURE)
  tf.io.gfile.copy(source, destination)

  # convert to tflite
  tflite_dir = os.path.join(scratch_dir, 'tflite')
  policy_saver.convert_mlgo_model(saved_model_dir, tflite_dir)
  return tflite_dir
