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
"""Combines two tf-agent policies with the given state and action spec."""

import tensorflow as tf
import hashlib

import tf_agents
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.trajectories import policy_step
import tensorflow_probability as tfp


class CombinedTFPolicy(tf_agents.policies.TFPolicy):
  """Policy which combines two target policies."""

  def __init__(self, *args, tf_policies: dict[str, tf_agents.policies.TFPolicy],
               **kwargs):
    super().__init__(*args, **kwargs)

    self.tf_policies = []
    self.tf_policy_names = []
    for name, policy in tf_policies.items():
      self.tf_policies.append(policy)
      self.tf_policy_names.append(name)

    self.expected_signature = self.time_step_spec
    self.sorted_keys = sorted(self.expected_signature.observation.keys())

    high_low_tensors = []
    for name in self.tf_policy_names:
      m = hashlib.md5()
      m.update(name.encode("utf-8"))
      high_low_tensors.append(
          tf.stack([
              tf.constant(
                  int.from_bytes(m.digest()[8:], "little"), dtype=tf.uint64),
              tf.constant(
                  int.from_bytes(m.digest()[:8], "little"), dtype=tf.uint64)
          ]))
    self.high_low_tensors = tf.stack(high_low_tensors)
    # Related LLVM commit: https://github.com/llvm/llvm-project/pull/96276
    m = hashlib.md5()
    m.update(self.tf_policy_names[0].encode("utf-8"))
    self.high = int.from_bytes(m.digest()[8:], "little")
    self.low = int.from_bytes(m.digest()[:8], "little")
    self.high_low_tensor = tf.constant([self.high, self.low], dtype=tf.uint64)

  def _process_observation(
      self, observation: types.NestedSpecTensorOrArray
  ) -> tuple[types.NestedSpecTensorOrArray, types.TensorOrArray]:
    assert "model_selector" in self.sorted_keys
    high_low_tensor = self.high_low_tensor
    for name in self.sorted_keys:
      if name in ["model_selector"]:
        # model_selector is a Tensor of shape (1,) which requires indexing [0]
        switch_tensor = observation.pop(name)[0]
        high_low_tensor = switch_tensor

        tf.debugging.Assert(
            tf.equal(
                tf.reduce_any(
                    tf.reduce_all(
                        tf.equal(high_low_tensor, self.high_low_tensors),
                        axis=1)), True),
            [high_low_tensor, self.high_low_tensors])

    return observation, high_low_tensor

  def _action(self,
              time_step: ts.TimeStep,
              policy_state: types.NestedTensorSpec,
              seed: types.Seed | None = None) -> policy_step.PolicyStep:
    new_observation = time_step.observation
    new_observation, switch_tensor = self._process_observation(new_observation)
    updated_step = ts.TimeStep(
        step_type=time_step.step_type,
        reward=time_step.reward,
        discount=time_step.discount,
        observation=new_observation)

    # TODO(359): We only support combining two policies. Generalize this to
    # handle multiple policies.
    def f0():
      return tf.cast(
          self.tf_policies[0].action(updated_step).action[0], dtype=tf.int64)

    def f1():
      return tf.cast(
          self.tf_policies[1].action(updated_step).action[0], dtype=tf.int64)

    action = tf.cond(
        tf.math.reduce_all(tf.equal(switch_tensor, self.high_low_tensor)), f0,
        f1)
    return policy_step.PolicyStep(action=action, state=policy_state)

  def _distribution(
      self, time_step: ts.TimeStep,
      policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
    """Placeholder for distribution as every TFPolicy requires it."""
    return policy_step.PolicyStep(
        action=tfp.distributions.Deterministic(2.), state=policy_state)
