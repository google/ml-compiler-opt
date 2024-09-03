from typing import Dict, List, Optional, Tuple

import gin
import tensorflow as tf
import hashlib

import tf_agents
from tf_agents.trajectories import time_step
from tf_agents.typing import types
from tf_agents.trajectories import policy_step
import tensorflow_probability as tfp
from tf_agents.specs import tensor_spec


class CombinedTFPolicy(tf_agents.policies.TFPolicy):

  def __init__(self, *args,
               tf_policies: Dict[str, tf_agents.policies.TFPolicy],
               **kwargs):
    super(CombinedTFPolicy, self).__init__(*args, **kwargs)

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
      m.update(name.encode('utf-8'))
      high_low_tensors.append(tf.stack([
          tf.constant(int.from_bytes(m.digest()[8:], 'little'), dtype=tf.uint64),
          tf.constant(int.from_bytes(m.digest()[:8], 'little'), dtype=tf.uint64)
          ])
      )
    self.high_low_tensors = tf.stack(high_low_tensors)

    m = hashlib.md5()
    m.update(self.tf_policy_names[0].encode('utf-8'))
    self.high = int.from_bytes(m.digest()[8:], 'little')
    self.low = int.from_bytes(m.digest()[:8], 'little')
    self.high_low_tensor = tf.constant([self.high, self.low], dtype=tf.uint64)

  def _process_observation(self, observation):
    for name in self.sorted_keys:
      if name in ['model_selector']:
        switch_tensor = observation.pop(name)[0]
        high_low_tensor = switch_tensor
    
        tf.debugging.Assert(
            tf.equal(
                tf.reduce_any(
                    tf.reduce_all(
                        tf.equal(high_low_tensor, self.high_low_tensors), axis=1
                        )
                    ),True
                ),
                 [high_low_tensor, self.high_low_tensors])
        return observation, switch_tensor

  def _create_distribution(self, inlining_prediction):
    probs = [inlining_prediction, 1.0 - inlining_prediction]
    logits = [[0.0, tf.math.log(probs[1]/(1.0 - probs[1]))]]
    return tfp.distributions.Categorical(logits=logits)

  def _action(self, time_step: time_step.TimeStep,
              policy_state: types.NestedTensorSpec,
              seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
    new_observation = time_step.observation
    new_observation, switch_tensor = self._process_observation(new_observation)
    updated_step = tf_agents.trajectories.TimeStep(step_type=time_step.step_type,
                                      reward=time_step.reward,
                                      discount=time_step.discount,
                                      observation=new_observation)
    def f0():
      return tf.cast(
          self.tf_policies[0].action(updated_step).action[0], dtype=tf.int64)
    def f1():
      return tf.cast(
          self.tf_policies[1].action(updated_step).action[0], dtype=tf.int64)
    action = tf.cond(
        tf.math.reduce_all(
            tf.equal(switch_tensor, self.high_low_tensor)),
        f0,
        f1
        )
    return tf_agents.trajectories.PolicyStep(action=action, state=policy_state)

  def _distribution(
      self, time_step: time_step.TimeStep,
      policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
    new_observation = time_step.observation
    new_observation, switch_tensor = self._process_observation(new_observation)
    updated_step = tf_agents.trajectories.TimeStep(step_type=time_step.step_type,
                                      reward=time_step.reward,
                                      discount=time_step.discount,
                                      observation=new_observation)
    def f0():
      return tf.cast(
          self.tf_policies[0].distribution(updated_step).action.cdf(0)[0],
          dtype=tf.float32)
    def f1():
      return tf.cast(
          self.tf_policies[1].distribution(updated_step).action.cdf(0)[0],
          dtype=tf.float32)
    distribution = tf.cond(
        tf.math.reduce_all(
            tf.equal(switch_tensor, self.high_low_tensor)),
        f0,
        f1
        )
    return tf_agents.trajectories.PolicyStep(
        action=self._create_distribution(distribution),
        state=policy_state)



@gin.configurable()
def get_input_signature():
    """Returns the list of features for LLVM inlining to be used in combining models."""
    # int64 features
    inputs = dict(
        (key,tf.TensorSpec(dtype=tf.int64, shape=(), name=key))
        for key in [
            "caller_basic_block_count",
            "caller_conditionally_executed_blocks",
            "caller_users",
            "callee_basic_block_count",
            "callee_conditionally_executed_blocks",
            "callee_users",
            "nr_ctant_params",
            "node_count",
            "edge_count",
            "callsite_height",
            "cost_estimate",
            "inlining_default",
            "sroa_savings",
            "sroa_losses",
            "load_elimination",
            "call_penalty",
            "call_argument_setup",
            "load_relative_intrinsic",
            "lowered_call_arg_setup",
            "indirect_call_penalty",
            "jump_table_penalty",
            "case_cluster_penalty",
            "switch_penalty",
            "unsimplified_common_instructions",
            "num_loops",
            "dead_blocks",
            "simplified_instructions",
            "constant_args",
            "constant_offset_ptr_args",
            "callsite_cost",
            "cold_cc_penalty",
            "last_call_to_static_bonus",
            "is_multiple_blocks",
            "nested_inlines",
            "nested_inline_cost_estimate",
            "threshold",
            "is_callee_avail_external",
            "is_caller_avail_external",
        ]
    )
    inputs.update({'model_selector': tf.TensorSpec(shape=(2,), dtype=tf.uint64, name='model_selector')})
    return time_step.time_step_spec(inputs)

@gin.configurable()
def get_action_spec():
  return tensor_spec.BoundedTensorSpec(
    dtype=tf.int64, shape=(), name='inlining_decision', minimum=0, maximum=1
  )