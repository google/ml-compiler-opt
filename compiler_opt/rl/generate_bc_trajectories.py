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
"""Module for running compilation and collect data for behavior cloning."""

import concurrent.futures
import contextlib
import functools
import gin
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Generator, Union
import json

from absl import app
from absl import flags
from absl import logging
import bisect
import dataclasses
import os
import shutil
import timeit

import math
import numpy as np
import scipy
import scipy.special
import tensorflow as tf
from tf_agents import policies
from tf_agents.typing import types as tf_types
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing as multiprocessing

from compiler_opt.rl import corpus
from compiler_opt.rl import env

from compiler_opt.distributed import worker
from compiler_opt.distributed import buffered_scheduler
from compiler_opt.distributed.local import local_worker_manager

from compiler_opt.tools import generate_test_model  # pylint:disable=unused-import

flags.FLAGS['gin_files'].allow_override = True
flags.FLAGS['gin_bindings'].allow_override = True

FLAGS = flags.FLAGS

ProfilingDictValueType = Dict[str, Union[str, float, int]]


@dataclasses.dataclass
class SequenceExampleFeatureNames:
  """Feature names for features that are always added to seq example."""
  action: str = 'action'
  reward: str = 'reward'
  module_name: str = 'module_name'


def get_loss(seq_example: tf.train.SequenceExample,
             reward_key: str = SequenceExampleFeatureNames.reward) -> int:
  """Return the last loss/reward of a trajectory written in a SequenceExample.

  Args:
    seq_example: tf.train.SequenceExample which contains the trajectory with
      all features, including a reward feature
    reward_key: the name of the feature that contains the loss/reward.

  Returns:
    The loss/reward of a trajectory written in a SequenceExample.
  """
  return (seq_example.feature_lists.feature_list[reward_key].feature[-1]
          .float_list.value[0])


def add_int_feature(
    sequence_example: tf.train.SequenceExample,
    feature_value: np.int64,
    feature_name: str,
):
  """Add an int feature to feature list.

  Args:
    sequence_example: sequence example to use instead of the one belonging to
      the instance
    feature_value: np.int64 value of feature, this is the required type by
      tf.train.SequenceExample for an int list
    feature_name: name of feature
  """
  f = sequence_example.feature_lists.feature_list[feature_name].feature.add()
  lst = f.int64_list.value
  lst.extend([feature_value])


def add_float_feature(
    sequence_example: tf.train.SequenceExample,
    feature_value: np.float32,
    feature_name: str,
):
  """Add a float feature to feature list.

  Args:
    sequence_example: sequence example to use instead of the one belonging to
      the instance
    feature_value: np.float32 value of feature, this is the required type by
      tf.train.SequenceExample for an float list
    feature_name: name of feature
  """
  f = sequence_example.feature_lists.feature_list[feature_name].feature.add()
  lst = f.float_list.value
  lst.extend([feature_value])


def add_string_feature(
    sequence_example: tf.train.SequenceExample,
    feature_value: str,
    feature_name: str,
):
  """Add a string feature to feature list.

  Args:
    sequence_example: sequence example to use instead of the one
    feature_value: tf.string value of feature
    feature_name: name of feature
  """
  f = sequence_example.feature_lists.feature_list[feature_name].feature.add()
  lst = f.bytes_list.value
  lst.extend([feature_value.encode('utf-8')])


def add_feature_list(seq_example: tf.train.SequenceExample,
                     feature_list: List[Any], feature_name: str):
  """Add the feature_list to the sequence example under feature name.

  Args:
    seq_example: sequence example to update
    feature_list: list of feature values to add to seq_example
    feature_name: name of the feature to add the list under
  """
  if (type(feature_list[0]) not in [
      np.dtype(np.int64),
      np.dtype(np.float32),
      str,
  ]):
    raise AssertionError((f'Unsupported type for feautre {feature_name}'
                          f' of type {type(feature_list[0])}. '
                          'Supported types are np.int64, np.float32, str'))
  if isinstance(feature_list[0], np.float32):
    add_function = add_float_feature
  elif isinstance(feature_list[0], (int, np.int64)):
    add_function = add_int_feature
  else:
    add_function = add_string_feature
  for feature in feature_list:
    add_function(seq_example, feature, feature_name)


def policy_action_wrapper(tf_policy) -> Callable[[Any], np.ndarray]:
  """Return a wrapper for a loaded policy action.

  The returned function maps from an (optional) state to an np.array
      that represents the action.

  Args:
    tf_policy: a policy (optionally can be tf_policy)

  Returns:
    wrap_function: function mapping a state to an np.array action.
  """

  def wrap_function(*args, **kwargs):
    return np.array(tf_policy.action(*args, **kwargs).action)

  return wrap_function


def policy_distr_wrapper(
    tf_policy: policies.TFPolicy
) -> Callable[[time_step.TimeStep, Optional[tf_types.NestedTensor]],
              policy_step.PolicyStep]:
  """Return a wrapper for a loaded tf policy distribution.

  The returned function maps from a state to a distribution over all actions.

  Args:
    tf_policy: A loaded tf policy.

  Returns:
    wrap_function: function mapping a state to a distribution over all actions.
  """

  def wrap_function(*args, **kwargs) -> policy_step.PolicyStep:
    return tf_policy.distribution(*args, **kwargs)

  return wrap_function


class ExplorationWithPolicy:
  """Policy which selects states for exploration.

  Exploration is facilitated in the following way. First the policy plays
  all actions from the replay_prefix. At the following state the policy computes
  a gap which is difference between the most likely action and the second most
  likely action according to the randomized exploration policy (distr).
  If the current gap is smaller than previously maintained gap, the gap is
  updated and the exploration state is set to the current state.
  The trajectory is completed by following following the policy from the
  constructor.

  Attributes:
    replay_prefix: a replay buffer of actions
    policy: policy to follow after exhausting the replay buffer
    explore_policy: randomized policy which is used to compute the gap
    curr_step: current step of the trajectory
    explore_step: current candidate for exploration step
    explore_state: current candidate state for exploration at explore_step
    gap: current difference at explore step between probability of most likely
      action according to explore_policy and second most likely action
    explore_on_features: dict of feature names and functions which specify
      when to explore on the respective feature
  """

  def __init__(
      self,
      replay_prefix: List[np.ndarray],
      policy: Callable[[time_step.TimeStep], np.ndarray],
      explore_policy: Callable[[time_step.TimeStep], policy_step.PolicyStep],
      explore_on_features: Optional[Dict[str, Callable[[tf.Tensor],
                                                       bool]]] = None,
  ):
    self._explore_step: int = len(replay_prefix) - 1
    self._explore_state: Optional[time_step.TimeStep] = None
    self._replay_prefix = replay_prefix
    self._policy = policy
    self._explore_policy = explore_policy
    self._curr_step = 0
    self._gap = np.inf
    self._explore_on_features: Optional[Dict[str, Callable[
        [tf.Tensor], bool]]] = explore_on_features
    self._stop_exploration = False

  def _compute_gap(self, distr: np.ndarray) -> np.float32:
    if distr.shape[0] < 2:
      return np.inf
    sorted_distr = np.sort(distr)
    return sorted_distr[-1] - sorted_distr[-2]

  def get_explore_step(self) -> int:
    return self._explore_step

  def get_explore_state(self) -> Optional[time_step.TimeStep]:
    return self._explore_state

  def get_advice(self, state: time_step.TimeStep) -> np.ndarray:
    """Action function for the policy.

    Args:
      state: current state in the trajectory

    Returns:
      policy_action: action to take at the current state.

    """
    if self._curr_step < len(self._replay_prefix):
      self._curr_step += 1
      return np.array(self._replay_prefix[self._curr_step - 1])
    policy_action = self._policy(state)
    # explore_policy(state) should play at least one action per state and so
    # self._explore_policy(state).action.logits should have at least one entry
    distr = tf.nn.softmax(self._explore_policy(state).action.logits).numpy()[0]
    curr_gap = self._compute_gap(distr)
    # selecting explore_step is done based on smallest encountered gap in the
    # play of self.policy. This logic can be changed to have different type
    # of exploration.
    if (not self._stop_exploration and distr.shape[0] > 1 and
        self._gap > curr_gap):
      self._gap = curr_gap
      self._explore_step = self._curr_step
      self._explore_state = state
    if not self._stop_exploration and self._explore_on_features is not None:
      for feature_name, explore_on_feature in self._explore_on_features.items():
        if explore_on_feature(state.observation[feature_name]):
          self._explore_step = self._curr_step
          self._stop_exploration = True
          break
    self._curr_step += 1
    return policy_action


class ModuleExplorer:
  """Class which implements the exploration for the given module.

  Attributes:
    loaded_module_spec: the module to be compiled and explored
    env: MLGO environment.
    exploration_frac: how often to explore in a trajectory
    max_exploration_steps: maximum number of exploration steps
    max_horizon_to_explore: if the horizon under policy is greater than this
      we do not do exploration
    explore_on_features: dict of feature names and functions which specify
      when to explore on the respective feature
    reward_key: which reward binary to use, must be specified as part of
      additional task args (kwargs).
  """

  def __init__(
      self,
      loaded_module_spec: corpus.LoadedModuleSpec,
      clang_path: str,
      mlgo_task_type: Type[env.MLGOTask],
      exploration_frac: float = 1.0,
      max_exploration_steps: int = 10,
      max_horizon_to_explore=np.inf,
      explore_on_features: Optional[Dict[str, Callable[[tf.Tensor],
                                                       bool]]] = None,
      obs_action_specs: Optional[Tuple[time_step.TimeStep,
                                       tensor_spec.BoundedTensorSpec,]] = None,
      reward_key: str = '',
      **kwargs,
  ):
    self._loaded_module_spec = loaded_module_spec
    if not obs_action_specs:
      obs_spec = None
      action_spec = None
    else:
      obs_spec = obs_action_specs[0].observation
      action_spec = obs_action_specs[1]

    if reward_key == '':
      raise TypeError(
          'reward_key not specified in ModuleExplorer initialization.')
    self._reward_key = reward_key
    kwargs.pop('reward_key', None)
    self._working_dir = None

    self._env = env.MLGOEnvironmentBase(
        clang_path=clang_path,
        task_type=mlgo_task_type,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    if self._env.action_spec:
      if self._env.action_spec.dtype != tf.int64:
        raise TypeError(
            ('Environment action_spec type '
             f'{self._env.action_spec.dtype} does not match tf.int64'))
    self._exploration_frac = exploration_frac
    self._max_exploration_steps = max_exploration_steps
    self._max_horizon_to_explore = max_horizon_to_explore
    self._explore_on_features = explore_on_features
    logging.info('Reward key in exploration worker: %s', self._reward_key)

  def compile_module(
      self,
      policy: Callable[[Optional[time_step.TimeStep]], np.ndarray],
  ) -> tf.train.SequenceExample:
    """Compiles the module with the given policy and outputs a seq. example.

    Args:
      policy: policy to compile with

    Returns:
      sequence_example: a tf.train.SequenceExample containing the trajectory
        from compilation. In addition to the features returned from the env
        tbe sequence_example adds the following extra features: action,
        reward and module_name. action is the action taken at any given step,
        reward is the reward specified by reward_key, not necessarily the
        reward returned by the environment and module_name is the name of
        the module processed by the compiler.
    """
    sequence_example = tf.train.SequenceExample()
    curr_obs_dict = self._env.reset(self._loaded_module_spec)
    try:
      curr_obs = curr_obs_dict.obs
      self._process_obs(curr_obs, sequence_example)
      while curr_obs_dict.step_type != env.StepType.LAST:
        timestep = self._create_timestep(curr_obs_dict)
        action = policy(timestep)
        add_int_feature(sequence_example, int(action),
                        SequenceExampleFeatureNames.action)
        curr_obs_dict = self._env.step(action)
        curr_obs = curr_obs_dict.obs
        if curr_obs_dict.step_type == env.StepType.LAST:
          break
        self._process_obs(curr_obs, sequence_example)
    except AssertionError as e:
      logging.error('AssertionError: %s', e)
    horizon = len(sequence_example.feature_lists.feature_list[
        SequenceExampleFeatureNames.action].feature)
    self._working_dir = curr_obs_dict.working_dir
    if horizon <= 0:
      working_dir_head = os.path.split(self._working_dir)[0]
      shutil.rmtree(working_dir_head)
    if horizon <= 0:
      raise ValueError(('Policy did not take any inlining decision for module '
                        f'{self._loaded_module_spec.name}.'))
    if curr_obs_dict.step_type != env.StepType.LAST:
      raise ValueError(('Compilation loop exited at step type'
                        f'{curr_obs_dict.step_type} before last step'))
    reward = curr_obs_dict.score_policy[self._reward_key]
    reward_list = np.float32(reward) * np.float32(np.ones(horizon))
    add_feature_list(sequence_example, reward_list,
                     SequenceExampleFeatureNames.reward)
    module_name_list = [self._loaded_module_spec.name for _ in range(horizon)]
    add_feature_list(sequence_example, module_name_list,
                     SequenceExampleFeatureNames.module_name)
    return sequence_example

  def explore_function(
      self,
      policy: Callable[[Optional[time_step.TimeStep]], np.ndarray],
      explore_policy: Optional[Callable[[time_step.TimeStep],
                                        policy_step.PolicyStep]] = None,
  ) -> Tuple[List[tf.train.SequenceExample], List[str], int, float]:
    """Explores the module using the given policy and the exploration distr.

    Args:
      policy: policy which acts on all states outside of the exploration states.
      explore_policy: randomized policy which is used to compute the gap for
        exploration and can be used for deciding which actions to explore at
        the exploration state.

    Returns:
      seq_example_list: a tf.train.SequenceExample list containing the all
        trajectories from exploration.
      working_dir_names: the directories of the compiled binaries
      loss_idx: idx of the smallest loss trajectory in the seq_example_list.
      base_seq_loss: loss of the best trajectory compiled with policy.
    """
    seq_example_list = []
    working_dir_names = []
    loss_idx = 0
    exploration_steps = 0

    if not explore_policy:
      base_seq = self.compile_module(policy)
      seq_example_list.append(base_seq)
      working_dir_names.append(self._working_dir)
      return (
          seq_example_list,
          working_dir_names,
          loss_idx,
          get_loss(base_seq),
      )

    base_policy = ExplorationWithPolicy(
        [],
        policy,
        explore_policy,
        self._explore_on_features,
    )
    base_seq = self.compile_module(base_policy.get_advice)
    seq_example_list.append(base_seq)
    working_dir_names.append(self._working_dir)
    base_seq_loss = get_loss(base_seq)
    horizon = len(base_seq.feature_lists.feature_list[
        SequenceExampleFeatureNames.action].feature)
    num_states = int(math.ceil(self._exploration_frac * horizon))
    num_states = min(num_states, self._max_exploration_steps)
    if num_states < 1 or horizon > self._max_horizon_to_explore:
      return seq_example_list, working_dir_names, loss_idx, base_seq_loss

    seq_losses = [base_seq_loss]
    for num_steps in range(num_states):
      explore_step = base_policy.get_explore_step()
      if explore_step >= horizon:
        break
      replay_prefix = base_seq.feature_lists.feature_list[
          SequenceExampleFeatureNames.action].feature
      replay_prefix = self._build_replay_prefix_list(
          replay_prefix[:explore_step + 1])
      explore_state = base_policy.get_explore_state()
      for base_seq, base_policy in self.explore_at_state_generator(
          replay_prefix, explore_step, explore_state, policy, explore_policy):
        exploration_steps += 1
        seq_example_list.append(base_seq)
        working_dir_names.append(self._working_dir)
        seq_loss = get_loss(base_seq)
        seq_losses.append(seq_loss)
        # <= biases towards more exploration in the dataset, < towards less expl
        if seq_loss < base_seq_loss:
          base_seq_loss = seq_loss
          loss_idx = num_steps + 1
        logging.info('module exploration losses: %s', seq_losses)
        if exploration_steps > self._max_exploration_steps:
          return seq_example_list, working_dir_names, loss_idx, base_seq_loss
      horizon = len(base_seq.feature_lists.feature_list[
          SequenceExampleFeatureNames.action].feature)
      # check if we are at the end of the trajectory and the last was explored
      if (explore_step == base_policy.get_explore_step() and
          explore_step == horizon - 1):
        return seq_example_list, working_dir_names, loss_idx, base_seq_loss

    return seq_example_list, working_dir_names, loss_idx, base_seq_loss

  def explore_at_state_generator(
      self, replay_prefix: List[np.ndarray], explore_step: int,
      explore_state: time_step.TimeStep,
      policy: Callable[[Optional[time_step.TimeStep]], np.ndarray],
      explore_policy: Callable[[time_step.TimeStep], policy_step.PolicyStep],
      num_samples: int=1,
  ) -> Generator[Tuple[tf.train.SequenceExample, ExplorationWithPolicy], None,
                 None]:
    """Generate sequence examples and next exploration policy while exploring.

    Generator that defines how to explore at the given explore_step. This
    implementation assumes the action set is only {0,1} and will just switch
    the action played at explore_step.

    Args:
      replay_prefix: a replay buffer of actions
      explore_step: exploration step in the previous compiled trajectory
      explore_state: state for exploration at explore_step
      policy: policy which acts on all states outside of the exploration states.
      explore_policy: randomized policy which is used to compute the gap for
        exploration and can be used for deciding which actions to explore at
        the exploration state.
      num_samples: the number of samples to generate

    Yields:
      base_seq: a tf.train.SequenceExample containing a compiled trajectory
      base_policy: the policy used to determine the next exploration step
    """

    distr_logits = explore_policy(explore_state).action.logits.numpy()[0]
    for _ in range(num_samples):
      distr_logits[replay_prefix[explore_step]] = -np.Inf
      if all(-np.Inf==logit for logit in distr_logits):
        break
      replay_prefix[explore_step] = np.random.choice(
        range(distr_logits.shape[0]), p=scipy.special.softmax(distr_logits))
      base_policy = ExplorationWithPolicy(
          replay_prefix,
          policy,
          explore_policy,
          self._explore_on_features,
      )
      base_seq = self.compile_module(base_policy.get_advice)
      yield base_seq, base_policy

  def _build_replay_prefix_list(self, seq_ex):
    ret_list = []
    for int_list in seq_ex:
      ret_list.append(int_list.int64_list.value[0])
    return ret_list

  def _create_timestep(self, curr_obs_dict: env.TimeStep):
    curr_obs = curr_obs_dict.obs
    curr_obs_step = curr_obs_dict.step_type
    step_type_converter = {
        env.StepType.FIRST: 0,
        env.StepType.MID: 1,
        env.StepType.LAST: 2,
    }
    if curr_obs_dict.step_type == env.StepType.LAST:
      reward = np.array(curr_obs_dict.score_policy[self._reward_key])
    else:
      reward = np.array(0.)
    curr_obs_step = step_type_converter[curr_obs_step]
    timestep = time_step.TimeStep(
        step_type=tf.convert_to_tensor([curr_obs_step],
                                       dtype=tf.int32,
                                       name='step_type'),
        reward=tf.convert_to_tensor([reward], dtype=tf.float32, name='reward'),
        discount=tf.convert_to_tensor([0.0], dtype=tf.float32, name='discount'),
        observation=curr_obs,
    )
    return timestep

  def _process_obs(self, curr_obs, sequence_example):
    for curr_obs_feature_name in curr_obs:
      if not self._env.obs_spec:
        obs_dtype = tf.int64
      else:
        if curr_obs_feature_name not in self._env.obs_spec.keys():
          raise AssertionError(
              (f'Feature name {curr_obs_feature_name} not in obs_spec {1}'
               f'{self._env.obs_spec.keys()}'))
        if curr_obs_feature_name in [
            SequenceExampleFeatureNames.action,
            SequenceExampleFeatureNames.reward,
            SequenceExampleFeatureNames.module_name
        ]:
          raise AssertionError(
              (f'Feature name {curr_obs_feature_name} part of '
               f'SequenceExampleFeatureNames {self._env.obs_spec.keys()}'))
        obs_dtype = self._env.obs_spec[curr_obs_feature_name].dtype
      curr_obs_feature = curr_obs[curr_obs_feature_name]
      curr_obs[curr_obs_feature_name] = tf.convert_to_tensor(
          curr_obs_feature, dtype=obs_dtype, name=curr_obs_feature_name)
      add_feature_list(sequence_example, curr_obs_feature,
                       curr_obs_feature_name)


class ModuleWorkerResultProcessor:
  """Utility class to process ModuleExplorer results for ModuleWorker."""

  def __init__(self, base_path: Optional[str] = None):
    self._base_path = base_path

  def _partition_for_loss(self, seq_example: tf.train.SequenceExample,
                          partitions: List[float], label_name: str):
    """Adds a feature to seq_example to partition the examples into buckets.

    Given a tuple of partition limits (a_1, a_2, ..., a_n) we create n+1
    bucekts with limits [0,a_1), [a_1, a_2), ..., [a_n-1, a_n), [a_n, +infty).
    The i-th bucket contains all sequence examples with loss in (a_i-1, a_i].

    Args:
      seq_example: sequence example from the compiled module
      partitions: a tuple of limits defining the buckets
      label_name: name of the feature which will contain the bucket index."""
    seq_loss = get_loss(seq_example)

    label = bisect.bisect_right(partitions, seq_loss)
    horizon = len(seq_example.feature_lists.feature_list['action'].feature)
    label_list = [label for _ in range(horizon)]
    add_feature_list(seq_example, label_list, label_name)

  def process_succeeded(
      self,
      succeeded: List[Tuple[List, List[str], int, float]],
      spec_name: str,
      partitions: List[float],
      label_name: str = 'label'
  ) -> Tuple[tf.train.SequenceExample, ProfilingDictValueType,
             ProfilingDictValueType]:
    seq_example_list = [exploration_res[0] for exploration_res in succeeded]
    working_dir_list = [(exploration_res[1], exploration_res[2])
                        for exploration_res in succeeded]
    seq_example_losses = [exploration_res[3] for exploration_res in succeeded]

    best_policy_idx = np.argmin(seq_example_losses)
    best_exploration_idx = working_dir_list[best_policy_idx][1]

    # comparator is the last policy in the policy_paths list
    module_dict_pol = self._profiling_dict(spec_name, seq_example_list[-1][0])

    module_dict_max = self._profiling_dict(
        spec_name, seq_example_list[best_policy_idx][best_exploration_idx])

    seq_example = seq_example_list[best_policy_idx][best_exploration_idx]
    best_exploration_idxs = [
        exploration_res[2] for exploration_res in succeeded
    ]
    logging.info('best policy idx: %s, best exploration idxs %s',
                 best_policy_idx, best_exploration_idxs)

    if self._base_path:
      # as long as we have one process handles one module this can stay here
      temp_working_dir_idx = working_dir_list[best_policy_idx][1]
      temp_working_dir_list = working_dir_list[best_policy_idx][0]
      temp_working_dir = temp_working_dir_list[temp_working_dir_idx]
      self._save_binary(self._base_path, spec_name, temp_working_dir)

    self._partition_for_loss(seq_example, partitions, label_name)

    return seq_example, module_dict_max, module_dict_pol

  def _profiling_dict(
      self, module_name: str,
      feature_list: tf.train.SequenceExample) -> ProfilingDictValueType:
    """Return a dictionary for the module containing the name, loss and horizon.

    Args:
      module_name: name of module
      feature_list: tf.train.SequenceExample of the compiled module

    Returns:
      per_module_dict: dictionary containing the name, loss and horizon of
        compiled module.
    """

    per_module_dict = {
        'module_name':
            module_name,
        'loss':
            float(get_loss(feature_list)),
        'horizon':
            len(feature_list.feature_lists.feature_list['action'].feature),
    }
    return per_module_dict

  def _save_binary(self, base_path: str, save_path: str, binary_path: str):
    path_head_tail = os.path.split(save_path)
    path_head = path_head_tail[0]
    path_tail = path_head_tail[1]
    save_dir = os.path.join(base_path, path_head)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir, exist_ok=True)
    shutil.copy(
        os.path.join(binary_path, 'comp_binary'),
        os.path.join(save_dir, path_tail))


@gin.configurable
class ModuleWorker(worker.Worker):
  """Class which sets up an exploration worker and processes the results.

  Given a list of policies and an exploration policy, the class processes
  modules, one a time, returning the maximum reward trajectory, where maximum
  is taken over the list of policies together with exploration fascilitated
  by the exploration policy if given.

  Attributes:
    module_explorer_type: type of the module explorer, 
    clang_path: path to clang
    mlgo_task_type: the type of compilation task
    policy_paths: list of policies to load and use for forming the trajectories
    exploration_policy_paths: list of policies to be used for exploration,
      i-th policy in exploration_policy_paths explores when using i-th policy
      in policy_paths for compilation
    exploration_frac: how often to explore in a trajectory
    max_exploration_steps: maximum number of exploration steps
    tf_policy_action: list of the action/advice function from loaded policies
    exploration_policy_paths: paths to load exploration policies.
    explore_on_features: dict of feature names and functions which specify
      when to explore on the respective feature
    obs_action_specs: optional observation spec annotating TimeStep
    base_path: root path to save best compiled binaries for linking
    partitions: a tuple of limits defining the buckets, see partition_for_loss
    env_args: additional arguments to pass to the ModuleExplorer, used in creating
      the environment. This has to include the reward_key
  """

  def __init__(
      #  pylint: disable=dangerous-default-value
      self,
      module_explorer_type: ModuleExplorer=ModuleExplorer,
      clang_path: str = gin.REQUIRED,
      mlgo_task_type: Type[env.MLGOTask] = gin.REQUIRED,
      policy_paths: List[Optional[str]] = [],
      exploration_frac: float = gin.REQUIRED,
      max_exploration_steps: int = 7,
      callable_policies: List[Optional[Callable[[Any], np.ndarray]]] = [],
      exploration_policy_paths: Optional[str] = None,
      explore_on_features: Optional[Dict[str, Callable[[tf.Tensor],
                                                       bool]]] = None,
      obs_action_specs: Optional[Tuple[time_step.TimeStep,
                                       tensor_spec.BoundedTensorSpec,]] = None,
      base_path: Optional[str] = None,
      partitions: List[float] = [
          0.,
      ],
      **envargs,
  ):
    if not policy_paths and not callable_policies:
      raise AssertionError("""At least one policy needs to be specified in
                           policy paths or callable_policies""")
    logging.info('Environment args: %s', envargs)
    self._module_explorer_type: ModuleExplorer=module_explorer_type
    self._clang_path: str = clang_path
    self._mlgo_task_type: Type[env.MLGOTask] = mlgo_task_type
    self._policy_paths: List[Optional[str]] = policy_paths
    self._exploration_policy_paths: Optional[str] = exploration_policy_paths
    self._exploration_frac: float = exploration_frac
    self._max_exploration_steps: int = max_exploration_steps
    self._tf_policy_action: List[Optional[Callable[[Any], np.ndarray]]] = []
    self._exploration_policy_distrs: List[Optional[Callable[
        [time_step.TimeStep, Optional[tf_types.NestedTensor]],
        policy_step.PolicyStep]]] = [
            None for _ in range(len(policy_paths) + len(callable_policies))
        ]
    self._explore_on_features: Optional[Dict[str, Callable[
        [tf.Tensor], bool]]] = explore_on_features
    self._obs_action_specs: Optional[Tuple[
        time_step.TimeStep, tensor_spec.BoundedTensorSpec]] = obs_action_specs
    self._mw_utility = ModuleWorkerResultProcessor(base_path)
    self._partitions = partitions
    self._envargs = envargs

    for policy_path in policy_paths:
      tf_policy = tf.saved_model.load(policy_path, tags=None, options=None)
      self._tf_policy_action.append(policy_action_wrapper(tf_policy))
    for policy in callable_policies:
      self._tf_policy_action.append(policy)
    if exploration_policy_paths:
      if len(exploration_policy_paths) > (len(policy_paths) +
                                          len(callable_policies)):
        raise AssertionError(
            (f'Number of exploration policies: {len(exploration_policy_paths)},'
             'greater than number of policies: '
             f'{len(policy_paths) + len(callable_policies)}'))
      self._exploration_policy_distrs = []
      for exploration_policy_path in exploration_policy_paths:
        expl_policy = tf.saved_model.load(
            exploration_policy_path, tags=None, options=None)
        self._exploration_policy_distrs.append(
            policy_distr_wrapper(expl_policy))
      if len(exploration_policy_paths) < (len(policy_paths) +
                                          len(callable_policies)):
        logging.warning(('Number of exploration policies: %d, '
                         'does not match number of policies: %d '
                         'remaining exploration policies will be set to None'),
                        len(exploration_policy_paths),
                        len(policy_paths) + len(callable_policies))
        for _ in range(
            len(policy_paths) + len(callable_policies) -
            len(exploration_policy_paths)):
          self._exploration_policy_distrs.append(None)

  def select_best_exploration(
      self,
      loaded_module_spec: corpus.LoadedModuleSpec,
  ) -> Tuple[Tuple[int, ProfilingDictValueType, ProfilingDictValueType],
             tf.train.SequenceExample]:

    num_calls = len(self._tf_policy_action)
    time_call_compiler = 0
    logging.info('Processing module: %s', loaded_module_spec.name)
    start = timeit.default_timer()
    work = list(zip(self._tf_policy_action, self._exploration_policy_distrs))
    exploration_worker = self._module_explorer_type(
        loaded_module_spec=loaded_module_spec,
        clang_path=self._clang_path,
        mlgo_task_type=self._mlgo_task_type,
        exploration_frac=self._exploration_frac,
        max_exploration_steps=self._max_exploration_steps,
        explore_on_features=self._explore_on_features,
        obs_action_specs=self._obs_action_specs,
        **self._envargs)
    succeeded = []
    for policy_action, explore_policy in work:
      exploration_res = None
      try:
        exploration_res = exploration_worker.explore_function(
            policy_action, explore_policy)
      except Exception as e:  # pylint: disable=broad-except
        logging.info('Compilation exception %s at %s', e,
                     loaded_module_spec.name)
      if exploration_res is not None:
        succeeded.append(exploration_res)

    end = timeit.default_timer()
    time_call_compiler += end - start
    logging.info('Processed module %s in time %s', loaded_module_spec.name,
                 time_call_compiler)
    (seq_example, module_dict_max,
     module_dict_pol) = self._mw_utility.process_succeeded(
         succeeded, loaded_module_spec.name, self._partitions)

    working_dir_list = [exploration_res[1] for exploration_res in succeeded]

    for temp_dirs in working_dir_list:
      for temp_dir in temp_dirs:
        temp_dir_head = os.path.split(temp_dir)[0]
        shutil.rmtree(temp_dir_head)

    return (
        num_calls,
        module_dict_max,
        module_dict_pol,
    ), seq_example.SerializeToString()


@gin.configurable
def gen_trajectories(
    #  pylint: disable=dangerous-default-value
    data_path: str = gin.REQUIRED,
    delete_flags: Tuple[str, ...] = gin.REQUIRED,
    output_file_name: str = gin.REQUIRED,
    output_path: str = gin.REQUIRED,
    mlgo_task_type: Type[env.MLGOTask] = gin.REQUIRED,
    callable_policies: List[Optional[Callable[[Any], np.ndarray]]] = [],
    obs_action_spec: Optional[Tuple[time_step.TimeStep,
                                    tensor_spec.BoundedTensorSpec]] = None,
    num_workers: Optional[int] = None,
    num_output_files: int = 1,
    profiling_file_path: Optional[str] = None,
    worker_wait_sec: int = 100,
    worker_class_type=ModuleWorker,
    worker_manager_class=local_worker_manager.LocalWorkerPoolManager,
):
  """Generates all trajectories for imitation learning training.

  Args:
    data_path: path to the corpus of modules.
    delete_flags: flags to be deleted during compilation.
    output_file_name: name of the files to write to.
    output_path: path to save the files to.
    mlgo_task_type: task type for the clang environment
    callable_policies: list of policies in the form of callable functions,
      this supplements the loaded policies from policy_paths given
      in ModuleWorker
    obs_action_spec: optional observation and action spec annotating the state
      (TimeStep) for training a policy
    num_workers: number of distributed workers to process the corpus.
    num_output_files: number of files to partition the outputs into, if set to n
      then each file is names output_file_name-i-of-n.tfrecord
    profiling_file_path: path + name of file to save the policy and max of all
      policies profiling dictionaries returned by
      ModuleWorker.select_best-exploration
    worker_wait_sec: max number of seconds to wait for a worker to terminate
    worker_class_type: the class that will process each module
    worker_class_type: allows for overrriding ModuleWorker
    worker_manager_class: A pool of workers hosted on the local machines, each
      in its own process.
  """
  cps = corpus.Corpus(data_path=data_path, delete_flags=delete_flags)
  logging.info('Done loading module specs from corpus.')

  corpus_elements = cps.module_specs
  work = [
      cps.load_module_spec(corpus_element) for corpus_element in corpus_elements
  ]

  modules_processed = 0
  time_compiler_calls = 0
  total_successful_examples = 0
  total_work = len(corpus_elements)
  total_failed_examples = 0
  total_profiles_max: List[Optional[ProfilingDictValueType]] = []
  total_profiles_pol: List[Optional[ProfilingDictValueType]] = []
  size_per_file = total_work // num_output_files

  worker_count = (
      min(os.cpu_count(), num_workers) if num_workers else os.cpu_count())
  with worker_manager_class(
      worker_class_type,
      worker_count,
      obs_action_specs=obs_action_spec,
      mlgo_task_type=mlgo_task_type,
      callable_policies=callable_policies,
  ) as lwm:

    _, result_futures = buffered_scheduler.schedule_on_worker_pool(
        action=lambda w, j: w.select_best_exploration(loaded_module_spec=j),
        jobs=work,
        worker_pool=lwm,
    )
    not_done = result_futures
    succeeded_idx = 0
    succeeded: List[Optional[Tuple[Tuple[int, ProfilingDictValueType,
                                         ProfilingDictValueType],
                                   tf.train.SequenceExample]]] = []

    for written_files_idx in range(num_output_files):
      written_per_file = 0
      file_name = (f'{output_file_name}-{written_files_idx}'
                   f'-of-{num_output_files}.tfrecord')
      tf_rec_path = (
          os.path.join(output_path, file_name)
          if output_path else contextlib.nullcontext())
      tfrecord_context = (
          tf.io.TFRecordWriter(tf_rec_path)
          if output_path else contextlib.nullcontext())

      with tfrecord_context as tfrecord_writer:
        time_compiler_start = timeit.default_timer()
        while not_done or succeeded:
          (done, not_done) = concurrent.futures.wait(not_done, worker_wait_sec)
          succeeded.extend(
              [r for r in done if not r.cancelled() and r.exception() is None])
          failed = [r for r in done if r.exception() is not None]
          for f in failed:
            logging.info('Module failed with: %s', f.exception())
          total_successful_examples += len(succeeded)
          total_failed_examples += len(done) - len(succeeded)
          modules_processed += len(done)
          while written_per_file < size_per_file:
            logging.log_every_n_seconds(
                logging.INFO,
                ('%d success, %d failed out of %d, modules processed'
                 ' %d\n timing compiler: %f'),
                10,
                total_successful_examples,
                total_failed_examples,
                total_work,
                modules_processed,
                time_compiler_calls,
            )
            if not succeeded:
              break
            extra, records = succeeded.pop().result()
            total_profiles_max.append(extra[1])
            total_profiles_pol.append(extra[2])
            time_compiler_calls = timeit.default_timer() - time_compiler_start
            if tfrecord_writer:
              tfrecord_writer.write(records)
            written_per_file += 1
            succeeded_idx += 1
          if written_per_file >= size_per_file:
            break

  max_profiles_path = ''
  pol_profiles_path = ''
  if profiling_file_path:
    max_profiles_path = profiling_file_path + '_max.json'
    pol_profiles_path = profiling_file_path + '_pol.json'
    with open(max_profiles_path, 'w+', encoding='utf-8') as prof_writer_max:
      with open(pol_profiles_path, 'w+', encoding='utf-8') as prof_writer_pol:
        json.dump(total_profiles_max, prof_writer_max, indent=2)
        json.dump(total_profiles_pol, prof_writer_pol, indent=2)

  logging.info(
      ('%d success, %d failed out of %d, modules processed %d\n timing'
       ' compiler: %f'),
      total_successful_examples,
      total_failed_examples,
      total_work,
      modules_processed,
      time_compiler_calls,
  )


def main(_):
  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=FLAGS.gin_bindings, skip_unknown=True)
  logging.info(gin.config_str())

  gen_trajectories()


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
