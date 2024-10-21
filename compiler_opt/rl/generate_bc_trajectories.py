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

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from absl import logging
import dataclasses
import os
import shutil

import numpy as np
import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.specs import tensor_spec

from compiler_opt.distributed import worker
from compiler_opt.rl import corpus
from compiler_opt.rl import env


@dataclasses.dataclass
class SequenceExampleFeatureNames:
  """Feature names for features that are always added to seq example."""
  action: str = 'action'
  reward: str = 'reward'
  module_name: str = 'module_name'


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
    raise AssertionError(f'''Unsupported type for feautre {0} of type {1}.
      Supported types are np.int64, np.float32, str'''.format(
        feature_name, type(feature_list[0])))
  if isinstance(feature_list[0], np.float32):
    add_function = add_float_feature
  elif isinstance(feature_list[0], (int, np.int64)):
    add_function = add_int_feature
  else:
    add_function = add_string_feature
  for feature in feature_list:
    add_function(seq_example, feature, feature_name)


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
    if not self._stop_exploration and self._explore_on_features is not None:
      for feature_name, explore_on_feature in self._explore_on_features.items():
        if explore_on_feature(state.observation[feature_name]):
          self._explore_step = self._curr_step
          self._stop_exploration = True
          break
    self._curr_step += 1
    return policy_action


class ExplorationWorker(worker.Worker):
  """Class which implements the exploration for the given module.

  Attributes:
    loaded_module_spec: the module to be compiled and explored
    use_greedy: indicates if the default/greedy policy is used to compile the
      module
    env: MLGO environment.
    exploration_frac: how often to explore in a trajectory
    max_exploration_steps: maximum number of exploration steps
    exploration_policy_distr: distribution function from exploration policy.
    reward_key: which reward binary to use, must be specified as part of
      additional task args (kwargs).
  """

  def __init__(
      self,
      loaded_module_spec: corpus.LoadedModuleSpec,
      clang_path: str,
      mlgo_task: Type[env.MLGOTask],
      use_greedy: bool = False,
      exploration_frac: float = 1.0,
      max_exploration_steps: int = 10,
      exploration_policy_distr=None,
      obs_action_specs: Optional[Tuple[time_step.TimeStep,
                                       tensor_spec.BoundedTensorSpec,]] = None,
      reward_key: str = '',
      **kwargs,
  ):
    self._loaded_module_spec = loaded_module_spec
    self._use_greedy = use_greedy
    if not obs_action_specs:
      obs_spec = None
      action_spec = None
    else:
      obs_spec = obs_action_specs[0].observation
      action_spec = obs_action_specs[1]

    if reward_key == '':
      raise TypeError(
          'reward_key not specified in ExplorationWorker initialization.')
    self._reward_key = reward_key
    kwargs.pop('reward_key', None)
    self._working_dir = None

    self._env = env.MLGOEnvironmentBase(
        clang_path=clang_path,
        task_type=mlgo_task,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
    if self._env.action_spec:
      if self._env.action_spec.dtype != tf.int64:
        raise TypeError(
            f'Environment action_spec type {0} does not match tf.int64'.format(
                self._env.action_spec.dtype))
    self._exploration_frac = exploration_frac
    self._max_exploration_steps = max_exploration_steps
    self._exploration_policy_distr = exploration_policy_distr
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
      raise ValueError(
          f'Policy did not take any inlining decision for module {0}.'.format(
              self._loaded_module_spec.name))
    if curr_obs_dict.step_type != env.StepType.LAST:
      raise ValueError(
          f'Compilation loop exited at step type {0} before last step'.format(
              curr_obs_dict.step_type))
    reward = curr_obs_dict.score_policy[self._reward_key]
    reward_list = np.float32(reward) * np.float32(np.ones(horizon))
    add_feature_list(sequence_example, reward_list,
                     SequenceExampleFeatureNames.reward)
    module_name_list = [self._loaded_module_spec.name for _ in range(horizon)]
    add_feature_list(sequence_example, module_name_list,
                     SequenceExampleFeatureNames.module_name)
    return sequence_example

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
          raise AssertionError(f'Feature name {0} not in obs_spec {1}'.format(
              curr_obs_feature_name, self._env.obs_spec.keys()))
        if curr_obs_feature_name in [
            SequenceExampleFeatureNames.action,
            SequenceExampleFeatureNames.reward,
            SequenceExampleFeatureNames.module_name
        ]:
          raise AssertionError(
              f'Feature name {0} already part of SequenceExampleFeatureNames'
              .format(curr_obs_feature_name, self._env.obs_spec.keys()))
        obs_dtype = self._env.obs_spec[curr_obs_feature_name].dtype
      curr_obs_feature = curr_obs[curr_obs_feature_name]
      curr_obs[curr_obs_feature_name] = tf.convert_to_tensor(
          curr_obs_feature, dtype=obs_dtype, name=curr_obs_feature_name)
      add_feature_list(sequence_example, curr_obs_feature,
                       curr_obs_feature_name)
