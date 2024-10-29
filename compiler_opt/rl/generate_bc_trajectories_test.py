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
"""Tests for compiler_opt.rl.generate_bc_trajectories."""

from typing import List
from unittest import mock

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step

from google.protobuf import text_format  # pytype: disable=pyi-error

from compiler_opt.rl import generate_bc_trajectories
from compiler_opt.rl import env
from compiler_opt.rl import env_test

_eps = 1e-5


def _get_state_list() -> List[time_step.TimeStep]:

  state_0 = time_step.TimeStep(
      discount=tf.constant(np.array([0.]), dtype=tf.float32),
      observation={
          'feature_1': tf.constant(np.array([0]), dtype=tf.int64),
          'feature_2': tf.constant(np.array([50]), dtype=tf.int64),
          'feature_3': tf.constant(np.array([0]), dtype=tf.int64),
      },
      reward=tf.constant(np.array([0]), dtype=tf.float32),
      step_type=tf.constant(np.array([0]), dtype=tf.int32))
  state_1 = time_step.TimeStep(
      discount=tf.constant(np.array([0.]), dtype=tf.float32),
      observation={
          'feature_1': tf.constant(np.array([1]), dtype=tf.int64),
          'feature_2': tf.constant(np.array([25]), dtype=tf.int64),
          'feature_3': tf.constant(np.array([0]), dtype=tf.int64),
      },
      reward=tf.constant(np.array([0]), dtype=tf.float32),
      step_type=tf.constant(np.array([0]), dtype=tf.int32))
  state_2 = time_step.TimeStep(
      discount=tf.constant(np.array([0.]), dtype=tf.float32),
      observation={
          'feature_1': tf.constant(np.array([0]), dtype=tf.int64),
          'feature_2': tf.constant(np.array([25]), dtype=tf.int64),
          'feature_3': tf.constant(np.array([1]), dtype=tf.int64),
      },
      reward=tf.constant(np.array([0]), dtype=tf.float32),
      step_type=tf.constant(np.array([0]), dtype=tf.int32))
  state_3 = time_step.TimeStep(
      discount=tf.constant(np.array([0.]), dtype=tf.float32),
      observation={
          'feature_1': tf.constant(np.array([0]), dtype=tf.int64),
          'feature_2': tf.constant(np.array([25]), dtype=tf.int64),
          'feature_3': tf.constant(np.array([0]), dtype=tf.int64),
      },
      reward=tf.constant(np.array([0]), dtype=tf.float32),
      step_type=tf.constant(np.array([0]), dtype=tf.int32))

  return [state_0, state_1, state_2, state_3]


def _policy(state: time_step.TimeStep) -> np.ndarray:
  feature_sum = np.array([0])
  for feature in state.observation.values():
    feature_sum += feature.numpy()
  return np.mod(feature_sum, 5)


class ExplorationWithPolicyTest(tf.test.TestCase):

  def _explore_policy(self,
                      state: time_step.TimeStep) -> policy_step.PolicyStep:
    probs = [
        0.5 * float(state.observation['feature_3'].numpy()),
        1 - 0.5 * float(state.observation['feature_3'].numpy())
    ]
    logits = [[0.0, tf.math.log(probs[1] / (1.0 - probs[1] + _eps))]]
    return policy_step.PolicyStep(
        action=tfp.distributions.Categorical(logits=logits))

  def test_explore_policy(self):
    prob = 1.
    state = _get_state_list()[3]
    logits = [[0.0, tf.math.log(prob / (1.0 - prob + _eps))]]
    action = tfp.distributions.Categorical(logits=logits)
    self.assertAllClose(action.logits,
                        self._explore_policy(state).action.logits)

  def test_explore_with_gap(self):
    # pylint: disable=protected-access
    explore_with_policy = generate_bc_trajectories.ExplorationWithPolicy(
        replay_prefix=[np.array([1])],
        policy=_policy,
        explore_policy=self._explore_policy,
    )
    for state in _get_state_list():
      _ = explore_with_policy.get_advice(state)[0]

    self.assertAllClose(0, explore_with_policy._gap, atol=2 * _eps)
    self.assertEqual(2, explore_with_policy.get_explore_step())

    explore_with_policy = generate_bc_trajectories.ExplorationWithPolicy(
        replay_prefix=[np.array([1]),
                       np.array([1]),
                       np.array([1])],
        policy=_policy,
        explore_policy=self._explore_policy,
    )
    for state in _get_state_list():
      _ = explore_with_policy.get_advice(state)[0]

    self.assertAllClose(1, explore_with_policy._gap, atol=2 * _eps)
    self.assertEqual(3, explore_with_policy.get_explore_step())

  def test_explore_with_feature(self):
    # pylint: disable=protected-access
    def explore_on_feature_1_val(feature_val):
      return feature_val.numpy()[0] > 0

    def explore_on_feature_2_val(feature_val):
      return feature_val.numpy()[0] > 25

    explore_on_features = {
        'feature_1': explore_on_feature_1_val,
        'feature_2': explore_on_feature_2_val
    }

    explore_with_policy = generate_bc_trajectories.ExplorationWithPolicy(
        replay_prefix=[],
        policy=_policy,
        explore_policy=self._explore_policy,
        explore_on_features=explore_on_features)
    for state in _get_state_list():
      _ = explore_with_policy.get_advice(state)[0]
    self.assertEqual(0, explore_with_policy.get_explore_step())

    explore_with_policy = generate_bc_trajectories.ExplorationWithPolicy(
        replay_prefix=[np.array([1])],
        policy=_policy,
        explore_policy=self._explore_policy,
        explore_on_features=explore_on_features,
    )

    for state in _get_state_list():
      _ = explore_with_policy.get_advice(state)[0]
    self.assertEqual(1, explore_with_policy.get_explore_step())


class AddToFeatureListsTest(tf.test.TestCase):

  def test_add_int_feature(self):
    sequence_example_text = """
      feature_lists {
        feature_list {
          key: "feature_0"
          value {
            feature { int64_list { value: 1 } }
            feature { int64_list { value: 2 } }
          }
        }
        feature_list {
          key: "feature_1"
          value {
            feature { int64_list { value: 3 } }
          }
        }
      }"""
    sequence_example_comp = text_format.Parse(sequence_example_text,
                                              tf.train.SequenceExample())

    sequence_example = tf.train.SequenceExample()
    generate_bc_trajectories.add_int_feature(
        sequence_example=sequence_example,
        feature_value=1,
        feature_name='feature_0')
    generate_bc_trajectories.add_int_feature(
        sequence_example=sequence_example,
        feature_value=2,
        feature_name='feature_0')
    generate_bc_trajectories.add_int_feature(
        sequence_example=sequence_example,
        feature_value=3,
        feature_name='feature_1')

    self.assertEqual(sequence_example, sequence_example_comp)

  def test_add_float_feature(self):
    sequence_example_text = """
      feature_lists {
        feature_list {
          key: "feature_0"
          value {
            feature { float_list { value: .1 } }
            feature { float_list { value: .2 } }
          }
        }
        feature_list {
          key: "feature_1"
          value {
            feature { float_list { value: .3 } }
          }
        }
      }"""
    sequence_example_comp = text_format.Parse(sequence_example_text,
                                              tf.train.SequenceExample())

    sequence_example = tf.train.SequenceExample()
    generate_bc_trajectories.add_float_feature(
        sequence_example=sequence_example,
        feature_value=.1,
        feature_name='feature_0')
    generate_bc_trajectories.add_float_feature(
        sequence_example=sequence_example,
        feature_value=.2,
        feature_name='feature_0')
    generate_bc_trajectories.add_float_feature(
        sequence_example=sequence_example,
        feature_value=.3,
        feature_name='feature_1')

    self.assertEqual(sequence_example, sequence_example_comp)

  def test_add_string_feature(self):
    sequence_example_text = """
      feature_lists {
        feature_list {
          key: "feature_0"
          value {
            feature { bytes_list { value: "1" } }
            feature { bytes_list { value: "2" } }
          }
        }
        feature_list {
          key: "feature_1"
          value {
            feature { bytes_list { value: "3" } }
          }
        }
      }"""
    sequence_example_comp = text_format.Parse(sequence_example_text,
                                              tf.train.SequenceExample())

    sequence_example = tf.train.SequenceExample()
    generate_bc_trajectories.add_string_feature(
        sequence_example=sequence_example,
        feature_value='1',
        feature_name='feature_0')
    generate_bc_trajectories.add_string_feature(
        sequence_example=sequence_example,
        feature_value='2',
        feature_name='feature_0')
    generate_bc_trajectories.add_string_feature(
        sequence_example=sequence_example,
        feature_value='3',
        feature_name='feature_1')

    self.assertEqual(sequence_example, sequence_example_comp)


class ExploreModuleTest(tf.test.TestCase):
  # pylint: disable=protected-access
  @mock.patch('subprocess.Popen')
  def test_create_timestep(self, mock_popen):
    mock_popen.side_effect = env_test.mock_interactive_clang

    def create_timestep_comp(step_type, reward, obs):
      timestep_comp = time_step.TimeStep(
          step_type=tf.convert_to_tensor([step_type],
                                         dtype=tf.int32,
                                         name='step_type'),
          reward=tf.convert_to_tensor([reward], dtype=tf.float32,
                                      name='reward'),
          discount=tf.convert_to_tensor([0.0],
                                        dtype=tf.float32,
                                        name='discount'),
          observation=obs,
      )
      return timestep_comp

    test_env = env.MLGOEnvironmentBase(
        clang_path=env_test._CLANG_PATH,
        task_type=env_test.MockTask,
        obs_spec={},
        action_spec={},
    )

    exploration_worker = generate_bc_trajectories.ExploreModule(
        loaded_module_spec=env_test._MOCK_MODULE,
        clang_path=env_test._CLANG_PATH,
        mlgo_task=env_test.MockTask,
        reward_key='default',
    )

    curr_step_obs = test_env.reset(env_test._MOCK_MODULE)
    timestep = exploration_worker._create_timestep(curr_step_obs)
    timestep_comp = create_timestep_comp(0, 0., curr_step_obs.obs)
    self.assertEqual(timestep, timestep_comp)

    for step_itr in range(env_test._NUM_STEPS - 1):
      del step_itr
      curr_step_obs = test_env.step(np.array([1], dtype=np.int64))
      timestep = exploration_worker._create_timestep(curr_step_obs)
      timestep_comp = create_timestep_comp(1, 0., curr_step_obs.obs)
      self.assertEqual(timestep, timestep_comp)

    curr_step_obs = test_env.step(np.array([1], dtype=np.int64))
    timestep = exploration_worker._create_timestep(curr_step_obs)
    timestep_comp = create_timestep_comp(2, 47., curr_step_obs.obs)
    self.assertEqual(timestep, timestep_comp)

  @mock.patch('subprocess.Popen')
  def test_compile_module(self, mock_popen):
    mock_popen.side_effect = env_test.mock_interactive_clang

    seq_example_comp = tf.train.SequenceExample()
    for i in range(10):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 47.0,
                                                 'reward')
      generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 5),
                                               'action')

    exploration_worker = generate_bc_trajectories.ExploreModule(
        loaded_module_spec=env_test._MOCK_MODULE,
        clang_path=env_test._CLANG_PATH,
        mlgo_task=env_test.MockTask,
        reward_key='default',
    )

    seq_example = exploration_worker.compile_module(_policy)
    self.assertEqual(seq_example, seq_example_comp)

  def _get_seq_example_list_comp(self):
    seq_example_list_comp = []

    # no exploration
    seq_example_comp = tf.train.SequenceExample()
    for i in range(10):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 47.0,
                                                 'reward')
      generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 5),
                                               'action')
    seq_example_list_comp.append(seq_example_comp)

    # first exploration trajectory, tests explore with gap
    seq_example_comp = tf.train.SequenceExample()
    for i in range(10):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 47.0,
                                                 'reward')
      if i == 4:
        generate_bc_trajectories.add_int_feature(seq_example_comp, -3, 'action')
      else:
        generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 5),
                                                 'action')
    seq_example_list_comp.append(seq_example_comp)

    # second exploration trajectory, tests explore on feature
    seq_example_comp = tf.train.SequenceExample()
    for i in range(10):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 47.0,
                                                 'reward')
      if i == 4:
        generate_bc_trajectories.add_int_feature(seq_example_comp, -3, 'action')
      elif i == 5:
        generate_bc_trajectories.add_int_feature(seq_example_comp, 1, 'action')
      else:
        generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 5),
                                                 'action')
    seq_example_list_comp.append(seq_example_comp)

    # third exploration trajectory, tests explore with gap
    seq_example_comp = tf.train.SequenceExample()
    for i in range(10):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 47.0,
                                                 'reward')
      if i == 4:
        generate_bc_trajectories.add_int_feature(seq_example_comp, -3, 'action')
      elif i == 5:
        generate_bc_trajectories.add_int_feature(seq_example_comp, 1, 'action')
      elif i == 9:
        generate_bc_trajectories.add_int_feature(seq_example_comp, -3, 'action')
      else:
        generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 5),
                                                 'action')
    seq_example_list_comp.append(seq_example_comp)

    return seq_example_list_comp

  @mock.patch('subprocess.Popen')
  def test_explore_function(self, mock_popen):
    mock_popen.side_effect = env_test.mock_interactive_clang

    def _explore_on_feature_func(feature_val) -> bool:
      return feature_val[0] in [4, 5]

    exploration_worker = generate_bc_trajectories.ExploreModule(
        loaded_module_spec=env_test._MOCK_MODULE,
        clang_path=env_test._CLANG_PATH,
        mlgo_task=env_test.MockTask,
        reward_key='default',
        explore_on_features={'times_called': _explore_on_feature_func})

    def _explore_policy(state: time_step.TimeStep):
      times_called = state.observation['times_called'][0]
      # will explore every 4-th step
      logits = [[
          4.0 + 1e-3 * float(env_test._NUM_STEPS - times_called),
          float(np.mod(times_called, 5))
      ]]
      return policy_step.PolicyStep(
          action=tfp.distributions.Categorical(logits=logits))

    (seq_example_list, working_dir_names, loss_idx,
     base_seq_loss) = exploration_worker.explore_function(
         _policy, _explore_policy)
    del working_dir_names

    self.assertEqual(loss_idx, 0)
    self.assertEqual(base_seq_loss, 47.0)
    seq_example_list_comp = self._get_seq_example_list_comp()
    self.assertListEqual(seq_example_list, seq_example_list_comp)


class ModuleWorkerResultProcessorTest(tf.test.TestCase):
  # pylint: disable=protected-access
  mw = generate_bc_trajectories.ModuleWorkerResultProcessor()

  def _get_succeded(self):
    seq_example_list_comp = []
    succeeded = []
    # no exploration
    seq_example_comp = tf.train.SequenceExample()
    for i in range(10):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 47.0,
                                                 'reward')
      generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 5),
                                               'action')
    seq_example_list_comp.append(seq_example_comp)

    # first exploration trajectory, tests explore with gap
    seq_example_comp = tf.train.SequenceExample()
    for i in range(10):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 45.0,
                                                 'reward')
      if i == 4:
        generate_bc_trajectories.add_int_feature(seq_example_comp, -3, 'action')
      else:
        generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 5),
                                                 'action')
    seq_example_list_comp.append(seq_example_comp)

    # second exploration trajectory, tests explore on feature
    seq_example_comp = tf.train.SequenceExample()
    for i in range(10):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 36.0,
                                                 'reward')
      if i == 4:
        generate_bc_trajectories.add_int_feature(seq_example_comp, -3, 'action')
      elif i == 5:
        generate_bc_trajectories.add_int_feature(seq_example_comp, 1, 'action')
      else:
        generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 5),
                                                 'action')
    seq_example_list_comp.append(seq_example_comp)
    working_dir_list_comp = ['policy0_0', 'policy0_1', 'policy0_2']
    loss_idx = 2
    seq_loss = 36.0
    succeeded.append(
        (seq_example_list_comp, working_dir_list_comp, loss_idx, seq_loss))

    seq_example_list_comp2 = []
    # no exploration
    seq_example_comp = tf.train.SequenceExample()
    for i in range(5):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 39.0,
                                                 'reward')
      generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 3),
                                               'action')
    seq_example_list_comp2.append(seq_example_comp)

    # third exploration trajectory, tests explore with gap
    seq_example_comp = tf.train.SequenceExample()
    for i in range(5):
      generate_bc_trajectories.add_int_feature(seq_example_comp, i,
                                               'times_called')
      generate_bc_trajectories.add_string_feature(seq_example_comp, 'module',
                                                  'module_name')
      generate_bc_trajectories.add_float_feature(seq_example_comp, 37.0,
                                                 'reward')
      if i == 4:
        generate_bc_trajectories.add_int_feature(seq_example_comp, -3, 'action')
      elif i == 5:
        generate_bc_trajectories.add_int_feature(seq_example_comp, 1, 'action')
      elif i == 9:
        generate_bc_trajectories.add_int_feature(seq_example_comp, -3, 'action')
      else:
        generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 4),
                                                 'action')
    seq_example_list_comp2.append(seq_example_comp)
    working_dir_list_comp2 = ['policy1_0', 'policy1_1']
    loss_idx2 = 1
    seq_loss2 = 37.0
    succeeded.append(
        (seq_example_list_comp2, working_dir_list_comp2, loss_idx2, seq_loss2))

    return succeeded

  def test_partition_for_loss(self):
    seq_example_base_list = []
    for i in range(3):
      seq_example = tf.train.SequenceExample()
      for _ in range(3):
        generate_bc_trajectories.add_float_feature(seq_example, 47.0 + i,
                                                   'reward')
        generate_bc_trajectories.add_int_feature(seq_example, np.mod(i, 5),
                                                 'action')
      seq_example_base_list.append(seq_example)

    seq_example_comp_list = []
    for i in range(3):
      seq_example_comp = tf.train.SequenceExample()
      for _ in range(3):
        generate_bc_trajectories.add_float_feature(seq_example_comp, 47.0 + i,
                                                   'reward')
        generate_bc_trajectories.add_int_feature(seq_example_comp, np.mod(i, 5),
                                                 'action')
        generate_bc_trajectories.add_int_feature(seq_example_comp, i + 1,
                                                 'label')
      seq_example_comp_list.append(seq_example_comp)

    partitions = [47.0, 48.0, 49.0]

    for i in range(3):
      self.mw._partition_for_loss(
          seq_example_base_list[i], partitions, label_name='label')

    self.assertListEqual(seq_example_base_list, seq_example_comp_list)

  def test_process_succeeded(self):
    succeeded = self._get_succeded()
    succeeded_comp = self._get_succeded()
    partitions = [47.0, 48.0, 49.0]

    (seq_example, module_dict_max, module_dict_pol) = self.mw.process_succeeded(
        succeeded=succeeded, spec_name='mw_test', partitions=partitions)

    self.assertEqual(module_dict_max, {
        'module_name': 'mw_test',
        'loss': 36.0,
        'horizon': 10
    })
    self.assertEqual(module_dict_pol, {
        'module_name': 'mw_test',
        'loss': 39.0,
        'horizon': 5
    })
    self.mw._partition_for_loss(
        succeeded_comp[0][0][2], partitions, label_name='label')
    self.assertEqual(seq_example, succeeded_comp[0][0][2])
