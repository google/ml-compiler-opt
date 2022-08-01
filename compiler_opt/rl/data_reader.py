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
"""util function to create training datasets."""

from typing import Callable, List

import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.typing import types

from compiler_opt.rl import constant


def _get_policy_info_parsing_dict(agent_name, action_spec):
  """Function to get parsing dict for policy info."""
  if agent_name == constant.AgentName.PPO:
    if tensor_spec.is_discrete(action_spec):
      return {
          'CategoricalProjectionNetwork_logits':
              tf.io.FixedLenSequenceFeature(
                  shape=(action_spec.maximum - action_spec.minimum + 1),
                  dtype=tf.float32)
      }
    else:
      return {
          'NormalProjectionNetwork_scale':
              tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32),
          'NormalProjectionNetwork_loc':
              tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32)
      }
  return {}


def _process_parsed_sequence_and_get_policy_info(parsed_sequence, agent_name,
                                                 action_spec):
  """Function to process parsed_sequence and to return policy_info.

  Args:
    parsed_sequence: A dict from feature_name to feature_value parsed from TF
      SequenceExample.
    agent_name: AgentName, enum type of the agent.
    action_spec: action spec of the optimization problem.

  Returns:
    policy_info: A nested policy_info for given agent.
  """
  if agent_name == constant.AgentName.PPO:
    if tensor_spec.is_discrete(action_spec):
      policy_info = {
          'dist_params': {
              'logits': parsed_sequence['CategoricalProjectionNetwork_logits']
          }
      }
      del parsed_sequence['CategoricalProjectionNetwork_logits']
    else:
      policy_info = {
          'dist_params': {
              'scale': parsed_sequence['NormalProjectionNetwork_scale'],
              'loc': parsed_sequence['NormalProjectionNetwork_loc']
          }
      }
      del parsed_sequence['NormalProjectionNetwork_scale']
      del parsed_sequence['NormalProjectionNetwork_loc']
    return policy_info
  else:
    return ()


def create_parser_fn(
    agent_name: constant.AgentName, time_step_spec: types.NestedSpec,
    action_spec: types.NestedSpec) -> Callable[[str], trajectory.Trajectory]:
  """Create a parser function for reading from a serialized tf.SequenceExample.

  Args:
    agent_name: AgentName, enum type of the agent.
    time_step_spec: time step spec of the optimization problem.
    action_spec: action spec of the optimization problem.

  Returns:
    A callable that takes scalar serialized proto Tensors and emits
    `Trajectory` objects containing parsed tensors.
  """

  def _parser_fn(serialized_proto):
    """Helper function that is returned by create_`parser_fn`."""
    # We copy through all context features at each frame, so even though we know
    # they don't change from frame to frame, they are still sequence features
    # and stored in the feature list.
    context_features = {}
    # pylint: disable=g-complex-comprehension
    sequence_features = dict(
        (tensor_spec.name,
         tf.io.FixedLenSequenceFeature(
             shape=tensor_spec.shape, dtype=tensor_spec.dtype))
        for tensor_spec in time_step_spec.observation.values())
    sequence_features[action_spec.name] = tf.io.FixedLenSequenceFeature(
        shape=action_spec.shape, dtype=action_spec.dtype)
    sequence_features[
        time_step_spec.reward.name] = tf.io.FixedLenSequenceFeature(
            shape=time_step_spec.reward.shape,
            dtype=time_step_spec.reward.dtype)
    sequence_features.update(
        _get_policy_info_parsing_dict(agent_name, action_spec))

    # pylint: enable=g-complex-comprehension
    with tf.name_scope('parse'):
      _, parsed_sequence = tf.io.parse_single_sequence_example(
          serialized_proto,
          context_features=context_features,
          sequence_features=sequence_features)
      # TODO(yundi): make the transformed reward configurable.
      action = parsed_sequence[action_spec.name]
      reward = tf.cast(parsed_sequence[time_step_spec.reward.name], tf.float32)

      policy_info = _process_parsed_sequence_and_get_policy_info(
          parsed_sequence, agent_name, action_spec)

      del parsed_sequence[time_step_spec.reward.name]
      del parsed_sequence[action_spec.name]
      full_trajectory = trajectory.from_episode(
          observation=parsed_sequence,
          action=action,
          policy_info=policy_info,
          reward=reward)
      return full_trajectory

  return _parser_fn


def create_sequence_example_dataset_fn(
    agent_name: constant.AgentName, time_step_spec: types.NestedSpec,
    action_spec: types.NestedSpec, batch_size: int,
    train_sequence_length: int) -> Callable[[List[str]], tf.data.Dataset]:
  """Get a function that creates a dataset from serialized sequence examples.

  Args:
    agent_name: AgentName, enum type of the agent.
    time_step_spec: time step spec of the optimization problem.
    action_spec: action spec of the optimization problem.
    batch_size: int, batch_size B.
    train_sequence_length: int, trajectory sequence length T.

  Returns:
    A callable that takes a list of serialized sequence examples and returns
      a `tf.data.Dataset`.  Treating this dataset as an iterator yields batched
      `trajectory.Trajectory` instances with shape `[B, T, ...]`.
  """
  trajectory_shuffle_buffer_size = 1024

  parser_fn = create_parser_fn(agent_name, time_step_spec, action_spec)

  def _sequence_example_dataset_fn(sequence_examples):
    # Data collector returns empty strings for corner cases, filter them out
    # here.
    # yapf: disable - Looks better hand formatted
    dataset = (tf.data.Dataset
                .from_tensor_slices(sequence_examples)
                .filter(lambda string: tf.strings.length(string) > 0)
                .map(parser_fn)
                .filter(lambda traj: tf.size(traj.reward) > 2)
                .unbatch()
                .batch(train_sequence_length, drop_remainder=True)
                .cache()
                .shuffle(trajectory_shuffle_buffer_size)
                .batch(batch_size, drop_remainder=True)
               )
    # yapf: enable
    return dataset

  return _sequence_example_dataset_fn


# TODO(yundi): PyType check of input_dataset as Type[tf.data.Dataset] is not
# working.
def create_file_dataset_fn(
    agent_name: constant.AgentName, time_step_spec: types.NestedSpec,
    action_spec: types.NestedSpec, batch_size: int, train_sequence_length: int,
    input_dataset) -> Callable[[List[str]], tf.data.Dataset]:
  """Get a function that creates an dataset from files.

  Args:
    agent_name: AgentName, enum type of the agent.
    time_step_spec: time step spec of the optimization problem.
    action_spec: action spec of the optimization problem.
    batch_size: int, batch_size B.
    train_sequence_length: int, trajectory sequence length T.
    input_dataset: A tf.data.Dataset subclass object.

  Returns:
    A callable that takes file path(s) and returns a `tf.data.Dataset`.
      Iterating over this dataset yields `trajectory.Trajectory` instances with
      shape `[B, T, ...]`.
  """
  files_buffer_size = 100
  num_readers = 10
  num_map_threads = 8
  shuffle_buffer_size = 1024
  trajectory_shuffle_buffer_size = 1024

  parser_fn = create_parser_fn(agent_name, time_step_spec, action_spec)

  def _file_dataset_fn(data_path):
    dataset = (
        tf.data.Dataset.list_files(data_path).shuffle(
            files_buffer_size).interleave(
                input_dataset, cycle_length=num_readers, block_length=1)
        # Due to a bug in collection, we sometimes get empty rows.
        .filter(lambda string: tf.strings.length(string) > 0).apply(
            tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size)).map(
                parser_fn, num_parallel_calls=num_map_threads)
        # Only keep sequences of length 2 or more.
        .filter(lambda traj: tf.size(traj.reward) > 2))

    # TODO(yundi): window and subsample data.
    # TODO(yundi): verify the shuffling is correct.
    dataset = (
        dataset.unbatch().batch(
            train_sequence_length,
            drop_remainder=True).shuffle(trajectory_shuffle_buffer_size).batch(
                batch_size,
                drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    return dataset

  return _file_dataset_fn


def create_tfrecord_dataset_fn(
    agent_name: constant.AgentName, time_step_spec: types.NestedSpec,
    action_spec: types.NestedSpec, batch_size: int,
    train_sequence_length: int) -> Callable[[List[str]], tf.data.Dataset]:
  """Get a function that creates an dataset from tfrecord.

  Args:
    agent_name: AgentName, enum type of the agent.
    time_step_spec: time step spec of the optimization problem.
    action_spec: action spec of the optimization problem.
    batch_size: int, batch_size B.
    train_sequence_length: int, trajectory sequence length T.

  Returns:
    A callable that takes tfrecord path(s) and returns a `tf.data.Dataset`.
      Iterating over this dataset yields `trajectory.Trajectory` instances with
      shape `[B, T, ...]`.
  """
  return create_file_dataset_fn(
      agent_name,
      time_step_spec,
      action_spec,
      batch_size,
      train_sequence_length,
      input_dataset=tf.data.TFRecordDataset)
