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

"""util function to create training data iterator."""

from typing import Callable, List, Iterator

import tensorflow as tf
from tf_agents.trajectories import trajectory

from compiler_opt.rl import config as config_lib


# TODO(yundi): define enum type for agent_name.

_POLICY_INFO_PARSING_DICT = {
    'ppo': {
        'CategoricalProjectionNetwork_logits':
            tf.io.FixedLenSequenceFeature(shape=(2), dtype=tf.float32)
    },
    'dqn': {},
    'behavioral_cloning': {},
    'actor_behavioral_cloning': {}
}


def _process_parsed_sequence_and_get_policy_info(parsed_sequence, agent_name):
  """Function to process parsed_sequence and to return policy_info.

  Args:
    parsed_sequence: A dict from feature_name to feature_value parsed from TF
      SequenceExample.
    agent_name: str, name of the agent.

  Returns:
    policy_info: A nested policy_info for given agent.
  """
  if agent_name == 'ppo':
    policy_info = {
        'dist_params': {
            'logits': parsed_sequence['CategoricalProjectionNetwork_logits']
        }
    }
    del parsed_sequence['CategoricalProjectionNetwork_logits']
    return policy_info
  else:
    return ()


def create_parser_fn(
    agent_name: str,
    config: config_lib.Config) -> Callable[[str], trajectory.Trajectory]:
  """Create a parser function for reading from a serialized tf.SequenceExample.

  Args:
    agent_name: str, name of the agent.
    config: An instance of `config.Config`.

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
        (key.name,
         tf.io.FixedLenSequenceFeature(shape=key.shape, dtype=key.dtype))
        for key in (config.feature_keys +
                    (config.action_key, config.reward_key)))
    sequence_features.update(_POLICY_INFO_PARSING_DICT[agent_name])

    # pylint: enable=g-complex-comprehension
    with tf.name_scope('parse'):
      _, parsed_sequence = tf.io.parse_single_sequence_example(
          serialized_proto,
          context_features=context_features,
          sequence_features=sequence_features)
      # TODO(yundi): make the transformed reward configurable.
      action = parsed_sequence[config.action_key.name]
      reward = tf.cast(parsed_sequence[config.reward_key.name], tf.float32)

      policy_info = _process_parsed_sequence_and_get_policy_info(
          parsed_sequence, agent_name)

      del parsed_sequence[config.reward_key.name]
      del parsed_sequence[config.action_key.name]
      full_trajectory = trajectory.from_episode(
          observation=parsed_sequence,
          action=action,
          policy_info=policy_info,
          reward=reward)
      return full_trajectory

  return _parser_fn


def create_sequence_example_iterator_fn(
    agent_name: str, config: config_lib.Config, batch_size: int,
    train_sequence_length: int
) -> Callable[[List[str]], Iterator[trajectory.Trajectory]]:
  """Get a function that creates an iterator from serialized sequence examples.

  Args:
    agent_name: str, name of the agent.
    config: An instance of `config.Config`.
    batch_size: int, batch_size B.
    train_sequence_length: int, trajectory sequence length T.

  Returns:
    A callable that takes a list of serialized sequence examples and returns
      iterator yielding batched trajectory.Trajectory instances with shape
      [B, T, ...].
  """
  trajectory_shuffle_buffer_size = 1024

  parser_fn = create_parser_fn(agent_name, config)

  def _sequence_example_iterator_fn(sequence_examples):
    # Data collector returns empty strings for corner cases, filter them out
    # here.
    dataset = tf.data.Dataset.from_tensor_slices(sequence_examples).filter(
        lambda string: tf.strings.length(string) > 0).map(parser_fn).filter(
            lambda traj: tf.size(traj.reward) > 2)
    dataset = (
        dataset.unbatch().batch(
            train_sequence_length,
            drop_remainder=True).shuffle(trajectory_shuffle_buffer_size).batch(
                batch_size,
                drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    return iter(dataset.repeat())

  return _sequence_example_iterator_fn


# TODO(yundi): PyType check of input_dataset as Type[tf.data.Dataset] is not
# working.
def create_file_iterator_fn(
    agent_name: str, config: config_lib.Config, batch_size: int,
    train_sequence_length: int,
    input_dataset) -> Callable[[List[str]], Iterator[trajectory.Trajectory]]:
  """Get a function that creates an iterator from files.

  Args:
    agent_name: str, name of the agent.
    config: An instance of `config.Config`.
    batch_size: int, batch_size B.
    train_sequence_length: int, trajectory sequence length T.
    input_dataset: A tf.data.Dataset subclass object.

  Returns:
    A callable that takes file path(s) and returns iterator yielding
      trajectory.Trajectory instances with shape [B, T, ...].
  """
  files_buffer_size = 100
  num_readers = 10
  num_map_threads = 8
  shuffle_buffer_size = 1024
  trajectory_shuffle_buffer_size = 1024

  parser_fn = create_parser_fn(agent_name, config)

  def _file_iterator_fn(data_path):
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
    return iter(dataset.repeat())

  return _file_iterator_fn


def create_tfrecord_iterator_fn(
    agent_name: str, config: config_lib.Config, batch_size: int,
    train_sequence_length: int
) -> Callable[[List[str]], Iterator[trajectory.Trajectory]]:
  """Get a function that creates an iterator from tfrecord.

  Args:
    agent_name: str, name of the agent.
    config: An instance of `config.Config`.
    batch_size: int, batch_size B.
    train_sequence_length: int, trajectory sequence length T.

  Returns:
    A callable that takes tfrecord path(s) and returns iterator yielding
      trajectory.Trajectory instances with shape [B, T, ...].
  """
  return create_file_iterator_fn(
      agent_name,
      config,
      batch_size,
      train_sequence_length,
      input_dataset=tf.data.TFRecordDataset)
