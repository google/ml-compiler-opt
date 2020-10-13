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

_REWARD_FOR_TIME_OUT = -10000.0


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
    agent_name: str, config: config_lib.Config, extra_inlining_reward: float,
    reward_shaping: bool) -> Callable[[str], trajectory.Trajectory]:
  """Create a parser function for reading from a serialized tf.SequenceExample.

  Args:
    agent_name: str, name of the agent.
    config: An instance of `config.Config`.
    extra_inlining_reward: Floating point scalar. Additional reward added when
      the inlining decision is `1`, to promote inlining.
    reward_shaping: bool, whether to apply sqrt reward shaping.

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
      # Notice that the delta size is "post - pre" size; and we want to
      # *minimize* this, so the reward should be the negative.
      # Reward are set to be "negative infinite" for early termination cases. We
      # perform a reward transformation on these cases.
      # TODO(yundi): make the transformed reward configurable.
      action = parsed_sequence[config.action_key.name]
      reward = -1 * parsed_sequence[config.reward_key.name]
      reward = tf.cast(reward, tf.float32)
      reward = reward + tf.where(action == 1, float(extra_inlining_reward), 0.0)
      reward = tf.where(reward < _REWARD_FOR_TIME_OUT, _REWARD_FOR_TIME_OUT,
                        reward)
      if reward_shaping:
        reward = tf.where(reward > 0.0, tf.sqrt(reward), reward)
        reward = tf.where(reward < 0.0, -tf.sqrt(-reward), reward)

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
    train_sequence_length: int, extra_inlining_reward: float,
    reward_shaping: bool
) -> Callable[[List[str]], Iterator[trajectory.Trajectory]]:
  """Get a function that creates an iterator from serialized sequence examples.

  Args:
    agent_name: str, name of the agent.
    config: An instance of `config.Config`.
    batch_size: int, batch_size B.
    train_sequence_length: int, trajectory sequence length T.
    extra_inlining_reward: Floating point scalar. Additional reward added when
      the inlining decision is `1`, to promote inlining.
    reward_shaping: bool, whether to apply sqrt reward shaping.

  Returns:
    A callable that takes a list of serialized sequence examples and returns
      iterator yielding batched trajectory.Trajectory instances with shape
      [B, T, ...].
  """
  trajectory_shuffle_buffer_size = 1024

  parser_fn = create_parser_fn(agent_name, config, extra_inlining_reward,
                               reward_shaping)

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
