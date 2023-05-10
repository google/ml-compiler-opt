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
"""Library to train a policy with PPO"""

import time
import os

from absl import logging

import gin
import reverb
import tensorflow.compat.v2 as tf

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.train import learner as actor_learner
from tf_agents.train import triggers
from tf_agents.utils import common

from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import agent_config
from compiler_opt.rl import registry
from compiler_opt.rl.distributed import learner as learner_lib

_SHUFFLE_BUFFER_EPISODE_LEN = 3


@gin.configurable
def train(
    root_dir: str,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    # Training params
    per_replica_batch_size,
    num_epochs: int,
    num_iterations: int,
    num_episodes_per_iteration: int,
    sequence_length: int,
    summary_interval: int):
  """Trains a PPO agent."""
  # Get the specs from the environment.
  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()

  # Create the agent.
  with strategy.scope():
    train_step = tf.compat.v1.train.get_or_create_global_step()
    agent = agent_config.create_agent(
        agent_config.DistributedPPOAgentConfig(
            time_step_spec=time_step_spec, action_spec=action_spec),
        preprocessing_layer_creator=problem_config
        .get_preprocessing_layer_creator())
    model_id = common.create_variable('model_id')
    # The model_id should equal to the iteration number.
    model_id.assign(0)
    agent.initialize()

  # Create the policy saver which saves the initial model now, then it
  # periodically checkpoints the policy weigths.
  saved_model_dir = os.path.join(root_dir, actor_learner.POLICY_SAVED_MODEL_DIR)
  save_model_trigger = triggers.PolicySavedModelTrigger(
      saved_model_dir, agent, train_step, interval=1000)

  # Create the variable container.
  variables = {
      reverb_variable_container.POLICY_KEY: agent.collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])
  variable_container.push(variables)

  # Create the replay buffer.
  reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=sequence_length,
      table_name='training_table',
      server_address=replay_buffer_server_address)

  # Initialize the dataset.
  def experience_dataset_fn():
    get_dtype = lambda x: x.dtype  # pylint: disable=unnecessary-lambda-assignment
    get_shape = lambda x: (None,) + x.shape  # pylint: disable=unnecessary-lambda-assignment

    shapes = tf.nest.map_structure(get_shape, agent.collect_data_spec)
    dtypes = tf.nest.map_structure(get_dtype, agent.collect_data_spec)

    dataset = reverb.TrajectoryDataset(
        server_address=replay_buffer_server_address,
        table='training_table',
        dtypes=dtypes,
        shapes=shapes,
        # Menger uses learner_iterations_per_call (256). Using 8 here instead
        # because we do not need that much data in the buffer (they have to be
        # filtered out for the next iteration anyways). The rule of thumb is
        # 2-3x batch_size.
        max_in_flight_samples_per_worker=8,
        num_workers_per_iterator=-1,
        max_samples_per_stream=-1,
        rate_limiter_timeout_ms=-1,
    )

    def broadcast_info(info_traj):
      # Assumes that the first element of traj is shaped
      # (sequence_length, ...); and we extract this length.
      info, traj = info_traj
      first_elem = tf.nest.flatten(traj)[0]
      length = first_elem.shape[0] or tf.shape(first_elem)[0]
      info = tf.nest.map_structure(lambda t: tf.repeat(t, [length]), info)
      return reverb.ReplaySample(info, traj)

    dataset = dataset.map(broadcast_info)
    return dataset

  # Create the learner.
  learning_triggers = [
      save_model_trigger,
      triggers.StepPerSecondLogTrigger(train_step, interval=100)
  ]

  def per_sequence_fn(sample):
    # At this point, each sample data contains a sequence of trajectories.
    data, info = sample.data, sample.info
    data = agent.preprocess_sequence(data)
    return data, info

  learner = learner_lib.MLGOPPOLearner(
      root_dir,
      train_step,
      model_id,
      agent,
      experience_dataset_fn,
      sequence_length,
      num_episodes_per_iteration=num_episodes_per_iteration,
      minibatch_size=per_replica_batch_size,
      shuffle_buffer_size=(_SHUFFLE_BUFFER_EPISODE_LEN * sequence_length),
      triggers=learning_triggers,
      summary_interval=summary_interval,
      strategy=strategy,
      num_epochs=num_epochs,
      per_sequence_fn=per_sequence_fn,
      allow_variable_length_episodes=False)

  # Run the training loop.
  for i in range(num_iterations):
    step_val = train_step.numpy()
    logging.info('Training. Iteration: %d', i)
    start_time = time.time()
    learner.wait_for_data()
    data_wait_time = time.time() - start_time
    logging.info('Data wait time sec: %s', data_wait_time)
    loss_info = learner.run()
    logging.info('Loss info: %s', str(loss_info))
    num_steps = train_step.numpy() - step_val
    run_time = time.time() - start_time
    logging.info('Steps per sec: %s', num_steps / run_time)
    logging.info('Pushing variables at model_id: %d', model_id.numpy())
    variable_container.push(variables)
    logging.info('clearing replay buffer')
    reverb_replay_train.clear()
