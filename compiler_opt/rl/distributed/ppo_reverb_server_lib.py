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
"""Library to launch a stand alone Reverb RB server."""

import os

from absl import logging

import reverb
import gin
import tensorflow.compat.v2 as tf

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.specs import tensor_spec
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.utils import common


@gin.configurable
def run_reverb_server(root_dir: str,
                      port: int,
                      replay_buffer_capacity: int,
                      num_epochs: int = 5):
  """Start the server after the initial policy becomes available."""
  # Wait for the collect policy to become available, then load it.
  collect_policy_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR,
                                    learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  collect_policy = train_utils.wait_for_policy(
      collect_policy_dir, load_specs_from_pbtxt=True)

  # Create the signature for the variable container holding the policy weights.
  train_step = train_utils.create_train_step()
  model_id = common.create_variable('model_id')
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container_signature = tf.nest.map_structure(
      lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
      variables)
  logging.info('Signature of variables: \n%s', variable_container_signature)

  # Create the signature for the replay buffer holding observed experience.
  replay_buffer_signature = tensor_spec.from_spec(
      collect_policy.collect_data_spec)
  replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
  logging.info('Signature of experience: \n%s', replay_buffer_signature)

  # Create and start the replay buffer and variable container server.
  server = reverb.Server(
      tables=[
          # The remover does not matter because we clear the table and the end
          # of each global step. We assume that the table is large enough to
          # contain the data collected from one step.
          reverb.Table(
              name='training_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=replay_buffer_capacity,
              max_times_sampled=num_epochs,
              signature=replay_buffer_signature,
          ),
          reverb.Table(
              name=reverb_variable_container.DEFAULT_TABLE,
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=1,
              max_times_sampled=0,
              signature=variable_container_signature,
          ),
      ],
      port=port)
  server.wait()
