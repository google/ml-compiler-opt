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
"""Tests for tf_agents.networks.actor_distribution_network."""

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.specs import tensor_spec

from compiler_opt.rl.regalloc import config
from compiler_opt.rl.regalloc import regalloc_network


def _observation_processing_layer(obs_spec):
  """Creates the layer to process observation given obs_spec."""

  def expand_progress(obs):
    if obs_spec.name == 'progress':
      obs = tf.expand_dims(obs, -1)
      obs = tf.tile(obs, [1, config.get_num_registers()])
    return tf.expand_dims(tf.cast(obs, tf.float32), -1)

  return tf.keras.layers.Lambda(expand_progress)


class RegAllocNetworkTest(tf.test.TestCase):

  def setUp(self):
    time_step_spec, action_spec = config.get_regalloc_signature_spec()
    random_observation = tensor_spec.sample_spec_nest(
        time_step_spec, outer_dims=(2, 3))
    super().setUp()
    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    self._random_observation = random_observation

  def testBuilds(self):
    layers = tf.nest.map_structure(_observation_processing_layer,
                                   self._time_step_spec.observation)

    net = regalloc_network.RegAllocNetwork(
        self._time_step_spec.observation,
        self._action_spec,
        preprocessing_layers=layers,
        preprocessing_combiner=tf.keras.layers.Concatenate(),
        fc_layer_params=(10,))

    action_distributions, _ = net(
        self._random_observation.observation,
        step_type=self._random_observation.step_type)
    self.assertIsInstance(action_distributions, tfp.distributions.Categorical)
    self.assertEqual([2, 3], action_distributions.mode().shape.as_list())
    self.assertAllInRange(action_distributions.mode(), 0,
                          config.get_num_registers() - 1)


if __name__ == '__main__':
  tf.test.main()
