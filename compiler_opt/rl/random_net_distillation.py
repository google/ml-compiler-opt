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
"""Random Network Distillation Implementation."""
import gin
import tensorflow as tf
from tf_agents.utils import tensor_normalizer


@gin.configurable
class RandomNetworkDistillation():
  """The Random Network Distillation class."""

  def __init__(self,
               time_step_spec=None,
               preprocessing_layer_creator=None,
               encoding_network=None,
               learning_rate=1e-4,
               update_frequency=4,
               fc_layer_params=(32,),
               initial_intrinsic_reward_scale=1.0,
               half_decay_steps=10000):
    """Initilization for RandomNetworkDistillation class.

    Args:
      time_step_spec: the time step spec for raw observation
      preprocessing_layer_creator: A callable returns feature processing layer
        given observation_spec.
      encoding_network: A tf_agents.networks.Network class.
      learning_rate: the learning rate for optimizer.
      update_frequency: the update frequency for the predictor network.
      fc_layer_params: list of fully_connected parameters, where each item is
        the number of units in the layer.
      initial_intrinsic_reward_scale: the scale of the initial intrinsic reward.
      half_decay_steps: the steps for intrinsic reward scale to decay by half.
    """

    feature_extractor_layer = tf.nest.map_structure(preprocessing_layer_creator,
                                                    time_step_spec.observation)

    self._target_net = encoding_network(
        input_tensor_spec=time_step_spec.observation,
        preprocessing_layers=feature_extractor_layer,
        preprocessing_combiner=tf.keras.layers.Concatenate(),
        fc_layer_params=fc_layer_params,
        name='ObsNormalizationNetwork')

    self._predict_net = encoding_network(
        input_tensor_spec=time_step_spec.observation,
        preprocessing_layers=feature_extractor_layer,
        preprocessing_combiner=tf.keras.layers.Concatenate(),
        fc_layer_params=fc_layer_params,
        name='ObsNormalizationNetwork')

    self._predict_net_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate)
    self._intrinsic_reward_normalizer = (
        tensor_normalizer.StreamingTensorNormalizer(
            tf.TensorSpec([], tf.float32)))
    self._update_frequency = update_frequency
    self._decay_scale = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_intrinsic_reward_scale,
        decay_steps=half_decay_steps,
        decay_rate=0.5,
        staircase=False)

    self._global_step = tf.compat.v1.train.get_or_create_global_step()
    self._normalized_intrinsic_reward_mean = tf.keras.metrics.Mean()
    self._intrinsic_reward_mean = tf.keras.metrics.Mean()
    self._external_reward_mean = tf.keras.metrics.Mean()

  def _get_intrinsic_reward(self, observation):
    """Compute the intrisic reward.

    Args:
      observation: raw observation in observation_spec format

    Returns:
      intrinsic_reward: the intrinsic reward
    """
    with tf.GradientTape() as tape:
      # make the predict network parameters trainable
      # Compute the feature embedding loss (for next obseravtion trajectory)
      feature_target, _ = self._target_net(observation)
      feature_predict, _ = self._predict_net(observation)
      feature_target = tf.stop_gradient(feature_target)

      # compute the embedding loss on a portion of the batch data
      # _update_frequency denotes the stride slicing
      emb_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.math.square(feature_target[::self._update_frequency] -
                             feature_predict[::self._update_frequency]),
              axis=-1))

      # log the predictor loss
      with tf.name_scope('Losses/'):
        tf.summary.scalar(
            name='random_net_predictor_loss',
            data=emb_loss,
            step=self._global_step)

      # compute the gradient and optimize the predictor function
      pred_grad = tape.gradient(emb_loss, self._predict_net.trainable_variables)
      self._predict_net_optimizer.apply_gradients(
          zip(pred_grad, self._predict_net.trainable_variables))

    # compute the intrinsic reward using the l2 norm square difference
    # only consider the next state trajectory, whose length = original length-1
    intrinsic_reward = tf.reduce_sum(
        tf.math.square(feature_target - feature_predict), axis=-1)[:, 1:]

    return intrinsic_reward

  def _update_metrics(self, experience, intrinsic_reward,
                      normalized_intrinsic_reward):
    """Updates metrics and exports to Tensorboard.

    Args:
      experience: the experience trajectory in shape of `[batch_size,
        time_steps]`.
      intrinsic_reward: the intrinsic reward in shape of `[batch_size,
        time_steps - 1]` (intrinsic_reward is based on the 'next-state'
        trajectory).
      normalized_intrinsic_reward: the scaled normalized intrinsic reward in
        shape of `[batch_size, time_steps - 1]`.
    """
    is_action = ~experience.is_boundary()
    # normalized_intrinsic_reward will assign for the first time_steps - 1
    # length trajectory states. The sample_weight is also based on the first
    # time_steps - 1 length trajectory.
    self._normalized_intrinsic_reward_mean.update_state(
        normalized_intrinsic_reward, sample_weight=is_action[:, :-1])
    self._intrinsic_reward_mean.update_state(
        intrinsic_reward, sample_weight=is_action[:, :-1])
    self._external_reward_mean.update_state(
        experience.reward, sample_weight=is_action)
    with tf.name_scope('random_network_distillation/'):
      tf.summary.scalar(
          name='data_normalized_intrinsic_reward_mean',
          data=self._normalized_intrinsic_reward_mean.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='data_intrinsic_reward_mean',
          data=self._intrinsic_reward_mean.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='data_external_reward_mean',
          data=self._external_reward_mean.result(),
          step=self._global_step)
      tf.summary.scalar(
          name='intrinsic_reward_scale',
          data=self._decay_scale(self._global_step),
          step=self._global_step)

    tf.summary.histogram(
        name='external_reward', data=experience.reward, step=self._global_step)
    tf.summary.histogram(
        name='intrinsic_reward', data=intrinsic_reward, step=self._global_step)
    tf.summary.histogram(
        name='normalized_intrinsic_reward',
        data=normalized_intrinsic_reward,
        step=self._global_step)

  def train(self, experience):
    """Train the predictor on the batched next state trajectory.

    Args:
      experience: Trajectory

    Returns:
      expereince_new: new Trajectory (the new trajectory modified from the
      original experience trajectory, where the reward is updated as the
      addition of external reward + intrinsic reward).
    """
    # compute intrinsic reward for length - 1 horizon
    intrinsic_reward = self._get_intrinsic_reward(experience.observation)

    normalized_intrinsic_reward = self._intrinsic_reward_normalizer.normalize(
        intrinsic_reward, clip_value=0, center_mean=False) * self._decay_scale(
            self._global_step)
    self._intrinsic_reward_normalizer.update(intrinsic_reward)

    # update the log
    self._update_metrics(experience, intrinsic_reward,
                         normalized_intrinsic_reward)

    batch_size = experience.reward.shape[0]
    # assign the last time step reward = 0 (no intrinsic reward)
    # pylint: disable=unexpected-keyword-arg
    normalized_intrinsic_reward = tf.concat(
        [normalized_intrinsic_reward,
         tf.zeros([batch_size, 1])], axis=1)

    # reconstruct the reward: external + intrinsic
    reconstructed_reward = experience.reward + normalized_intrinsic_reward

    return experience.replace(reward=reconstructed_reward)
