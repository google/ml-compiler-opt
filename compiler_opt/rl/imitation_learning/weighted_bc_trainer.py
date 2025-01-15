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
"""Module for training an inlining policy with imitation learning."""

from absl import app
from absl import flags

from typing import List, Optional

import bisect
import copy
import gin
import json
import logging
import os
from functools import partial

from compiler_opt.rl import policy_saver
from compiler_opt.rl.imitation_learning.generate_bc_trajectories_lib import ProfilingDictValueType
from compiler_opt.rl.imitation_learning.generate_bc_trajectories_lib import SequenceExampleFeatureNames
from compiler_opt.rl.inlining import imitation_learning_config as config
from compiler_opt.rl import feature_ops

import keras
import numpy as np
import tensorflow as tf

import tf_agents
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.trajectories import policy_step
import tensorflow_probability as tfp

_QUANTILE_MAP_PATH = flags.DEFINE_string(
    'quantile_map_path', None,
    ('Directory containing the quantile map for normalizing features'
     'in feature_ops.build_quantile_map.'))
_TRAINING_DATA = flags.DEFINE_multi_string(
    'training_data', None, 'Training data for one step of BC-Max')
_PROFILING_DATA = flags.DEFINE_multi_string(
    'profiling_data', None,
    ('Paths to profile files for computing the TrainingWeights'
     'If specified the order for each pair of json files is'
     'comparator.json followed by eval.json and the number of'
     'files should always be even.'))
_SAVE_MODEL_DIR = flags.DEFINE_string(
    'save_model_dir', None, 'Location to save the keras and TFAgents policies.')
# _GIN_FILES = flags.DEFINE_multi_string(
#     'gin_files', [], 'List of paths to gin configuration files.')
# _GIN_BINDINGS = flags.DEFINE_multi_string(
#     'gin_bindings', [],
#     'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS

# Pytype cannot pick up the pyi file for tensorflow.summary. Disable the error
# here as these errors are false positives.
# pytype: disable=pyi-error


@gin.configurable
class TrainingWeights:
  """Class for computing weights for training."""

  def __init__(
      #  pylint: disable=dangerous-default-value
      self,
      partitions: list[float] = [0.],
      weights: Optional[np.ndarray] = None):
    self._weights = weights
    if not weights:
      self._weights = np.ones(len(partitions) + 1)
    self._probs: np.ndarray = np.exp(self._weights) / np.sum(
        np.exp(self._weights))
    self._partitions: List[float] = partitions
    self._round: int = 1

  def _bucket_by_feature(
      self, data: List[ProfilingDictValueType],
      feature_name: str) -> List[List[ProfilingDictValueType]]:
    """Partitions the profiles according to the feature name.
    
    Partitions the profiles according to the feature name and the
    buckets defined by self._partitions.

    Args:
      data: list of ProfilingDictValueType to partition
        feature_name: feature according to which the partition happens

    Returns:
      buckets: partitioned profiles according to the feature name
    """
    buckets = [[] for i in range(len(self._partitions) + 1)]

    for prof in data:
      idx = bisect.bisect_right(self._partitions, prof[feature_name])
      buckets[idx].append(prof)

    return buckets

  def _eg_step(self, loss, step_size) -> np.ndarray:
    """Exponentiated gradient step.
    
    Args:
      loss: observed losses to update the weights
      step_size: step size for the update
    
    Returns:
      probability distribution from the updated weights
    """
    self._weights = self._weights - step_size * loss

    return np.exp(self._weights) / np.sum(np.exp(self._weights))

  def create_new_profile(self,
                         data_comparator: List[ProfilingDictValueType],
                         data_eval: List[ProfilingDictValueType],
                         eps: float = 1e-5) -> List[ProfilingDictValueType]:
    """Create a new profile which contains the regret and relative reward.
    
    The regret is measured as the difference between the loss of the data_eval
    profiles and of the data_comparator profiles. The reward is the negative
    regret normalized by the loss of hte data_comparator profiles.

    Args:
      data_comparator: baseline profiles to measure improvement against
      data_eval: profiles to evaluate for improvement
    Returns:
      new profile containing regret and reward
    """

    func_key_dict = {}
    for prof in data_eval:
      if not isinstance(prof[SequenceExampleFeatureNames.module_name], str):
        raise ValueError(
            'SequenceExampleFeatureNames.module_name has to be str.')
      func_key_dict[prof[
          SequenceExampleFeatureNames.module_name]] = copy.deepcopy(prof)
    for prof in data_comparator:
      try:
        new_prof = func_key_dict[prof[SequenceExampleFeatureNames.module_name]]
      except KeyError as k:
        print(k)
        continue
      if isinstance(prof[SequenceExampleFeatureNames.loss], str):
        raise ValueError(('prof[SequenceExampleFeatureNames.loss] is a string'
                          'but it should be numeric.'))
      if isinstance(new_prof[SequenceExampleFeatureNames.loss], str):
        raise ValueError(
            ('new_prof[SequenceExampleFeatureNames.loss] is a string'
             'but it should be numeric.'))
      new_prof[SequenceExampleFeatureNames
               .regret] = new_prof[SequenceExampleFeatureNames.loss] - prof[
                   SequenceExampleFeatureNames.loss]
      new_prof[SequenceExampleFeatureNames
               .reward] = -new_prof[SequenceExampleFeatureNames.regret] / (
                   prof[SequenceExampleFeatureNames.loss] + eps)

    return list(func_key_dict.values())

  def update_weights(
      self, comparator_profile: List[ProfilingDictValueType],
      policy_profile: List[ProfilingDictValueType]) -> np.ndarray:
    """Constructs a new profile and uses the loss to update self._probs with EG.
    
    Args:
      comparator_profile: baseline profiles to measure improvement against
      policy_profile: profiles to evaluate for improvement
    
    Returns:
      Updated probabilities to use as weights in training.
    """
    comp_prof = self.create_new_profile(comparator_profile, policy_profile)
    losses_per_bucket = []
    ppo_loss_buckets = self._bucket_by_feature(comp_prof,
                                               SequenceExampleFeatureNames.loss)
    for bucket in ppo_loss_buckets:
      bucket_loss = 0
      for prof in bucket:
        bucket_loss += np.maximum(prof[SequenceExampleFeatureNames.regret], 0)
      losses_per_bucket.append(bucket_loss)
    logging.info('Losses per bucket: %s', losses_per_bucket)
    losses_per_bucket_normalized = losses_per_bucket / np.max(
        np.abs(losses_per_bucket))
    pt = self._eg_step(losses_per_bucket_normalized, 1.0)
    self._round += 1
    self._probs = (self._probs * (self._round - 1) + pt) / self._round

    return self._probs

  def get_weights(self) -> np.ndarray:
    """Returns the current weights.

    Returns:
      self._probs: the current weights."""
    return np.float64(self._probs)


@gin.configurable
class ImitationLearningTrainer:
  """Implements one iteration of the BC-Max algorithm.
  
  BC-Max can be found at https://arxiv.org/pdf/2403.19462."""

  def __init__(
      #  pylint: disable=dangerous-default-value
      self,
      width: int = 100,
      layers: int = 4,
      batch_size: int = 128,
      epochs: int = 1,
      log_interval: int = 1000,
      optimizer: Optional[keras.optimizers.Optimizer] = None,
      save_model_dir: Optional[str] = None,
      shuffle_size: int = 131072,
      training_weights: Optional[TrainingWeights] = None,
      features_to_remove: Optional[List[str]] = [
      'policy_label', 'inlining_default'
      ]):
    self._width = width
    self._layers = layers
    self._batch_size = batch_size
    self._epochs = epochs
    self._log_interval = log_interval
    self._optimizer = optimizer
    if not self._optimizer:
      self._optimizer = keras.optimizers.SGD(learning_rate=0.01)
    self._save_model_dir = save_model_dir
    self._shuffle_size = shuffle_size
    self._trainig_weights = training_weights
    if not self._trainig_weights:
      self._trainig_weights = TrainingWeights()
    self._features_to_remove = features_to_remove
    self._global_step = 0

    observation_spec, action_spec = config.get_inlining_signature_spec()
    sequence_features = dict(
        (tensor_spec.name,
         tf.io.FixedLenSequenceFeature(
             shape=tensor_spec.shape, dtype=tensor_spec.dtype))
        for tensor_spec in observation_spec[-1].values())
    sequence_features.update({
        action_spec.name:
            tf.io.FixedLenSequenceFeature(
                shape=action_spec.shape, dtype=action_spec.dtype)
    })
    self._sorted_features_dict = dict(sorted(sequence_features.items()))
    if not _QUANTILE_MAP_PATH.value:
      raise ValueError('quantile_map_path needs to be specified for training')
    quantile_map = feature_ops.build_quantile_map(_QUANTILE_MAP_PATH.value)
    self._normalize_func_dict = {
        name:
            feature_ops.get_normalize_fn(
                qm,
                with_sqrt=True,
                with_z_score_normalization=False)
        for name, qm in quantile_map.items()
    }

    self._num_threads = os.cpu_count()
    self._num_processors = 10

  def _initialize_model(self, input_shape=None):
    inputs = keras.layers.Input(shape=(input_shape,))
    x = keras.layers.Normalization(axis=-1)(inputs)
    for _ in range(self._layers):
      x = keras.layers.Dense(
          self._width,
          activation='relu',
          kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))(
              x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    self._model = keras.Model(inputs=inputs, outputs=outputs)

  def _parse_func(self, raw_record, sequence_features):
    parsed_example = tf.io.parse_sequence_example(
        raw_record, sequence_features=sequence_features)
    return parsed_example[1]

  def _make_feature_label(self, parsed_example, num_processors):
    """Function to pre-process the parsed examples from dataset.
    
    Removes certein features not used for training and reshapes
    features appropriately."""
    concat_arr = []
    for name, feature in parsed_example.items():
      if name == SequenceExampleFeatureNames.action:
        label = tf.cast(feature, tf.float32)
        label = tf.reshape(label, [num_processors, 1])
      if name == SequenceExampleFeatureNames.label_name:
        weight_label = tf.cast(feature, tf.float32)
        weight_label = tf.reshape(weight_label, [num_processors, 1])
      if name in self._features_to_remove + [
          SequenceExampleFeatureNames.action,
          SequenceExampleFeatureNames.label_name,
          SequenceExampleFeatureNames.module_name
      ]:
        continue
      feature = tf.cast(feature, tf.float32)
      normalize_func = self._normalize_func_dict[name]
      feature = normalize_func(feature)
      if len(feature.shape) == 1:
        feature = tf.reshape(feature, [num_processors, 1])
      concat_arr.append(tf.cast(feature, dtype=tf.float32))
      if len(tf.where(tf.math.is_nan(tf.cast(feature[0],
                                             dtype=tf.float32)))) > 0:
        logging.warning('Feature %s is nan', name)
    return tf.concat(concat_arr, -1), tf.concat([label, weight_label], -1)

  def load_dataset(self, filepaths: List[str]) -> tf.data.TFRecordDataset:
    """Load datasets from specified filepaths for training.
    
    Args:
      filepaths: paths to dataset files
      
    Returns:
      dataset: loaded tf dataset"""
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    raw_data = tf.data.TFRecordDataset(filepaths)
    dataset = raw_data.map(
        partial(self._parse_func, sequence_features=self._sorted_features_dict),
        num_parallel_calls=self._num_threads)
    dataset = dataset.unbatch().batch(
        self._num_processors, drop_remainder=True).map(
            partial(
                self._make_feature_label, num_processors=self._num_processors))
    dataset = dataset.unbatch().shuffle(self._shuffle_size).batch(
        self._batch_size, drop_remainder=True)  # 4194304
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    return dataset

  def _create_weights(self, labels, weights_arr):
    p_norm = max(weights_arr)
    weights_arr = tf.map_fn(lambda x: p_norm / x, tf.constant(weights_arr))
    int_labels = tf.cast(labels, tf.int32)
    return tf.gather(weights_arr, int_labels)

  def _loss_fn(self, y_true, y_pred, labels, weights_arr):
    weights = tf.ones_like(y_true, dtype=tf.float64)
    for l, wa in zip(labels, weights_arr):
      w = self._create_weights(l, wa)
      w = tf.reshape(w, [-1, 1])
      weights = tf.math.multiply(w, weights)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    return bce(y_true, y_pred, sample_weight=weights), weights

  def _initialize_metrics(self):
    """Initializes metrics."""
    self._metrics = [keras.metrics.AUC(name='AUC')]
    self._metrics.append(keras.metrics.BinaryAccuracy(name='binary_acc'))
    self._metrics.append(
        keras.metrics.BinaryAccuracy(name='binary_acc_weighted'))
    self._metrics.append(keras.metrics.Mean(name='mean_loss'))

  def _update_metrics(self, y_true, y_pred, loss, weights):
    """Updates metrics and exports to Tensorboard."""
    self._metrics[0].update_state(y_true, y_pred)
    self._metrics[1].update_state(y_true, y_pred)
    self._metrics[2].update_state(y_true, y_pred, sample_weight=weights)
    self._metrics[3].update_state(loss)
    # Check earlier rather than later if we should record summaries.
    # TF also checks it, but much later. Needed to avoid looping through
    # the dict so gave the if a bigger scope
    if tf.summary.should_record_summaries():
      with tf.name_scope('default/'):
        for metric in self._metrics:
          tf.summary.scalar(
              name=metric.name, data=metric.result(), step=self._global_step)

  def _train_step(self, example, label, weight_labels, weights_arr):
    y_true = label[:, 0]
    y_true = tf.reshape(y_true, [self._batch_size, 1])
    with tf.GradientTape() as tape:
      y_pred = self._model(example, training=True)
      loss_value, weights = self._loss_fn(y_true, y_pred, weight_labels,
                                          weights_arr)
    grads = tape.gradient(loss_value, self._model.trainable_weights)
    self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
    self._update_metrics(y_true, y_pred, loss_value, weights)
    return loss_value

  def train(self, filepaths: List[str]):
    """Train the model for number of the specified number of epochs."""
    dataset = self.load_dataset(filepaths)
    logging.info('Datasets loaded from %s', str(filepaths))
    input_shape = dataset.element_spec[0].shape[-1]
    self._initialize_model(input_shape=input_shape)
    self._initialize_metrics()
    for _ in range(self._epochs):
      for metric in self._metrics:
        metric.reset_states()
      for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        weight_labels = [y_batch_train[:, 1]]
        weights_arr = [self._trainig_weights.get_weights()]
        # context management is implemented in decorator
        # pytype: disable=attribute-error
        # pylint: disable=not-context-manager
        # pylint: disable=cell-var-from-loop
        with tf.summary.record_if(
            lambda: tf.math.equal(step % self._log_interval, 0)):
          # pytype: enable=attribute-error
          self._train_step(x_batch_train, y_batch_train, weight_labels,
                           weights_arr)
          self._global_step += 1
          if step % self._log_interval == 0:
            logging.info('\n\nExamples so far %s',
                         (step + 1) * self._batch_size)
            for metric in self._metrics:
              logging.info('%s: %s', metric.name, metric.result())
        if step > 1000:  # debugging
          break

    if self._save_model_dir:
      keras.models.save_model(self._model,
                              os.path.join(self._save_model_dir, 'keras_model'))

  def get_policy(self):
    return self._model


class WrapKerasModel(tf_agents.policies.TFPolicy):
  """Create a TFPolicy from a trained keras model."""

  def __init__(
      #  pylint: disable=dangerous-default-value
      self,
      *args,
      keras_policy: tf.keras.Model,
      features_to_remove: Optional[List[str]] = ['inlining_default'],
      **kwargs):
    super().__init__(*args, **kwargs)
    self._keras_policy = keras_policy
    self._expected_signature = self.time_step_spec
    self._sorted_keys = sorted(self._expected_signature.observation.keys())
    self._quantile_map = feature_ops.build_quantile_map(
        _QUANTILE_MAP_PATH.value)
    self._features_to_remove = features_to_remove
    logging.info('Feature spec %s:', self._sorted_keys)

  def _process_observation(self, observation):
    concat_arr = []
    for name in self._sorted_keys:
      if name in self._features_to_remove:
        continue
      feature = tf.cast(observation[name], dtype=tf.float32)
      normalize_func = feature_ops.get_normalize_fn(
          self._quantile_map[name],
          with_sqrt=True,
          with_z_score_normalization=False)
      feature = normalize_func(feature)
      concat_arr.append(feature)
    return tf.concat(concat_arr, -1)

  def _create_distribution(self, inlining_prediction):
    probs = [1.0 - inlining_prediction[0], inlining_prediction[0]]
    logits = [[0.0, tf.math.log(probs[1] / (1.0 - probs[1]))[0]]]
    return tfp.distributions.Categorical(logits=logits)

  def _create_action(self, inlining_prediction):
    return tf.cast(inlining_prediction >= 0.5, dtype=tf.int64)

  def _action(self,
              time_step: ts.TimeStep,
              policy_state: types.NestedTensor,
              seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
    new_observation = time_step.observation
    keras_model_input = self._process_observation(new_observation)
    inlining_predict = self._keras_policy(keras_model_input)[0]
    return policy_step.PolicyStep(
        action=self._create_action(inlining_predict), state=policy_state)

  def _distribution(
      self, time_step: ts.TimeStep,
      policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
    new_observation = time_step.observation
    keras_model_input = self._process_observation(new_observation)
    inlining_predict = self._keras_policy(keras_model_input)
    return policy_step.PolicyStep(
        action=self._create_distribution(inlining_predict), state=policy_state)


def train():
  training_weights = None
  if _PROFILING_DATA.value:
    if len(_PROFILING_DATA.value) % 2 != 0:
      raise ValueError('Profiling file paths should always be an even number.')
    training_weights = TrainingWeights()
    for i in range(len(_PROFILING_DATA.value) // 2):
      with open(
          _PROFILING_DATA.value[2 * i], encoding='utf-8') as comp_f, open(
              _PROFILING_DATA.value[2 * i + 1], encoding='utf-8') as eval_f:
        comparator_prof = json.load(comp_f)
        eval_prof = json.load(eval_f)
        training_weights.update_weights(
            comparator_profile=comparator_prof, policy_profile=eval_prof)
  trainer = ImitationLearningTrainer(
      save_model_dir=_SAVE_MODEL_DIR.value, training_weights=training_weights)
  trainer.train(filepaths=_TRAINING_DATA.value)
  if _SAVE_MODEL_DIR.value:
    keras_policy = trainer.get_policy()
    expected_signature, action_spec = config.get_input_signature()
    wrapped_keras_model = WrapKerasModel(
        keras_policy=keras_policy,
        time_step_spec=expected_signature,
        action_spec=action_spec)
    policy_dict = {'tf_agents_policy': wrapped_keras_model}
    saver = policy_saver.PolicySaver(policy_dict=policy_dict)
    saver.save(_SAVE_MODEL_DIR.value)


def main(_):
  gin.parse_config_files_and_bindings(
      FLAGS.get_flag_value('gin_files'),
      FLAGS.get_flag_value('gin_bindings'),
      skip_unknown=False)
  logging.info(gin.config_str())

  train()


if __name__ == '__main__':
  app.run(main)
