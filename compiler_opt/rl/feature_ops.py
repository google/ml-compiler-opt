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

"""operations to transform features (observations)."""

import os
import re

import gin
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.networks import expand_dims_layer


def _build_quantile_map(quantile_file_dir):
  """build feature quantile map by reading from files in quantile_file_dir."""
  quantile_map = {}
  pattern = os.path.join(quantile_file_dir, '(.*).buckets')
  for quantile_file_path in tf.io.gfile.glob(
      os.path.join(quantile_file_dir, '*.buckets')):
    m = re.fullmatch(pattern, quantile_file_path)
    assert m
    feature_name = m.group(1)
    with tf.io.gfile.GFile(quantile_file_path, 'r') as quantile_file:
      raw_quantiles = [float(x) for x in quantile_file]
    quantile_map[feature_name] = (raw_quantiles, np.mean(raw_quantiles),
                                  np.std(raw_quantiles))
  return quantile_map


@gin.configurable
def get_observation_processing_layer_creator(quantile_file_dir,
                                             with_sqrt=False,
                                             with_z_score_normalization=False,
                                             eps=1e-8):
  """Wrapper for observation_processing_layer."""
  quantile_map = _build_quantile_map(quantile_file_dir)
  expand_dims_op = expand_dims_layer.ExpandDims(-1)

  def observation_processing_layer(obs_spec):
    """Creates the layer to process observation given obs_spec."""
    if obs_spec.name not in quantile_map:

      # Return empty will cause tf.keras.layers.Concatenate to fail.
      def discard_feature(obs):
        expanded_obs = expand_dims_op(obs)
        return tf.zeros_like(expanded_obs, dtype=tf.float32)

      func = discard_feature

    else:
      quantile, mean, std = quantile_map[obs_spec.name]

      def normalization(obs):
        expanded_obs = expand_dims_op(obs)
        x = tf.cast(
            tf.raw_ops.Bucketize(input=expanded_obs, boundaries=quantile),
            tf.float32) / len(quantile)
        features = [x, x * x]
        if with_sqrt:
          features.append(tf.sqrt(x))
        if with_z_score_normalization:
          y = tf.cast(expanded_obs, tf.float32)
          y = (y - mean) / (std + eps)
          features.append(y)
        return tf.concat(features, axis=-1)

      func = normalization

    return tf.keras.layers.Lambda(func)

  return observation_processing_layer
