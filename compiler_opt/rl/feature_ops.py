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

import numpy as np
import tensorflow.compat.v2 as tf


def build_quantile_map(quantile_file_dir):
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


def discard_fn(obs):
  """discard the input feature by setting it to 0."""
  return tf.zeros(shape=obs.shape + [1], dtype=tf.float32)


def get_normalize_fn(quantile,
                     mean,
                     std,
                     with_sqrt,
                     with_z_score_normalization,
                     eps=1e-8):
  """Return a normalization function to normalize the input feature."""

  def normalize(obs):
    obs = tf.expand_dims(obs, -1)
    x = tf.cast(
        tf.raw_ops.Bucketize(input=obs, boundaries=quantile),
        tf.float32) / len(quantile)
    features = [x, x * x]
    if with_sqrt:
      features.append(tf.sqrt(x))
    if with_z_score_normalization:
      y = tf.cast(obs, tf.float32)
      y = (y - mean) / (std + eps)
      features.append(y)
    return tf.concat(features, axis=-1)

  return normalize
