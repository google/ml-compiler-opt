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

import json
import os
import re
import functools

from collections.abc import Callable

import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.typing import types


def build_quantile_map(quantile_file_dir: str):
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
    quantile_map[feature_name] = raw_quantiles

  return quantile_map


def discard_fn(obs: types.Float):
  """discard the input feature by setting it to 0."""
  zeros = tf.zeros_like(obs, dtype=tf.float32)
  return tf.expand_dims(zeros, axis=-1)


def identity_fn(obs: types.Float):
  """Return the same value with expanding the last dimension."""
  return tf.cast(tf.expand_dims(obs, -1), tf.float32)


def get_normalize_fn(quantile: list[float],
                     with_sqrt: bool,
                     with_z_score_normalization: bool,
                     eps: float = 1e-8,
                     preprocessing_fn: Callable[[types.Tensor], types.Float]
                     | None = None):
  """Return a normalization function to normalize the input feature."""

  if not preprocessing_fn:
    # pylint: disable=unnecessary-lambda-assignment
    preprocessing_fn = lambda x: x
  processed_quantile = [preprocessing_fn(x) for x in quantile]
  mean = np.mean(processed_quantile)
  std = np.std(processed_quantile)

  def normalize(obs: types.Float):
    obs = tf.expand_dims(obs, -1)
    x = tf.cast(
        tf.raw_ops.Bucketize(input=obs, boundaries=quantile),
        tf.float32) / len(quantile)
    features = [x, x * x]
    if with_sqrt:
      features.append(tf.sqrt(x))
    if with_z_score_normalization:
      y = preprocessing_fn(tf.cast(obs, tf.float32))
      y = (y - mean) / (std + eps)
      features.append(y)
    return tf.concat(features, axis=-1)

  return normalize


def get_ir2vec_normalize_fn(
    ir2vec_with_z_score_normalization: bool = False,
    eps: float = 1e-8) -> Callable[[tf.Tensor], tf.Tensor]:
  """Return a normalization function for embeddings.

  Args:
    ir2vec_with_z_score_normalization: If True, applies z-score standardization
      (whitening) to the embeddings per batch. This transforms the embeddings
      to have zero mean and unit variance across the batch dimension.
      Setting this to True would reduce the impact of outlier embeddings and
      improve training stability, especially when embeddings from different
      batches have significantly different scales.
    eps: Small epsilon value added to standard deviation to prevent division
      by zero during standardization.

  Returns:
    A function that normalizes IR2Vec embeddings according to the specified
    parameters.
  """
  if ir2vec_with_z_score_normalization:

    def z_score_normalize(batch_embeddings: tf.Tensor) -> tf.Tensor:
      """Whitens the embeddings per batch."""
      mean = tf.math.reduce_mean(batch_embeddings, axis=0, keepdims=True)
      std = tf.math.reduce_std(batch_embeddings, axis=0, keepdims=True)
      standardized_embeddings = (batch_embeddings - mean) / (std + eps)
      return standardized_embeddings

    return z_score_normalize
  # Currently, we just return the identity function for embeddings.
  # We can extend this to include other normalizations like L2
  # normalization if needed.
  return functools.partial(tf.cast, dtype=tf.float32)


def get_ir2vec_dimensions_from_vocab_file(vocab_file_path: str) -> int:
  """Read the IR2Vec vocabulary file and get embedding dimensions from the
  first embedding in the first section.

  Args:
    vocab_file_path: Path to the IR2Vec vocabulary JSON file.

  Returns:
    The number of dimensions in the embeddings.

  Raises:
    ValueError: If the vocabulary file structure is invalid.
  """
  # Load the vocabulary file and get the length of the first embedding.
  # Robust structure checks are done by IR2Vec within LLVM.
  # This method could be replaced to use IR2Vec Python APIs, when available.
  with tf.io.gfile.GFile(vocab_file_path, 'r') as f:
    vocab_data = json.load(f)

  # Check if vocab_data is a dict with sections
  if not isinstance(vocab_data, dict) or not vocab_data:
    raise ValueError('Vocabulary file must contain a non-empty dictionary')

  # Get the first section
  sections = vocab_data.values()
  first_section = next(iter(sections), None)
  if not isinstance(first_section, dict) or not first_section:
    raise ValueError('Vocabulary file sections must be non-empty '
                     'dictionaries')

  # Get the first embedding
  embeddings = first_section.values()
  first_embedding = next(iter(embeddings), None)
  if not isinstance(first_embedding, list):
    raise ValueError('Vocabulary file embeddings must be lists')

  if not first_embedding:
    raise ValueError('Vocabulary file embeddings must be non-empty')

  # Find any embedding array and return its length
  return len(first_embedding)
