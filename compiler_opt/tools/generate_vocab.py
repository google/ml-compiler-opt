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
"""Generate sparse buckets - aka "vocab".

Generate numerical features' X (1000) quantile based on their distributions.
"""

import math
import multiprocessing as mp
import os
from typing import Callable, Dict, List, Iterable

from absl import app
from absl import flags
from absl import logging
import gin

import numpy as np
import tensorflow as tf

from compiler_opt.rl import registry

flags.DEFINE_string('input', None,
                    'Path to input file containing tf record datasets.')
flags.DEFINE_string('output_dir', None,
                    'Path to output directory to store quantiles per feature.')
flags.DEFINE_float('sampling_fraction', 1.0,
                   'Fraction to downsample input data.', 0.0, 1.0)
flags.DEFINE_integer(
    'parallelism', None, 'Number of parallel processes to spawn.'
    'Each process does vocab generation for each feature.', 1)
flags.DEFINE_integer('num_buckets', 1000,
                     'Number of quantiles to bucketize feature values into.')
flags.DEFINE_multi_string('gin_files', [],
                          'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


def _get_feature_info(
    serialized_proto: tf.Tensor,
    features_to_not_process: Iterable[str]) -> Dict[str, tf.io.RaggedFeature]:
  """Provides feature information by analyzing a single serialized example.

  Args:
    serialized_proto: serialized SequenceExample.
    features_to_not_process: A list of feature names that should not be
                             processed.

  Returns:
    Dictionary of Tensor formats indexed by feature name.
  """
  example = tf.train.SequenceExample()
  example.ParseFromString(serialized_proto.numpy())
  sequence_features = {}
  for key, feature_list in example.feature_lists.feature_list.items():
    if key in features_to_not_process:
      continue
    feature = feature_list.feature[0]
    kind = feature.WhichOneof('kind')
    if kind == 'float_list':
      sequence_features[key] = tf.io.RaggedFeature(
          partitions=(), dtype=tf.float32)
    elif kind == 'int64_list':
      sequence_features[key] = tf.io.RaggedFeature(
          partitions=(), dtype=tf.int64)
  return sequence_features


def create_tfrecord_parser_fn(
    sequence_features: Dict[str, tf.io.RaggedFeature]
) -> Callable[[str], List[tf.Tensor]]:
  """Create a parser function for reading serialized tf.data.TFRecordDataset.

  Args:
    sequence_features: Dict having feature type info indexed by feature name.

  Returns:
    A callable that takes scalar serialized proto Tensors
    and emits a list of parsed Tensors for all features.
  """

  def _parser_fn(serialized_proto):
    """Helper function that is returned by create_tfrecord_parser_fn."""
    context_features = {}

    # pylint: enable=g-complex-comprehension
    with tf.name_scope('parse'):
      try:
        _, parsed_sequence = tf.io.parse_single_sequence_example(
            serialized_proto,
            context_features=context_features,
            sequence_features=sequence_features)

        parsed_array = []
        for feature_name in sorted(sequence_features):
          v = parsed_sequence[feature_name]
          if isinstance(v, tf.RaggedTensor):
            v = v.to_tensor()
          parsed_array.append(tf.reshape(v, [-1]))
        return parsed_array
      except ValueError as e:
        # ignore malformed or invalid serialized_proto inputs
        print(f'Error: {e}')

  return _parser_fn


def _generate_vocab(feature_values_arrays, feature_name):
  """Downsample and generate vocab using brute force method."""
  feature_values = np.concatenate(feature_values_arrays)
  sample_length = math.floor(
      np.shape(feature_values)[0] * FLAGS.sampling_fraction)
  values = np.random.choice(feature_values, sample_length, replace=False)
  bin_edges = np.quantile(values, np.linspace(0, 1, FLAGS.num_buckets))
  filename = os.path.join(FLAGS.output_dir, f'{feature_name}.buckets')
  with open(filename, 'w', encoding='utf-8') as f:
    for edge in bin_edges:
      f.write(f'{edge}\n')


def main(_) -> None:
  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=FLAGS.gin_bindings, skip_unknown=False)
  logging.info(gin.config_str())
  problem_config = registry.get_configuration()

  # Generate num_buckets quantiles for each feature.
  tf.io.gfile.makedirs(FLAGS.output_dir)
  dataset = tf.data.Dataset.list_files(FLAGS.input)
  dataset = tf.data.TFRecordDataset(dataset)
  features_to_not_process = problem_config.get_nonnormalized_features()

  sequence_features = {}
  # TODO(b/222775595): need to fix this after update to logic for handling
  # empty examples during trace generation.
  for raw_example in dataset:
    try:
      sequence_features = _get_feature_info(raw_example,
                                            features_to_not_process)
      logging.info('Found valid sequence_features dict: %s', sequence_features)
      break
    except IndexError:
      # modules with no results have empty feature values and
      # raise an IndexError. For example, in the inliner case, maybe there were
      # no inlining opportunities (very small modules)
      # continue until a module with non-empty feature values is found.
      logging.warn('Found module with empty feature values.')
      continue
  if not sequence_features:
    raise ValueError('No module with non-empty sequence_features values found.')

  parser_fn = create_tfrecord_parser_fn(sequence_features)
  dataset = dataset.map(parser_fn, num_parallel_calls=tf.data.AUTOTUNE)
  data_list = np.array(list(dataset.as_numpy_iterator()), dtype=object)
  data_list = data_list.swapaxes(0, 1)

  with mp.Pool(FLAGS.parallelism) as pool:
    feature_names = list(sorted(sequence_features))
    for i, feature_values_arrays in enumerate(data_list):
      pool.apply_async(_generate_vocab, (
          feature_values_arrays,
          feature_names[i],
      ))
    pool.close()
    pool.join()


if __name__ == '__main__':
  flags.mark_flag_as_required('input')
  flags.mark_flag_as_required('output_dir')
  app.run(main)
