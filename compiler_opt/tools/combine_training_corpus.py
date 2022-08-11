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
r"""Combine multiple training corpus into a single training corpus.

Usage: we'd like to combine training corpus corpus1 and corpus2 into
combinedcorpus; we first structure the files as follows:

combinedcorpus
combinedcorpus/corpus1
combinedcorpus/corpus2

Running this script with

python3 \
compiler_opt/tools/combine_training_corpus.py \
  --root_dir=$PATH_TO_combinedcorpus

generates combinedcorpus/module_path file. In this way corpus1 and
corpus2 are combined into combinedcorpus.
"""

import json
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

flags.DEFINE_string('root_dir', '', 'root dir of module paths to combine.')

FLAGS = flags.FLAGS

_FILE_NAME = 'corpus_description.json'


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  module_names = []

  for sub_dir in tf.io.gfile.listdir(FLAGS.root_dir):
    path = os.path.join(FLAGS.root_dir, sub_dir, _FILE_NAME)

    logging.info('processing %s', path)

    if not tf.io.gfile.exists(path):
      logging.error('%s does not exist.', path)
      continue

    with tf.io.gfile.GFile(path, 'r') as f:
      corpus_description = json.load(f)
      module_names.extend([
          os.path.join(sub_dir, name) for name in corpus_description['modules']
      ])

  # Assume other configs the same as the last corpus_decsription loaded.
  corpus_description['modules'] = module_names

  with tf.io.gfile.GFile(
      os.path.join(FLAGS.root_dir, _FILE_NAME), 'w') as f:
    json.dump(corpus_description, f, indent=2)


if __name__ == '__main__':
  app.run(main)
