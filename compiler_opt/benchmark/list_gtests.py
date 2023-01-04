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
"""Lists gtests in a JSON format
This script takes in a gtest executable and outputs all of the tests
contained within that gtest executable to the command line separated by
line breaks or in a JSON format.

Usage:
PYTHONPATH=$PYTHONPATH:. python3 \
compiler_opt/benchmark/list_gtests.py \
  --gtest_executable=/path/to/executable \
  --output_type=json \
  --output_file=/path/to/output

The gtest_executable flag is required. The output_type by default is set
to JSON. If you don't provide an output file path, the output will just
be printed to the terminal (fully compatible with shell pipes).
"""

import json
import sys
import os

from absl import app
from absl import flags
from absl import logging

from compiler_opt.benchmark import gtest_executable_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('gtest_executable', None,
                    'The path to the gtest executable')
flags.DEFINE_enum(
    'output_type', 'json', ['json', 'default'],
    """The output type. JSON produces JSON style output with
                  the tests being in an array with the key tests. Default
                  prints all tests separated by line breaks""")
flags.DEFINE_string(
    'output_file', None, """The output path. If not set, all output will get
                    dumped to the terminal""")

flags.mark_flag_as_required('gtest_executable')


def main(_):
  test_list_raw_output = gtest_executable_utils.get_gtest_testlist_raw(
      FLAGS.gtest_executable)
  test_list = gtest_executable_utils.parse_gtest_tests(test_list_raw_output)

  output = ''
  if FLAGS.output_type == 'json':
    test_json = {
        'executable': os.path.basename(FLAGS.gtest_executable),
        'tests': test_list
    }
    output = json.dumps(test_json, indent=4)
  elif FLAGS.output_type == 'default':
    for test in test_list:
      output = output + f'{test}\n'
    # get rid of extra last newline
    output = output[:-1]
  else:
    logging.fatal('output_type should only be json or default')

  if FLAGS.output_file:
    with open(FLAGS.output_file, 'w', encoding='UTF-8') as output_file:
      output_file.write(output)
      print(f'wrote tests to {FLAGS.output_file}', file=sys.stderr)
  else:
    print(output)


if __name__ == '__main__':
  app.run(main)
