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
"""A script for filtering gtests based on whether or not they pass/fail

Within Chromium specifically, there are some test executables that have a lot
of tests that are excellent for benchmarking, but running these test suites
in their entirety can sometimes be problematic as some subsets of the tests
available in the executable might require certain hardware configurations
such as an X configuration with working graphics drivers, and we would prefer
to avoid those tests. This exectuable goes through an entire test suite
description and returns another test suite description containing only tests
that pass.

Usage:
PYTHONPATH=$PYTHONPATH:. python3 \
  ./compiler_opt/benchmark/filter_tests.py \
  --input_tests=./compiler_opt/benchmark/chromium_test_descriptions \
    /browser_tests.json \
  --output_tests=./browser_tests_filtered.json \
  --num_threads=32 \
  --executable_path=/chromium/src/out/Release/browser_tests
"""

import json
import os

from absl import flags
from absl import app
from absl import logging

from compiler_opt.benchmark import gtest_executable_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('input_tests', '',
                    'The path to the test description JSON to filter.')
flags.DEFINE_string(
    'output_tests', '',
    'The path to the JSON file to place the output test suite '
    'description.')
flags.DEFINE_integer(
    'num_threads', 1, 'The number of threads to use for running tests in '
    'parallel.')
flags.DEFINE_string(
    'executable_path', '',
    'The path to the Chromium build directory where all the '
    'test executables are stored')


def main(_):
  if not os.path.exists(FLAGS.executable_path):
    logging.fatal('Executable path does not exist.')
  with open(FLAGS.input_tests, encoding='UTF-8') as test_description_file:
    test_suite_description = json.load(test_description_file)
    test_outputs = gtest_executable_utils.run_test_suite(
        test_suite_description, FLAGS.executable_path, [], FLAGS.num_threads)
    test_list = []
    for test_output in test_outputs:
      test_list.append(test_output['name'])
    # copy the old test suite and just replace the tests array
    new_test_suite_description = test_suite_description
    new_test_suite_description['tests'] = test_list
    with open(FLAGS.output_tests, 'w', encoding='UTF-8') as tests_output_file:
      json.dump(new_test_suite_description, tests_output_file)


if __name__ == '__main__':
  app.run(main)
