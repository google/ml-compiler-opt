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
"""Utilities for working with test executables using gtest (ie Chromium)"""

import sys
import subprocess
import re

from joblib import Parallel, delayed
from typing import Tuple, List, Optional, Dict
from absl import logging


def run_test(test_executable: str, test_name: str, perf_counters: List[str]):
  """Runs a specific test

  This function executes a specific test in a gtest executable using
  the --gtest_filter flag to select a specific test. All performance
  counter information is captured using perf stat.

  Args:
    test_executable: the path to the gtest executable containing the test
    test_name: the name of the test to be run
    perf_counters: A string list of performance counters recognized by
      perf stat (platform dependent)
  """
  command_vector = ['perf', 'stat']
  for perf_counter in perf_counters:
    command_vector.extend(['-e', perf_counter])
  command_vector.extend([test_executable, f'--gtest_filter={test_name}'])
  with subprocess.Popen(
      command_vector, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE) as process:
    out, err = process.communicate()
    decoded_stderr = err.decode('UTF-8')
    decoded_stdout = out.decode('UTF-8')
    if process.returncode != 0:
      logging.warning('test %s failed', test_name)
      raise RuntimeError(f'Test executable failed while running {test_name}')
    elif 'PASSED' not in decoded_stdout:
      logging.warning('test %s does not exist', test_name)
      raise RuntimeError(f'No test {test_name} exists in test executable')
    # all of the output from perf stat is on STDERR
    return decoded_stderr


def parse_perf_stat_output(perf_stat_output: str, perf_counters: List[str]):
  """Parses raw output from perf stat

  This function takes in the raw decoded output from perf stat
  and parses it into a dictionary containing each of the requested
  performance counters as a key.

  Args:
    perf_stat_output: raw decoded output from a perf stat run
    perf_counters: A list of strings of valid perf stat
      performance counters
  """
  counters_dict = {}
  for line in perf_stat_output.split('\n'):
    for perf_counter in perf_counters:
      if perf_counter in line:
        count_string = re.findall(r'^\s*\d*', line)[0].replace(' ', '')
        count = int(count_string)
        counters_dict[perf_counter] = count
  return counters_dict


def run_and_parse(test_description: Tuple[str, str, List[str]]):
  """Runs a test and processes the output of an individual test

  This function takes in a description of an individual test, runs the test
  to get the perf stat output, and then returns the parsed perf stat output
  in the form of a dictionary

  Args:
    test_description: a tuple in the form of (executable path, test name,
      performance counters to collect) that describes the test
  """
  test_executable, test_name, performance_counters = test_description
  try:
    test_output = run_test(test_executable, test_name, performance_counters)
    print(f'Finished running test {test_name}', file=sys.stderr)
    return (test_name, parse_perf_stat_output(test_output,
                                              performance_counters))
  except RuntimeError:
    return None


def run_test_suite(test_suite_description: Dict[str, List[str]],
                   test_executable: str, perf_counters: List[str],
                   num_threads: Optional[int]):
  """Runs an entire test suite

  This function takes in a test set description in the form of a path to a JSON
  file and runs all of the tests within that test suite description, capturing
  all of the performance counters requested. This function also allows different
  tests to be run in parallel for non-parallel sensitive performance counters
  such as ones tracking overall loads and stores, but parallelism should be used
  with extreme caution while benchmarking.

  Args:
    test_suite_description: A python dictionary containing an array with the
      key tests which has all the tests to run
    test_executable: A path to the gtest executable being described by the test
      description JSON
    perf_counters: A list of strings of valid perf performance counters that
      are to be collected.
    num_threads: The number of threads to use when running tests. Set to 1 by
      default. Be very cautious about running benchmarks in parallel.
  """

  if num_threads is None:
    num_threads = 1

  test_descriptions = []
  for test in test_suite_description['tests']:
    test_descriptions.append((test_executable, test, perf_counters))

  test_data_output = Parallel(n_jobs=num_threads)(
      delayed(run_and_parse)(test_description)
      for test_description in test_descriptions)

  formatted_test_data = []
  for test_instance in test_data_output:
    if test_instance:
      test_info = {'name': test_instance[0], 'iterations': 1}
      test_info.update(test_instance[1])
      formatted_test_data.append(test_info)

  return formatted_test_data


def get_gtest_testlist_raw(path_to_executable: str):
  """Gets raw output of a gtest executable's test list

  Takes in a path to a gtest executable and uses the flag --gtest_list_tests
  to get a list of all the tests and then returns the raw output so that it
  can later be parsed.

  Args:
    path_to_executable: A path to the gtest executable for which a test list
      is desired
  """
  command_vector = [path_to_executable, '--gtest_list_tests']
  with subprocess.Popen(
      command_vector, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE) as process:
    out = process.communicate()[0]
    return out.decode('UTF-8')


def parse_gtest_tests(gtest_output_raw: str):
  """Parses gtest test list output into a Python list

  Loops through each line in the raw output obtained from the
  get_gtest_testlist_raw functions and relies on the current output
  structure in order to be able to parse the test names.

  Args:
    gtest_output_raw: A string containing the decoded
      raw output from a gtest executable run with the
      --gtest_list_tests flag
  """
  test_list = []
  current_test_prefix = ''
  gtest_output_split = gtest_output_raw.split('\n')
  current_index = 0
  # skip to the actual test list
  while current_index < len(gtest_output_split):
    current_string = gtest_output_split[current_index]
    test_matches = re.findall(r'^[a-zA-Z]*\.$', current_string)
    if len(test_matches) != 0:
      break
    current_index += 1
  while current_index < len(gtest_output_split):
    current_string = gtest_output_split[current_index]
    if len(current_string) == 0:
      current_index += 1
      continue
    # get the test name
    test_match = re.findall(r'^\s*\S*', current_string)[0].replace(' ', '')
    if test_match[len(test_match) - 1] == '.':
      # We've found a new prefix
      current_test_prefix = test_match
      current_index += 1
      continue
    test_list.append(current_test_prefix + test_match)
    current_index += 1
  return test_list
