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

import json
import sys
import subprocess
import re

from joblib import Parallel, delayed

def load_test_set(test_set_file_name):
  """Loads a set of tests to run from a JSON file

  Args:
    test_set_file_name: the path to the file to load the test set from
  """
  with open(test_set_file_name) as test_set_file:
    return json.load(test_set_file)

def run_test(test_executable, test_name, perf_counters):
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
  command_vector = ["perf", "stat"]
  for perf_counter in perf_counters:
    command_vector.extend(["-e", perf_counter])
  command_vector.extend([test_executable, "--gtest_filter={filter}".format(filter=test_name)])
  process = subprocess.Popen(command_vector, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = process.communicate()
  # all of the output from perf stat is on STDERR
  return err.decode("UTF-8")

def parse_perf_stat_output(perf_stat_output, perf_counters):
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
  for line in perf_stat_output.split("\n"):
    for perf_counter in perf_counters:
      if perf_counter in line:
        count_string = re.findall("^\s*\d*", line)[0].replace(" ","")
        count = int(count_string)
        counters_dict[perf_counter] = count
  return counters_dict

def run_and_parse(test_description):
  """Runs a test and processes the output of an individual test

  This function takes in a description of an individual test, runs the test
  to get the perf stat output, and then returns the parsed perf stat output
  in the form of a dictionary

  Args:
    test_description: a tuple in the form of (executable path, test name,
      performance counters to collect) that describes the test
  """
  test_executable, test_name, performance_counters = test_description
  test_output = run_test(test_executable, test_name, performance_counters)
  print("Finished running test {test}".format(test=test_name), file=sys.stderr)
  return (test_name, parse_perf_stat_output(test_output, performance_counters))

def run_test_suite(test_suite_description_path, test_executable, perf_counters, num_threads=1):
  """Runs an entire test suite

  This function takes in a test suite description in the form of a path to a JSON
  file and runs all of the tests within that test suite description, capturing
  all of the performance counters requested. This function also allows different
  tests to be run in parallel for non-parallel sensitive performance counters
  such as ones tracking overall loads and stores, but parallelism should be used
  with extreme caution while benchmarking.

  Args:
    test_suite_description_path: A path to a JSON file containing a list of tests
      to be run for the test suite
    test_executable: A path to the gtest executable being described by the test
      description JSON
    perf_counters: A list of strings of valid perf performance counters that
      are to be collected.
    num_threads: The number of threads to use when running tests. Set to 1 by
      default. Be very cautious about running benchmarks in parallel.
  """
  tests_list = load_test_set(test_suite_description_path)

  test_descriptions = []
  for test in tests_list["tests"]:
    test_descriptions.append((test_executable, test, perf_counters))
  
  test_data_output = Parallel(n_jobs=num_threads)(delayed(run_and_parse)(test_description) for test_description in test_descriptions)

  formatted_test_data = {}
  for test_instance in test_data_output:
    formatted_test_data[test_instance[0]] = test_instance[1]
  
  return formatted_test_data

def get_gtest_testlist_raw(path_to_executable):
  """Gets raw output of a gtest executable's test list

  Takes in a path to a gtest executable and uses the flag --gtest_list_tests
  to get a list of all the tests and then returns the raw output so that it
  can later be parsed.

  Args:
    path_to_executable: A path to the gtest executable for which a test list
      is desired
  """
  command_vector = [path_to_executable, "--gtest_list_tests"]
  process = subprocess.Popen(command_vector, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = process.communicate()
  return out.decode("UTF-8")

def parse_gtest_tests(gtest_output_raw):
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
  current_test_prefix = ""
  gtest_output_split = gtest_output_raw.split("\n")
  current_index = 0
  # skip to the actual test list
  while current_index < len(gtest_output_split):
    current_string = gtest_output_split[current_index]
    test_matches = re.findall("^[a-zA-Z]*\.$", current_string)
    if len(test_matches) != 0:
      break
    current_index += 1
  while current_index < len(gtest_output_split):
    current_string = gtest_output_split[current_index]
    if len(current_string) == 0:
      current_index += 1
      continue
    # get the test name
    test_match = re.findall("^\s*\S*", current_string)[0].replace(" ","")
    if test_match[len(test_match) - 1] == ".":
      # We've found a new prefix
      current_test_prefix = test_match
      current_index += 1
      continue
    test_list.append(current_test_prefix + test_match)
    current_index += 1
  return test_list