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
"""A script for running chromium based benchmarks

This script allows for running chromium benchmarks to evaluate the performance
of MLGO regalloc models in a highly automated fashion. It will automatically
recompile LLVM if requested with the release mode model specified, recompile
the chromium benchmarks using the correct MLGO model/advisor, and then run
a specified subset of those tests designed to minimize run to run variability.

Usage:
PYTHONPATH=$PYTHONPATH:. python3 \
  ./compiler_opt/benchmark/benchmark_chromium.py \
  --compile_tests \
  --advisor=release \
  --chromium_src_path=/chromium/src \
  --depot_tools_path=/depot_tools \
  --llvm_build_path=/llvm-build \
  --compile_llvm \
  --model_path=/tmp/model \
  --chromium_build_path=./out/Release \
  --output_file=./output.json \
  --perf_counters=mem_uops_retired.all_loads \
  --perf_counters=mem_uops_retired.all_stores

Note that --perf_counters can be defined multiple times to grab more than one
performance counter. Also note that the chromium_build_path is a relative
directory. It is relative to the chromium source dir and specified this way
as there appears to be problems building chromium outside of the source
directory.
"""

import os
import shutil
import subprocess
import json

from absl import flags
from absl import app

from compiler_opt.benchmark import gtest_executable_utils
from compiler_opt.benchmark import benchmarking_utils

from typing import List, Dict, Union

FLAGS = flags.FLAGS

test_prefix = './compiler_opt/benchmark/chromium_test_descriptions/'

test_description_files = [
    'base_perftests.json', 'browser_tests.json', 'components_perftests.json',
    'base_unittests.json', 'cc_unittests.json', 'components_unittests.json',
    'content_unittests.json'
]

default_test_descriptions = [
    f'{test_prefix}{test_dsc}' for test_dsc in test_description_files
]

flags.DEFINE_multi_string(
    'test_description', default_test_descriptions,
    '(Can be defined multiple times) A path to a test'
    'description JSON file containing the test executable'
    'and the tests to run')
flags.DEFINE_boolean('compile_tests', True,
                     'Whether or not to compile the tests from scratch')
flags.DEFINE_enum('advisor', None, ['release', 'default'],
                  'The advisor to use when compiling chromium')
flags.DEFINE_string('chromium_src_path', '/chromium/src',
                    'The path to the chromium source')
flags.DEFINE_string('depot_tools_path', '/depot_tools',
                    'The path to your depot tools checkout')
flags.DEFINE_string('llvm_build_path', '/llvm-build',
                    'The path to your llvm build')
flags.DEFINE_boolean('compile_llvm', True,
                     'whether or not to compile llvm using the new model')
flags.DEFINE_boolean(
    'llvm_use_incremental', True,
    'whether or not to use an incremental build while'
    'compiling llvm')
flags.DEFINE_string('llvm_source_path', '/llvm-project',
                    'The root path of your local llvm-project checkout')
flags.DEFINE_string('model_path', '',
                    'The path to the model to use when compiling llvm')
flags.DEFINE_string(
    'chromium_build_path', './out/Release',
    'The chromium build path, relative to the chromium source'
    'directory')
flags.DEFINE_string('output_file', 'output.json',
                    'The path to the output file (in JSON format)')
flags.DEFINE_integer(
    'num_threads', 1, 'The number of threads to use when running benchmarks.'
    'Should be used with caution')
flags.DEFINE_multi_string(
    'perf_counters',
    ['mem_uops_retired.all_loads', 'mem_uops_retired.all_stores'],
    'The performance counters to use')


def build_chromium_tests(regalloc_advisor: str, chromium_build_path: str,
                         chromium_source_path: str, depot_tools_path: str,
                         llvm_build_path: str, tests_to_build: List[str]):
  """Builds the chromium test suite

  This function will build the specified chromium tests using the specified
  regalloc advisor. This function configures some default build options using gn
  as shown below in the gn_args list, and then builds the needed targets using
  autoninja.

  Args:
    regalloc_advisor: The regalloc advisor to use when compiling
    chromium_build_path: The path (relative to the chromium source dir) to use
      for building chromium
    chromium_source_path: The path to the chromium source
    depot_tools_path: The path to the root of your depot tools checkout
    llvm_build_path: The path to the root of the directory where llvm was built
    tests_to_build: An array of test targets that are to be built
  """
  chromium_absolute_build_path = os.path.join(chromium_source_path,
                                              chromium_build_path)
  if os.path.exists(chromium_absolute_build_path):
    shutil.rmtree(chromium_absolute_build_path)

  new_environment = os.environ.copy()
  new_environment['PATH'] += ':' + depot_tools_path
  new_environment['CC'] = os.path.join(llvm_build_path, './bin/clang')
  new_environment['CXX'] = os.path.join(llvm_build_path, './bin/clang++')
  new_environment['AR'] = os.path.join(llvm_build_path, './bin/llvm-ar')
  new_environment['NM'] = os.path.join(llvm_build_path, './bin/llvm-nm')
  new_environment['CPPFLAGS'] = \
    f'-mllvm -regalloc-enable-advisor={regalloc_advisor}'

  gn_args = [
      'is_official_build=true', 'use_thin_lto=false', 'is_cfi=false',
      'use_cfi_icall=false', 'use_cfi_cast=false',
      'clang_use_chrome_plugins=false', 'is_debug=false', 'symbol_level=0',
      'custom_toolchain=\\\"//build/toolchain/linux/unbundle:default\\\"',
      'host_toolchain=\\\"//build/toolchain/linux/unbundle:default\\\"'
  ]

  gn_args_string = '--args="'
  for arg in gn_args:
    gn_args_string += arg + ' '
  gn_args_string += '"'

  gn_config_command = 'gn gen ' + chromium_build_path + ' ' + gn_args_string
  with subprocess.Popen(
      gn_config_command,
      env=new_environment,
      cwd=chromium_source_path,
      shell=True) as gn_config_process:
    gn_config_process.wait()

  ninja_compile_command = ['autoninja', '-C', chromium_build_path]
  ninja_compile_command.extend(tests_to_build)
  with subprocess.Popen(
      ninja_compile_command, env=new_environment,
      cwd=chromium_source_path) as ninja_compile_process:
    ninja_compile_process.wait()


def run_tests(tests_to_run: List[Dict[str, Union[str, List[str]]]],
              chromium_absolute_build_path: str, num_threads: int,
              perf_counters: List[str]):
  """A utility to run a set of chromium tests

  This function takes in a list of test descriptions containing the
  name of a chromium test target as well as a list of all the tests
  within that test executable that are to be run. It executes each test
  (in parallel if specified for performance counters that aren't highly
  sensitive to that environment), grabbing the perf counters for the test
  that are specified.

  Args:
    tests_to_run: A list of python dictionaries containing the test descriptions
    chromium_absolute_build_path: The absolute build path to the chromium
      build dir
    num_threads: The number of threads to use when running tests
    perf_counters: A list of perf compatible performance counters
  """
  test_data = []
  for test in tests_to_run:
    executable_path = os.path.join(chromium_absolute_build_path,
                                   test['executable'])
    test_data.extend(
        gtest_executable_utils.run_test_suite(test, executable_path,
                                              perf_counters, num_threads))
  return test_data


def main(_):
  test_descriptions = []
  for test_description in FLAGS.test_description:
    with open(test_description, encoding='UTF-8') as test_description_file:
      print(test_description)
      test_descriptions.append(json.load(test_description_file))
  test_executables = []
  for test_description in test_descriptions:
    test_executables.append(test_description['executable'])

  if FLAGS.compile_llvm:
    benchmarking_utils.build_llvm(FLAGS.model_path, FLAGS.llvm_use_incremental,
                                  FLAGS.llvm_build_path, FLAGS.llvm_source_path)

  if FLAGS.compile_tests:
    build_chromium_tests(FLAGS.advisor, FLAGS.chromium_build_path,
                         FLAGS.chromium_src_path, FLAGS.depot_tools_path,
                         FLAGS.llvm_build_path, test_executables)

  chromium_absolute_build_path = os.path.join(FLAGS.chromium_src_path,
                                              FLAGS.chromium_build_path)
  test_data = run_tests(test_descriptions, chromium_absolute_build_path,
                        FLAGS.num_threads, FLAGS.perf_counters)

  with open(FLAGS.output_file, 'w', encoding='UTF-8') as output_file:
    output_data = {'benchmarks': test_data}
    output_file.write(json.dumps(output_data, indent=4))


if __name__ == '__main__':
  app.run(main)
