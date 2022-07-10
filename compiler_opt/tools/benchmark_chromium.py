# coding=utf-8
# Copyright 2022 Google LLC
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

import os
import shutil
import subprocess
import json

from joblib import Parallel, delayed

from absl import flags
from absl import app

from compiler_opt.tools import gtest_executable_utils
from compiler_opt.tools import benchmarking_utils

FLAGS = flags.FLAGS

default_test_descriptions = [
  "./compiler_opt/tools/chromium_test_descriptions/base_perftests.json",
  "./compiler_opt/tools/chromium_test_descriptions/browser_tests.json",
  "./compiler_opt/tools/chromium_test_descriptions/components_perftests.json"
]

flags.DEFINE_multi_string("test_description", default_test_descriptions, "(Can be defined multiple times) A path to a test description JSON file containing the test executable and the tests to run")
flags.DEFINE_boolean("compile_tests", True, "Whether or not to compile the tests from scratch")
flags.DEFINE_enum("advisor", None, ["release", "default"], "The advisor to use when compiling chromium")
flags.DEFINE_string("chromium_src_path", "/chromium/src", "The path to the chromium source")
flags.DEFINE_string("depot_tools_path", "/depot_tools", "The path to your depot tools checkout")
flags.DEFINE_string("llvm_build_path", "/llvm-build", "The path to your llvm build")
flags.DEFINE_boolean("compile_llvm", True, "whether or not to compile llvm using the new model")
flags.DEFINE_boolean("llvm_use_incremental", True, "whether or not to use an incremental build while compiling llvm")
flags.DEFINE_string("llvm_source_path", "/llvm-project", "The root path of your local llvm-project checkout")
flags.DEFINE_string("model_path", "", "The path to the model to use when compiling llvm")
flags.DEFINE_string("tensorflow_c_lib_path", "/tmp/tensorflow", "The path to an extracted copy of the tensorflow c library")
flags.DEFINE_string("chromium_build_path", "./out/Release", "The chromium build path, relative to the chromium source directory")
flags.DEFINE_string("output_file", "output.json", "The path to the output file (in JSON format)")
flags.DEFINE_integer("num_threads", 1, "The number of threads to use when running benchmarks. Should be used with caution")
flags.DEFINE_multi_string("perf_counters", ["mem_uops_retired.all_loads","mem_uops_retired.all_stores"], "The performance counters to use")

def build_chromium_tests(regalloc_advisor, chromium_build_path, chromium_source_path, depot_tools_path, llvm_build_path, tests_to_build):
  chromium_absolute_build_path = os.path.join(chromium_source_path, chromium_build_path)
  if os.path.exists(chromium_absolute_build_path):
    shutil.rmtree(chromium_absolute_build_path)
    
  new_environment = os.environ.copy()
  new_environment["PATH"] += ":" + depot_tools_path
  new_environment["CC"] = os.path.join(llvm_build_path, "./bin/clang")
  new_environment["CXX"] = os.path.join(llvm_build_path, "./bin/clang++")
  new_environment["AR"] = os.path.join(llvm_build_path, "./bin/llvm-ar")
  new_environment["NM"] = os.path.join(llvm_build_path, "./bin/llvm-nm")
  new_environment["CPPFLAGS"] = "-mllvm -regalloc-enable-advisor={advisor}".format(advisor=regalloc_advisor)

  gn_args = [
    "is_official_build=true",
    "use_thin_lto=false",
    "is_cfi=false",
    "use_cfi_icall=false",
    "use_cfi_cast=false",
    "clang_use_chrome_plugins=false",
    "is_debug=false",
    "symbol_level=0",
    "custom_toolchain=\\\"//build/toolchain/linux/unbundle:default\\\"",
    "host_toolchain=\\\"//build/toolchain/linux/unbundle:default\\\""
  ]

  gn_args_string = '--args="'
  for arg in gn_args:
    gn_args_string += arg + " "
  gn_args_string += '"'

  gn_config_command = "gn gen " + chromium_build_path + " " + gn_args_string

  print(gn_config_command)

  gn_config_process = subprocess.Popen(gn_config_command, env=new_environment, cwd=chromium_source_path, shell=True)
  gn_config_process.wait()

  ninja_compile_command = ["autoninja", "-C", chromium_build_path]
  ninja_compile_command.extend(tests_to_build)
  ninja_compile_process = subprocess.Popen(ninja_compile_command, env=new_environment, cwd=chromium_source_path)
  ninja_compile_process.wait()

def run_tests(tests_to_run, chromium_absolute_build_path, num_threads, perf_counters):
  test_data = {}
  for test in tests_to_run:
    executable_path = os.path.join(chromium_absolute_build_path, test["executable"])
    test_data.update(gtest_executable_utils.run_test_suite(test, executable_path, perf_counters, num_threads))
  return test_data

def main(argv):
  test_descriptions = []
  for test_description in FLAGS.test_description:
    with open(test_description) as test_description_file:
      print(test_description)
      test_descriptions.append(json.load(test_description_file))
  test_executables = []
  for test_description in test_descriptions:
    test_executables.append(test_description["executable"])
  
  if FLAGS.compile_llvm:
    benchmarking_utils.build_llvm(FLAGS.model_path, FLAGS.llvm_use_incremental, FLAGS.llvm_build_path, FLAGS.llvm_source_path, FLAGS.tensorflow_c_lib_path)

  if FLAGS.compile_tests:
    build_chromium_tests(FLAGS.advisor, FLAGS.chromium_build_path, FLAGS.chromium_src_path, FLAGS.depot_tools_oath, FLAGS.llvm_build_path, test_executables)

  chromium_absolute_build_path = os.path.join(FLAGS.chromium_src_path, FLAGS.chromium_build_path)
  test_data = run_tests(test_descriptions, chromium_absolute_build_path, FLAGS.num_threads, FLAGS.perf_counters)

  with open(FLAGS.output_file, "w") as output_file:
    output_file.write(json.dumps(test_data, indent=4))

if __name__ == "__main__":
  app.run(main)