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

from absl import flags
from absl import app

from compiler_opt.tools import benchmarking_utils

default_tests = [
  "harris/harris",
  "SLPVectorization/SLPVectorizationBenchmarks",
  "MemFunctions/MemFunctions",
  "LoopVectorization/LoopVectorizationBenchmarks",
  "LoopInterchange/LoopInterchange",
  "LCALS/SubsetALambdaLoops/lcalsALambda",
  "LCALS/SubsetARawLoops/lcalsARaw",
  "LCALS/SubsetBLambdaLoops/lcalsBLambda",
  "LCALS/SubsetBRawLoops/lcalsBRaw",
  "LCALS/SubsetCLambdaLoops/lcalsCLambda",
  "LCALS/SubsetCRawLoops/lcalsCRaw",
  "ImageProcessing/AnisotropicDiffusion/AnisotropicDiffusion",
  "ImageProcessing/BilateralFiltering/BilateralFilter",
  "ImageProcessing/Blur/blur",
  "ImageProcessing/Dilate/Dilate",
  "ImageProcessing/Dither/Dither",
  "ImageProcessing/Interpolation/Interpolation",
  "Builtins/Int128/Builtins"
]

FLAGS = flags.FLAGS

flags.DEFINE_enum("advisor", None, ["default", "release"], "The regalloc advisor to be used for compiling the test suite")
flags.DEFINE_string("model_path", "", "The path to the regalloc model for testing")
flags.DEFINE_boolean("compile_llvm", True, "compiles llvm using the specified model path")
flags.DEFINE_boolean("llvm_use_incremental", True, "recompile LLVM incrementally rather than doing a whole build")
flags.DEFINE_boolean("compile_testsuite", True, "compiles the test suite using the specified advisor and model path")
flags.DEFINE_string("output_path", None, "The output JSON file containing the test results")
flags.DEFINE_string("llvm_test_suite_path", "/llvm-test-suite", "The path to the LLVM test suite repository root")
flags.DEFINE_string("llvm_build_path", "/llvm-build", "The path to the llvm build")
flags.DEFINE_string("llvm_source_path", "/llvm-project", "The path to the root of the llvm-project repositoy")
flags.DEFINE_string("llvm_test_suite_build_path", None, "The path to the llvm test suite build")
flags.DEFINE_string("tensorflow_c_lib_path", "/tmp/tensorflow", "The path to the tensorflow c lib library")
flags.DEFINE_multi_string("tests_to_run", default_tests, "Tests compiled with google benchmark to run, with paths from the base of the ./microbenchmarks directory")
flags.DEFINE_multi_string("perf_counter", [], "A perf counter to be used (may be defined more than once).")

flags.mark_flag_as_required("advisor")
flags.mark_flag_as_required("output_path")

def build_test_suite(regalloc_advisor, llvm_test_suite_build_path, llvm_build_path):
  """Builds the LLVM test suite using the specified regalloc advisor
  
  This function just builds the llvm test suite from scratch. The only two advisor
  modes are release and default as the main function automatically compiles the advisor
  to be evaluated as the release mode model into llvm. This script then compiles the test
  suite using PGO so that benchmarks will return valid results.

  Args:
    regalloc_advisor: The regalloc advisor (default or release) to use when compiling
    llvm_test_suite_build_path: the path to place the llvm test suite build in
    llvm_build_path: The directory to where the LLVM/clang build has been completed
  """
  llvm_c_compiler_path = os.path.join(llvm_build_path, "./bin/clang")
  llvm_cxx_compiler_path = os.path.join(llvm_build_path, "./bin/clang++")
  llvm_lit_path = os.path.join(llvm_build_path, "./bin/llvm-lit")

  if os.path.exists(llvm_test_suite_build_path):
    shutil.rmtree(llvm_test_suite_build_path)
    
  os.makedirs(llvm_test_suite_build_path)

  cmake_config_command_stage_1 = ["cmake", "-G", "Ninja",
    "-DTEST_SUITE_PROFILE_GENERATE=ON",
    "-DTEST_SUITE_RUN_TYPE=train",
    "-DCMAKE_C_COMPILER={clang}".format(clang=llvm_c_compiler_path),
    "-DCMAKE_CXX_COMPILER={clangxx}".format(clangxx=llvm_cxx_compiler_path),
    "-DCMAKE_CXX_FLAGS='-mllvm -regalloc-enable-advisor={regalloc_advisor}'".format(regalloc_advisor=regalloc_advisor),
    "-DCMAKE_C_FLAGS='-mllvm -regalloc-enable-advisor={regalloc_advisor}'".format(regalloc_advisor=regalloc_advisor),
    "-DBENCHMARK_ENABLE_LIBPFM=ON",
    "-DTEST_SUITE_BENCHMARKING_ONLY=ON",
    "../"]

  cmake_stage_1_process = subprocess.Popen(cmake_config_command_stage_1, cwd=llvm_test_suite_build_path)
  cmake_stage_1_process.wait()

  cmake_compile_command = ["cmake", "--build", "."]
  cmake_stage_1_build_process = subprocess.Popen(cmake_compile_command, cwd=llvm_test_suite_build_path)
  cmake_stage_1_build_process.wait()

  lit_test_runner_command = ["{lit_path}".format(lit_path=llvm_lit_path), "."]
  
  lit_test_runner_process = subprocess.Popen(lit_test_runner_command, cwd=llvm_test_suite_build_path)
  lit_test_runner_process.wait()

  cmake_config_command_stage_2 = ["cmake", "-G", "Ninja",
    "-DTEST_SUITE_PROFILE_GENERATE=OFF",
    "-DTEST_SUITE_PROFILE_USE=ON",
    "-DTEST_SUITE_RUN_TYPE=ref",
    "."]
  
  cmake_stage_2_process = subprocess.Popen(cmake_config_command_stage_2, cwd=llvm_test_suite_build_path)
  cmake_stage_2_process.wait()

  cmake_stage_2_build_process = subprocess.Popen(cmake_compile_command, cwd=llvm_test_suite_build_path)
  cmake_stage_2_build_process.wait()

def main(argv):
  if FLAGS.llvm_test_suite_build_path == None:
    FLAGS.llvm_test_suite_build_path = os.path.join(FLAGS.llvm_test_suite_path, "./build")
  if FLAGS.compile_llvm:
    benchmarking_utils.build_llvm(FLAGS.model_path, FLAGS.llvm_use_incremental, FLAGS.llvm_build_path, FLAGS.llvm_source_path, FLAGS.tensorflow_c_lib_path)
  if FLAGS.compile_testsuite:
    build_test_suite(FLAGS.advisor, FLAGS.llvm_test_suite_build_path, FLAGS.llvm_build_path)
  completed_benchmarks = []
  for test in FLAGS.tests_to_run:
    test_path = os.path.join(FLAGS.llvm_test_suite_build_path, "./MicroBenchmarks/" + test)
    completed_benchmarks.extend(benchmarking_utils.run_microbenchmark(test_path, FLAGS.perf_counter))
  with open(FLAGS.output_path, "w") as output_file:
    output_file.write(json.dumps({"benchmarks": completed_benchmarks}, indent=4))


if __name__ == "__main__":
  app.run(main)