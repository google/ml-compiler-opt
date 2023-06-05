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
"""Common benchmarking utilities between chromium and the llvm test suite
"""

import subprocess
import os
import shutil
import tensorflow
import json

from typing import Optional, List


def build_llvm(model_path: str, use_existing_build: bool, llvm_build_path: str,
               llvm_source_path: Optional[str]):
  """Builds LLVM/clang with the specified model and the correct settings

  This function invokes CMake with all the correct build flags specified
  so that the resulting LLVM build is fully setup for the rest of the
  benchmarking process.

  Args:
    model_path: The path to the TF saved model that will be benchmarked
    use_existing_build: Whether or not to do an incremental build
    llvm_build_path: The path to where the LLVM build will go
    llvm_source_path: The path to the root of the llvm-project repository

  Note: llvm_source_path is not necessary if you have set use_existing_build to
  true, you just need to make sure that the existing build is already set up to
  enable the necessary MLGO flags.
  """
  if not use_existing_build and os.path.exists(llvm_build_path):
    shutil.rmtree(llvm_build_path)

  if not os.path.exists(llvm_build_path):
    os.makedirs(llvm_build_path)

  cmake_config_command = [
      "cmake", "-G", "Ninja", f"-DLLVM_RAEVICT_MODEL_PATH={model_path}"
  ]

  if use_existing_build:
    cmake_config_command.append(".")
  else:
    tensorflow_aot_path = os.path.dirname(tensorflow.__file__)
    cmake_config_command.extend([
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DTENSORFLOW_AOT_PATH='{tensorflow_aot_path}'",
        "-DLLVM_ENABLE_PROJECTS='clang;lld'",
        "-DLLVM_ENABLE_RUNTIMES='compiler-rt'", f"{llvm_source_path}"
    ])

  with subprocess.Popen(
      cmake_config_command, cwd=llvm_build_path) as cmake_config_process:
    cmake_config_process.wait()

  cmake_compile_command = ["cmake", "--build", "."]
  with subprocess.Popen(
      cmake_compile_command, cwd=llvm_build_path) as cmake_compile_process:
    cmake_compile_process.wait()


def run_microbenchmark(executable: str, perf_counters: List[str]):
  """Runs all the tests in a specific google benchmark binary

  This function takes in an executable and performance counters according to the
  libpfm naming scheme and then returns the output in the google benchmark
  format.

  Args:
    executable: path to the google benchmark executable to run tests from
    perf_counters: a list of strings of perf counters in the libpfm format
  """
  perf_counters_string = ""
  for perf_counter in perf_counters:
    perf_counters_string = perf_counters_string + perf_counter + ","
  perf_counters_string = perf_counters_string[:-1]
  test_runner_command = [
      executable, "--benchmark_out_format=console",
      "--benchmark_out=/dev/stderr", "--benchmark_format=json",
      f"--benchmark_perf_counters={perf_counters_string}"
  ]

  with subprocess.Popen(
      test_runner_command, stdout=subprocess.PIPE) as test_runner_process:
    out = test_runner_process.communicate()[0]

    out_json = json.loads(out)
    return out_json["benchmarks"]
