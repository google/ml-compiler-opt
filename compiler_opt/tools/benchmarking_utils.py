import subprocess
import os
import shutil
import tensorflow
import json

def build_llvm(model_path, use_existing_build, llvm_build_path, llvm_source_path=None, tensorflow_c_lib_path=None):
  """Builds LLVM/clang with the specified model and the correct settings

  This function invokes CMake with all the correct build flags specified
  so that the resulting LLVM build is fully setup for the rest of the
  benchmarking process.

  Args:
    model_path: The path to the TF saved model that will be benchmarked
    use_existing_build: Whether or not to do an incremental build
    llvm_build_path: The path to where the LLVM build will go
    llvm_source_path: The path to the root of the llvm-project repository
    tensorflow_c_lib_path: The path to the tensorflow c lib path

  Note: llvm_source_path and tensorflow_c_lib_path aren't necessary if you have
  set use_existing_build to true, you just need to make sure that the existing
  build is already set up to enable the necessary MLGO flags.
  """
  if not use_existing_build and os.path.exists(llvm_build_path):
    shutil.rmtree(llvm_build_path)

  cmake_config_command = ["cmake", "-G", "Ninja",
    "-DLLVM_RAEVICT_MODEL_PATH={model_path}".format(model_path=model_path)]

  if use_existing_build:
    cmake_config_command.append(".")
  else:
    tensorflow_aot_path = os.path.dirname(tensorflow.__file__)
    cmake_config_command.extend([
      "-DCMAKE_BUILD_TYPE=Release",
      f"-DTENSORFLOW_C_LIB_PATH={tensorflow_c_lib_path}",
      f"-DTENSORFLOW_AOT_PATH='{tensorflow_aot_path}'",
      "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
      "-DLLVM_ENABLE_PROJECTS='clang'",
      "-DLLVM_ENABLE_RUNTIMES='compiler-rt'",
      f"{llvm_source_path}"
    ])
  
  cmake_config_process = subprocess.Popen(cmake_config_command, cwd=llvm_build_path)
  cmake_config_process.wait()

  cmake_compile_command = ["cmake", "--build", "."]
  cmake_compile_process = subprocess.Popen(cmake_compile_command, cwd=llvm_build_path)
  cmake_compile_process.wait()

"""Runs all the tests in a specific google benchmark binary

This function takes in an executable and performance counters according to the
libpfm naming scheme and then returns the output in the google benchmark format.

Args:
  executable: path to the google benchmark executable to run tests from
  perf_counters: a list of strings of perf counters in the libpfm format
"""
def run_microbenchmark(executable, perf_counters):
  perf_counters_string = ""
  for perf_counter in perf_counters:
    perf_counters_string = perf_counters_string + perf_counter + ","
  perf_counters_string = perf_counters_string[:-1]
  test_runner_command = [
    executable,
    "--benchmark_out_format=console",
    "--benchmark_out=/dev/stderr",
    "--benchmark_format=json",
    f"--benchmark_perf_counters={perf_counters_string}"
  ]

  test_runner_process = subprocess.Popen(test_runner_command, stdout=subprocess.PIPE)
  out, err = test_runner_process.communicate()

  out_json = json.loads(out)
  return out_json["benchmarks"]
