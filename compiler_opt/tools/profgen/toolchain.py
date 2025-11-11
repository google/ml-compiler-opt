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
"""Generate stand-alone runnable C++ from example IR"""

from typing import Any
import asyncio
import json
import os
from absl import logging
import tensorflow as tf

gfile = tf.io.gfile

_DEFAULT_LLVM_TIMEOUT = 120


async def _run(command: list[str],
               timeout: float = _DEFAULT_LLVM_TIMEOUT,
               env=None):
  logging.info("Running: %s in env: %s", command, env)
  process = await asyncio.create_subprocess_exec(
      *command,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      env=env,
  )
  stdout, stderr = await asyncio.wait_for(
      process.communicate(), timeout=timeout)
  [stdout, stderr] = [stdout.decode(), stderr.decode()]
  logging.info("stdout: %s", stdout)
  logging.info("stderr: %s", stderr)
  return stdout, stderr


class CompilationError(Exception):
  pass


class BuildEnv:

  def __init__(
      self,
      toolchain_path: str,
      workingdir: str,
      tool_suffix="",
      extra_build_flags=[],
  ):
    tool_names = ["clang", "llc", "llvm-extract", "llvm-profdata"]
    tools = [
        os.path.join(toolchain_path, tool + tool_suffix) for tool in tool_names
    ]
    absent = [tool for tool in tools if not os.path.exists(tool)]
    if absent:
      raise FileNotFoundError(
          f"following tools are absent: {', '.join(absent)}")
    [self._clang, self._llc, self._llvm_extract, self._llvm_profdata] = tools
    self._workingdir = workingdir
    self._common_build_flags = [
        "-std=c++17",
        "-Werror",
        "-O2",
    ] + extra_build_flags

  async def verify_cpp(self, cpp_filename: str):
    output_filename = os.path.join(self._workingdir, "temp.exe")
    _, errors = await _run([
        self._clang,
        cpp_filename,
        "-o",
        output_filename,
    ] + self._common_build_flags)
    if not gfile.Exists(output_filename):
      raise CompilationError(errors)

  async def generate_compiled_ir_with_profile(self, cpp_filename: str):
    output_filename = os.path.join(self._workingdir, "temp.exe")
    await _run([
        self._clang,
        cpp_filename,
        "-o",
        output_filename,
        "-fprofile-generate",
        "-Wno-gcc-compat",
    ] + self._common_build_flags)
    prof_raw_file = os.path.join(self._workingdir, "prof.profraw")
    await _run([output_filename], env={"LLVM_PROFILE_FILE": prof_raw_file})
    profdata_file = os.path.join(self._workingdir, "profile.prof")
    await _run([
        self._llvm_profdata,
        "merge",
        "-sparse",
        prof_raw_file,
        "-o",
        profdata_file,
    ])
    full_ir_output = os.path.join(self._workingdir, "full.ll")
    await _run([
        self._clang,
        "-S",
        "-emit-llvm",
        cpp_filename,
        f"-fprofile-use={profdata_file}",
        "-o",
        full_ir_output,
        "-Wno-profile-instr-unprofiled",
        "-Wno-ignored-attributes",
    ] + self._common_build_flags)
    ir_output = os.path.join(self._workingdir, "test.ll")
    await _run([
        self._llvm_extract,
        full_ir_output,
        "-func",
        "main",
        "-delete",
        "-o",
        ir_output,
        "-S",
    ])
    return ir_output

  async def get_ir_stats(self, ir_file) -> dict[str, Any]:
    _, jsondata = await _run([
        self._llc,
        ir_file,
        "-stats",
        "-o",
        "/dev/null",
        "-stats-json",
        "-O2",
    ])
    if not jsondata:
      return {}
    return json.loads(jsondata)
