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
"""Test for RegallocTraceWorker."""

import os
import json
from pathlib import Path
import stat
import textwrap

from absl.testing import absltest
import numpy as np
import gin

from compiler_opt.es.regalloc_trace import regalloc_trace_worker
from compiler_opt.rl import corpus


def _setup_corpus(corpus_dir: str,
                  has_thinlto: bool = False,
                  cli_flags: tuple = ()) -> list[corpus.ModuleSpec]:
  modules = [
      corpus.ModuleSpec("module_a.o", 1, ("-fmodule-a", *cli_flags), True),
      corpus.ModuleSpec("module_b.o", 1, ("-fmodule-b", *cli_flags), True)
  ]

  corpus_description = {
      "has_thinlto": has_thinlto,
      "modules": [module.name for module in modules]
  }

  with open(
      os.path.join(corpus_dir, "corpus_description.json"),
      "w",
      encoding="utf-8") as corpus_description_handle:
    json.dump(corpus_description, corpus_description_handle)

  for module in ["module_a.o", "module_b.o"]:
    extensions = [".cmd", ".bc"]
    if has_thinlto:
      extensions.append(".thinlto.bc")

    for extension in extensions:
      module_path = os.path.join(corpus_dir, module + extension)
      Path(module_path).touch()

  return modules


def _create_test_binary(binary_path: str, output_path: str):
  test_binary = textwrap.dedent(f"""\
  #!/bin/bash
  echo "$@" >> {output_path}
  echo 1
  echo 1
  """)

  with open(binary_path, "w", encoding="utf-8") as binary_handle:
    binary_handle.write(test_binary)
  binary_stat = os.stat(binary_path)
  os.chmod(binary_path, binary_stat.st_mode | stat.S_IEXEC)


class RegallocTraceWorkerTest(absltest.TestCase):

  def setUp(self):
    gin.parse_config_file(
        "compiler_opt/es/regalloc_trace/gin_configs/regalloc_trace.gin")

  def test_build_corpus_and_evaluate(self):
    corpus_dir = self.create_tempdir("corpus")
    corpus_modules = _setup_corpus(corpus_dir.full_path)
    fake_clang_binary = self.create_tempfile("fake_clang")
    fake_clang_invocations = self.create_tempfile("fake_clang_invocations")
    _create_test_binary(fake_clang_binary.full_path,
                        fake_clang_invocations.full_path)
    fake_bb_trace_model_binary = self.create_tempfile(
        "fake_basic_block_trace_model")
    fake_bb_trace_model_invocations = self.create_tempfile(
        "fake_basic_block_trace_model_invocations")
    _create_test_binary(fake_bb_trace_model_binary.full_path,
                        fake_bb_trace_model_invocations.full_path)

    worker = regalloc_trace_worker.RegallocTraceWorker(
        gin_config="",
        clang_path=fake_clang_binary.full_path,
        basic_block_trace_model_path=fake_bb_trace_model_binary.full_path,
        thread_count=1,
        corpus_path=corpus_dir.full_path)
    total_cost = worker.compile_corpus_and_evaluate(corpus_modules,
                                                    "function_index_path.pb",
                                                    "bb_trace_path.pb", None)
    self.assertEqual(total_cost, 2)

    # Check that we are compiling the modules with the appropriate flags and
    # the default regalloc advisor given we did not pass in any TFLite policy.
    clang_command_lines = fake_clang_invocations.read_text().split("\n")
    clang_command_lines.remove("")
    self.assertLen(clang_command_lines, 2)
    self.assertTrue("-fmodule-a" in clang_command_lines[0])
    self.assertTrue(
        "-regalloc-enable-advisor=default" in clang_command_lines[0])
    self.assertTrue("-fmodule-b" in clang_command_lines[1])
    self.assertTrue(
        "-regalloc-enable-advisor=default" in clang_command_lines[1])

    # Check that we pass the expected flags to basic_block_trace_model.
    bb_trace_model_command_line = fake_bb_trace_model_invocations.read_text(
    ).split("\n")[0].split()
    self.assertLen(bb_trace_model_command_line, 5)
    self.assertTrue("--corpus_path" in bb_trace_model_command_line[0])
    self.assertEqual("--function_index_path=function_index_path.pb",
                     bb_trace_model_command_line[1])
    self.assertEqual("--thread_count=1", bb_trace_model_command_line[2])
    self.assertEqual("--bb_trace_path=bb_trace_path.pb",
                     bb_trace_model_command_line[3])
    self.assertEqual("--model_type=mca", bb_trace_model_command_line[4])

  def test_compile_corpus_and_evaluate_with_tflite(self):
    corpus_dir = self.create_tempdir("corpus")
    corpus_modules = _setup_corpus(corpus_dir.full_path)
    fake_clang_binary = self.create_tempfile("fake_clang")
    fake_clang_invocations = self.create_tempfile("fake_clang_invocations")
    _create_test_binary(fake_clang_binary.full_path,
                        fake_clang_invocations.full_path)
    fake_bb_trace_model_binary = self.create_tempfile(
        "fake_basic_block_trace_model")
    fake_bb_trace_model_invocations = self.create_tempfile(
        "fake_basic_block_trace_model_invocations")
    _create_test_binary(fake_bb_trace_model_binary.full_path,
                        fake_bb_trace_model_invocations.full_path)

    test_policy = np.ones(6777, dtype=np.float32)

    worker = regalloc_trace_worker.RegallocTraceWorker(
        gin_config="",
        clang_path=fake_clang_binary.full_path,
        basic_block_trace_model_path=fake_bb_trace_model_binary.full_path,
        thread_count=1,
        corpus_path=corpus_dir.full_path)
    worker.compile_corpus_and_evaluate(corpus_modules, "function_index_path.pb",
                                       "bb_trace_path.pb",
                                       test_policy.tobytes())

    # Assert that we pass the TFLite model to the clang invocations.
    clang_command_lines = fake_clang_invocations.read_text().split("\n")
    clang_command_lines.remove("")
    self.assertLen(clang_command_lines, 2)
    self.assertTrue(
        "-regalloc-enable-advisor=development" in clang_command_lines[0])
    self.assertTrue("-regalloc-model=" in clang_command_lines[0])
    self.assertTrue(
        "-regalloc-enable-advisor=development" in clang_command_lines[1])
    self.assertTrue("-regalloc-model=" in clang_command_lines[1])

  def test_copy_corpus_locally(self):
    corpus_copy_base_dir = self.create_tempdir("corpus_copy")
    corpus_copy_dir = os.path.join(corpus_copy_base_dir.full_path,
                                   "corpus_copy")
    corpus_dir = self.create_tempdir("corpus")
    _ = _setup_corpus(corpus_dir.full_path)
    worker = regalloc_trace_worker.RegallocTraceWorker(
        gin_config="",
        clang_path="/fake/path/to/clamg",
        basic_block_trace_model_path="/fake/path/to/basic_block_trace_model",
        thread_count=1,
        corpus_path=corpus_dir.full_path,
        copy_corpus_locally_path=corpus_copy_dir)

    self.assertTrue(
        os.path.exists(os.path.join(corpus_copy_dir, "module_a.o.bc")))
    self.assertTrue(
        os.path.exists(os.path.join(corpus_copy_dir, "module_a.o.cmd")))
    self.assertTrue(
        os.path.exists(os.path.join(corpus_copy_dir, "module_b.o.bc")))
    self.assertTrue(
        os.path.exists(os.path.join(corpus_copy_dir, "module_b.o.cmd")))

    # Check that the worker cleans up after itself upon deletion.
    del worker
    self.assertFalse(os.path.exists(corpus_copy_dir))

  def test_copy_corpus_locally_thinlto(self):
    corpus_copy_base_dir = self.create_tempdir("corpus_copy")
    corpus_copy_dir = os.path.join(corpus_copy_base_dir.full_path,
                                   "corpus_copy")
    corpus_dir = self.create_tempdir("corpus")
    _ = _setup_corpus(corpus_dir.full_path, True)
    _ = regalloc_trace_worker.RegallocTraceWorker(
        gin_config="",
        clang_path="/fake/path/to/clamg",
        basic_block_trace_model_path="/fake/path/to/basic_block_trace_model",
        thread_count=1,
        corpus_path=corpus_dir.full_path,
        copy_corpus_locally_path=corpus_copy_dir)

    self.assertTrue(
        os.path.exists(os.path.join(corpus_copy_dir, "module_a.o.thinlto.bc")))
    self.assertTrue(
        os.path.exists(os.path.join(corpus_copy_dir, "module_b.o.thinlto.bc")))

  def test_remote_corpus_replacement_flags(self):
    corpus_copy_base_dir = self.create_tempdir("corpus_copy")
    corpus_copy_dir = os.path.join(corpus_copy_base_dir.full_path,
                                   "corpus_copy")
    corpus_dir = self.create_tempdir("corpus")
    profile_path = os.path.join(corpus_dir, "profile.prof")
    Path(profile_path).touch()
    corpus_modules = _setup_corpus(corpus_dir.full_path, False,
                                   ("-fprofile-instr-use={prof}",))

    fake_clang_binary = self.create_tempfile("fake_clang")
    fake_clang_invocations = self.create_tempfile("fake_clang_invocations")
    _create_test_binary(fake_clang_binary.full_path,
                        fake_clang_invocations.full_path)
    fake_bb_trace_model_binary = self.create_tempfile(
        "fake_basic_block_trace_model")
    fake_bb_trace_model_invocations = self.create_tempfile(
        "fake_basic_block_trace_model_invocations")
    _create_test_binary(fake_bb_trace_model_binary.full_path,
                        fake_bb_trace_model_invocations.full_path)

    worker = regalloc_trace_worker.RegallocTraceWorker(
        gin_config="",
        clang_path=fake_clang_binary.full_path,
        basic_block_trace_model_path=fake_bb_trace_model_binary.full_path,
        thread_count=1,
        corpus_path=corpus_dir.full_path,
        copy_corpus_locally_path=corpus_copy_dir,
        aux_file_replacement_flags={"prof": profile_path})

    copied_profile_path = os.path.join(corpus_copy_dir, "profile.prof")
    self.assertTrue(os.path.exists(copied_profile_path))
    _ = worker.compile_corpus_and_evaluate(corpus_modules,
                                           "function_index_path.pb",
                                           "bb_trace_path.pb", None)
    clang_command_lines = fake_clang_invocations.read_text().split("\n")
    clang_command_lines.remove("")
    self.assertLen(clang_command_lines, 2)
    self.assertTrue(
        f"-fprofile-instr-use={copied_profile_path}" in clang_command_lines[0])
    self.assertTrue(
        f"-fprofile-instr-use={copied_profile_path}" in clang_command_lines[1])
