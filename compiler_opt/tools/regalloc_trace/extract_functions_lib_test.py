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
"""Tests for the extract_functions_lib library."""

import os

from absl.testing import absltest

from compiler_opt.testing import corpus_test_utils
from compiler_opt.tools.regalloc_trace import extract_functions_lib


class ExtractFunctionsTest(absltest.TestCase):

  def test_get_function_module_map(self):
    fake_llvm_nm_binary = self.create_tempfile("fake_llvm_nm")
    fake_llvm_nm_invocations = self.create_tempfile("fake_llvm_nm_invocations")
    corpus_test_utils.create_test_binary(
        fake_llvm_nm_binary.full_path, fake_llvm_nm_invocations.full_path,
        ["echo \"a T 1140 b\"", "echo \"main T 1150 16\""])
    corpus_dir = self.create_tempdir("corpus")
    _ = corpus_test_utils.setup_corpus(corpus_dir.full_path)

    function_module_map = extract_functions_lib.get_function_module_map(
        corpus_dir.full_path, fake_llvm_nm_binary.full_path)
    self.assertDictEqual(
        function_module_map, {
            "a": os.path.join(corpus_dir.full_path, "module_b.o.bc"),
            "main": os.path.join(corpus_dir.full_path, "module_b.o.bc")
        })

  def test_extract_functions(self):
    output_dir = self.create_tempdir(
        "output", cleanup=absltest.TempFileCleanup.OFF)
    fake_llvm_extract_binary = self.create_tempfile(
        "fake_llvm_extract", cleanup=absltest.TempFileCleanup.OFF)
    fake_llvm_extract_invocations = self.create_tempfile(
        "fake_llvm_extract_invocations")
    corpus_test_utils.create_test_binary(
        fake_llvm_extract_binary.full_path,
        fake_llvm_extract_invocations.full_path, [
            f"touch {os.path.join(output_dir.full_path, '0.bc.fat')}",
            f"touch {os.path.join(output_dir.full_path, '1.bc.fat')}"
        ])
    fake_opt_binary = self.create_tempfile("fake_opt")
    fake_opt_invocations = self.create_tempfile("fake_opt_invocations")
    corpus_test_utils.create_test_binary(fake_opt_binary.full_path,
                                         fake_opt_invocations.full_path)
    corpus_dir = self.create_tempdir("corpus")
    _ = corpus_test_utils.setup_corpus(corpus_dir.full_path)
    functions_to_extract = ["a", "b"]
    function_to_module = {
        "a": os.path.join(corpus_dir, "module_a.o.bc"),
        "b": os.path.join(corpus_dir, "module_b.o.bc")
    }

    extract_functions_lib.extract_functions(functions_to_extract,
                                            function_to_module,
                                            fake_llvm_extract_binary.full_path,
                                            fake_opt_binary.full_path, 1,
                                            output_dir.full_path)
    llvm_extract_invocations = fake_llvm_extract_invocations.read_text().split(
        "\n")
    llvm_extract_invocations.remove("")
    self.assertEqual(
        llvm_extract_invocations[0],
        f"-func a -o {os.path.join(output_dir.full_path, '0.bc.fat')} "
        f"{os.path.join(corpus_dir.full_path, 'module_a.o.bc')}")
    self.assertEqual(
        llvm_extract_invocations[1],
        f"-func b -o {os.path.join(output_dir.full_path, '1.bc.fat')} "
        f"{os.path.join(corpus_dir.full_path, 'module_b.o.bc')}")
    opt_invocations = fake_opt_invocations.read_text().split("\n")
    opt_invocations.remove("")
    self.assertEqual(
        opt_invocations[0],
        f"-strip-debug -o {os.path.join(output_dir.full_path, '0.bc')} "
        f"{os.path.join(output_dir.full_path, '0.bc.fat')}")
    self.assertEqual(
        opt_invocations[1],
        f"-strip-debug -o {os.path.join(output_dir.full_path, '1.bc')} "
        f"{os.path.join(output_dir.full_path, '1.bc.fat')}")


if __name__ == "__main__":
  absltest.main()
