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
"""Test for InliningWorker."""

import gin
import numpy as np
import tensorflow as tf
from absl.testing import absltest
from unittest import mock

from compiler_opt.es.inlining import inlining_worker
from compiler_opt.es import policy_utils
from compiler_opt.rl import corpus
from compiler_opt.testing import corpus_test_utils


def mock_get_observation_processing_layer_creator_for_patch():
  """Mock for get_observation_processing_layer_creator."""

  def mock_layer_creator_fn(obs_spec):
    return tf.keras.layers.Lambda(lambda x: x, name=f"identity_{obs_spec.name}")

  return mock_layer_creator_fn


class InliningWorkerTest(absltest.TestCase):

  def setUp(self):
    gin.parse_config_file("compiler_opt/es/inlining/gin_configs/inlining.gin")

  def test_compile_and_get_size_no_policy(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[
            corpus.ModuleSpec(name="smth1", size=1, command_line=("-cc1",)),
            corpus.ModuleSpec(name="smth2", size=1, command_line=("-cc1",))
        ])
    loaded_modules = [
        cps.load_module_spec(cps.module_specs[0]),
        cps.load_module_spec(cps.module_specs[1])
    ]

    fake_clang_binary = self.create_tempfile("fake_clang")
    fake_clang_invocations = self.create_tempfile("fake_clang_invocations")
    corpus_test_utils.create_test_binary(fake_clang_binary.full_path,
                                         fake_clang_invocations.full_path)

    fake_llvm_size_binary = self.create_tempfile("fake_llvm_size")
    fake_llvm_size_invocations = self.create_tempfile(
        "fake_llvm_size_invocations")
    llvm_size_output_script = ["echo 'dummy_output:'", "echo '150'"]
    corpus_test_utils.create_test_binary(
        fake_llvm_size_binary.full_path,
        fake_llvm_size_invocations.full_path,
        commands_to_run=llvm_size_output_script)

    # pylint: disable=line-too-long
    with mock.patch(
        "compiler_opt.rl.inlining.config.get_observation_processing_layer_creator",
        new=mock_get_observation_processing_layer_creator_for_patch):
      # pylint: enable=line-too-long
      worker = inlining_worker.InliningWorker(
          gin_config="",
          clang_path=fake_clang_binary.full_path,
          llvm_size_path=fake_llvm_size_binary.full_path,
          thread_count=1)

    total_size = worker.compile(policy=None, modules=loaded_modules)
    self.assertEqual(total_size, 150 + 150)

    # Check for inliner flags when no policy is given
    clang_invocations_content = fake_clang_invocations.read_text()
    self.assertIn("-mllvm -enable-ml-inliner=development",
                  clang_invocations_content)
    self.assertNotIn("-ml-inliner-model-under-training",
                     clang_invocations_content)

    # Two modules, so two llvm-size calls.
    llvm_size_invocations_content = fake_llvm_size_invocations.read_text()
    self.assertEqual(
        llvm_size_invocations_content.strip().count("\n") +
        (1 if llvm_size_invocations_content.strip() else 0), 2)

  def test_compile_and_get_size_with_policy(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[
            corpus.ModuleSpec(name="smth1", size=1, command_line=("-cc1",)),
            corpus.ModuleSpec(name="smth2", size=1, command_line=("-cc1",))
        ])
    loaded_modules = [
        cps.load_module_spec(cps.module_specs[0]),
        cps.load_module_spec(cps.module_specs[1])
    ]

    fake_clang_binary = self.create_tempfile("fake_clang")
    fake_clang_invocations = self.create_tempfile("fake_clang_invocations")
    corpus_test_utils.create_test_binary(fake_clang_binary.full_path,
                                         fake_clang_invocations.full_path)

    fake_llvm_size_binary = self.create_tempfile("fake_llvm_size")
    fake_llvm_size_invocations = self.create_tempfile(
        "fake_llvm_size_invocations")
    llvm_size_output_script = ["echo 'dummy_output:'", "echo '100'"]
    corpus_test_utils.create_test_binary(
        fake_llvm_size_binary.full_path,
        fake_llvm_size_invocations.full_path,
        commands_to_run=llvm_size_output_script)

    # pylint: disable=line-too-long
    with mock.patch(
        "compiler_opt.rl.inlining.config.get_observation_processing_layer_creator",
        new=mock_get_observation_processing_layer_creator_for_patch):
      # pylint: enable=line-too-long
      worker = inlining_worker.InliningWorker(
          gin_config="",
          clang_path=fake_clang_binary.full_path,
          llvm_size_path=fake_llvm_size_binary.full_path,
          thread_count=1)

    mock_tflite_path = "/fake/policy.tflite"
    with mock.patch.object(
        policy_utils, "convert_to_tflite",
        return_value=mock_tflite_path) as mock_convert:
      dummy_policy_bytes = np.array([1.0, 2.0], dtype=np.float32).tobytes()
      total_size = worker.compile(
          policy=dummy_policy_bytes, modules=loaded_modules)

      self.assertEqual(total_size, 200)

      mock_convert.assert_called_once()
      self.assertEqual(mock_convert.call_args[0][0], dummy_policy_bytes)
      # pylint: disable=protected-access
      self.assertEqual(mock_convert.call_args[0][2],
                       worker._tf_base_policy_path)
      # pylint: enable=protected-access

    # Check for inliner flags when a policy is given
    clang_invocations_content = fake_clang_invocations.read_text()
    self.assertIn(f"-mllvm -ml-inliner-model-under-training={mock_tflite_path}",
                  clang_invocations_content)
    self.assertIn("-mllvm -enable-ml-inliner=development",
                  clang_invocations_content)

  def test_compile_failure_returns_none(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[
            corpus.ModuleSpec(name="smth1", size=1, command_line=("-cc1",))
        ])
    loaded_modules = [cps.load_module_spec(cps.module_specs[0])]

    fake_clang_binary = self.create_tempfile("fake_clang")
    fake_clang_invocations = self.create_tempfile("fake_clang_invocations")
    corpus_test_utils.create_test_binary(fake_clang_binary.full_path,
                                         fake_clang_invocations.full_path)

    fake_llvm_size_binary = self.create_tempfile("fake_llvm_size")
    fake_llvm_size_invocations = self.create_tempfile(
        "fake_llvm_size_invocations")
    llvm_size_output_script = ["echo 'not a valid output'"]

    corpus_test_utils.create_test_binary(
        fake_llvm_size_binary.full_path,
        fake_llvm_size_invocations.full_path,
        commands_to_run=llvm_size_output_script)

    # pylint: disable=line-too-long
    with mock.patch(
        "compiler_opt.rl.inlining.config.get_observation_processing_layer_creator",
        new=mock_get_observation_processing_layer_creator_for_patch):
      # pylint: enable=line-too-long
      worker = inlining_worker.InliningWorker(
          gin_config="",
          clang_path=fake_clang_binary.full_path,
          llvm_size_path=fake_llvm_size_binary.full_path,
          thread_count=1)

    # Test with llvm-size causing an error during parsing
    total_size = worker.compile(policy=None, modules=loaded_modules)
    self.assertIsNone(total_size)

    # Test with _compile_module_and_get_size mocked to return float('inf')
    with mock.patch.object(
        worker, "_compile_module_and_get_size", return_value=float("inf")):
      self.assertRaises(
          ValueError, worker.compile, policy=None, modules=loaded_modules)

    # Test with _compile_module_and_get_size mocked to raise an exception
    with mock.patch.object(
        worker,
        "_compile_module_and_get_size",
        side_effect=ValueError("Simulated compilation error")):
      total_size_exception = worker.compile(policy=None, modules=loaded_modules)
      self.assertIsNone(total_size_exception)


if __name__ == "__main__":
  absltest.main()
