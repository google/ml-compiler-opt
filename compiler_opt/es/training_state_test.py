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
"""Tests for training_state management."""

import json
import multiprocessing
import pickle
import traceback
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from compiler_opt.es import blackbox_optimizers
from compiler_opt.es import training_state
from compiler_opt.jax import dense_nn_bc
from compiler_opt.jax import litert_adapter
from etils import epath
import jax
import jax.flatten_util
import numpy as np


def _worker_process_fn(pipe, model_spec, flat_weights, inputs):
  try:
    pipe.send(("progress", "started"))

    params = jax.tree_util.tree_map(jax.device_put, model_spec.params)
    sample_inputs = jax.tree_util.tree_map(jax.device_put,
                                           model_spec.sample_inputs)
    flat_weights = jax.device_put(flat_weights)
    inputs = jax.tree_util.tree_map(jax.device_put, inputs)
    pipe.send(("progress", "converted_to_jax"))

    _, unravel_fn = jax.flatten_util.ravel_pytree(params)
    params = unravel_fn(flat_weights)
    pipe.send(("progress", "unravelled"))

    edge_model = litert_adapter.convert(
        model_spec.model.apply, params, jax_fn_kwargs=sample_inputs)
    pipe.send(("progress", "converted"))

    outputs = edge_model(**inputs)
    pipe.send(("progress", "evaluated"))

    pipe.send(("success", outputs))
  except Exception as e:  # pylint: disable=broad-exception-caught
    pipe.send(("failure", f"{str(e)}\n{traceback.format_exc()}"))
  finally:
    pipe.close()


class FakeOptimizer(blackbox_optimizers.BlackboxOptimizer):

  def __init__(self):
    self.state = "initial_state"

  def get_state(self):
    return self.state

  def set_state(self, state):
    self.state = state

  def get_hyperparameters(self):
    raise NotImplementedError()

  def run_step(self, current_weights, perturbations, rewards):
    raise NotImplementedError()

  def update_state(self, rewards, perturbations):
    raise NotImplementedError()


def create_dummy_model_spec(num_features=2, hidden_size=4, seed=0):
  constants = {
      "internal_x_mean": np.zeros(num_features, dtype=np.float32),
      "internal_x_std": np.ones(num_features, dtype=np.float32),
  }
  model = dense_nn_bc.DenseModel(
      hidden_layers=[hidden_size], constants=constants)
  sample_inputs = {"x": np.zeros((1, num_features), dtype=np.float32)}
  key = jax.random.key(seed)
  params = model.init(key, **sample_inputs)
  return dense_nn_bc.ModelSpec.fromJax(
      model=model, params=params, sample_inputs=sample_inputs)


class TrainingStateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = epath.Path(self.create_tempdir().full_path)
    self.spec_a = create_dummy_model_spec(2, 4, seed=0)
    self.spec_b = create_dummy_model_spec(3, 2, seed=1)

    self.get_initial_models = lambda: {
        "model_a": self.spec_a,
        "model_b": self.spec_b,
    }
    self.optimizer = FakeOptimizer()

  @mock.patch.object(litert_adapter, "convert")
  def test_constructor_initializes_new_state(self, mock_convert):
    mock_edge_model = mock.MagicMock()
    mock_edge_model.tflite_model.return_value = b"dummy_tflite_bytes"
    mock_convert.return_value = mock_edge_model

    state = training_state.TrainingState(
        output_path=str(self.test_dir),
        get_initial_models=self.get_initial_models,
        optimizer=self.optimizer,
    )

    self.assertEqual(state.current_step, 0)
    flat_a, _ = jax.flatten_util.ravel_pytree(self.spec_a.params)
    flat_b, _ = jax.flatten_util.ravel_pytree(self.spec_b.params)
    expected_weights = np.concatenate([flat_a, flat_b])
    np.testing.assert_array_equal(state.current_weights, expected_weights)

    self.assertTrue(
        (self.test_dir / "initial_model" / "model_a.tflite").exists())
    self.assertTrue(
        (self.test_dir / "initial_model" / "model_b.tflite").exists())

    pkl_path = (
        self.test_dir / "initial_model" /
        training_state._PICKLED_MODULES_FILENAME)
    self.assertTrue(pkl_path.exists())
    with pkl_path.open("rb") as f:
      saved_specs = pickle.load(f)
    self.assertEqual(saved_specs.keys(), {"model_a", "model_b"})
    np.testing.assert_array_equal(
        jax.flatten_util.ravel_pytree(saved_specs["model_a"].params)[0], flat_a)

    checkpoint_path = self.test_dir / "checkpoint.json"
    self.assertTrue(checkpoint_path.exists())
    ckpt_data = json.loads(checkpoint_path.read_text())
    self.assertEqual(ckpt_data["step"], -1)
    self.assertEqual(ckpt_data["latest_models_dir"], "initial_model")
    self.assertEqual(ckpt_data["model_keys"], ["model_a", "model_b"])

  @mock.patch.object(litert_adapter, "convert")
  def test_save_policies_creates_checkpoints_and_best_policy(
      self, mock_convert):
    mock_edge_model = mock.MagicMock()
    mock_edge_model.tflite_model.return_value = b"dummy_tflite_bytes"
    mock_convert.return_value = mock_edge_model

    state = training_state.TrainingState(
        output_path=str(self.test_dir),
        get_initial_models=self.get_initial_models,
        optimizer=self.optimizer,
    )

    flat_a, _ = jax.flatten_util.ravel_pytree(self.spec_a.params)
    flat_b, _ = jax.flatten_util.ravel_pytree(self.spec_b.params)
    size_a = flat_a.shape[0]
    size_b = flat_b.shape[0]
    total_size = size_a + size_b

    rng = np.random.default_rng(0)
    new_weights = rng.random(total_size, dtype=np.float32)
    perturbations = [
        rng.random(total_size, dtype=np.float32),
        rng.random(total_size, dtype=np.float32),
    ]

    max_reward = state.save_policies(
        rewards=[10.0, 20.0],
        perturbations=perturbations,
        new_model_weights=new_weights,
    )

    self.assertEqual(max_reward, 20.0)
    self.assertEqual(state.current_step, 1)
    np.testing.assert_array_equal(state.current_weights, new_weights)

    best_dir = self.test_dir / "best_policy_20.0_0"
    self.assertTrue(best_dir.exists())
    self.assertTrue((best_dir / "model_a.tflite").exists())
    self.assertTrue((best_dir / "model_b.tflite").exists())

    iter_dir = self.test_dir / "iteration_0"
    self.assertTrue(iter_dir.exists())
    self.assertTrue((iter_dir / "model_a.tflite").exists())
    self.assertTrue((iter_dir / "model_b.tflite").exists())
    self.assertTrue(
        (iter_dir / training_state._PICKLED_MODULES_FILENAME).exists())

    checkpoint_path = self.test_dir / "checkpoint.json"
    ckpt_data = json.loads(checkpoint_path.read_text())
    self.assertEqual(ckpt_data["step"], 0)
    self.assertEqual(ckpt_data["latest_models_dir"], "iteration_0")
    self.assertEqual(ckpt_data["max_reward"], 20.0)

  def test_restores_from_checkpoint(self):
    checkpoint_data = {
        "step": 42,
        "latest_models_dir": "iteration_42",
        "model_keys": ["model_a", "model_b"],
        "max_reward": 99.0,
        "optimizer_state": "checkpointed_state",
    }
    checkpoint_path = self.test_dir / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint_data))

    iter_dir = self.test_dir / "iteration_42"
    iter_dir.mkdir(parents=True, exist_ok=True)

    restored_spec_a = create_dummy_model_spec(2, 4, seed=2)
    restored_spec_b = create_dummy_model_spec(3, 2, seed=3)

    model_specs = {
        "model_a": restored_spec_a,
        "model_b": restored_spec_b,
    }
    with (iter_dir / training_state._PICKLED_MODULES_FILENAME).open("wb") as f:
      pickle.dump(model_specs, f)

    state = training_state.TrainingState(
        output_path=str(self.test_dir),
        get_initial_models=self.get_initial_models,
        optimizer=self.optimizer,
    )

    self.assertEqual(state.current_step, 43)
    self.assertEqual(self.optimizer.get_state(), "checkpointed_state")

    flat_a, _ = jax.flatten_util.ravel_pytree(restored_spec_a.params)
    flat_b, _ = jax.flatten_util.ravel_pytree(restored_spec_b.params)
    expected_weights = np.concatenate([flat_a, flat_b])
    np.testing.assert_array_equal(state.current_weights, expected_weights)

  @mock.patch.object(litert_adapter, "convert")
  def test_restores_from_given_checkpoint(self, mock_convert):
    mock_edge_model = mock.MagicMock()
    mock_edge_model.tflite_model.return_value = b"dummy_tflite_bytes"
    mock_convert.return_value = mock_edge_model

    other_dir = epath.Path(self.create_tempdir().full_path)
    checkpoint_data = {
        "step": 42,
        "latest_models_dir": "iteration_42",
        "model_keys": ["model_a", "model_b"],
        "max_reward": 99.0,
        "optimizer_state": "checkpointed_state",
    }
    given_checkpoint_path = other_dir / "checkpoint.json"
    given_checkpoint_path.write_text(json.dumps(checkpoint_data))

    iter_dir = other_dir / "iteration_42"
    iter_dir.mkdir(parents=True, exist_ok=True)

    restored_spec_a = create_dummy_model_spec(2, 4, seed=2)
    restored_spec_b = create_dummy_model_spec(3, 2, seed=3)
    model_specs = {
        "model_a": restored_spec_a,
        "model_b": restored_spec_b,
    }
    with (iter_dir / training_state._PICKLED_MODULES_FILENAME).open("wb") as f:
      pickle.dump(model_specs, f)

    with flagsaver.flagsaver(init_from_checkpoint=str(given_checkpoint_path)):
      state = training_state.TrainingState(
          output_path=str(self.test_dir),
          get_initial_models=self.get_initial_models,
          optimizer=self.optimizer,
      )

    self.assertEqual(state.current_step, 43)
    self.assertEqual(self.optimizer.get_state(), "checkpointed_state")

    flat_a, _ = jax.flatten_util.ravel_pytree(restored_spec_a.params)
    flat_b, _ = jax.flatten_util.ravel_pytree(restored_spec_b.params)
    expected_weights = np.concatenate([flat_a, flat_b])
    np.testing.assert_array_equal(state.current_weights, expected_weights)

    self.assertTrue(
        (self.test_dir / "initial_model" / "model_a.tflite").exists())
    self.assertTrue(
        (self.test_dir / "initial_model" / "model_b.tflite").exists())
    self.assertTrue((self.test_dir / "initial_model" /
                     training_state._PICKLED_MODULES_FILENAME).exists())

  def test_worker_simulation_different_process(self):
    constants = {
        "internal_input_a_mean": np.array([1.0], dtype=np.float32),
        "internal_input_a_std": np.array([2.0], dtype=np.float32),
        "internal_input_b_mean": np.array([3.0], dtype=np.float32),
    }
    model = dense_nn_bc.DenseModel(hidden_layers=[2], constants=constants)
    sample_inputs = {
        "input_a": np.zeros((1, 1), dtype=np.float32),
        "input_b": np.zeros((1, 1), dtype=np.float32),
    }
    key = jax.random.key(0)
    params_jax = model.init(key, **sample_inputs)

    params_jax["params"]["Dense_0"]["kernel"] = np.array(
        [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    params_jax["params"]["Dense_0"]["bias"] = np.array([0.5, -0.5],
                                                       dtype=np.float32)
    params_jax["params"]["Dense_1"]["kernel"] = np.array([[0.5], [1.5]],
                                                         dtype=np.float32)
    params_jax["params"]["Dense_1"]["bias"] = np.array([1.0], dtype=np.float32)

    model_spec = dense_nn_bc.ModelSpec(
        model=model, params=params_jax, sample_inputs=sample_inputs)

    flat_weights, _ = jax.flatten_util.ravel_pytree(params_jax)

    perturbation = np.arange(1, len(flat_weights) + 1, dtype=np.float32)

    perturbed_weights_numpy = np.array(flat_weights) + perturbation

    def make_inputs(a, b):
      return {
          "input_a": np.array([[a]], dtype=np.float32),
          "input_b": np.array([[b]], dtype=np.float32),
      }

    def make_jax_inputs(a, b):
      return jax.tree_util.tree_map(jax.device_put, make_inputs(a, b))

    eval_inputs = make_inputs(5.0, 7.0)
    eval_inputs_jax = make_jax_inputs(5.0, 7.0)

    _, unravel_fn = jax.flatten_util.ravel_pytree(params_jax)
    perturbed_params = unravel_fn(perturbed_weights_numpy)
    perturbed_params_jax = jax.tree_util.tree_map(jax.device_put,
                                                  perturbed_params)
    expected_output = model.apply(perturbed_params_jax, **eval_inputs_jax)

    variations = [
        expected_output,
        model.apply(params_jax, **eval_inputs_jax),
        model.apply(params_jax, **make_jax_inputs(-1.0, 3.0)),
        model.apply(perturbed_params_jax, **make_jax_inputs(-1.0, 3.0)),
    ]
    self.assertTrue(all(v.size == 1 for v in variations))
    variations = [v.item() for v in variations]
    self.assertEmpty([v for v in variations if np.isnan(v)])
    self.assertLen(set(variations), 4)

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(
        target=_worker_process_fn,
        args=(child_conn, model_spec, perturbed_weights_numpy, eval_inputs),
    )
    p.start()

    success = False
    value = None
    while True:
      if parent_conn.poll(timeout=120):
        msg_type, msg_val = parent_conn.recv()
        if msg_type == "progress":
          print(f"Worker progress: {msg_val}", flush=True)
        elif msg_type == "success":
          success = True
          value = msg_val
          break
        elif msg_type == "failure":
          self.fail(f"Worker failed with: {msg_val}")
          break
      else:
        p.terminate()
        self.fail("Worker process hung (no progress for 120s)")
        break

    p.join()

    self.assertTrue(success)

    if isinstance(value, dict):
      value = list(value.values())[0]

    expected_output_np = np.array(expected_output)
    np.testing.assert_allclose(value, expected_output_np, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
