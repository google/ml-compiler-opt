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
"""Training state management library.

Manages serialization and parameter history for multi-model evolution strategy
training. Checkpoints are saved in `checkpoint.json` alongside subdirectories
containing the active JAX models and params (in `models.pkl`) and optionally
`.tflite` model files.

Checkpoint JSON Schema:
  step: Current training iteration index.
  latest_models_dir: Subdirectory name containing the active model set.
  model_keys: List of model identifier strings matching the loaded state.
  max_reward: Maximum reward observed so far.
  optimizer_state: Internal state configuration of the blackbox optimizer.
"""

from collections.abc import Callable
import dataclasses
import json
import logging
import os

from absl import flags
import cloudpickle
from compiler_opt.es import blackbox_optimizers
from compiler_opt.jax import dense_nn_bc
from etils import epath
import jax
import jax.flatten_util
import numpy as np
import numpy.typing as npt

_INIT_FROM_CHECKPOINT = flags.DEFINE_string(
    "init_from_checkpoint",
    None,
    "Path to a checkpoint to initialize the training state from if no "
    "checkpoint exists in the output directory.",
)

_PICKLED_MODULES_FILENAME = "models.pkl"


class TrainingState:
  """Container managing historical training parameters and serialization."""

  def __init__(
      self,
      output_path: str,
      get_initial_models: Callable[
          [],
          dict[str, dense_nn_bc.ModelSpec],
      ],
      optimizer: blackbox_optimizers.BlackboxOptimizer,
  ):
    self._output_path = epath.Path(output_path)
    self._optimizer = optimizer
    self._checkpoint_path = self._output_path / "checkpoint.json"

    self._model_specs = {}
    self._model_keys = []

    ckpt = None
    ckpt_source_path = None
    if self._checkpoint_path.exists():
      logging.info("Loading checkpoint from %s", self._checkpoint_path)
      ckpt = json.loads(self._checkpoint_path.read_text())
      ckpt_source_path = self._checkpoint_path
    elif _INIT_FROM_CHECKPOINT.value:
      logging.info("Loading initial checkpoint from %s",
                   _INIT_FROM_CHECKPOINT.value)
      init_ckpt_path = epath.Path(_INIT_FROM_CHECKPOINT.value)
      if not init_ckpt_path.exists():
        raise FileNotFoundError(
            f"init_from_checkpoint {_INIT_FROM_CHECKPOINT.value} does not exist"
        )
      ckpt = json.loads(init_ckpt_path.read_text())
      ckpt_source_path = init_ckpt_path

    if ckpt is not None:
      logging.info("Restoring models from checkpoint %s", ckpt_source_path)
      assert ckpt_source_path is not None
      ckpt_dir = ckpt_source_path.parent
      latest_models_dir = ckpt["latest_models_dir"]
      self._model_keys = ckpt["model_keys"]
      pickle_path = ckpt_dir / latest_models_dir / _PICKLED_MODULES_FILENAME
      with pickle_path.open("rb") as f:
        self._model_specs = cloudpickle.load(f)

      self._step = ckpt["step"] + 1
      self._max_reward = ckpt["max_reward"]
      self._optimizer.set_state(ckpt["optimizer_state"])
    else:
      initial_models = get_initial_models()
      if not isinstance(initial_models, dict):
        raise ValueError("get_initial_model must return a dict")
      self._model_specs = initial_models
      self._model_keys = sorted(self._model_specs.keys())
      self._step = 0
      self._max_reward = 0.0

    self._unravel_fns = {}
    self._flat_shapes = {}
    for k in self._model_keys:
      flat, unravel = jax.flatten_util.ravel_pytree(self._model_specs[k].params)
      self._unravel_fns[k] = unravel
      self._flat_shapes[k] = flat.shape[0]

    self._model_weights = np.concatenate([
        jax.flatten_util.ravel_pytree(self._model_specs[k].params)[0]
        for k in self._model_keys
    ])

    if not self._checkpoint_path.exists():
      init_dir_name = "initial_model"
      init_dir = self._output_path / init_dir_name
      init_dir.mkdir(parents=True, exist_ok=True)
      for k in self._model_keys:
        path = os.fspath(init_dir / f"{k}.tflite")
        self._model_specs[k].save_tflite(path)

      pickle_path = init_dir / _PICKLED_MODULES_FILENAME
      with pickle_path.open("wb") as f:
        cloudpickle.dump(self._model_specs, f)

      self._save_checkpoint(self._step - 1, init_dir_name)

  @property
  def current_step(self) -> int:
    return self._step

  @property
  def current_weights(self) -> npt.NDArray[np.float32]:
    return self._model_weights

  @property
  def model_specs(self) -> dict[str, dense_nn_bc.ModelSpec]:
    return self._model_specs

  def save_policies(
      self,
      rewards: list[float],
      perturbations: list[npt.NDArray[np.float32]],
      new_model_weights: npt.NDArray[np.float32],
  ) -> float:
    max_reward = np.max(rewards)

    if self._max_reward < max_reward:
      self._max_reward = max_reward
      logging.info(
          "Found new best model with reward %f at step %d, saving.",
          max_reward,
          self._step,
      )
      max_index = np.argmax(rewards)
      perturbation = perturbations[max_index]

      best_weights = self._model_weights + perturbation
      best_dir_name = f"best_policy_{max_reward}_{self._step}"
      best_dir = self._output_path / best_dir_name
      best_dir.mkdir(parents=True, exist_ok=True)
      offset = 0
      for k in self._model_keys:
        size = self._flat_shapes[k]
        model_weights = best_weights[offset:offset + size]
        offset += size

        model_params = self._unravel_fns[k](model_weights)
        best_model = dense_nn_bc.ModelSpec(
            model=self._model_specs[k].model,
            params=model_params,
            sample_inputs=self._model_specs[k].sample_inputs,
        )
        best_model.save_tflite(os.fspath(best_dir / f"{k}.tflite"))

    iter_dir_name = f"iteration_{self._step}"
    iter_dir = self._output_path / iter_dir_name
    iter_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    for k in self._model_keys:
      size = self._flat_shapes[k]
      model_flat_weights = new_model_weights[offset:offset + size]
      offset += size

      new_params = self._unravel_fns[k](model_flat_weights)
      self._model_specs[k] = dataclasses.replace(
          self._model_specs[k], params=new_params)

      self._model_specs[k].save_tflite(os.fspath(iter_dir / f"{k}.tflite"))

    pickle_path = iter_dir / _PICKLED_MODULES_FILENAME
    with pickle_path.open("wb") as f:
      cloudpickle.dump(self._model_specs, f)

    self._save_checkpoint(self._step, iter_dir_name)

    self._model_weights = new_model_weights
    ret_max = self._max_reward
    self._step += 1

    return ret_max

  def _save_checkpoint(self, step: int, latest_models_dir: str) -> None:
    ckpt_data = {
        "step": step,
        "latest_models_dir": latest_models_dir,
        "model_keys": self._model_keys,
        "max_reward": self._max_reward,
        "optimizer_state": self._optimizer.get_state(),
    }

    tmp_checkpoint = self._output_path / "checkpoint.json.tmp"
    tmp_checkpoint.write_text(json.dumps(ckpt_data, indent=2))
    tmp_checkpoint.replace(self._checkpoint_path)
