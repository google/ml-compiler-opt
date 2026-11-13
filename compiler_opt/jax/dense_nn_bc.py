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
"""Behavioral cloning for dense neural networks."""

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import logging
from typing import Any

from compiler_opt.jax import litert_adapter
from etils import epath
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

FeatureTransformFn = Callable[[Any], Any]


def _identity_fn(x: Any) -> Any:
  return x


def _default_feature_transforms(*_) -> FeatureTransformFn:
  return _identity_fn


def _extract_features_and_constants(
    records: Iterable[Any],
    label_name: str,
    feature_transforms: Callable[
        [str, Any], FeatureTransformFn] = _default_feature_transforms,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
  """Extracts raw features, target label, and feature normalization constants."""
  features_accum: dict[str, list[Any]] = {}

  for record in records:
    for name, val in record.features.items():
      if name not in features_accum:
        features_accum[name] = []
      features_accum[name].append(val.value)

  if not features_accum:
    raise ValueError("No data found in records")

  all_features: dict[str, np.ndarray] = {}
  for name, values in features_accum.items():
    all_features[name] = np.array(values, dtype=np.float32)

  labels = all_features.pop(label_name)

  constants: dict[str, np.ndarray] = {}
  for name, data in all_features.items():
    transformed_data = feature_transforms(name, np)(data)

    mean = np.mean(transformed_data, axis=0)
    std = np.std(transformed_data, axis=0)
    std = np.where(std == 0, 1.0, std)
    constants[f"internal_{name}_mean"] = mean
    constants[f"internal_{name}_std"] = std

  return all_features, labels, constants


def preprocess_data(
    records: Iterable[Any],
    label_name: str,
    feature_transforms: Callable[
        [str, Any], FeatureTransformFn] = _default_feature_transforms,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
  """Preprocesses records into features, normalized targets, and normalization constants."""
  all_features, target_scores, constants = _extract_features_and_constants(
      records, label_name, feature_transforms)

  target_mean = np.mean(target_scores, axis=0, keepdims=True)
  target_std = np.std(target_scores, axis=0, keepdims=True)
  target_std = np.where(target_std == 0, 1.0, target_std)

  normalized_targets = (target_scores - target_mean) / target_std
  return all_features, normalized_targets, constants


class DenseModel(nn.Module):
  """MLP scoring with baked-in normalization."""

  hidden_layers: Sequence[int]
  constants: dict[str, np.ndarray]
  feature_transforms: Callable[[str, Any], FeatureTransformFn] = (
      _default_feature_transforms)

  @nn.compact
  def __call__(self, **features):
    normalized_features = {}
    for k, v in features.items():
      transformed_feature = self.feature_transforms(k, jnp)(v)
      mean_key = f"internal_{k}_mean"
      std_key = f"internal_{k}_std"
      if mean_key in self.constants and std_key in self.constants:
        mean = self.constants[mean_key]
        std = self.constants[std_key]
        normalized_features[k] = (transformed_feature - mean) / std
      else:
        normalized_features[k] = transformed_feature

    x = jnp.concatenate(
        [normalized_features[k] for k in sorted(normalized_features.keys())],
        axis=-1,
    )
    for feat_size in self.hidden_layers:
      x = nn.Dense(features=feat_size)(x)
      x = jax.nn.relu(x)
    x = nn.Dense(features=1)(x)
    return x


class DenseClassifierModel(DenseModel):
  """MLP binary classifier with baked-in normalization and step decision."""

  threshold: float = 0.0
  export_binary_decision: bool = False
  feature_transforms: Callable[[str, Any], FeatureTransformFn] = (
      _default_feature_transforms)

  @nn.compact
  def __call__(self, **features):
    logits = super().__call__(**features)
    if self.export_binary_decision:
      return jnp.where(logits > self.threshold, 1.0, 0.0)
    return logits


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  model: nn.Module
  params: dict[str, Any]
  sample_inputs: dict[str, Any]

  @staticmethod
  def fromJax(
      model: nn.Module,
      params: dict[str, Any],
      sample_inputs: dict[str, Any],
  ) -> "ModelSpec":
    return ModelSpec(
        model=model,
        params=jax.tree_util.tree_map(np.array, params),
        sample_inputs=jax.tree_util.tree_map(np.array, sample_inputs),
    )

  def save_tflite(self, filename: str):
    edge_model = litert_adapter.convert(
        self.model.apply,
        self.params,
        jax_fn_kwargs=self.sample_inputs,
    )
    epath.Path(filename).write_bytes(edge_model.tflite_model())


def _train_model(
    model: nn.Module,
    all_features: dict[str, np.ndarray],
    targets: np.ndarray,
    loss_fn: Callable[[Any, dict[str, Any], Any], Any],
    steps: int,
    sample_inputs: dict[str, Any],
    batch_size: int = 128,
    learning_rate: float = 0.001,
    seed: int = 0,
    balanced_sampling: bool = False,
    pos_indices: np.ndarray | None = None,
    neg_indices: np.ndarray | None = None,
    loss_name: str = "Loss",
    log_prefix: str = "Training",
) -> dict[str, Any]:
  """Trains a Flax model on batch features and targets."""
  key = jax.random.key(seed)
  params = model.init(key, **sample_inputs)

  grad_fn = jax.value_and_grad(loss_fn)
  tx = optax.adam(learning_rate=learning_rate)
  opt_state = tx.init(params)

  @jax.jit
  def train_step(p, opt_s, feat, target):
    loss_val, grads = grad_fn(p, feat, target)
    updates, opt_s = tx.update(grads, opt_s, p)
    p = optax.apply_updates(p, updates)
    return p, opt_s, loss_val

  rng = np.random.default_rng(seed)
  num_examples = targets.shape[0]
  use_balanced = (
      balanced_sampling and pos_indices is not None and
      neg_indices is not None and len(pos_indices) > 0 and len(neg_indices) > 0)

  logging.info("%s for %d steps", log_prefix, steps)
  for i in range(steps):
    if use_balanced:
      pos_batch_size = batch_size // 2
      neg_batch_size = batch_size - pos_batch_size
      batch_pos = rng.choice(pos_indices, size=pos_batch_size, replace=True)
      batch_neg = rng.choice(neg_indices, size=neg_batch_size, replace=True)
      indices = np.concatenate([batch_pos, batch_neg])
      rng.shuffle(indices)
    else:
      batch_idx = (i * batch_size) % num_examples
      indices = np.arange(batch_idx, batch_idx + batch_size) % num_examples

    batch_feat = {k: v[indices] for k, v in all_features.items()}
    batch_target = targets[indices]

    params, opt_state, loss_val = train_step(params, opt_state, batch_feat,
                                             batch_target)
    if i % 100 == 0:
      logging.info("Step %d, %s: %f", i, loss_name, loss_val)

  return params


def behavioral_clone(
    records: Iterable[Any],
    steps: int,
    sample_inputs: dict[str, Any],
    hidden_layers: Sequence[int],
    label_name: str,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    feature_transforms: Callable[
        [str, Any], FeatureTransformFn] = _default_feature_transforms,
) -> ModelSpec:
  """Behavioral clones the model from feature records (regression with MSE)."""
  all_features, normalized_targets, constants = preprocess_data(
      records, label_name, feature_transforms)

  model = DenseModel(
      hidden_layers=hidden_layers,
      constants=constants,
      feature_transforms=feature_transforms,
  )

  def mse_loss(p, feat, target):
    predictions = model.apply(p, **feat)
    return jnp.mean(
        (jnp.squeeze(predictions, axis=-1) - jnp.squeeze(target, axis=-1))**2)

  params = _train_model(
      model=model,
      all_features=all_features,
      targets=normalized_targets,
      loss_fn=mse_loss,
      steps=steps,
      sample_inputs=sample_inputs,
      batch_size=batch_size,
      learning_rate=learning_rate,
      log_prefix="Training",
  )

  return ModelSpec.fromJax(
      model=model, params=params, sample_inputs=sample_inputs)


def behavioral_clone_classifier(
    records: Iterable[Any],
    steps: int,
    sample_inputs: dict[str, Any],
    hidden_layers: Sequence[int],
    label_name: str,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    pos_weight: float | None = None,
    threshold: float = 0.0,
    balanced_sampling: bool = False,
    seed: int = 0,
    feature_transforms: Callable[
        [str, Any], FeatureTransformFn] = _default_feature_transforms,
) -> ModelSpec:
  """Behavioral clones a binary decision classifier from feature records."""
  all_features, labels, constants = _extract_features_and_constants(
      records, label_name, feature_transforms)

  train_model = DenseClassifierModel(
      hidden_layers=hidden_layers,
      constants=constants,
      threshold=threshold,
      export_binary_decision=False,
      feature_transforms=feature_transforms,
  )

  labels = np.array(labels, dtype=np.float32)
  if labels.ndim == 1:
    labels = np.expand_dims(labels, axis=-1)

  flat_labels = labels.flatten()
  pos_indices = np.where(flat_labels > 0.5)[0]
  neg_indices = np.where(flat_labels <= 0.5)[0]
  logging.info(
      "Classifier BC: %d positive labels, %d negative labels",
      len(pos_indices),
      len(neg_indices),
  )
  if pos_weight is None and len(pos_indices) > 0 and len(neg_indices) > 0:
    pos_weight = float(len(neg_indices)) / float(len(pos_indices))
    logging.info("Auto-calculated pos_weight: %f", pos_weight)

  def bce_loss(p, feat, target):
    logits = train_model.apply(p, **feat)
    loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=target)
    if pos_weight is not None:
      w = jnp.where(target > 0.5, pos_weight, 1.0)
      loss = loss * w
    return jnp.mean(loss)

  params = _train_model(
      model=train_model,
      all_features=all_features,
      targets=labels,
      loss_fn=bce_loss,
      steps=steps,
      sample_inputs=sample_inputs,
      batch_size=batch_size,
      learning_rate=learning_rate,
      seed=seed,
      balanced_sampling=balanced_sampling,
      pos_indices=pos_indices,
      neg_indices=neg_indices,
      loss_name="BCE Loss",
      log_prefix="Training classifier",
  )

  inference_model = DenseClassifierModel(
      hidden_layers=hidden_layers,
      constants=constants,
      threshold=threshold,
      export_binary_decision=True,
      feature_transforms=feature_transforms,
  )
  return ModelSpec.fromJax(
      model=inference_model, params=params, sample_inputs=sample_inputs)
