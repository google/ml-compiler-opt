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
"""Tests for dense_nn_bc."""

from absl.testing import absltest
from compiler_opt.jax import dense_nn_bc
from compiler_opt.jax import litert_adapter
import jax
import jax.numpy as jnp
import numpy as np


class DummyRecord:

  def __init__(self, features):
    self.features = features


class DummyVal:

  def __init__(self, value):
    self.value = value


class DenseNnBcTest(absltest.TestCase):

  def test_behavioral_clone(self):
    rng = np.random.default_rng(0)
    records = []
    for _ in range(10):
      features = {
          "feat1": DummyVal(rng.random(4)),
          "feat2": DummyVal(rng.random(2)),
          "score": DummyVal(rng.random(1)),
      }
      records.append(DummyRecord(features))

    sample_inputs = {
        "feat1": jnp.zeros(4),
        "feat2": jnp.zeros(2),
    }

    spec = dense_nn_bc.behavioral_clone(
        records=records,
        steps=2,
        sample_inputs=sample_inputs,
        hidden_layers=[8, 4],
        label_name="score",
    )

    self.assertIsNotNone(spec.model)
    self.assertIsNotNone(spec.params)

    edge_model = litert_adapter.convert(
        spec.model.apply, spec.params, jax_fn_kwargs=spec.sample_inputs)
    self.assertIsNotNone(edge_model)

  def test_from_jax(self):
    constants = {
        "internal_feat1_mean": np.array([0.0]),
        "internal_feat1_std": np.array([1.0]),
    }
    model = dense_nn_bc.DenseModel(hidden_layers=[4], constants=constants)
    sample_inputs = {"feat1": jnp.zeros((1, 1))}
    key = jax.random.key(0)
    params = model.init(key, **sample_inputs)

    spec = dense_nn_bc.ModelSpec.fromJax(
        model=model, params=params, sample_inputs=sample_inputs)

    self.assertIsInstance(spec.params["params"]["Dense_0"]["kernel"],
                          np.ndarray)
    self.assertIsInstance(spec.sample_inputs["feat1"], np.ndarray)

  def test_save(self):
    constants = {
        "internal_feat1_mean": np.array([0.0]),
        "internal_feat1_std": np.array([1.0]),
    }
    model = dense_nn_bc.DenseModel(hidden_layers=[4], constants=constants)
    sample_inputs = {"feat1": jnp.zeros((1, 1))}
    key = jax.random.key(0)
    params = model.init(key, **sample_inputs)

    spec = dense_nn_bc.ModelSpec.fromJax(
        model=model, params=params, sample_inputs=sample_inputs)
    temp_file = self.create_tempfile()
    spec.save_tflite(temp_file.full_path)
    self.assertGreater(len(temp_file.read_bytes()), 0)

  def test_preprocess_data_normal(self):
    records = [
        DummyRecord({
            "feat1": DummyVal(np.array([1.0, 2.0])),
            "score": DummyVal(np.array([10.0])),
        }),
        DummyRecord({
            "feat1": DummyVal(np.array([3.0, 4.0])),
            "score": DummyVal(np.array([20.0])),
        }),
    ]
    features, normalized_targets, constants = dense_nn_bc.preprocess_data(
        records, "score")

    self.assertIn("feat1", features)
    np.testing.assert_array_almost_equal(features["feat1"],
                                         np.array([[1.0, 2.0], [3.0, 4.0]]))
    self.assertNotIn("score", features)

    np.testing.assert_array_almost_equal(normalized_targets,
                                         np.array([[-1.0], [1.0]]))

    np.testing.assert_array_almost_equal(constants["internal_feat1_mean"],
                                         np.array([2.0, 3.0]))
    np.testing.assert_array_almost_equal(constants["internal_feat1_std"],
                                         np.array([1.0, 1.0]))

  def test_preprocess_data_zero_variance_target(self):
    records = [
        DummyRecord({
            "feat1": DummyVal(np.array([1.0, 2.0])),
            "score": DummyVal(np.array([10.0])),
        }),
        DummyRecord({
            "feat1": DummyVal(np.array([3.0, 4.0])),
            "score": DummyVal(np.array([10.0])),
        }),
    ]
    _, normalized_targets, _ = dense_nn_bc.preprocess_data(records, "score")

    np.testing.assert_array_almost_equal(normalized_targets,
                                         np.array([[0.0], [0.0]]))

  def test_preprocess_data_zero_variance_feature(self):
    records = [
        DummyRecord({
            "feat1": DummyVal(np.array([1.0, 2.0])),
            "score": DummyVal(np.array([10.0])),
        }),
        DummyRecord({
            "feat1": DummyVal(np.array([1.0, 4.0])),
            "score": DummyVal(np.array([20.0])),
        }),
    ]
    _, _, constants = dense_nn_bc.preprocess_data(records, "score")

    np.testing.assert_array_almost_equal(constants["internal_feat1_mean"],
                                         np.array([1.0, 3.0]))
    np.testing.assert_array_almost_equal(constants["internal_feat1_std"],
                                         np.array([1.0, 1.0]))

  def test_behavioral_clone_classifier(self):
    rng = np.random.default_rng(0)
    records = []
    for i in range(10):
      label = 1.0 if i < 2 else 0.0
      features = {
          "feat1": DummyVal(rng.random(4)),
          "feat2": DummyVal(rng.random(2)),
          "decision": DummyVal(np.array([label])),
      }
      records.append(DummyRecord(features))

    sample_inputs = {
        "feat1": jnp.zeros(4),
        "feat2": jnp.zeros(2),
    }

    spec = dense_nn_bc.behavioral_clone_classifier(
        records=records,
        steps=5,
        sample_inputs=sample_inputs,
        hidden_layers=[8, 4],
        label_name="decision",
        pos_weight=4.0,
        balanced_sampling=True,
    )

    self.assertIsNotNone(spec.model)
    self.assertIsNotNone(spec.params)

    output = spec.model.apply(spec.params, **sample_inputs)
    self.assertIn(float(output[0]), (0.0, 1.0))

    temp_file = self.create_tempfile()
    spec.save_tflite(temp_file.full_path)
    self.assertGreater(len(temp_file.read_bytes()), 0)

  def test_behavioral_clone_classifier_auto_pos_weight(self):
    rng = np.random.default_rng(0)
    records = []
    for i in range(10):
      label = 1.0 if i < 2 else 0.0
      features = {
          "feat1": DummyVal(rng.random(4)),
          "feat2": DummyVal(rng.random(2)),
          "decision": DummyVal(np.array([label])),
      }
      records.append(DummyRecord(features))

    sample_inputs = {
        "feat1": jnp.zeros(4),
        "feat2": jnp.zeros(2),
    }

    spec = dense_nn_bc.behavioral_clone_classifier(
        records=records,
        steps=5,
        sample_inputs=sample_inputs,
        hidden_layers=[8, 4],
        label_name="decision",
        pos_weight=None,
    )

    self.assertIsNotNone(spec.model)
    self.assertIsNotNone(spec.params)
    output = spec.model.apply(spec.params, **sample_inputs)
    self.assertIn(float(output[0]), (0.0, 1.0))

  def test_extract_features_and_constants(self):
    records = [
        DummyRecord({
            "feat1": DummyVal(np.array([1.0, 2.0])),
            "decision": DummyVal(np.array([1.0])),
        }),
        DummyRecord({
            "feat1": DummyVal(np.array([3.0, 4.0])),
            "decision": DummyVal(np.array([0.0])),
        }),
    ]
    features, labels, constants = dense_nn_bc._extract_features_and_constants(
        records, "decision")

    self.assertIn("feat1", features)
    self.assertNotIn("decision", features)
    np.testing.assert_array_equal(labels, np.array([[1.0], [0.0]]))
    np.testing.assert_array_almost_equal(constants["internal_feat1_mean"],
                                         np.array([2.0, 3.0]))
    np.testing.assert_array_almost_equal(constants["internal_feat1_std"],
                                         np.array([1.0, 1.0]))

  def test_preprocess_data_custom_feature_transforms(self):

    def custom_feature_transforms(name, mod):
      if name == "feat1":
        return lambda x: mod.exp(x)
      return lambda x: x

    records = [
        DummyRecord({
            "feat1": DummyVal(np.array([0.0, 0.0])),
            "score": DummyVal(np.array([10.0])),
        }),
        DummyRecord({
            "feat1": DummyVal(np.array([np.log(3.0), np.log(5.0)])),
            "score": DummyVal(np.array([20.0])),
        }),
    ]
    _, _, constants = dense_nn_bc.preprocess_data(
        records, "score", feature_transforms=custom_feature_transforms)

    np.testing.assert_array_almost_equal(constants["internal_feat1_mean"],
                                         np.array([2.0, 3.0]))
    np.testing.assert_array_almost_equal(constants["internal_feat1_std"],
                                         np.array([1.0, 2.0]))

  def test_behavioral_clone_custom_feature_transforms(self):

    def custom_feature_transforms(name, mod):
      if name == "feat1":
        return lambda x: mod.exp(x)
      return lambda x: x

    records = [
        DummyRecord({
            "feat1": DummyVal(np.array([0.0, 0.0])),
            "score": DummyVal(np.array([10.0])),
        }),
        DummyRecord({
            "feat1": DummyVal(np.array([np.log(3.0), np.log(5.0)])),
            "score": DummyVal(np.array([20.0])),
        }),
    ]
    sample_inputs = {
        "feat1": jnp.zeros((1, 2)),
    }

    spec = dense_nn_bc.behavioral_clone(
        records=records,
        steps=2,
        sample_inputs=sample_inputs,
        hidden_layers=[4],
        label_name="score",
        feature_transforms=custom_feature_transforms,
    )

    self.assertIsNotNone(spec.model)
    self.assertIsNotNone(spec.params)
    self.assertEqual(spec.model.feature_transforms, custom_feature_transforms)

    output = spec.model.apply(spec.params, **sample_inputs)
    self.assertEqual(output.shape, (1, 1))

    temp_file = self.create_tempfile()
    spec.save_tflite(temp_file.full_path)
    self.assertGreater(len(temp_file.read_bytes()), 0)

  def test_behavioral_clone_classifier_custom_feature_transforms(self):

    def custom_feature_transforms(name, mod):
      if name == "feat1":
        return lambda x: mod.exp(x)
      return lambda x: x

    records = [
        DummyRecord({
            "feat1": DummyVal(np.array([0.0, 0.0])),
            "decision": DummyVal(np.array([1.0])),
        }),
        DummyRecord({
            "feat1": DummyVal(np.array([np.log(3.0), np.log(5.0)])),
            "decision": DummyVal(np.array([0.0])),
        }),
    ]
    sample_inputs = {
        "feat1": jnp.zeros((1, 2)),
    }

    spec = dense_nn_bc.behavioral_clone_classifier(
        records=records,
        steps=2,
        sample_inputs=sample_inputs,
        hidden_layers=[4],
        label_name="decision",
        feature_transforms=custom_feature_transforms,
    )

    self.assertIsNotNone(spec.model)
    self.assertIsNotNone(spec.params)
    self.assertEqual(spec.model.feature_transforms, custom_feature_transforms)

    output = spec.model.apply(spec.params, **sample_inputs)
    self.assertIn(float(output[0, 0]), (0.0, 1.0))

    temp_file = self.create_tempfile()
    spec.save_tflite(temp_file.full_path)
    self.assertGreater(len(temp_file.read_bytes()), 0)


if __name__ == "__main__":
  absltest.main()
