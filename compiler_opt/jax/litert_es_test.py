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
"""Tests for litert_es."""

import os

from absl.testing import absltest
from compiler_opt.jax import litert_adapter
from compiler_opt.jax import litert_es
import jax.numpy as jnp
import numpy as np


class LiteRtEsTest(absltest.TestCase):

  def test_flow(self):

    def model_fn(params, x):
      return jnp.dot(x, params["w"]) + params["b"]

    params = {
        "w": jnp.array([[2.0], [3.0]], dtype=jnp.float32),
        "b": jnp.array([1.0], dtype=jnp.float32),
    }
    x = jnp.array([[1.0, 2.0]], dtype=jnp.float32)

    edge_model = litert_adapter.convert(
        jax_fn=model_fn, params=params, jax_fn_args=(x,))

    weights = litert_es.get_weights(edge_model)
    self.assertEqual(weights.size, 3)

    orig_output = edge_model(x)
    if isinstance(orig_output, list):
      orig_output = orig_output[0]
    np.testing.assert_allclose(orig_output, np.array([[9.0]]), rtol=1e-5)

    doubled_weights = weights * 2

    new_model_path = os.path.join(self.create_tempdir().full_path,
                                  "new_model.tflite")
    litert_es.save_with_weights(edge_model, doubled_weights, new_model_path)

    new_model = litert_adapter.Model.load(new_model_path)

    new_output = new_model(x)
    if isinstance(new_output, list):
      new_output = new_output[0]
    np.testing.assert_allclose(new_output, np.array([[18.0]]), rtol=1e-5)

  def test_curry_constant_inputs(self):

    def model_fn(params, **kwargs):
      return jnp.dot(kwargs["x"], params["w"]) + params["b"] + kwargs["c"]

    params = {
        "w": jnp.array([[2.0], [3.0]], dtype=jnp.float32),
        "b": jnp.array([1.0], dtype=jnp.float32),
    }
    x = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    c = jnp.array([[5.0]], dtype=jnp.float32)

    edge_model = litert_adapter.convert(
        jax_fn=model_fn, params=params, jax_fn_kwargs={
            "x": x,
            "c": c
        })

    curried_model = litert_es.curry_constant_inputs(
        edge_model, {"c": np.array([[5.0]], dtype=np.float32)})

    sig_list = litert_adapter.get_signature_list(curried_model)
    self.assertIn("serving_default", sig_list)
    self.assertEqual(list(sig_list["serving_default"]["inputs"]), ["x"])

    out = curried_model(x=x)
    if isinstance(out, dict):
      out = list(out.values())[0]
    elif isinstance(out, list):
      out = out[0]
    np.testing.assert_allclose(out, np.array([[14.0]]), rtol=1e-5)

    weights = litert_es.get_weights(curried_model)
    self.assertEqual(weights.size, 3)

    doubled_weights = weights * 2
    new_model_path = os.path.join(self.create_tempdir().full_path,
                                  "curried_new_model.tflite")
    litert_es.save_with_weights(curried_model, doubled_weights, new_model_path)

    new_model = litert_adapter.Model.load(new_model_path)
    new_out = new_model(x=x)
    if isinstance(new_out, dict):
      new_out = list(new_out.values())[0]
    elif isinstance(new_out, list):
      new_out = new_out[0]
    np.testing.assert_allclose(new_out, np.array([[23.0]]), rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
