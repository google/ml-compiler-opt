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
"""Open-source LiteRT conversion and execution adapter for JAX models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ai_edge_litert.interpreter import Interpreter
from etils import epath
from jax.experimental import jax2tf
import numpy as np
import tensorflow as tf


class TfLiteModel:
  """Lightweight wrapper around serialized LiteRT flatbuffer bytes."""

  def __init__(self, model_bytes: bytes):
    self._model_bytes = bytes(model_bytes)
    self._interpreter: Interpreter | None = None

  def tflite_model(self) -> bytes:
    return self._model_bytes

  def _get_interpreter(self) -> Interpreter:
    if self._interpreter is None:
      self._interpreter = Interpreter(model_content=self._model_bytes)
      self._interpreter.allocate_tensors()
    return self._interpreter

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    interp = self._get_interpreter()
    if kwargs:
      runner = interp.get_signature_runner("serving_default")
      np_kwargs = {k: np.asarray(v, dtype=np.float32) for k, v in kwargs.items()}
      outputs = runner(**np_kwargs)
      return outputs

    input_details = interp.get_input_details()
    if len(args) != len(input_details):
      raise ValueError(
          f"Expected {len(input_details)} positional inputs, got {len(args)}")
    for detail, arg in zip(input_details, args):
      interp.set_tensor(detail["index"], np.asarray(arg, dtype=detail["dtype"]))
    interp.invoke()
    output_details = interp.get_output_details()
    outputs = [interp.get_tensor(detail["index"]) for detail in output_details]
    return outputs[0] if len(outputs) == 1 else outputs


class Model:
  """Loader matching the internal LiteRT model interface."""

  @staticmethod
  def load(path: str | epath.Path) -> TfLiteModel:
    return TfLiteModel(epath.Path(path).read_bytes())


def get_signature_list(edge_model: TfLiteModel) -> dict[str, Any]:
  """Returns the signature list dictionary for a loaded LiteRT model."""
  return edge_model._get_interpreter().get_signature_list()  # pylint: disable=protected-access


def convert(
    jax_fn: Callable[..., Any],
    params: Any,
    jax_fn_args: Sequence[Any] = (),
    jax_fn_kwargs: Mapping[str, Any] | None = None,
) -> TfLiteModel:
  """Converts a JAX callable and params into a TfLiteModel."""
  kwargs_dict = dict(jax_fn_kwargs) if jax_fn_kwargs is not None else {}

  if kwargs_dict and jax_fn_args:
    raise ValueError("Specify either jax_fn_args or jax_fn_kwargs, not both.")

  if kwargs_dict:
    kw_keys = sorted(kwargs_dict.keys())
    specs = [
        tf.TensorSpec(
            shape=np.asarray(kwargs_dict[k]).shape,
            dtype=tf.dtypes.as_dtype(np.asarray(kwargs_dict[k]).dtype),
            name=k,
        ) for k in kw_keys
    ]

    def _fn_pos(*pos_args):
      bound = dict(zip(kw_keys, pos_args))
      return jax_fn(params, **bound)

    tf_fn = jax2tf.convert(_fn_pos, with_gradient=False)

    class _Module(tf.Module):

      @tf.function(input_signature=specs)
      def serving_default(self, *pos_args):
        return {"output_0": tf_fn(*pos_args)}

    mod = _Module()
    concrete_fn = mod.serving_default.get_concrete_function(*specs)
  else:
    args_list = [np.asarray(a) for a in jax_fn_args]
    specs = [
        tf.TensorSpec(
            shape=a.shape,
            dtype=tf.dtypes.as_dtype(a.dtype),
            name=f"arg_{i}",
        ) for i, a in enumerate(args_list)
    ]

    def _fn_args(*pos_args):
      return jax_fn(params, *pos_args)

    tf_fn = jax2tf.convert(_fn_args, with_gradient=False)

    class _Module(tf.Module):

      @tf.function(input_signature=specs)
      def serving_default(self, *pos_args):
        return tf_fn(*pos_args)

    mod = _Module()
    concrete_fn = mod.serving_default.get_concrete_function(*specs)

  converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn],
                                                              mod)
  tflite_bytes = converter.convert()
  return TfLiteModel(tflite_bytes)
