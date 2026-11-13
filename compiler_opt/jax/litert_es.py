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
"""Library for LiteRT evolutionary strategy APIs."""

from collections.abc import Generator
from typing import Any

from ai_edge_litert import schema_py_generated as model_schema
from ai_edge_litert.tools import flatbuffer_utils
from compiler_opt.jax import litert_adapter
from etils import epath
import numpy as np
import numpy.typing as npt

_CURRIED_CONSTANT_BUFFERS_META_KEY = "curried_constant_buffers"


def _tensor_type_to_np_dtype(tensor_type: int) -> npt.DTypeLike:
  """Maps TFLite tensor type enum to numpy dtype."""
  if tensor_type == model_schema.TensorType.FLOAT32:
    return np.float32
  if tensor_type == model_schema.TensorType.INT32:
    return np.int32
  if tensor_type == model_schema.TensorType.INT8:
    return np.int8
  if tensor_type == model_schema.TensorType.UINT8:
    return np.uint8
  if tensor_type == model_schema.TensorType.FLOAT16:
    return np.float16
  if tensor_type == model_schema.TensorType.INT64:
    return np.int64
  if tensor_type == model_schema.TensorType.BOOL:
    return np.bool_
  raise ValueError(f"Unsupported tensor type {tensor_type}")


def _get_indices_of_ignored_buffers(model_t: model_schema.ModelT) -> set[int]:
  """Returns the set of buffer indices that represent curried constants."""
  if not model_t.metadata:
    return set()
  for meta in model_t.metadata:
    meta_name = (
        meta.name.decode("utf-8")
        if isinstance(meta.name, bytes) else meta.name)
    if meta_name == _CURRIED_CONSTANT_BUFFERS_META_KEY:
      buf_idx = meta.buffer
      if buf_idx < 0 or buf_idx >= len(model_t.buffers):
        raise ValueError(f"Invalid metadata buffer index {buf_idx}")
      buf = model_t.buffers[buf_idx]
      if buf.data is None or len(buf.data) == 0:
        raise ValueError(
            f"Metadata buffer {buf_idx} for {_CURRIED_CONSTANT_BUFFERS_META_KEY}"
            " is empty")
      return set(np.frombuffer(buf.data, dtype=np.int32).tolist())
  return set()


def _get_weight_tensors(
    model_t: model_schema.ModelT,
) -> Generator[
    tuple[model_schema.TensorT, model_schema.BufferT, type[Any], np.ndarray],
    None,
    None,
]:
  """Yields weight tensors, buffers, dtypes, and numpy arrays."""
  ignored_buffers = _get_indices_of_ignored_buffers(model_t)
  for subgraph in model_t.subgraphs:
    for tensor in subgraph.tensors:
      if tensor.buffer == 0:
        # the input
        continue
      if tensor.buffer in ignored_buffers:
        continue
      if tensor.buffer >= len(model_t.buffers):
        raise ValueError(f"Invalid buffer index {tensor.buffer}")
      buffer = model_t.buffers[tensor.buffer]
      if buffer.data is not None and len(buffer.data) > 0:
        dtype = _tensor_type_to_np_dtype(tensor.type)
        arr = np.frombuffer(buffer.data, dtype=dtype)
        yield tensor, buffer, dtype, arr


def get_weights(model: litert_adapter.TfLiteModel) -> npt.NDArray[np.float32]:
  """Extracts weights from a loaded LiteRT model into a numpy array."""
  model_bytes = model.tflite_model()

  model_t = flatbuffer_utils.convert_bytearray_to_object(model_bytes)

  weight_buffers = []
  for _, _, _, arr in _get_weight_tensors(model_t):
    weight_buffers.append(arr.flatten())

  return (np.concatenate(weight_buffers)
          if weight_buffers else np.array([], dtype=np.float32))


def save_with_weights(
    model: litert_adapter.TfLiteModel,
    weights: npt.NDArray[np.float32],
    path: str,
) -> None:
  """Saves a LiteRT model with updated weights to a file."""
  model_bytes = model.tflite_model()

  model_t = flatbuffer_utils.convert_bytearray_to_object(model_bytes)

  weight_tensors = []
  total_size = 0
  for tensor, buffer, dtype, arr in _get_weight_tensors(model_t):
    weight_tensors.append((tensor, buffer, dtype, arr.size))
    total_size += arr.size

  if weights.size != total_size:
    raise ValueError(
        f"Weight size mismatch. Expected {total_size}, got {weights.size}")

  offset = 0
  for _, buffer, dtype, size in weight_tensors:
    chunk = weights[offset:offset + size]
    offset += size
    new_data = chunk.astype(dtype).tobytes()
    buffer.data = bytearray(new_data)

  new_bytes = flatbuffer_utils.convert_object_to_bytearray(model_t)

  epath.Path(path).write_bytes(bytes(new_bytes))


def curry_constant_inputs(
    model: litert_adapter.TfLiteModel,
    constants: dict[str, np.ndarray],
) -> litert_adapter.TfLiteModel:
  """Replaces named inputs with constant buffers and tags them with metadata."""
  model_bytes = model.tflite_model()
  model_t = flatbuffer_utils.convert_bytearray_to_object(model_bytes)

  updated_tensors = set()
  new_constant_buffers = []
  matched_constants = set()

  for signature in model_t.signatureDefs:
    subgraph_idx = signature.subgraphIndex
    subgraph = model_t.subgraphs[subgraph_idx]

    new_sig_inputs = []
    inputs_to_remove_from_subgraph = set()

    for tensor_map in signature.inputs:
      name = (
          tensor_map.name.decode("utf-8")
          if isinstance(tensor_map.name, bytes) else tensor_map.name)
      if name in constants:
        matched_constants.add(name)
        tensor_idx = tensor_map.tensorIndex
        tensor = subgraph.tensors[tensor_idx]

        if tensor_idx not in updated_tensors:
          val = constants[name]
          dtype = _tensor_type_to_np_dtype(tensor.type)
          val = np.asarray(val, dtype=dtype)

          new_buffer = model_schema.BufferT()
          new_buffer.data = bytearray(val.tobytes())
          model_t.buffers.append(new_buffer)
          new_buf_idx = len(model_t.buffers) - 1
          tensor.buffer = new_buf_idx

          new_constant_buffers.append(new_buf_idx)
          updated_tensors.add(tensor_idx)

        inputs_to_remove_from_subgraph.add(tensor_idx)
      else:
        new_sig_inputs.append(tensor_map)

    signature.inputs = new_sig_inputs

    new_subgraph_inputs = [
        idx for idx in subgraph.inputs
        if idx not in inputs_to_remove_from_subgraph
    ]
    subgraph.inputs = new_subgraph_inputs

  missing = set(constants.keys()) - matched_constants
  if missing:
    raise ValueError(
        f"Constant inputs not found in model signatures: {sorted(missing)}")

  if new_constant_buffers:
    meta_buf_data = np.array(new_constant_buffers, dtype=np.int32).tobytes()
    meta_buffer = model_schema.BufferT()
    meta_buffer.data = bytearray(meta_buf_data)
    model_t.buffers.append(meta_buffer)
    meta_buf_idx = len(model_t.buffers) - 1

    if model_t.metadata is None:
      model_t.metadata = []

    meta_entry = model_schema.MetadataT()
    meta_entry.name = _CURRIED_CONSTANT_BUFFERS_META_KEY
    meta_entry.buffer = meta_buf_idx
    model_t.metadata.append(meta_entry)

  new_bytes = bytes(flatbuffer_utils.convert_object_to_bytearray(model_t))
  return litert_adapter.TfLiteModel(new_bytes)
