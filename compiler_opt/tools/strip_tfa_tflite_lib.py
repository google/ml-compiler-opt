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
"""Strips tf_agents-introduced functions from a TFLite model."""

import os

import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools import flatbuffer_utils

ACTION_SIGNATURE_KEY = b'action'
TFA_EXTRA_FUNCTIONS = frozenset({
    b'get_train_step',
    b'get_initial_state',
    b'get_metadata',
})
_SUBGRAPH_INDEX_FIELDS = (
    'initSubgraphIndex',
    'subgraph',
    'thenSubgraphIndex',
    'elseSubgraphIndex',
    'condSubgraphIndex',
    'bodySubgraphIndex',
    'updateComputationSubgraphIndex',
    'comparatorSubgraphIndex',
)


def _extract_action_signature_and_subgraph(
    model: schema_fb.ModelT,
) -> tuple[schema_fb.SignatureDefT, schema_fb.SubGraphT]:
  """Validates and extracts the single action signature and subgraph."""
  if not model.signatureDefs:
    raise ValueError('TFLite model has no signatureDefs.')

  unexpected_sigs = [
      s.signatureKey
      for s in model.signatureDefs
      if s.signatureKey != ACTION_SIGNATURE_KEY and
      s.signatureKey not in TFA_EXTRA_FUNCTIONS
  ]
  if unexpected_sigs:
    raise ValueError(
        f'Unexpected signature keys in TFLite model: {unexpected_sigs}')

  action_sigs = [
      s for s in model.signatureDefs if s.signatureKey == ACTION_SIGNATURE_KEY
  ]
  if len(action_sigs) != 1:
    raise ValueError(f'Expected 1 {ACTION_SIGNATURE_KEY!r} signature, found '
                     f'{len(action_sigs)}.')
  action_sig = action_sigs[0]

  if not model.subgraphs:
    raise ValueError('TFLite model has no subgraphs.')
  if not 0 <= action_sig.subgraphIndex < len(model.subgraphs):
    raise ValueError(
        f'Invalid action subgraphIndex {action_sig.subgraphIndex} for '
        f'{len(model.subgraphs)} subgraphs.')

  action_sg = model.subgraphs[action_sig.subgraphIndex]
  if action_sg.name != ACTION_SIGNATURE_KEY:
    raise ValueError(
        f'Expected action subgraph name {ACTION_SIGNATURE_KEY!r}, got '
        f'{action_sg.name!r}.')

  for idx, sg in enumerate(model.subgraphs):
    if idx != action_sig.subgraphIndex and sg.name not in TFA_EXTRA_FUNCTIONS:
      raise ValueError(f'Unexpected subgraph {sg.name!r} at index {idx}.')

  for op_idx, op in enumerate(action_sg.operators or []):
    for opts in (op.builtinOptions, op.builtinOptions2):
      if opts is None:
        continue
      for field in _SUBGRAPH_INDEX_FIELDS:
        if hasattr(opts, field):
          raise ValueError(
              f'Operator {op_idx} in action subgraph references a secondary '
              f'subgraph via {type(opts).__name__}.{field}.')

  return action_sig, action_sg


def _prune_unused_buffers(model: schema_fb.ModelT,
                          action_sg: schema_fb.SubGraphT) -> None:
  """Removes unreferenced buffers and re-indexes remaining buffer references."""
  if not model.buffers:
    raise ValueError('TFLite model has no buffers.')
  num_buffers = len(model.buffers)
  used_buffers: set[int] = {0}
  for t in action_sg.tensors or []:
    if not 0 <= t.buffer < num_buffers:
      raise ValueError(
          f'Tensor {t.name!r} has out-of-range buffer index {t.buffer}.')
    used_buffers.add(t.buffer)
  for md in model.metadata or []:
    if not 0 <= md.buffer < num_buffers:
      raise ValueError(
          f'Metadata {md.name!r} has out-of-range buffer index {md.buffer}.')
    used_buffers.add(md.buffer)
  for b_idx in model.metadataBuffer or []:
    if not 0 <= b_idx < num_buffers:
      raise ValueError(f'metadataBuffer has out-of-range buffer index {b_idx}.')
    used_buffers.add(b_idx)

  sorted_buffers = sorted(used_buffers)
  buffer_map = {
      old_idx: new_idx for new_idx, old_idx in enumerate(sorted_buffers)
  }
  model.buffers = [model.buffers[i] for i in sorted_buffers]
  for t in action_sg.tensors or []:
    t.buffer = buffer_map[t.buffer]
  for md in model.metadata or []:
    md.buffer = buffer_map[md.buffer]
  if model.metadataBuffer is not None:
    model.metadataBuffer = [buffer_map[i] for i in model.metadataBuffer]


def _prune_unused_operator_codes(model: schema_fb.ModelT,
                                 action_sg: schema_fb.SubGraphT) -> None:
  """Removes unreferenced operatorCodes and re-indexes operators."""
  num_opcodes = len(model.operatorCodes or [])
  used_opcodes: set[int] = set()
  for op_idx, op in enumerate(action_sg.operators or []):
    if not 0 <= op.opcodeIndex < num_opcodes:
      raise ValueError(
          f'Operator {op_idx} has out-of-range opcodeIndex {op.opcodeIndex}.')
    used_opcodes.add(op.opcodeIndex)

  sorted_opcodes = sorted(used_opcodes)
  opcode_map = {
      old_idx: new_idx for new_idx, old_idx in enumerate(sorted_opcodes)
  }
  if model.operatorCodes is not None:
    model.operatorCodes = [model.operatorCodes[i] for i in sorted_opcodes]
  for op in action_sg.operators or []:
    op.opcodeIndex = opcode_map[op.opcodeIndex]


def strip_tfa_functions_from_object(
    model: schema_fb.ModelT,) -> schema_fb.ModelT:
  """Strips tf_agents helper functions from a TFLite ModelT object in place.

  Retains only the `action` subgraph and signatureDef (re-indexed to 0), and
  garbage-collects unreferenced buffers and operatorCodes.

  Args:
    model: A `schema_fb.ModelT` unpacked TFLite model.

  Returns:
    The mutated `schema_fb.ModelT` containing only the `action` function.

  Raises:
    ValueError: If required fields are missing, unexpected subgraphs/signatures
      are present, or any index invariant is violated.
  """
  action_sig, action_sg = _extract_action_signature_and_subgraph(model)
  action_sig.subgraphIndex = 0
  model.signatureDefs = [action_sig]
  model.subgraphs = [action_sg]
  _prune_unused_buffers(model, action_sg)
  _prune_unused_operator_codes(model, action_sg)
  return model


def strip_tfa_functions(model_bytes: bytes | bytearray) -> bytes:
  """Strips tf_agents helper functions from serialized TFLite bytes."""
  if not model_bytes:
    raise ValueError('Input TFLite model bytes are empty.')
  model = flatbuffer_utils.convert_bytearray_to_object(bytearray(model_bytes))
  strip_tfa_functions_from_object(model)
  output_bytes = flatbuffer_utils.convert_object_to_bytearray(model)
  if not output_bytes:
    raise RuntimeError('Serialized stripped TFLite model is empty.')
  return output_bytes


def strip_tfa_tflite_file(input_path: str, output_path: str) -> None:
  """Reads a TFLite file, strips tf_agents functions, and writes the result."""
  if not tf.io.gfile.exists(input_path):
    raise FileNotFoundError(f'Input TFLite file not found: {input_path}')
  if tf.io.gfile.stat(input_path).length == 0:
    raise ValueError(f'Input TFLite file is empty: {input_path}')

  model = flatbuffer_utils.read_model(input_path)
  strip_tfa_functions_from_object(model)

  output_dir = os.path.dirname(output_path)
  if output_dir:
    tf.io.gfile.makedirs(output_dir)
  flatbuffer_utils.write_model(model, output_path)

  if (not tf.io.gfile.exists(output_path) or
      tf.io.gfile.stat(output_path).length == 0):
    raise RuntimeError(
        f'Failed to write non-empty output TFLite file: {output_path}')
