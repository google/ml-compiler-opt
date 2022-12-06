# coding=utf-8
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
"""Reader for the simple log format (non-protobuf).

Refer to (in llvm repo) to llvm/lib/Analysis/models/log_reader.py
which is used there for testing.
"""

import ctypes
import dataclasses
import json
import math

from typing import Any, BinaryIO, Dict, Generator, List, Optional

_element_types = {
    'float': ctypes.c_float,
    'double': ctypes.c_double,
    'int8_t': ctypes.c_int8,
    'uint8_t': ctypes.c_uint8,
    'int16_t': ctypes.c_int16,
    'uint16_t': ctypes.c_uint16,
    'int32_t': ctypes.c_int32,
    'uint32_t': ctypes.c_uint32,
    'int64_t': ctypes.c_int64,
    'uint64_t': ctypes.c_uint64
}


@dataclasses.dataclass(frozen=True)
class TensorSpec:
  name: str
  port: int
  shape: List[int]
  element_type: type


def create_tensorspec(d: Dict[str, Any]) -> TensorSpec:
  name: str = d['name']
  port: int = int(d['port'])
  shape: List[int] = [int(e) for e in d['shape']]
  element_type_str: str = d['type']
  if element_type_str not in _element_types:
    raise ValueError(f'uknown type: {element_type_str}')
  return TensorSpec(
      name=name,
      port=port,
      shape=shape,
      element_type=_element_types[element_type_str])


class TensorValue:
  """The value of a tensor of a given spec.

  We root the bytes buffer which provide the underlying data, and index in
  the value based on the type of the tensor, thus the TensorValue can be used
  as a list-like object containing the scalar values, in row-major order, of
  the tensor.

  Endianness is assumed to be the same as the log producer's.
  """

  def __init__(self, spec: TensorSpec, buffer: bytes):
    self._spec = spec
    self._buffer = buffer
    # c_char_p is a nul-terminated string, so the more appropriate cast here
    # would be POINTER(c_char), but unfortunately, c_char_p is the only
    # type that can be constructed from a `bytes`. To capture our intent,
    # we cast the c_char_p to
    buffer_as_nul_ending_ptr = ctypes.c_char_p(self._buffer)
    buffer_as_naked_ptr = ctypes.cast(buffer_as_nul_ending_ptr,
                                      ctypes.POINTER(ctypes.c_char))
    self._view = ctypes.cast(buffer_as_naked_ptr,
                             ctypes.POINTER(self._spec.element_type))
    self._len = math.prod(self._spec.shape)

  def spec(self) -> TensorSpec:
    return self._spec

  def __len__(self) -> int:
    return self._len

  def __getitem__(self, index: int):
    if index < 0 or index >= self._len:
      raise IndexError(f'Index {index} out of range [0..{self._len})')
    # pytype believes `index` is an object, despite all the annotations to the
    # contrary.
    return self._view[index]  # pytype:disable=wrong-arg-types


@dataclasses.dataclass(frozen=True)
class Header:
  features: List[TensorSpec]
  score: Optional[TensorSpec]


def read_tensor(fs: BinaryIO, ts: TensorSpec) -> TensorValue:
  size = math.prod(ts.shape) * ctypes.sizeof(ts.element_type)
  data = fs.read(size)
  return TensorValue(ts, data)


def read_header(f: BinaryIO) -> Header:
  header = json.loads(f.readline())
  tensor_specs = [create_tensorspec(ts) for ts in header['features']]
  score_spec = create_tensorspec(header['score']) if 'score' in header else None
  return Header(features=tensor_specs, score=score_spec)


def pretty_print_tensor_value(tv: TensorValue):
  print(f'{tv.spec().name}: {",".join([str(v) for v in tv])}')


@dataclasses.dataclass(frozen=True)
class Record:
  context: str
  observation_id: int
  feature_values: List[TensorValue]
  score: Optional[TensorValue]


def enumerate_log_from_stream(f: BinaryIO,
                              header: Header) -> Generator[Record, None, None]:
  """A generator that returns Record objects from a log file.

  It is assumed the log file's header was read separately.
  """
  tensor_specs = header.features
  score_spec = header.score
  context = None
  while event_str := f.readline():
    event = json.loads(event_str)
    if 'context' in event:
      context = event['context']
      continue
    observation_id = int(event['observation'])
    features = []
    for ts in tensor_specs:
      features.append(read_tensor(f, ts))
    f.readline()
    score = None
    if score_spec is not None:
      score_header = json.loads(f.readline())
      assert int(score_header['outcome']) == observation_id
      score = read_tensor(f, score_spec)
      f.readline()
    yield Record(
        context=context,
        observation_id=observation_id,
        feature_values=features,
        score=score)
