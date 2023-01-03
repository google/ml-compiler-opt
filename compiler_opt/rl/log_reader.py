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

The "simple" log format consists of a sequence of one-line json strings and
raw dump of tensor data - with the assumption that the endianness of the
reader is the same as the endianness of the producer - as follows:

header: this is a json object describing the feature tensors and the outcome
tensor. The feature tensors are in an array - the order matters (as will become
apparent).
Example: {"features": [{tensor spec}, {tensor spec}], "score": {tensor spec}}

The tensor spec is a json object:
{"name":.., "port":... "shape":[..], "type":".."}

context: this is a json object indicating the context the observations that
follow refer to. Example: for inlining, the context is "default" (the module).
For regalloc, the context is a function name.
Example: {"context": "_ZNfoobar"}

observation: this is a json object indicating that an observation - i.e. feature
data - is following. It also contains the observation count (0, 1...)
Example: {"observation": 0}

A buffer containing the dump of tensor data, in the order given in the header,
follows here. A new line terminates it - just so that the next json string
appears at the beginning of the line, in case the log is opened with an editor
(so just for debugging). The reader should use the header data to know how much
data to read, and to which tensors it corresponds. The reader may not rely on
the terminating \n as indicator, and should just assume it there and consume it
upon finishing reading the tensor data.

outcome: this is a json object indicating the score/reward data follows. It also
has an id which should match that of the observation before (for debugging)
Example: {"outcome": 0}

A buffer containing the outcome tensor follows - same idea as for features.

The above repeats - either a new observation follows, or a new context.

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
  __slots__ = ('_buffer', '_elem_type', '_len', '_view')

  def __init__(self, spec: TensorSpec, buffer: bytes):
    self._buffer = buffer
    self._elem_type = spec.element_type
    self._len = math.prod(spec.shape)
    self._set_view()

  def _set_view(self):
    # c_char_p is a nul-terminated string, so the more appropriate cast here
    # would be POINTER(c_char), but unfortunately, c_char_p is the only
    # type that can be constructed from a `bytes`. To capture our intent,
    # we cast the c_char_p to
    buffer_as_nul_ending_ptr = ctypes.c_char_p(self._buffer)
    buffer_as_naked_ptr = ctypes.cast(buffer_as_nul_ending_ptr,
                                      ctypes.POINTER(ctypes.c_char))
    self._view = ctypes.cast(buffer_as_naked_ptr,
                             ctypes.POINTER(self._elem_type))

  def __getstate__(self):
    # _view wouldn't be picklable because it's a pointer. It's easily
    # recreatable when un-pickling.
    return (None, {
        '_buffer': self._buffer,
        '_view': None,
        '_len': self._len,
        '_elem_type': self._elem_type
    })

  def __setstate__(self, state):
    _, slots = state
    self._buffer = slots['_buffer']
    self._elem_type = slots['_elem_type']
    self._len = slots['_len']
    self._set_view()

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


def _read_header(f: BinaryIO) -> Header:
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


def _enumerate_log_from_stream(f: BinaryIO,
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


def read_log(fname: str) -> Generator[Record, None, None]:
  with open(fname, 'rb') as f:
    header = _read_header(f)
    yield from _enumerate_log_from_stream(f, header)
