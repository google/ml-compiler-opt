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

import collections
import ctypes
import dataclasses
import json
import math

from compiler_opt import type_map
from typing import Any, BinaryIO, Dict, Generator, List, Optional
import numpy as np
import tensorflow as tf

_element_type_name_to_dtype = {
    name: dtype for name, _, dtype in type_map.TYPE_ASSOCIATIONS
}

_dtype_to_ctype = {
    dtype: ctype for _, ctype, dtype in type_map.TYPE_ASSOCIATIONS
}


def create_tensorspec(d: Dict[str, Any]) -> tf.TensorSpec:
  name: str = d['name']
  shape: List[int] = [int(e) for e in d['shape']]
  element_type_str: str = d['type']
  if element_type_str not in _element_type_name_to_dtype:
    raise ValueError(f'uknown type: {element_type_str}')
  return tf.TensorSpec(
      name=name,
      shape=tf.TensorShape(shape),
      dtype=_element_type_name_to_dtype[element_type_str])


class LogReaderTensorValue:
  """The value of a tensor of a given spec.

  We root the bytes buffer which provide the underlying data, and index in
  the value based on the type of the tensor, thus the TensorValue can be used
  as a list-like object containing the scalar values, in row-major order, of
  the tensor.

  Endianness is assumed to be the same as the log producer's.
  """
  __slots__ = ('_buffer', '_spec', '_len', '_view')

  def __init__(self, spec: tf.TensorSpec, buffer: bytes):
    self._buffer = buffer
    self._spec = spec
    self._len = math.prod(spec.shape)
    self._set_view()

  @property
  def spec(self):
    return self._spec

  def to_numpy(self) -> np.ndarray:
    return np.frombuffer(
        self._buffer, dtype=_dtype_to_ctype[self._spec.dtype], count=self._len)

  def _set_view(self):
    # c_char_p is a nul-terminated string, so the more appropriate cast here
    # would be POINTER(c_char), but unfortunately, c_char_p is the only
    # type that can be constructed from a `bytes`. To capture our intent,
    # we cast the c_char_p to
    buffer_as_nul_ending_ptr = ctypes.c_char_p(self._buffer)
    buffer_as_naked_ptr = ctypes.cast(buffer_as_nul_ending_ptr,
                                      ctypes.POINTER(ctypes.c_char))
    self._view = ctypes.cast(buffer_as_naked_ptr,
                             ctypes.POINTER(_dtype_to_ctype[self.spec.dtype]))

  def __len__(self) -> int:
    return self._len

  def __getitem__(self, index: int):
    if index < 0 or index >= self._len:
      raise IndexError(f'Index {index} out of range [0..{self._len})')
    # pytype believes `index` is an object, despite all the annotations to the
    # contrary.
    return self._view[index]  # pytype:disable=wrong-arg-types


@dataclasses.dataclass(frozen=True)
class _Header:
  features: List[tf.TensorSpec]
  score: Optional[tf.TensorSpec]


def _read_tensor(fs: BinaryIO, ts: tf.TensorSpec) -> LogReaderTensorValue:
  size = math.prod(ts.shape) * ctypes.sizeof(_dtype_to_ctype[ts.dtype])
  data = fs.read(size)
  return LogReaderTensorValue(ts, data)


def _read_header(f: BinaryIO) -> Optional[_Header]:
  header_raw = f.readline()
  if not header_raw:
    # This is the path taken by empty files
    return None
  header = json.loads(header_raw)
  tensor_specs = [create_tensorspec(ts) for ts in header['features']]
  score_spec = create_tensorspec(header['score']) if 'score' in header else None
  return _Header(features=tensor_specs, score=score_spec)


@dataclasses.dataclass(frozen=True)
class ObservationRecord:
  context: str
  observation_id: int
  feature_values: List[LogReaderTensorValue]
  score: Optional[LogReaderTensorValue]


def _enumerate_log_from_stream(
    f: BinaryIO, header: _Header) -> Generator[ObservationRecord, None, None]:
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
      features.append(_read_tensor(f, ts))
    f.readline()
    score = None
    if score_spec is not None:
      score_header = json.loads(f.readline())
      assert int(score_header['outcome']) == observation_id
      score = _read_tensor(f, score_spec)
      f.readline()
    yield ObservationRecord(
        context=context,
        observation_id=observation_id,
        feature_values=features,
        score=score)


def read_log_from_file(f) -> Generator[ObservationRecord, None, None]:
  header = _read_header(f)
  if header:
    yield from _enumerate_log_from_stream(f, header)


def read_log(fname: str) -> Generator[ObservationRecord, None, None]:
  with open(fname, 'rb') as f:
    yield from read_log_from_file(f)


def _add_feature(se: tf.train.SequenceExample, spec: tf.TensorSpec,
                 value: LogReaderTensorValue):
  f = se.feature_lists.feature_list[spec.name].feature.add()
  # This should never happen: _add_feature is an implementation detail of
  # read_log_as_sequence_examples, and the only dtypes we should see here are
  # those in _element_type_name_map, or an exception would have been thrown
  # already.
  if spec.dtype not in _dtype_to_ctype:
    raise ValueError('Unsupported dtype: f{spec.dtype}')
  if spec.dtype in [tf.float32, tf.float64]:
    lst = f.float_list.value
  else:
    lst = f.int64_list.value
  lst.extend(value)


def read_log_as_sequence_examples(
    fname: str) -> Dict[str, tf.train.SequenceExample]:
  ret: Dict[str, tf.train.SequenceExample] = collections.defaultdict(
      tf.train.SequenceExample)
  # a record is an observation: the features and score for one step.
  # the records are in time order
  # the `context` is, for example, the function name for passes like regalloc.
  # we produce a dictionary keyed in contexts with SequenceExample values.
  for record in read_log(fname):
    se = ret[record.context]
    if record.score:
      _add_feature(se, record.score.spec, record.score)
    for t in record.feature_values:
      _add_feature(se, t.spec, t)
  return ret
