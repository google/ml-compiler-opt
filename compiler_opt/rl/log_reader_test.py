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
"""Tests for compiler_opt.rl.log_reader."""

import ctypes
import json
import pickle
import sys
from absl.testing import absltest
from compiler_opt.rl import log_reader
from typing import BinaryIO


def json_to_bytes(d) -> bytes:
  return json.dumps(d).encode('utf-8')


def write_buff(f: BinaryIO, buffer: list, ct):
  # we should get the ctypes array to bytes for pytype to be happy.
  f.write((ct * len(buffer))(*buffer))  # pytype:disable=wrong-arg-types


def create_example(fname: str):
  nl = '\n'.encode('utf-8')
  t0_val = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  t1_val = [1, 2, 3]
  s = [1.2]

  with open(fname, 'wb') as f:
    f.write(
        json_to_bytes({
            'features': [{
                'name': 'tensor_name2',
                'port': 0,
                'shape': [2, 3],
                'type': 'float'
            }, {
                'name': 'tensor_name1',
                'port': 0,
                'shape': [3, 1],
                'type': 'int64_t'
            }],
            'score': {
                'name': 'reward',
                'port': 0,
                'shape': [1],
                'type': 'float'
            }
        }))
    f.write(nl)
    f.write(json_to_bytes({'context': 'some_context'}))
    f.write(nl)
    f.write(json_to_bytes({'observation': 0}))
    f.write(nl)
    write_buff(f, t0_val, ctypes.c_float)
    write_buff(f, t1_val, ctypes.c_int64)
    f.write(nl)
    f.write(json_to_bytes({'outcome': 0}))
    f.write(nl)
    write_buff(f, s, ctypes.c_float)
    f.write(nl)

    t0_val = [v + 1 for v in t0_val]
    t1_val = [v + 1 for v in t1_val]
    s[0] += 1

    f.write(json_to_bytes({'observation': 1}))
    f.write(nl)
    write_buff(f, t0_val, ctypes.c_float)
    write_buff(f, t1_val, ctypes.c_int64)
    f.write(nl)
    f.write(json_to_bytes({'outcome': 1}))
    f.write(nl)
    write_buff(f, s, ctypes.c_float)


class LogReaderTest(absltest.TestCase):

  def test_create_tensorspec(self):
    ts = log_reader.create_tensorspec({
        'name': 'tensor_name',
        'port': 0,
        'shape': [2, 3],
        'type': 'float'
    })
    self.assertEqual(
        ts,
        log_reader.TensorSpec(
            name='tensor_name',
            port=0,
            shape=[2, 3],
            element_type=ctypes.c_float))

  def test_read_header(self):
    tf = self.create_tempfile()
    create_example(tf)
    with open(tf, 'rb') as f:
      header = log_reader._read_header(f)  # pylint:disable=protected-access
      self.assertEqual(header.features, [
          log_reader.TensorSpec(
              name='tensor_name2',
              port=0,
              shape=[2, 3],
              element_type=ctypes.c_float),
          log_reader.TensorSpec(
              name='tensor_name1',
              port=0,
              shape=[3, 1],
              element_type=ctypes.c_int64)
      ])
      self.assertEqual(
          header.score,
          log_reader.TensorSpec(
              name='reward', port=0, shape=[1], element_type=ctypes.c_float))

  def test_read_log(self):
    tf = self.create_tempfile()
    create_example(tf)
    obs_id = 0
    for record in log_reader.read_log(tf):
      self.assertEqual(record.observation_id, obs_id)
      self.assertAlmostEqual(record.score[0], 1.2 + obs_id)
      obs_id += 1
    self.assertEqual(obs_id, 2)

  def test_pickling(self):
    skip_size = sys.version_info.minor < 10
    tf = self.create_tempfile()
    create_example(tf)
    records = list(log_reader.read_log(tf))
    r1: log_reader.Record = records[0]
    fv1 = r1.feature_values[0]
    s = pickle.dumps(fv1)
    self.assertTrue(skip_size or len(s) == 202)
    fv2 = r1.feature_values[1]
    # illustrate that, while the size of the stand-alone pickled fv2 is the same
    # as fv1 (as expected given their shape), when we pickle them together, the
    # resulting size is smaller than the sum - because the TensorSpec is
    # referenced.
    self.assertTrue(skip_size or len(pickle.dumps(fv1)) == 202)
    # in particular, pickling references is quite cheap.
    self.assertTrue(skip_size or len(pickle.dumps([fv1, fv1])) == 208)
    self.assertTrue(skip_size or len(pickle.dumps([fv1, fv2])) == 305)
    o: log_reader.TensorValue = pickle.loads(s)
    self.assertEqual(len(fv1), len(o))
    for i in range(len(fv1)):
      self.assertEqual(fv1[i], o[i])


if __name__ == '__main__':
  absltest.main()
