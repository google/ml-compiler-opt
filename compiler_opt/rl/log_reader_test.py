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
from compiler_opt.rl import log_reader

# This is https://github.com/google/pytype/issues/764
from google.protobuf import text_format  # pytype: disable=pyi-error
from typing import BinaryIO

import tensorflow as tf


def json_to_bytes(d) -> bytes:
  return json.dumps(d).encode('utf-8')


nl = '\n'.encode('utf-8')


def write_buff(f: BinaryIO, buffer: list, ct):
  # we should get the ctypes array to bytes for pytype to be happy.
  f.write((ct * len(buffer))(*buffer))  # pytype:disable=wrong-arg-types


def write_context_marker(f: BinaryIO, name: str):
  f.write(nl)
  f.write(json_to_bytes({'context': name}))


def write_observation_marker(f: BinaryIO, obs_idx: int):
  f.write(nl)
  f.write(json_to_bytes({'observation': obs_idx}))


def begin_features(f: BinaryIO):
  f.write(nl)


def write_outcome_marker(f: BinaryIO, obs_idx: int):
  f.write(nl)
  f.write(json_to_bytes({'outcome': obs_idx}))


def create_example(fname: str, nr_contexts=1):

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
    for ctx_id in range(nr_contexts):
      t0_val = [v + ctx_id * 10 for v in t0_val]
      t1_val = [v + ctx_id * 10 for v in t1_val]
      write_context_marker(f, f'context_nr_{ctx_id}')
      write_observation_marker(f, 0)
      begin_features(f)
      write_buff(f, t0_val, ctypes.c_float)
      write_buff(f, t1_val, ctypes.c_int64)
      write_outcome_marker(f, 0)
      begin_features(f)
      write_buff(f, s, ctypes.c_float)

      t0_val = [v + 1 for v in t0_val]
      t1_val = [v + 1 for v in t1_val]
      s[0] += 1

      write_observation_marker(f, 1)
      begin_features(f)
      write_buff(f, t0_val, ctypes.c_float)
      write_buff(f, t1_val, ctypes.c_int64)
      write_outcome_marker(f, 1)
      begin_features(f)
      write_buff(f, s, ctypes.c_float)


class LogReaderTest(tf.test.TestCase):

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
    logfile = self.create_tempfile()
    create_example(logfile)
    with open(logfile, 'rb') as f:
      header = log_reader._read_header(f)  # pylint:disable=protected-access
      self.assertIsNotNone(header)
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
      ])  # pylint:disable=attribute-error,
      self.assertEqual(header.score,
                       log_reader.TensorSpec(
                           name='reward',
                           port=0,
                           shape=[1],
                           element_type=ctypes.c_float))  # pylint:disable=attribute-error

  def test_read_header_empty_file(self):
    logfile = self.create_tempfile()
    with open(logfile, 'rb') as f:
      header = log_reader._read_header(f)  # pylint:disable=protected-access
      self.assertIsNone(header)

  def test_read_log(self):
    logfile = self.create_tempfile()
    create_example(logfile)
    obs_id = 0
    for record in log_reader.read_log(logfile):
      self.assertEqual(record.observation_id, obs_id)
      self.assertAlmostEqual(record.score[0], 1.2 + obs_id)
      obs_id += 1
    self.assertEqual(obs_id, 2)

  def test_pickling(self):
    skip_size = sys.version_info.minor < 10
    logfile = self.create_tempfile()
    create_example(logfile)
    records = list(log_reader.read_log(logfile))
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

  def test_seq_example_conversion(self):
    logfile = self.create_tempfile()
    create_example(logfile, nr_contexts=2)
    seq_examples = log_reader.read_log_as_sequence_examples(logfile)
    self.assertIn('context_nr_0', seq_examples)
    self.assertIn('context_nr_1', seq_examples)
    self.assertEqual(
        seq_examples['context_nr_1'].feature_lists.feature_list['tensor_name1']
        .feature[0].int64_list.value, [12, 13, 14])
    # each context has 2 observations. The reward is scalar, the
    # 2 features' shapes are given in `create_example` above.
    expected_ctx_0 = text_format.Parse(
        """
feature_lists {
  feature_list {
    key: "reward"
    value {
      feature {
        float_list {
          value: 1.2000000476837158
        }
      }
      feature {
        float_list {
          value: 2.200000047683716
        }
      }
    }
  }
  feature_list {
    key: "tensor_name1"
    value {
      feature {
        int64_list {
          value: 1
          value: 2
          value: 3
        }
      }
      feature {
        int64_list {
          value: 2
          value: 3
          value: 4
        }
      }
    }
  }
  feature_list {
    key: "tensor_name2"
    value {
      feature {
        float_list {
          value: 0.10000000149011612
          value: 0.20000000298023224
          value: 0.30000001192092896
          value: 0.4000000059604645
          value: 0.5
          value: 0.6000000238418579
        }
      }
      feature {
        float_list {
          value: 1.100000023841858
          value: 1.2000000476837158
          value: 1.2999999523162842
          value: 1.399999976158142
          value: 1.5
          value: 1.600000023841858
        }
      }
    }
  }
}
""", tf.train.SequenceExample())
    self.assertProtoEquals(expected_ctx_0, seq_examples['context_nr_0'])


if __name__ == '__main__':
  tf.test.main()
