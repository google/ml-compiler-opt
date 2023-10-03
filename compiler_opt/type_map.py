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
"""Map between tf, ctypes, and string names for scalar types."""
import ctypes
from typing import List, Tuple, Union
import tensorflow as tf

ScalarCType = Union['type[ctypes.c_float]', 'type[ctypes.c_double]',
                    'type[ctypes.c_int8]', 'type[ctypes.c_int16]',
                    'type[ctypes.c_uint16]', 'type[ctypes.c_int32]',
                    'type[ctypes.c_uint32]', 'type[ctypes.c_int64]',
                    'type[ctypes.c_uint64]']

TYPE_ASSOCIATIONS: List[Tuple[str, ScalarCType,
                              tf.DType]] = [
                                  ('float', ctypes.c_float, tf.float32),
                                  ('double', ctypes.c_double, tf.float64),
                                  ('int8_t', ctypes.c_int8, tf.int8),
                                  ('uint8_t', ctypes.c_uint8, tf.uint8),
                                  ('int16_t', ctypes.c_int16, tf.int16),
                                  ('uint16_t', ctypes.c_uint16, tf.uint16),
                                  ('int32_t', ctypes.c_int32, tf.int32),
                                  ('uint32_t', ctypes.c_uint32, tf.uint32),
                                  ('int64_t', ctypes.c_int64, tf.int64),
                                  ('uint64_t', ctypes.c_uint64, tf.uint64)
                              ]
