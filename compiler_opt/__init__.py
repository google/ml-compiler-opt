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
"""Ensure flags are initialized for e.g. pytest harness case."""

import sys

from absl import flags

# When this module is loaded in an app, flags would have been parsed already
# (assuming the app's main uses directly or indirectly absl.app.main). However,
# when loaded in a test harness like pytest or unittest (e.g. python -m pytest)
# that won't happen.
# While tests shouldn't use the flags directly, some flags - like compilation
# timeout - have default values that need to be accessible.
# This makes sure flags are initialized, for this purpose.
if not flags.FLAGS.is_parsed():
  flags.FLAGS(sys.argv, known_only=True)
assert flags.FLAGS.is_parsed()
