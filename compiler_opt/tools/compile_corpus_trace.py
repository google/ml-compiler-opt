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
"""Tool for running the local trace data collector."""

from absl import flags
from absl import app
from absl import logging

from compiler_opt.rl import trace_data_collector

_CORPUS_PATH = flags.DEFINE_string('corpus_path', None, 'Path to the corpus.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None, 'Path to the output.')
_CLANG_PATH = flags.DEFINE_string('clang_path', None,
                                  'The path to the clang to use.')


def main(_):
  trace_data_collector.compile_corpus(_CORPUS_PATH.value, _OUTPUT_PATH.value,
                                      _CLANG_PATH.value)


if __name__ == '__main__':
  flags.mark_flag_as_required('corpus_path')
  flags.mark_flag_as_required('output_path')
  flags.mark_flag_as_required('clang_path')
  app.run(main)
