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
"""Strips tf_agents-introduced functions from a TFLite model file."""

from collections.abc import Sequence

from absl import app
from absl import flags

from compiler_opt.tools import strip_tfa_tflite_lib

_INPUT_PATH = flags.DEFINE_string(
    'input_path',
    None,
    'Path to the input TFLite model file.',
    required=True,
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Path to write the stripped TFLite model file.',
    required=True,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  strip_tfa_tflite_lib.strip_tfa_tflite_file(_INPUT_PATH.value,
                                             _OUTPUT_PATH.value)


if __name__ == '__main__':
  app.run(main)
