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
"""Local ES trainer."""

from absl import app, flags, logging
import gin

from compiler_opt.es import es_trainer_lib

_GIN_FILES = flags.DEFINE_multi_string(
    "gin_files", [], "List of paths to gin configuration files.")
_GIN_BINDINGS = flags.DEFINE_multi_string(
    "gin_bindings", [],
    "Gin bindings to override the values set in the config files.")


def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILES.value, bindings=_GIN_BINDINGS.value, skip_unknown=False)
  logging.info(gin.config_str())

  final_weights = es_trainer_lib.train()

  logging.info("Final Weights:")
  logging.info(", ".join(final_weights))


if __name__ == "__main__":
  app.run(main)
