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
"""Module for running compilation and collect data for behavior cloning."""

import functools
from absl import app
from absl import flags
from absl import logging
import gin

from compiler_opt.rl.imitation_learning import generate_bc_trajectories_lib
from compiler_opt.tools import generate_test_model  # pylint:disable=unused-import

from tf_agents.system import system_multiprocessing as multiprocessing

flags.FLAGS['gin_files'].allow_override = True
flags.FLAGS['gin_bindings'].allow_override = True

FLAGS = flags.FLAGS


def main(_):
  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=FLAGS.gin_bindings, skip_unknown=True)
  logging.info(gin.config_str())

  generate_bc_trajectories_lib.gen_trajectories()


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
