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
"""Implementation of the 'regalloc' problem."""

import gin

from compiler_opt.rl import env
from compiler_opt.rl import problem_configuration
from compiler_opt.rl.regalloc import config
from compiler_opt.rl.regalloc import regalloc_runner


@gin.register(module='configs')
class RegallocEvictionConfig(problem_configuration.ProblemConfiguration):
  """Expose the regalloc eviction configuration."""

  def get_env(self) -> env.MLGOEnvironmentBase:
    raise NotImplementedError

  def get_runner_type(self):
    return regalloc_runner.RegAllocRunner

  def get_signature_spec(self):
    return config.get_regalloc_signature_spec()

  def get_preprocessing_layer_creator(self):
    return config.get_observation_processing_layer_creator()

  def get_nonnormalized_features(self):
    return config.get_nonnormalized_features()
