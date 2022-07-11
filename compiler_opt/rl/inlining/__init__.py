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
"""Implementation of the 'inlining for size' problem."""

from typing import Tuple, List
import gin

from compiler_opt import adt
from compiler_opt.rl import problem_configuration
from compiler_opt.rl.inlining import config
from compiler_opt.rl.inlining import inlining_runner


@gin.register(module='configs')
class InliningConfig(problem_configuration.ProblemConfiguration):
  """Expose the regalloc eviction components."""

  def get_runner_type(self):
    return inlining_runner.InliningRunner

  def get_signature_spec(self):
    return config.get_inlining_signature_spec()

  def get_preprocessing_layer_creator(self):
    return config.get_observation_processing_layer_creator()

  def get_nonnormalized_features(self):
    return config.get_nonnormalized_features()

  def get_module_specs(
      self,
      data_path: str,
      additional_flags: Tuple[str, ...] = (),
      delete_flags: Tuple[str, ...] = ()
  ) -> List[adt.ModuleSpec]:
    """Fetch a list of ModuleSpecs for the corpus at data_path

    Args:
      data_path: base directory of corpus
      additional_flags: tuple of clang flags to add.
      delete_flags: tuple of clang flags to remove.
    """
    xopts = {
        'tf_policy_path':
            ('-mllvm', '-ml-inliner-model-under-training={path:s}'),
        'training_log': ('-mllvm', '-training-log={path:s}')
    }
    additional_flags += ('-mllvm', '-enable-ml-inliner=development')

    return adt.ModuleSpec.get(data_path, additional_flags, delete_flags, xopts)
