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
"""Compilation problem component model registry.

This allows tools just get a ProblemConfiguration object encapsulating all
necessary elements - compilation runner implementation & ML configuration:

`config = registry.get_configuration()`

See also problem_configuration.py
"""

from typing import Type

import gin
import tf_agents as tfa

from compiler_opt.rl import problem_configuration

# Register implementations. They appear unused, but they need to be imported
# to trigger gin registration.
import compiler_opt.rl.inlining  # pylint: disable=unused-import
import compiler_opt.rl.regalloc  # pylint: disable=unused-import

types = tfa.typing.types


@gin.configurable(module='config_registry')
def get_configuration(
    implementation: Type[problem_configuration.ProblemConfiguration]
) -> problem_configuration.ProblemConfiguration:
  return implementation()
