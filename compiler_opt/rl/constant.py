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
"""Constants for policy training."""

import dataclasses
import enum
import gin
import json

BASE_DIR = 'compiler_opt/rl'
BASE_MODULE_DIR = 'compiler_opt.rl'

# Delta to add when computing reward.
DELTA = 0.01

# Default of global_command_override in corpus_description.json
UNSPECIFIED_OVERRIDE = ['<UNSPECIFIED>']


@gin.constants_from_enum
class AgentName(enum.Enum):
  """Class that enumerates different types of agent names."""
  BEHAVIORAL_CLONE = 0
  DQN = 1
  PPO = 2
  PPO_DISTRIBUTED = 3


class DataClassJSONEncoder(json.JSONEncoder):

  def default(self, o):
    if dataclasses.is_dataclass(o):
      return dataclasses.asdict(o)
    return super().default(o)
