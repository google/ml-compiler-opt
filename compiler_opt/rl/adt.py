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
"""Abstract data types."""

from dataclasses import dataclass
import logging
from typing import List
import sys


@dataclass(frozen=True)
class ModuleSpec:
  """Dataclass describing an input module and its compilation command options.
  """
  _exec_cmd: List[str]
  _xopts: dict[str, List[str]]
  name: str

  def cmd(self, **kwargs) -> List[str]:
    """Retrieves the compiler execution options,
    optionally adding configurable, pre-set options.
    """
    ret_cmd: List[str] = list(self._exec_cmd) # Make a copy
    for k, substitutions in kwargs.items():
      if k in self._xopts:
        if substitutions is not None:
          ret_cmd += [opt.format(**substitutions) for opt in self._xopts[k]]
      else:
        logging.fatal('Command line addition of \'%s\' requested '
                      'but no such extra option exists', k)
        sys.exit(1)
    return ret_cmd
