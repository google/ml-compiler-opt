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
"""Implementation of the 'RegAlloc' subclass of ModuleSpec."""

from typing import Tuple

from compiler_opt.rl import corpus


class InliningSpec(corpus.ModuleSpec):
  """
  Dataclass describing an RegAlloc input module
  and its compilation command options.
  """

  @classmethod
  def get(cls, data_path: str, additional_flags: Tuple[str, ...],
          delete_flags: Tuple[str, ...]):
    """Fetch a list of RegAllocSpecs for the corpus at data_path

    Args:
      data_path: base directory of corpus
      additional_flags: tuple of clang flags to add.
      delete_flags: tuple of clang flags to remove.
    """
    xopts = {
        'tf_policy_path': ('-mllvm', '-regalloc-model={path:s}'),
        'training_log': ('-mllvm', '-regalloc-training-log={path:s}')
    }
    additional_flags += ('-mllvm', '-regalloc-enable-advisor=development')

    return cls._get(data_path, additional_flags, delete_flags, xopts)
