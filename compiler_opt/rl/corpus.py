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
"""ModuleSpec definition and utility command line parsing functions."""
import random
import re

from absl import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import json
import os

from compiler_opt.rl import constant


@dataclass(frozen=True)
class ModuleSpec:
  """Dataclass describing an input module and its compilation command options.
  """
  name: str
  exec_cmd: Tuple[str, ...] = ()
  size: int = 0


class Corpus:
  """Represents a corpus. Comes along with some utility functions."""

  def __init__(self,
               data_path: str,
               additional_flags: Tuple[str, ...] = (),
               delete_flags: Tuple[str, ...] = ()):
    self._module_specs = _build_modulespecs_from_datapath(
        data_path=data_path,
        additional_flags=additional_flags,
        delete_flags=delete_flags)
    self._root_dir = data_path
    self._module_specs.sort(key=lambda m: m.size, reverse=True)

  @classmethod
  def from_module_specs(cls, module_specs: List[ModuleSpec]):
    """Construct a Corpus from module specs. Mostly for testing purposes."""
    cps = cls.__new__(cls)  # Avoid calling __init__
    super(cls, cps).__init__()
    cps._module_specs = list(module_specs)  # Don't mutate the original list.
    cps._module_specs.sort(key=lambda m: m.size, reverse=True)
    cps.root_dir = None
    return cps

  def sample(self,
             k: int,
             sort: bool = False,
             sampler=random.sample) -> List[ModuleSpec]:
    """Samples `k` module_specs, sorting by size descending."""
    k = min(len(self._module_specs), k)
    if k < 1:
      raise ValueError('Attempting to sample <1 module specs from corpus.')
    sampled_specs = sampler(self._module_specs, k=k)
    if sort:
      sampled_specs.sort(key=lambda m: m.size, reverse=True)
    return sampled_specs

  def filter(self, p: re.Pattern):
    """Filters module specs, keeping those which match the provided pattern."""
    self._module_specs = [ms for ms in self._module_specs if p.match(ms.name)]

  def __len__(self):
    return len(self._module_specs)


def sampler_bucket_round_robin(module_specs: List[ModuleSpec],
                               k: int) -> List[ModuleSpec]:
  """Return a list of module_specs sampled randomly from n buckets, in
  randomized-round-robin order. The buckets are sequential sections of
  module_specs of roughly equal lengths."""
  n = 20

  bucket_size_int, remainder = divmod(len(module_specs), n)
  # Calculate "front-heavy" buckets. Being front-heavy is necessary in case
  # k == len(module_specs)- to prevent encountering a bucket where everything
  # has been selected while there's another bucket with a non-selected item.
  bucket_ranges = []
  start, end = 0, bucket_size_int
  for i in range(n):
    if i < remainder:
      end += 1
    bucket_ranges.append((start, end))
    start, end = end, end + bucket_size_int

  ret = []
  selected = set()
  selected_add = selected.add  # borrowed trick from random.sample
  rand_between = random.randrange

  while True:
    for start, end in bucket_ranges:
      if len(ret) == k:
        return ret
      while (j := rand_between(start, end)) in selected:
        pass
      selected_add(j)
      ret.append(module_specs[j])


def _build_modulespecs_from_datapath(
    data_path: str,
    additional_flags: Tuple[str, ...] = (),
    delete_flags: Tuple[str, ...] = ()
) -> List[ModuleSpec]:
  # TODO: (b/233935329) Per-corpus *fdo profile paths can be read into
  # {additional|delete}_flags here
  with open(
      os.path.join(data_path, 'corpus_description.json'), 'r',
      encoding='utf-8') as f:
    corpus_description: Dict[str, Any] = json.load(f)

  module_paths = corpus_description['modules']
  if len(module_paths) == 0:
    raise ValueError(f'{data_path}\'s corpus_description contains no modules.')

  has_thinlto: bool = corpus_description['has_thinlto']

  cmd_override = ()
  if 'global_command_override' in corpus_description:
    if corpus_description[
        'global_command_override'] == constant.UNSPECIFIED_OVERRIDE:
      raise ValueError(
          'global_command_override in corpus_description.json not filled.')
    cmd_override = tuple(corpus_description['global_command_override'])
    if len(additional_flags) > 0:
      logging.warning('Additional flags are specified together with override.')
    if len(delete_flags) > 0:
      logging.warning('Delete flags are specified together with override.')

  module_specs: List[ModuleSpec] = []

  # This takes ~7s for 30k modules
  for rel_module_path in module_paths:
    full_module_path = os.path.join(data_path, rel_module_path)
    exec_cmd = _load_and_parse_command(
        module_path=full_module_path,
        has_thinlto=has_thinlto,
        additional_flags=additional_flags,
        delete_flags=delete_flags,
        cmd_override=cmd_override)
    size = os.path.getsize(full_module_path + '.bc')
    module_specs.append(
        ModuleSpec(name=rel_module_path, exec_cmd=tuple(exec_cmd), size=size))

  return module_specs


def _load_and_parse_command(
    module_path: str,
    has_thinlto: bool,
    additional_flags: Tuple[str, ...] = (),
    delete_flags: Tuple[str, ...] = (),
    cmd_override: Tuple[str, ...] = ()
) -> List[str]:
  """Cleans up base command line.

  Remove certain unnecessary flags, and add the .bc file to compile and, if
  given, the thinlto index.

  Args:
    module_path: Absolute path to the module without extension (from corpus).
    has_thinlto: Whether to add thinlto flags.
    additional_flags: Tuple of clang flags to add.
    delete_flags: Tuple of clang flags to remove.
    cmd_override: Tuple of strings to use as the base command line.

  Returns:
    The argument list to pass to the compiler process.
  """
  cmdline = []

  if cmd_override:
    option_iterator = iter(cmd_override)
  else:
    with open(module_path + '.cmd', encoding='utf-8') as f:
      option_iterator = iter(f.read().split('\0'))
  option = next(option_iterator, None)

  while option is not None:
    if any(option.startswith(flag) for flag in delete_flags):
      if '=' not in option:
        next(option_iterator, None)
    else:
      cmdline.append(option)
    option = next(option_iterator, None)
  cmdline.extend(['-x', 'ir', module_path + '.bc'])

  if has_thinlto:
    cmdline.extend([
        f'-fthinlto-index={module_path}.thinlto.bc', '-mllvm',
        '-thinlto-assume-merged'
    ])

  cmdline.extend(additional_flags)

  # The options read from a .cmd file must be run with -cc1
  if not cmd_override and cmdline[0] != '-cc1':
    raise ValueError('-cc1 flag not present in .cmd file.')

  return cmdline
