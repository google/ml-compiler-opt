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
import math
import random
import re

from absl import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


class SamplerBucketRoundRobin:
  """Calls return a list of module_specs sampled randomly from n buckets, in
  round-robin order. The buckets are sequential sections of module_specs of
  roughly equal lengths."""

  def __init__(self):
    self._ranges = {}

  def __call__(self,
               module_specs: Tuple[ModuleSpec],
               k: int,
               n: int = 20) -> List[ModuleSpec]:
    """
    Args:
      module_specs: list of module_specs to sample from
      k: number of modules to sample
      n: number of buckets to use
    """
    # Credits to yundi@ for the highly optimized algo.
    # Essentially, split module_specs into k buckets, then define the order of
    # visiting the k buckets such that it approximates the behaviour of having
    # n buckets.
    specs_len = len(module_specs)
    if (specs_len, k, n) not in self._ranges:
      quotient = k // n
      # rev_map maps from bucket # (implicitly via index) to order of visiting.
      # lower values should be visited first, and earlier indices before later.
      rev_map = [i % quotient for i in range(k)] if quotient else [0] * k
      # mapping defines the order in which buckets should be visited.
      mapping = [t[0] for t in sorted(enumerate(rev_map), key=lambda x: x[1])]

      # generate the buckets ranges, in the order which they should be visited.
      bucket_size_float = specs_len / k
      self._ranges[(specs_len, k, n)] = tuple(
          (math.floor(bucket_size_float * i),
           math.floor(bucket_size_float * (i + 1))) for i in mapping)

    return [
        module_specs[random.randrange(start, end)]
        for start, end in self._ranges[(specs_len, k, n)]
    ]


class Corpus:
  """Represents a corpus. Comes along with some utility functions."""

  @dataclass(frozen=True)
  class ReplaceContext:
    """Context for 'replace' rules."""
    full_path_prefix: str

  def __init__(self,
               data_path: str,
               additional_flags: Tuple[str, ...] = (),
               delete_flags: Tuple[str, ...] = (),
               replace_flags: Optional[Dict[str, str]] = None):
    """
    Args:
      data_path: corpus directory.
      additional_flags: list of flags to append to the command line
      delete_flags: list of flags to remove (both `-flag=<value` and
        `-flag <value>` are supported).
      replace_flags: list of flags to be replaced. The key in the dictionary
        is the flag. The value is a string that will be `format`-ed with a
        `context` object - see `ReplaceContext`.
        We verify that flags in replace_flags are present, and do not appear
        in the additional_flags nor delete_flags.
        Thinlto index is handled this way, too.
    """
    self._module_specs = tuple(
        sorted(
            _build_modulespecs_from_datapath(
                data_path=data_path,
                additional_flags=additional_flags,
                delete_flags=delete_flags,
                replace_flags=replace_flags),
            key=lambda m: m.size,
            reverse=True))
    self._root_dir = data_path

  @classmethod
  def from_module_specs(cls, module_specs: Iterable[ModuleSpec]):
    """Construct a Corpus from module specs. Mostly for testing purposes."""
    cps = cls.__new__(cls)  # Avoid calling __init__
    super(cls, cps).__init__()
    cps._module_specs = tuple(
        sorted(module_specs, key=lambda m: m.size, reverse=True))
    cps.root_dir = None
    return cps

  def sample(self,
             k: int,
             sort: bool = False,
             sampler=SamplerBucketRoundRobin()) -> List[ModuleSpec]:
    """Samples `k` module_specs, optionally sorting by size descending."""
    # Note: sampler is intentionally defaulted to a mutable object, as the
    # only mutable attribute of SamplerBucketRoundRobin is its range cache.
    k = min(len(self._module_specs), k)
    if k < 1:
      raise ValueError('Attempting to sample <1 module specs from corpus.')
    sampled_specs = sampler(self._module_specs, k=k)
    if sort:
      sampled_specs.sort(key=lambda m: m.size, reverse=True)
    return sampled_specs

  def filter(self, p: re.Pattern):
    """Filters module specs, keeping those which match the provided pattern."""
    return Corpus.from_module_specs(
        (ms for ms in self._module_specs if p.match(ms.name)))

  @property
  def module_specs(self):
    return self._module_specs

  def __len__(self):
    return len(self._module_specs)


def _build_modulespecs_from_datapath(
    data_path: str,
    additional_flags: Tuple[str, ...] = (),
    delete_flags: Tuple[str, ...] = (),
    replace_flags: Optional[Dict[str, str]] = None) -> List[ModuleSpec]:
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
    if len(replace_flags) > 0:
      logging.warning('Replace flags are specified together with override.')

  module_specs: List[ModuleSpec] = []

  # This takes ~7s for 30k modules
  for rel_module_path in module_paths:
    full_module_path = os.path.join(data_path, rel_module_path)
    exec_cmd = _load_and_parse_command(
        module_path=full_module_path,
        has_thinlto=has_thinlto,
        additional_flags=additional_flags,
        delete_flags=delete_flags,
        replace_flags=replace_flags,
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
    replace_flags: Optional[Dict[str, str]] = None,
    cmd_override: Tuple[str, ...] = (),
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

  context = Corpus.ReplaceContext(full_path_prefix=module_path)
  replace_flags = replace_flags.copy() if replace_flags else {}
  if has_thinlto:
    additional_flags = ('-mllvm', '-thinlto-assume-merged') + additional_flags
    if cmd_override:
      additional_flags = (
          f'-fthinlto-index={context.full_path_prefix}.thinlto.bc',
      ) + additional_flags
    else:
      fthinlto_index = '-fthinlto-index'
      if fthinlto_index in replace_flags:
        raise ValueError(
            '-fthinlto-index must be handled by the infrastructure')
      replace_flags[fthinlto_index] = '{context.full_path_prefix}.thinlto.bc'

  additional_flags = ('-x', 'ir', module_path + '.bc') + additional_flags

  matched_replace_flags = set()
  # don't user add/remove for replace
  if set(additional_flags).intersection(
      set(replace_flags)) or set(delete_flags).intersection(set(replace_flags)):
    raise ValueError('do not use add/delete flags to replace')

  option = next(option_iterator, None)

  while option is not None:
    if any(option.startswith(flag) for flag in delete_flags):
      if '=' not in option:
        next(option_iterator, None)
    else:
      matching_replace = [
          flag for flag in replace_flags if option.startswith(flag)
      ]
      if not matching_replace:
        cmdline.append(option)
      else:
        assert len(matching_replace) == 1
        flag = matching_replace[0]
        if flag in matched_replace_flags:
          raise ValueError(f'{flag} was matched twice')
        matched_replace_flags.add(flag)

        if '=' not in option:
          next(option_iterator, None)
          cmdline.extend([option, replace_flags[flag].format(context=context)])
        else:
          cmdline.append(flag + '=' +
                         replace_flags[flag].format(context=context))

    option = next(option_iterator, None)
  if len(matched_replace_flags) != len(replace_flags):
    raise ValueError('flags that were expected to be replaced were not found')
  cmdline.extend(additional_flags)

  # The options read from a .cmd file must be run with -cc1
  if not cmd_override and cmdline[0] != '-cc1':
    raise ValueError('-cc1 flag not present in .cmd file.')

  return cmdline
