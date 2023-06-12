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
"""Corpus and related concepts."""
import abc
import concurrent.futures
import math
import random

from absl import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import json
import os
import tensorflow as tf

from compiler_opt.rl import constant

# Alias to better self-document APIs. Represents a complete, ready to use
# command line, where all the flags reference existing, local files.
FullyQualifiedCmdLine = Tuple[str, ...]


def _apply_cmdline_filters(
    orig_options: Tuple[str, ...],
    additional_flags: Tuple[str, ...] = (),
    delete_flags: Tuple[str, ...] = (),
    replace_flags: Optional[Dict[str, str]] = None) -> Tuple[str]:
  option_iterator = iter(orig_options)
  matched_replace_flags = set()
  replace_flags = replace_flags if replace_flags is not None else {}

  option = next(option_iterator, None)
  cmdline = []
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
          cmdline.extend([option, replace_flags[flag]])
        else:
          cmdline.append(flag + '=' + replace_flags[flag])

    option = next(option_iterator, None)
  if len(matched_replace_flags) != len(replace_flags):
    raise ValueError('flags that were expected to be replaced were not found')
  cmdline.extend(additional_flags)
  return tuple(cmdline)


@dataclass(frozen=True)
class LoadedModuleSpec:
  """Encapsulates the loaded data of a module and the rules to persist it.

  A LoadedModuleSpec can be passed to a remote location. There, given a local
  directory, to_module_spec can be called, resulting in the data being saved
  under that directory, the final compiler command line fully computed, and a
  ready-to-use FullyQualifiedCmdLine returned.
  """
  name: str
  loaded_ir: bytes
  loaded_thinlto_index: Optional[bytes] = None
  orig_options: Tuple[str, ...] = ()

  def _create_files_and_get_context(self, local_dir: str):
    root_dir = os.path.join(local_dir, self.name)
    os.makedirs(root_dir, exist_ok=True)
    module_path = os.path.join(root_dir, 'input.bc')
    thinlto_index_path = None
    with tf.io.gfile.GFile(module_path, 'wb') as f:
      f.write(self.loaded_ir)
    if self.loaded_thinlto_index is not None:
      thinlto_index_path = os.path.join(root_dir, 'index.thinlto.bc')
      with tf.io.gfile.GFile(thinlto_index_path, 'wb') as f:
        f.write(self.loaded_thinlto_index)
    context = Corpus.ReplaceContext(
        module_full_path=module_path, thinlto_full_path=thinlto_index_path)
    return context

  def build_command_line(self, local_dir: str) -> FullyQualifiedCmdLine:
    """Different LoadedModuleSpec objects must get different `local_dir`s."""
    context = self._create_files_and_get_context(local_dir)
    return tuple(option.format(context=context) for option in self.orig_options)


@dataclass(frozen=True)
class ModuleSpec:
  """Metadata of a compilation unit.
  This contains the necessary information to enable corpus operations like
  sampling or filtering, as well as to enable the corpus create
  a LoadedModuleSpec from a CorpusElement.
  """
  name: str
  size: int
  command_line: Tuple[str, ...] = ()
  has_thinlto: bool = False


class Sampler(metaclass=abc.ABCMeta):
  """Corpus sampler abstraction."""

  @abc.abstractmethod
  def __init__(self, module_specs: Tuple[ModuleSpec]):
    self._module_specs = module_specs

  @abc.abstractmethod
  def reset(self):
    pass

  @abc.abstractmethod
  def __call__(self, k: int, n: int = 20) -> List[ModuleSpec]:
    """
    Args:
      k: number of modules to sample
      n: number of buckets to use
    """
    raise NotImplementedError()


class SamplerBucketRoundRobin(Sampler):
  """Calls return a list of module_specs sampled randomly from n buckets, in
  round-robin order. The buckets are sequential sections of module_specs of
  roughly equal lengths."""

  def __init__(self, module_specs: Tuple[ModuleSpec]):
    self._ranges = {}
    super().__init__(module_specs)

  def reset(self):
    pass

  def __call__(self, k: int, n: int = 20) -> List[ModuleSpec]:
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
    specs_len = len(self._module_specs)
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
        self._module_specs[random.randrange(start, end)]
        for start, end in self._ranges[(specs_len, k, n)]
    ]


class CorpusExhaustedError(Exception):
  pass


class SamplerWithoutReplacement(Sampler):
  """Randomly samples the corpus, without replacement."""

  def __init__(self, module_specs: Tuple[ModuleSpec]):
    super().__init__(module_specs)
    self._idx = 0
    self._shuffle_order()

  def _shuffle_order(self):
    self._module_specs = tuple(
        random.sample(self._module_specs, len(self._module_specs)))

  def reset(self):
    self._shuffle_order()
    self._idx = 0

  def __call__(self, k: int, n: int = 10) -> List[ModuleSpec]:
    """
    Args:
      k: number of modules to sample
      n: ignored
    Raises:
      CorpusExhaustedError if there are fewer than k elements left to sample in
      the corpus.
    """
    endpoint = self._idx + k
    if endpoint > len(self._module_specs):
      raise CorpusExhaustedError()
    results = self._module_specs[self._idx:endpoint]
    self._idx = self._idx + k
    return list(results)


class Corpus:
  """Represents a corpus.

  A corpus is created from a corpus_description.json file, produced by
  extract_ir.py (for example).

  To use the corpus:
  - call sample to get a subset of modules (using the Sampler provided at
  initialization time). This returns a list of ModuleSpec objects
  - convert the ModuleSpecs to LoadedModuleSpecs. This loads the contents of the
  modules in memory (hence this lazy approach). The caller may want to perform
  this step with a threadpool
  - pass the LoadedModuleSpecs to Workers
  - to use a LoadedModuleSpec, create a unique directory (i.e. tempdir) and
  pass it to to_module_spec

  Example:

  corpus = Corpus(...)

  samples = corpus.sample(10)
  with ThreadPoolExecutor() as tp:
    futures = [tp.submit(corpus.load_module_spec, s) for s in samples]
    ...
    lms = [f.result() for f in futures]
    ...(pass lms values to workers)

  On the worker side:
  lm: LoadedModuleSpec = ...
  with tempfile.mkdir() as tempdir:
    final_cmd_line = lm.build_command_line(tempdir)
    ...(prepend executable to final_cmd_line, run it)

  """

  @dataclass(frozen=True)
  class ReplaceContext:
    """Context for 'replace' rules."""
    module_full_path: str
    thinlto_full_path: Optional[str] = None

  def __init__(self,
               *,
               data_path: str,
               module_filter: Optional[Callable[[str], bool]] = None,
               additional_flags: Tuple[str, ...] = (),
               delete_flags: Tuple[str, ...] = (),
               replace_flags: Optional[Dict[str, str]] = None,
               sampler_type: Type[Sampler] = SamplerBucketRoundRobin):
    """
    Prepares the corpus by pre-loading all the CorpusElements and preparing for
    sampling. Command line origin (.cmd file or override) is decided, and final
    command line transformation rules are set (i.e. thinlto flags handled, also
    output) and validated.

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
      module_filter: a regular expression used to filter 'in' modules with names
        matching it. None to include everything.
    """
    self._base_dir = data_path
    # TODO: (b/233935329) Per-corpus *fdo profile paths can be read into
    # {additional|delete}_flags here
    with tf.io.gfile.GFile(
        os.path.join(data_path, 'corpus_description.json'), 'r') as f:
      corpus_description: Dict[str, Any] = json.load(f)

    module_paths = corpus_description['modules']
    if len(module_paths) == 0:
      raise ValueError(
          f'{data_path}\'s corpus_description contains no modules.')

    has_thinlto: bool = corpus_description['has_thinlto']

    cmd_override = ()
    cmd_override_was_specified = False
    if 'global_command_override' in corpus_description:
      cmd_override_was_specified = True
      if corpus_description[
          'global_command_override'] == constant.UNSPECIFIED_OVERRIDE:
        raise ValueError(
            'global_command_override in corpus_description.json not filled.')
      cmd_override = tuple(corpus_description['global_command_override'])
      if len(additional_flags) > 0:
        logging.warning(
            'Additional flags are specified together with override.')
      if len(delete_flags) > 0:
        logging.warning('Delete flags are specified together with override.')
      if replace_flags:
        logging.warning('Replace flags are specified together with override.')

    replace_flags = replace_flags.copy() if replace_flags else {}
    fthinlto_index_flag = '-fthinlto-index'

    if has_thinlto:
      additional_flags = ('-mllvm', '-thinlto-assume-merged') + additional_flags
      if cmd_override_was_specified:
        additional_flags = (f'{fthinlto_index_flag}=' +
                            '{context.thinlto_full_path}',) + additional_flags
      else:
        if fthinlto_index_flag in replace_flags:
          raise ValueError(
              '-fthinlto-index must be handled by the infrastructure')
        replace_flags[fthinlto_index_flag] = '{context.thinlto_full_path}'

    additional_flags = ('-x', 'ir',
                        '{context.module_full_path}') + additional_flags

    # don't use add/remove for replace
    add_keys = set(k.split('=', maxsplit=1)[0] for k in additional_flags)
    if add_keys.intersection(
        set(replace_flags)) or set(delete_flags).intersection(
            set(replace_flags)) or add_keys.intersection(set(delete_flags)):
      raise ValueError('do not use add/delete flags to replace')

    if module_filter:
      module_paths = [name for name in module_paths if module_filter(name)]

    def get_cmdline(name: str):
      if cmd_override_was_specified:
        ret = cmd_override
      else:
        with tf.io.gfile.GFile(os.path.join(data_path, name + '.cmd')) as f:
          ret = tuple(f.read().replace(r'{', r'{{').replace(r'}',
                                                            r'}}').split('\0'))
          # The options read from a .cmd file must be run with -cc1
          if ret[0] != '-cc1':
            raise ValueError('-cc1 flag not present in .cmd file.')
      return _apply_cmdline_filters(
          orig_options=ret,
          additional_flags=additional_flags,
          delete_flags=delete_flags,
          replace_flags=replace_flags)

    # perform concurrently because fetching file size may be slow (remote)
    with concurrent.futures.ThreadPoolExecutor() as tp:
      contents = tp.map(
          lambda name: ModuleSpec(
              name=name,
              size=tf.io.gfile.GFile(os.path.join(data_path, name + '.bc')).
              size(),
              command_line=get_cmdline(name),
              has_thinlto=has_thinlto), module_paths)
    self._module_specs = tuple(
        sorted(contents, key=lambda m: m.size, reverse=True))
    self._sampler = sampler_type(self._module_specs)

  def reset(self):
    self._sampler.reset()

  def sample(self, k: int, sort: bool = False) -> List[ModuleSpec]:
    """Samples `k` module_specs, optionally sorting by size descending.

    Use load_corpus_element to get LoadedModuleSpecs - this allows the user
    decide how the loading should happen (e.g. may want to use a threadpool)
    """
    # Note: sampler is intentionally defaulted to a mutable object, as the
    # only mutable attribute of SamplerBucketRoundRobin is its range cache.
    k = min(len(self._module_specs), k)
    if k < 1:
      raise ValueError('Attempting to sample <1 module specs from corpus.')
    sampled_specs = self._sampler(k=k)
    if sort:
      sampled_specs.sort(key=lambda m: m.size, reverse=True)
    return sampled_specs

  def load_module_spec(self, module_spec: ModuleSpec) -> LoadedModuleSpec:
    with tf.io.gfile.GFile(
        os.path.join(self._base_dir, module_spec.name + '.bc'), 'rb') as f:
      module_bytes = f.read()
    thinlto_bytes = None
    if module_spec.has_thinlto:
      with tf.io.gfile.GFile(
          os.path.join(self._base_dir, module_spec.name + '.thinlto.bc'),
          'rb') as f:
        thinlto_bytes = f.read()
    return LoadedModuleSpec(
        name=module_spec.name,
        loaded_ir=module_bytes,
        loaded_thinlto_index=thinlto_bytes,
        orig_options=module_spec.command_line)

  @property
  def module_specs(self):
    return self._module_specs

  def __len__(self):
    return len(self._module_specs)


def create_corpus_for_testing(location: str,
                              elements: List[ModuleSpec],
                              cmdline: Tuple[str, ...] = ('-cc1',),
                              cmdline_is_override=False,
                              is_thinlto=False,
                              **kwargs) -> Corpus:
  os.makedirs(location, exist_ok=True)
  for element in elements:
    with tf.io.gfile.GFile(os.path.join(location, element.name + '.bc'),
                           'wb') as f:
      f.write(bytes([1] * element.size))
    if not cmdline_is_override:
      with tf.io.gfile.GFile(
          os.path.join(location, element.name + '.cmd'), 'w') as f:
        f.write('\0'.join(cmdline))
    if is_thinlto:
      with tf.io.gfile.GFile(
          os.path.join(location, element.name + '.thinlto.bc'), 'w') as f:
        f.write('')

  corpus_description = {
      'modules': [e.name for e in elements],
      'has_thinlto': is_thinlto,
  }
  if cmdline_is_override:
    corpus_description['global_command_override'] = cmdline
  with tf.io.gfile.GFile(
      os.path.join(location, 'corpus_description.json'), 'w') as f:
    f.write(json.dumps(corpus_description))
  return Corpus(data_path=location, **kwargs)
