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
"""Tests for corpus and related concepts."""
# pylint: disable=protected-access
import os
import re

import tensorflow as tf

from compiler_opt.rl import corpus


class CommandParsingTest(tf.test.TestCase):

  def test_deletion(self):
    self.assertEqual(
        ('-cc1',),
        corpus._apply_cmdline_filters(
            orig_options=('-cc1', '-fthinlto-index=bad', '-split-dwarf-file',
                          '/tmp/foo.dwo', '-split-dwarf-output',
                          'somepath/some.dwo'),
            delete_flags=('-split-dwarf-file', '-split-dwarf-output',
                          '-fthinlto-index', '-fprofile-sample-use',
                          '-fprofile-remapping-file')))

    # OK to not match deletion flags
    self.assertEqual(
        ('-cc1',),
        corpus._apply_cmdline_filters(
            orig_options=('-cc1',),
            delete_flags=('-split-dwarf-file', '-split-dwarf-output',
                          '-fthinlto-index', '-fprofile-sample-use',
                          '-fprofile-remapping-file')))

  def test_addition(self):
    self.assertEqual(
        ('-cc1', '-fix-all-bugs', '-something={context.module_full_path}'),
        corpus._apply_cmdline_filters(
            orig_options=('-cc1',),
            additional_flags=('-fix-all-bugs',
                              '-something={context.module_full_path}')))

  def test_replacement(self):

    # if we expect to be able to replace a flag, and it's not in the original
    # cmdline, raise.
    with self.assertRaises(
        ValueError,
        msg='flags that were expected to be replaced were not found'):
      corpus._apply_cmdline_filters(
          orig_options=('-cc1',),
          replace_flags={'-replace-me': '{context.some_field}.replaced'})

    self.assertEqual(
        ('-cc1', '-replace-me={context.some_field}.replaced'),
        corpus._apply_cmdline_filters(
            orig_options=('-cc1', '-replace-me=some_value'),
            replace_flags={'-replace-me': '{context.some_field}.replaced'}))
    # variant without '='
    self.assertEqual(
        ('-cc1', '-replace-me', '{context.some_field}.replaced'),
        corpus._apply_cmdline_filters(
            orig_options=('-cc1', '-replace-me', 'some_value'),
            replace_flags={'-replace-me': '{context.some_field}.replaced'}))


class ModuleSpecTest(tf.test.TestCase):

  def test_loadable_spec(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[corpus.ModuleSpec(name='smth', size=1)])
    lms = cps.load_module_spec(cps.module_specs[0])
    corpdir2 = self.create_tempdir()
    fqcmd = lms.build_command_line(corpdir2)
    bc_loc = os.path.join(corpdir2, 'smth', 'input.bc')
    self.assertEqual(('-cc1', '-x', 'ir', f'{bc_loc}'), fqcmd)
    with tf.io.gfile.GFile(bc_loc, 'rb') as f:
      self.assertEqual(f.readlines(), [bytes([1])])


class CorpusTest(tf.test.TestCase):

  def test_constructor(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[corpus.ModuleSpec(name='1', size=1)],
        additional_flags=('-add',))
    self.assertEqual(cps.module_specs, (corpus.ModuleSpec(
        name='1',
        size=1,
        command_line=('-cc1', '-x', 'ir', '{context.module_full_path}', '-add'),
        has_thinlto=False),))
    self.assertEqual(len(cps), 1)

  def test_invalid_args(self):
    with self.assertRaises(
        ValueError, msg='-cc1 flag not present in .cmd file'):
      corpus.create_corpus_for_testing(
          location=self.create_tempdir(),
          elements=[corpus.ModuleSpec(name='smol', size=1)],
          cmdline=('-hi',))

    with self.assertRaises(
        ValueError, msg='do not use add/delete flags to replace'):
      corpus.create_corpus_for_testing(
          location=self.create_tempdir(),
          elements=[corpus.ModuleSpec(name='smol', size=1)],
          cmdline=('-cc1',),
          additional_flags=('-fsomething',),
          replace_flags={'-fsomething': 'does not matter'})

    with self.assertRaises(
        ValueError, msg='do not use add/delete flags to replace'):
      corpus.create_corpus_for_testing(
          location=self.create_tempdir(),
          elements=[corpus.ModuleSpec(name='smol', size=1)],
          cmdline=('-cc1',),
          additional_flags=('-fsomething=new_value',),
          replace_flags={'-fsomething': 'does not matter'})

    with self.assertRaises(
        ValueError, msg='do not use add/delete flags to replace'):
      corpus.create_corpus_for_testing(
          location=self.create_tempdir(),
          elements=[corpus.ModuleSpec(name='smol', size=1)],
          cmdline=('-cc1',),
          delete_flags=('-fsomething',),
          replace_flags={'-fsomething': 'does not matter'})

    with self.assertRaises(
        ValueError, msg='do not use add/delete flags to replace'):
      corpus.create_corpus_for_testing(
          location=self.create_tempdir(),
          elements=[corpus.ModuleSpec(name='smol', size=1)],
          cmdline=('-cc1',),
          additional_flags=('-fsomething',),
          delete_flags=('-fsomething',))

    with self.assertRaises(
        ValueError, msg='do not use add/delete flags to replace'):
      corpus.create_corpus_for_testing(
          location=self.create_tempdir(),
          elements=[corpus.ModuleSpec(name='smol', size=1)],
          cmdline=('-cc1',),
          additional_flags=('-fsomething=value',),
          delete_flags=('-fsomething',))

  def test_empty_module_list(self):
    location = self.create_tempdir()
    with self.assertRaises(
        ValueError,
        msg=f'{location}\'s corpus_description contains no modules.'):
      corpus.create_corpus_for_testing(
          location=self.create_tempdir(), elements=[], cmdline=('-hello',))

  def test_ctor_thinlto(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[corpus.ModuleSpec(name='smol', size=1)],
        cmdline=('-cc1', '-fthinlto-index=foo'),
        is_thinlto=True)
    self.assertIn('-fthinlto-index={context.thinlto_full_path}',
                  cps.module_specs[0].command_line)
    self.assertEqual(cps.module_specs[0].command_line[-5:],
                     ('-x', 'ir', '{context.module_full_path}', '-mllvm',
                      '-thinlto-assume-merged'))

  def test_braces_in_cmd(self):
    corpusdir = self.create_tempdir()
    cps = corpus.create_corpus_for_testing(
        location=corpusdir,
        elements=[corpus.ModuleSpec(name='somename', size=1)],
        cmdline=('-cc1', r'-DMACRO(expr)=do {} while(0)'),
        additional_flags=('-additional_flag={context.module_full_path}',))
    mod_spec = cps.module_specs[0]
    loaded_spec = cps.load_module_spec(mod_spec)
    corpdir2 = self.create_tempdir()
    final_cmdline = loaded_spec.build_command_line(corpdir2)
    bcpath = os.path.join(corpdir2, 'somename/input.bc')
    self.assertEqual(final_cmdline,
                     ('-cc1', r'-DMACRO(expr)=do {} while(0)', '-x', 'ir',
                      bcpath, f'-additional_flag={bcpath}'))

  def test_cmd_override_thinlto(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[corpus.ModuleSpec(name='smol', size=1)],
        cmdline=(),
        cmdline_is_override=True,
        is_thinlto=True)
    self.assertNotIn('-fthinlto-index', cps.module_specs[0].command_line)
    self.assertEqual(cps.module_specs[0].command_line[-6:],
                     ('-x', 'ir', '{context.module_full_path}',
                      '-fthinlto-index={context.thinlto_full_path}', '-mllvm',
                      '-thinlto-assume-merged'))

    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[corpus.ModuleSpec(name='smol', size=1)],
        cmdline=('-something',),
        cmdline_is_override=True,
        is_thinlto=True)
    self.assertIn('-fthinlto-index={context.thinlto_full_path}',
                  cps.module_specs[0].command_line)
    self.assertEqual(cps.module_specs[0].command_line[-6:],
                     ('-x', 'ir', '{context.module_full_path}',
                      '-fthinlto-index={context.thinlto_full_path}', '-mllvm',
                      '-thinlto-assume-merged'))
    self.assertIn('-something', cps.module_specs[0].command_line)

  def test_sample(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[
            corpus.ModuleSpec(name='smol', size=1),
            corpus.ModuleSpec(name='middle', size=200),
            corpus.ModuleSpec(name='largest', size=500),
            corpus.ModuleSpec(name='small', size=100)
        ])
    sample = cps.sample(4, sort=True)
    self.assertLen(sample, 4)
    self.assertEqual(sample[0].name, 'largest')
    self.assertEqual(sample[1].name, 'middle')
    self.assertEqual(sample[2].name, 'small')
    self.assertEqual(sample[3].name, 'smol')

  def test_sample_without_replacement(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[
            corpus.ModuleSpec(name='smol', size=1),
            corpus.ModuleSpec(name='middle', size=200),
            corpus.ModuleSpec(name='largest', size=500),
            corpus.ModuleSpec(name='small', size=100)
        ],
        sampler_type=corpus.SamplerWithoutReplacement)
    samples = []
    samples.extend(cps.sample(1, sort=True))
    self.assertLen(samples, 1)
    samples.extend(cps.sample(1, sort=True))
    self.assertLen(samples, 2)
    # Can't sample 3 from the corpus because there are only 2 elements left
    with self.assertRaises(corpus.CorpusExhaustedError):
      samples.extend(cps.sample(3, sort=True))
    # But, we can sample exactly 2 more
    self.assertLen(samples, 2)
    samples.extend(cps.sample(2, sort=True))
    self.assertLen(samples, 4)
    with self.assertRaises(corpus.CorpusExhaustedError):
      samples.extend(cps.sample(1, sort=True))
    samples.sort(key=lambda m: m.size, reverse=True)
    self.assertEqual(samples[0].name, 'largest')
    self.assertEqual(samples[1].name, 'middle')
    self.assertEqual(samples[2].name, 'small')
    self.assertEqual(samples[3].name, 'smol')

  def test_filter(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[
            corpus.ModuleSpec(name='smol', size=1),
            corpus.ModuleSpec(name='middle', size=200),
            corpus.ModuleSpec(name='largest', size=500),
            corpus.ModuleSpec(name='small', size=100)
        ],
        module_filter=lambda name: re.compile(r'.+l').match(name))
    sample = cps.sample(999, sort=True)
    self.assertLen(sample, 3)
    self.assertEqual(sample[0].name, 'middle')
    self.assertEqual(sample[1].name, 'small')
    self.assertEqual(sample[2].name, 'smol')

  def test_sample_zero(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[corpus.ModuleSpec(name='smol', size=1)])

    self.assertRaises(ValueError, cps.sample, 0)
    self.assertRaises(ValueError, cps.sample, -213213213)

  def test_bucket_sample(self):
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[corpus.ModuleSpec(name=f'{i}', size=i) for i in range(100)])
    # Odds of passing once by pure luck with random.sample: 1.779e-07
    # Try 32 times, for good measure.
    for i in range(32):
      sample = cps.sample(k=20, sort=True)
      self.assertLen(sample, 20)
      for idx, s in enumerate(sample):
        # Each bucket should be size 5, since n=20 in the sampler
        self.assertEqual(s.size // 5, 19 - idx)

  def test_bucket_sample_all(self):
    # Make sure we can sample everything, even if it's not divisible by the
    # `n` in SamplerBucketRoundRobin.
    # Create corpus with a prime number of modules.
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[corpus.ModuleSpec(name=f'{i}', size=i) for i in range(101)])

    # Try 32 times, for good measure.
    for i in range(32):
      sample = cps.sample(k=101, sort=True)
      self.assertLen(sample, 101)
      for idx, s in enumerate(sample):
        # Since everything is sampled, it should be in perfect order.
        self.assertEqual(s.size, 100 - idx)

  def test_bucket_sample_small(self):
    # Make sure we can sample even when k < n.
    cps = corpus.create_corpus_for_testing(
        location=self.create_tempdir(),
        elements=[corpus.ModuleSpec(name=f'{i}', size=i) for i in range(100)])

    # Try all 19 possible values 0 < i < n
    for i in range(1, 20):
      sample = cps.sample(k=i, sort=True)
      self.assertLen(sample, i)


if __name__ == '__main__':
  tf.test.main()
