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
"""Tests for the ModuleSpec dataclass and its utility functions."""
# pylint: disable=protected-access

import tensorflow as tf

from compiler_opt.rl import corpus


class CommandParsingTest(tf.test.TestCase):

  def test_thinlto_file(self):
    data = ['-cc1', '-foo', '-bar=baz']
    argfile = self.create_tempfile(content='\0'.join(data))
    self.assertEqual(
        corpus._load_and_parse_command(argfile.full_path, 'my_file.bc'),
        ['-cc1', '-foo', '-bar=baz', '-x', 'ir', 'my_file.bc'])
    self.assertEqual(
        corpus._load_and_parse_command(argfile.full_path, 'my_file.bc',
                                       'the_index.bc'),
        [
            '-cc1', '-foo', '-bar=baz', '-x', 'ir', 'my_file.bc',
            '-fthinlto-index=the_index.bc', '-mllvm', '-thinlto-assume-merged'
        ])

  def test_deletion(self):
    delete_compilation_flags = ('-split-dwarf-file', '-split-dwarf-output',
                                '-fthinlto-index', '-fprofile-sample-use',
                                '-fprofile-remapping-file')
    data = [
        '-cc1', '-fthinlto-index=bad', '-split-dwarf-file', '/tmp/foo.dwo',
        '-split-dwarf-output', 'somepath/some.dwo'
    ]
    argfile = self.create_tempfile(content='\0'.join(data))
    self.assertEqual(
        corpus._load_and_parse_command(
            argfile.full_path, 'hi.bc', delete_flags=delete_compilation_flags),
        ['-cc1', '-x', 'ir', 'hi.bc'])

  def test_addition(self):
    additional_flags = ('-fix-all-bugs',)
    data = ['-cc1']
    argfile = self.create_tempfile(content='\0'.join(data))
    self.assertEqual(
        corpus._load_and_parse_command(
            argfile.full_path, 'hi.bc', additional_flags=additional_flags),
        ['-cc1', '-x', 'ir', 'hi.bc', '-fix-all-bugs'])

  def test_modification(self):
    delete_compilation_flags = ('-split-dwarf-file', '-split-dwarf-output',
                                '-fthinlto-index', '-fprofile-sample-use',
                                '-fprofile-remapping-file')
    additional_flags = ('-fix-all-bugs',)
    data = [
        '-cc1', '-fthinlto-index=bad', '-split-dwarf-file', '/tmp/foo.dwo',
        '-split-dwarf-output', 'somepath/some.dwo'
    ]
    argfile = self.create_tempfile(content='\0'.join(data))
    self.assertEqual(
        corpus._load_and_parse_command(
            argfile.full_path,
            'hi.bc',
            delete_flags=delete_compilation_flags,
            additional_flags=additional_flags),
        ['-cc1', '-x', 'ir', 'hi.bc', '-fix-all-bugs'])


class HasThinLTOIndexTest(tf.test.TestCase):

  def test_exists(self):
    tempdir = self.create_tempdir()
    tempdir.create_file(file_path='a.thinlto.bc')
    self.assertTrue(corpus.has_thinlto_index(iter([tempdir.full_path + '/a'])))

  def test_not_exists(self):
    self.assertFalse(
        corpus.has_thinlto_index(iter(['this#file$cant:possibly^exist'])))


class ModuleSpecTest(tf.test.TestCase):

  def test_cmd(self):
    data = ['-cc1', '-foo', '-bar=baz']
    argfile = self.create_tempfile(file_path='a.cmd', content='\0'.join(data))
    self.assertEqual(
        corpus._load_and_parse_command(argfile.full_path,
                                       argfile.full_path[:-4] + '.bc'),
        corpus.ModuleSpec(name=argfile.full_path[:-4]).cmd())
    self.assertEqual(
        corpus._load_and_parse_command(argfile.full_path,
                                       argfile.full_path[:-4] + '.bc',
                                       argfile.full_path[:-4] + '.thinlto.bc'),
        corpus.ModuleSpec(name=argfile.full_path[:-4], has_thinlto=True).cmd())
    self.assertEqual(
        corpus._load_and_parse_command(
            argfile.full_path,
            argfile.full_path[:-4] + '.bc',
            additional_flags=('-O5',)),
        corpus.ModuleSpec(name=argfile.full_path[:-4]).cmd(
            additional_flags=('-O5',)))
    self.assertEqual(
        corpus._load_and_parse_command(
            argfile.full_path,
            argfile.full_path[:-4] + '.bc',
            delete_flags=('-bar',)),
        corpus.ModuleSpec(name=argfile.full_path[:-4]).cmd(
            delete_flags=('-bar',)))
    self.assertEqual(
        corpus._load_and_parse_command(
            argfile.full_path,
            argfile.full_path[:-4] + '.bc',
            additional_flags=('-O5',),
            delete_flags=('-bar',)),
        corpus.ModuleSpec(name=argfile.full_path[:-4]).cmd(
            additional_flags=('-O5',), delete_flags=('-bar',)))
