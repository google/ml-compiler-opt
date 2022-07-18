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
import os

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


class LoadModulePathsTest(tf.test.TestCase):

  def test_exists(self):
    data = ['1', '2', '3']
    tempdir = self.create_tempdir()
    tempdir.create_file(file_path='module_paths', content='\n'.join(data))
    read_data = corpus._load_module_paths(tempdir.full_path)
    self.assertEqual([os.path.join(tempdir.full_path, p) for p in data],
                     read_data)

  def test_empty(self):
    tempdir = self.create_tempdir()
    tempdir.create_file(file_path='module_paths')
    self.assertRaises(ValueError, corpus._load_module_paths, tempdir.full_path)

  def test_not_exists_file(self):
    tempdir = self.create_tempdir()
    self.assertRaises(FileNotFoundError, corpus._load_module_paths,
                      tempdir.full_path)

  def test_not_exists_dir(self):
    self.assertRaises(FileNotFoundError, corpus._load_module_paths,
                      '/this#path$cant:possibly^exist')


class HasThinLTOIndexTest(tf.test.TestCase):

  def test_exists(self):
    tempdir = self.create_tempdir()
    tempdir.create_file(file_path='a.thinlto.bc')
    self.assertTrue(corpus._has_thinlto_index(iter([tempdir.full_path + '/a'])))

  def test_not_exists(self):
    self.assertFalse(
        corpus._has_thinlto_index(iter(['this#file$cant:possibly^exist'])))


class ModuleSpecTest(tf.test.TestCase):

  def test_cmd(self):
    ms = corpus.ModuleSpec(_exec_cmd=('-cc1', '-fix-all-bugs'), name='dummy')
    self.assertEqual(ms.name, 'dummy')
    self.assertEqual(ms.cmd(), ['-cc1', '-fix-all-bugs'])

  def test_get_without_thinlto(self):
    data = ['1', '2']
    tempdir = self.create_tempdir()
    tempdir.create_file('module_paths', content='\n'.join(data))
    tempdir.create_file('1.bc')
    tempdir.create_file('1.cmd', content='\0'.join(['-cc1']))
    tempdir.create_file('2.bc')
    tempdir.create_file('2.cmd', content='\0'.join(['-O3']))

    ms_list = corpus.read(tempdir.full_path, additional_flags=('-add',))
    self.assertEqual(len(ms_list), 2)
    ms1 = ms_list[0]
    ms2 = ms_list[1]
    self.assertEqual(ms1.name, tempdir.full_path + '/1')
    self.assertEqual(ms1._exec_cmd,
                     ('-cc1', '-x', 'ir', tempdir.full_path + '/1.bc', '-add'))

    self.assertEqual(ms2.name, tempdir.full_path + '/2')
    self.assertEqual(ms2._exec_cmd,
                     ('-O3', '-x', 'ir', tempdir.full_path + '/2.bc', '-add'))

  def test_get_with_thinlto(self):
    data = ['1', '2']
    tempdir = self.create_tempdir()
    tempdir.create_file('module_paths', content='\n'.join(data))
    tempdir.create_file('1.bc')
    tempdir.create_file('1.thinlto.bc')
    tempdir.create_file(
        '1.cmd', content='\0'.join(['-cc1', '-fthinlto-index=xyz']))
    tempdir.create_file('2.bc')
    tempdir.create_file('2.thinlto.bc')
    tempdir.create_file('2.cmd', content='\0'.join(['-fthinlto-index=abc']))

    ms_list = corpus.read(
        tempdir.full_path,
        additional_flags=('-add',),
        delete_flags=('-fthinlto-index',))
    self.assertEqual(len(ms_list), 2)
    ms1 = ms_list[0]
    ms2 = ms_list[1]
    self.assertEqual(ms1.name, tempdir.full_path + '/1')
    self.assertEqual(ms1._exec_cmd,
                     ('-cc1', '-x', 'ir', tempdir.full_path + '/1.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/1.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))

    self.assertEqual(ms2.name, tempdir.full_path + '/2')
    self.assertEqual(ms2._exec_cmd,
                     ('-x', 'ir', tempdir.full_path + '/2.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/2.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))


if __name__ == '__main__':
  tf.test.main()
