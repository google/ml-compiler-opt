# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the ModuleSpec dataclass and its utility functions."""
# pylint: disable=protected-access

import tensorflow as tf
import json
import os

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

  def test_override(self):
    cmd_override = ('-fix-all-bugs',)
    data = ['-cc1', '-fthinlto-index=bad']
    argfile = self.create_tempfile(content='\0'.join(data))
    self.assertEqual(
        corpus._load_and_parse_command(
            argfile.full_path,
            'hi.bc',
            cmd_override=cmd_override,
            additional_flags=('-add-bugs',),
            delete_flags=('-fthinlto-index',)),
        ['-fix-all-bugs', '-x', 'ir', 'hi.bc', '-add-bugs'])
    self.assertEqual(
        corpus._load_and_parse_command(
            None, 'hi.bc', cmd_override=cmd_override),
        ['-fix-all-bugs', '-x', 'ir', 'hi.bc'])

  def test_cmd_not_provided(self):
    self.assertRaises(
        ValueError,
        corpus._load_and_parse_command,
        None,
        'hi.bc',
        cmd_override=None,
        additional_flags=('-add-bugs',),
        delete_flags=('-fthinlto-index',))

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

  def test_add_cc1(self):
    data = ['-fix-all-bugs', '-xyz']
    argfile = self.create_tempfile(content='\0'.join(data))
    self.assertEqual(
        corpus._load_and_parse_command(argfile.full_path, 'hi.bc'),
        ['-cc1', '-fix-all-bugs', '-xyz', '-x', 'ir', 'hi.bc'])


class LoadMetadataTest(tf.test.TestCase):

  def test_exists(self):
    data = {'abc': 123}
    metadata_file = self.create_tempfile(content=json.dumps(data))
    read_data = corpus._load_metadata(metadata_file.full_path)
    self.assertEqual(data, read_data)

  def test_not_exists(self):
    read_data = corpus._load_metadata('this#file$cant:possibly^exist')
    self.assertEqual({}, read_data)


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


class HasCmdTest(tf.test.TestCase):

  def test_exists(self):
    tempdir = self.create_tempdir()
    tempdir.create_file(file_path='a.cmd')
    self.assertTrue(corpus._has_cmd(iter([tempdir.full_path + '/a'])))

  def test_not_exists(self):
    self.assertFalse(corpus._has_cmd(iter(['this#file$cant:possibly^exist'])))


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
    ms = corpus.ModuleSpec(
        exec_cmd=('-cc1', '-fix-all-bugs'),
        extra_opts={
            'mllvm': ('-mllvm', '{opt:s}'),
            'std': ('{opt:s}',)
        },
        name='dummy')
    self.assertEqual(ms.name, 'dummy')
    self.assertEqual(ms.cmd(), ['-cc1', '-fix-all-bugs'])
    self.assertEqual(ms.cmd([]), ['-cc1', '-fix-all-bugs'])
    self.assertEqual(
        ms.cmd([('-policy=path',)]), ['-cc1', '-fix-all-bugs', '-policy=path'])
    self.assertEqual(
        ms.cmd([('-mllvm', '-policy=path')]),
        ['-cc1', '-fix-all-bugs', '-mllvm', '-policy=path'])
    self.assertEqual(
        ms.cmd([('-mllvm', '-policy=path'), ('-nomllvm',)]),
        ['-cc1', '-fix-all-bugs', '-mllvm', '-policy=path', '-nomllvm'])
    self.assertRaises(ValueError, ms.cmd, [('-miivm', '-policy=path')])
    self.assertRaises(ValueError, ms.cmd,
                      [('-miivm', '-policy=path', '-extra-opt')])

  def test_get(self):
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
    self.assertEqual(ms1.exec_cmd,
                     ('-cc1', '-x', 'ir', tempdir.full_path + '/1.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/1.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))

    self.assertEqual(ms2.name, tempdir.full_path + '/2')
    self.assertEqual(ms2.exec_cmd,
                     ('-cc1', '-x', 'ir', tempdir.full_path + '/2.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/2.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))

  def test_get_with_metadata(self):
    data = ['1', '2']
    metadata = {'global_command_override': ['-O3', '-qrs']}
    tempdir = self.create_tempdir()
    tempdir.create_file('module_paths', content='\n'.join(data))
    tempdir.create_file('metadata.json', content=json.dumps(metadata))
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
    self.assertEqual(ms1.exec_cmd,
                     ('-O3', '-qrs', '-x', 'ir', tempdir.full_path + '/1.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/1.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))

    self.assertEqual(ms2.name, tempdir.full_path + '/2')
    self.assertEqual(ms2.exec_cmd,
                     ('-O3', '-qrs', '-x', 'ir', tempdir.full_path + '/2.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/2.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))


if __name__ == '__main__':
  tf.test.main()
