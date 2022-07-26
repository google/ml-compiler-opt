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
import json
import os

import tensorflow as tf

from compiler_opt.rl import corpus


class CommandParsingTest(tf.test.TestCase):

  def test_thinlto_file(self):
    data = ['-cc1', '-foo', '-bar=baz']
    argfile = self.create_tempfile(content='\0'.join(data), file_path='hi.cmd')
    module_path = argfile.full_path[:-4]
    self.assertEqual(
        corpus._load_and_parse_command(
            module_path=module_path, has_thinlto=False),
        ['-cc1', '-foo', '-bar=baz', '-x', 'ir', module_path + '.bc'])
    self.assertEqual(
        corpus._load_and_parse_command(
            module_path=module_path, has_thinlto=True), [
                '-cc1', '-foo', '-bar=baz', '-x', 'ir', module_path + '.bc',
                '-fthinlto-index=' + module_path + '.thinlto.bc', '-mllvm',
                '-thinlto-assume-merged'
            ])

  def test_deletion(self):
    delete_compilation_flags = ('-split-dwarf-file', '-split-dwarf-output',
                                '-fthinlto-index', '-fprofile-sample-use',
                                '-fprofile-remapping-file')
    data = [
        '-cc1', '-fthinlto-index=bad', '-split-dwarf-file', '/tmp/foo.dwo',
        '-split-dwarf-output', 'somepath/some.dwo'
    ]
    argfile = self.create_tempfile(content='\0'.join(data), file_path='hi.cmd')
    module_path = argfile.full_path[:-4]
    self.assertEqual(
        corpus._load_and_parse_command(
            module_path=module_path,
            has_thinlto=False,
            delete_flags=delete_compilation_flags),
        ['-cc1', '-x', 'ir', module_path + '.bc'])

  def test_addition(self):
    additional_flags = ('-fix-all-bugs',)
    data = ['-cc1']
    argfile = self.create_tempfile(content='\0'.join(data), file_path='hi.cmd')
    module_path = argfile.full_path[:-4]
    self.assertEqual(
        corpus._load_and_parse_command(
            module_path=module_path,
            has_thinlto=False,
            additional_flags=additional_flags),
        ['-cc1', '-x', 'ir', module_path + '.bc', '-fix-all-bugs'])

  def test_modification(self):
    delete_compilation_flags = ('-split-dwarf-file', '-split-dwarf-output',
                                '-fthinlto-index', '-fprofile-sample-use',
                                '-fprofile-remapping-file')
    additional_flags = ('-fix-all-bugs',)
    data = [
        '-cc1', '-fthinlto-index=bad', '-split-dwarf-file', '/tmp/foo.dwo',
        '-split-dwarf-output', 'somepath/some.dwo'
    ]
    argfile = self.create_tempfile(content='\0'.join(data), file_path='hi.cmd')
    module_path = argfile.full_path[:-4]
    self.assertEqual(
        corpus._load_and_parse_command(
            module_path=module_path,
            has_thinlto=False,
            delete_flags=delete_compilation_flags,
            additional_flags=additional_flags),
        ['-cc1', '-x', 'ir', module_path + '.bc', '-fix-all-bugs'])

  def test_override(self):
    cmd_override = ('-fix-all-bugs',)
    data = ['-cc1', '-fthinlto-index=bad']
    argfile = self.create_tempfile(content='\0'.join(data), file_path='hi.cmd')
    module_path = argfile.full_path[:-4]
    self.assertEqual(
        corpus._load_and_parse_command(
            module_path=module_path,
            has_thinlto=False,
            cmd_override=cmd_override,
            additional_flags=('-add-bugs',),
            delete_flags=('-fthinlto-index',)),
        ['-fix-all-bugs', '-x', 'ir', module_path + '.bc', '-add-bugs'])
    self.assertEqual(
        corpus._load_and_parse_command(
            module_path=os.path.join('this!path#cant$exist/'
                                     'hi'),
            has_thinlto=False,
            cmd_override=cmd_override),
        ['-fix-all-bugs', '-x', 'ir', 'this!path#cant$exist/hi.bc'])

  def test_add_cc1(self):
    data = ['-fix-all-bugs', '-xyz']
    argfile = self.create_tempfile(content='\0'.join(data), file_path='hi.cmd')
    module_path = argfile.full_path[:-4]
    self.assertEqual(
        corpus._load_and_parse_command(
            module_path=module_path, has_thinlto=False),
        ['-cc1', '-fix-all-bugs', '-xyz', '-x', 'ir', module_path + '.bc'])


class ModuleSpecTest(tf.test.TestCase):

  def test_cmd(self):
    ms = corpus.ModuleSpec(exec_cmd=('-cc1', '-fix-all-bugs'), name='dummy')
    self.assertEqual(ms.name, 'dummy')
    self.assertEqual(ms.exec_cmd, ('-cc1', '-fix-all-bugs'))

  def test_get_without_thinlto(self):
    corpus_description = {'modules': ['1', '2'], 'has_thinlto': False}
    tempdir = self.create_tempdir()
    tempdir.create_file(
        'corpus_description.json', content=json.dumps(corpus_description))
    tempdir.create_file('1.bc')
    tempdir.create_file('1.cmd', content='\0'.join(['-cc1']))
    tempdir.create_file('2.bc')
    tempdir.create_file('2.cmd', content='\0'.join(['-O3']))

    ms_list = corpus.build_modulespecs_from_datapath(
        tempdir.full_path, additional_flags=('-add',))
    self.assertEqual(len(ms_list), 2)
    ms1 = ms_list[0]
    ms2 = ms_list[1]
    self.assertEqual(ms1.name, '1')
    self.assertEqual(ms1.exec_cmd,
                     ('-cc1', '-x', 'ir', tempdir.full_path + '/1.bc', '-add'))

    self.assertEqual(ms2.name, '2')
    self.assertEqual(
        ms2.exec_cmd,
        ('-cc1', '-O3', '-x', 'ir', tempdir.full_path + '/2.bc', '-add'))

  def test_get_with_thinlto(self):
    corpus_description = {'modules': ['1', '2'], 'has_thinlto': True}
    tempdir = self.create_tempdir()
    tempdir.create_file(
        'corpus_description.json', content=json.dumps(corpus_description))
    tempdir.create_file('1.bc')
    tempdir.create_file('1.thinlto.bc')
    tempdir.create_file(
        '1.cmd', content='\0'.join(['-cc1', '-fthinlto-index=xyz']))
    tempdir.create_file('2.bc')
    tempdir.create_file('2.thinlto.bc')
    tempdir.create_file('2.cmd', content='\0'.join(['-fthinlto-index=abc']))

    ms_list = corpus.build_modulespecs_from_datapath(
        tempdir.full_path,
        additional_flags=('-add',),
        delete_flags=('-fthinlto-index',))
    self.assertEqual(len(ms_list), 2)
    ms1 = ms_list[0]
    ms2 = ms_list[1]
    self.assertEqual(ms1.name, '1')
    self.assertEqual(ms1.exec_cmd,
                     ('-cc1', '-x', 'ir', tempdir.full_path + '/1.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/1.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))

    self.assertEqual(ms2.name, '2')
    self.assertEqual(ms2.exec_cmd,
                     ('-cc1', '-x', 'ir', tempdir.full_path + '/2.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/2.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))

  def test_get_with_override(self):
    corpus_description = {
        'modules': ['1', '2'],
        'has_thinlto': True,
        'global_command_override': ['-O3', '-qrs']
    }
    tempdir = self.create_tempdir()
    tempdir.create_file(
        'corpus_description.json', content=json.dumps(corpus_description))
    tempdir.create_file('1.bc')
    tempdir.create_file('1.thinlto.bc')
    tempdir.create_file(
        '1.cmd', content='\0'.join(['-cc1', '-fthinlto-index=xyz']))
    tempdir.create_file('2.bc')
    tempdir.create_file('2.thinlto.bc')
    tempdir.create_file('2.cmd', content='\0'.join(['-fthinlto-index=abc']))

    ms_list = corpus.build_modulespecs_from_datapath(
        tempdir.full_path,
        additional_flags=('-add',),
        delete_flags=('-fthinlto-index',))
    self.assertEqual(len(ms_list), 2)
    ms1 = ms_list[0]
    ms2 = ms_list[1]
    self.assertEqual(ms1.name, '1')
    self.assertEqual(ms1.exec_cmd,
                     ('-O3', '-qrs', '-x', 'ir', tempdir.full_path + '/1.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/1.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))

    self.assertEqual(ms2.name, '2')
    self.assertEqual(ms2.exec_cmd,
                     ('-O3', '-qrs', '-x', 'ir', tempdir.full_path + '/2.bc',
                      '-fthinlto-index=' + tempdir.full_path + '/2.thinlto.bc',
                      '-mllvm', '-thinlto-assume-merged', '-add'))


if __name__ == '__main__':
  tf.test.main()
