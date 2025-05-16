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
"""Unit tests for group_functions_lib."""

import os

from absl.testing import absltest

from compiler_opt.tools.regalloc_trace import group_functions_lib
from compiler_opt.rl import corpus
from compiler_opt.testing import corpus_test_utils


class GroupFunctionsTest(absltest.TestCase):

  def test_get_chunks_one_command_line(self):
    corpus_folder = self.create_tempdir()
    corpus.create_corpus_for_testing(
        corpus_folder.full_path,
        elements=[
            corpus.ModuleSpec(name='module1', size=5, command_line=('-cc1',)),
            corpus.ModuleSpec(name='module2', size=5, command_line=('-cc1',))
        ])
    corpus_chunks = group_functions_lib.get_chunks(corpus_folder.full_path, (),
                                                   2)
    self.assertDictEqual(
        corpus_chunks, {
            ('-cc1',): [[
                os.path.join(corpus_folder.full_path, 'module1.bc'),
                os.path.join(corpus_folder.full_path, 'module2.bc')
            ]]
        })

  def test_get_chunks_two_command_lines(self):
    corpus_folder = self.create_tempdir()
    corpus.create_corpus_for_testing(
        corpus_folder.full_path,
        elements=[
            corpus.ModuleSpec(name='module1', size=5, command_line=('-cc1',)),
            corpus.ModuleSpec(
                name='module2', size=5, command_line=('-cc1', '-O3'))
        ])
    corpus_chunks = group_functions_lib.get_chunks(corpus_folder.full_path, (),
                                                   2)
    self.assertDictEqual(
        corpus_chunks, {
            ('-cc1',): [[
                os.path.join(corpus_folder.full_path, 'module1.bc'),
            ]],
            ('-cc1', '-O3'):
                [[os.path.join(corpus_folder.full_path, 'module2.bc')]]
        })

  def test_get_chunks_multiple_chunks(self):
    corpus_folder = self.create_tempdir()
    corpus.create_corpus_for_testing(
        corpus_folder.full_path,
        elements=[
            corpus.ModuleSpec(name='module1', size=5, command_line=('-cc1',)),
            corpus.ModuleSpec(name='module2', size=5, command_line=('-cc1',))
        ])
    corpus_chunks = group_functions_lib.get_chunks(corpus_folder.full_path, (),
                                                   1)
    self.assertDictEqual(
        corpus_chunks, {
            ('-cc1',): [[os.path.join(corpus_folder.full_path, 'module1.bc')],
                        [os.path.join(corpus_folder.full_path, 'module2.bc')]]
        })

  def test_get_chunks_multiple_uneven_chunks(self):
    corpus_folder = self.create_tempdir()
    corpus.create_corpus_for_testing(
        corpus_folder.full_path,
        elements=[
            corpus.ModuleSpec(name='module1', size=5, command_line=('-cc1',)),
            corpus.ModuleSpec(name='module2', size=5, command_line=('-cc1',)),
            corpus.ModuleSpec(name='module3', size=5, command_line=('-cc1',)),
        ])
    corpus_chunks = group_functions_lib.get_chunks(corpus_folder.full_path, (),
                                                   2)
    self.assertDictEqual(
        corpus_chunks, {
            ('-cc1',): [[
                os.path.join(corpus_folder.full_path, 'module1.bc'),
                os.path.join(corpus_folder.full_path, 'module3.bc')
            ], [os.path.join(corpus_folder.full_path, 'module2.bc')]]
        })

  def test_combine_chunks(self):
    corpus_folder = self.create_tempdir()
    corpus.create_corpus_for_testing(
        corpus_folder.full_path,
        elements=[
            corpus.ModuleSpec(name='module1', size=5, command_line=('-cc1',)),
            corpus.ModuleSpec(name='module2', size=5, command_line=('-cc1',))
        ])
    corpus_chunks = group_functions_lib.get_chunks(corpus_folder.full_path, (),
                                                   2)
    fake_llvm_link_binary = self.create_tempfile('fake_llvm_link')
    fake_llvm_link_invocations = self.create_tempfile(
        'fake_llvm_link_invocations')
    corpus_test_utils.create_test_binary(fake_llvm_link_binary.full_path,
                                         fake_llvm_link_invocations.full_path,
                                         ['touch $2'])

    output_folder = self.create_tempdir()
    group_functions_lib.combine_chunks(corpus_chunks,
                                       fake_llvm_link_binary.full_path,
                                       output_folder.full_path)
    self.assertContainsSubsequence(
        os.listdir(output_folder.full_path),
        ['corpus_description.json', '0.bc', '0.cmd'])


if __name__ == '__main__':
  absltest.main()
