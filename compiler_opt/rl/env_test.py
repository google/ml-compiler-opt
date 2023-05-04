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
"""Tests for compiler_opt.rl.env."""

import io
import contextlib
import ctypes
from unittest import mock
import subprocess

from typing import Dict, List

import tensorflow as tf
import numpy as np

from compiler_opt.rl import env
from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader_test

_CLANG_PATH = '/test/clang/path'

_MOCK_MODULE = corpus.LoadedModuleSpec(
    name='module',
    loaded_ir=b'asdf',
    orig_options=('--opt_a', 'a', '--opt_b', 'b'),
)

_NUM_STEPS = 10


class MockTask(env.MLGOTask):
  """Implementation of mock task for testing."""

  def get_clang_args(self) -> List[str]:
    return ['a', 'b', 'c']

  def get_interactive_clang_args(self, interactive_base_path: str) -> List[str]:
    return [f'--interactive={interactive_base_path}']

  def get_module_scores(self, compiled_module_path: str) -> Dict[str, float]:
    return {'default': 47}


def mock_task_factory(working_dir: str):
  return MockTask(working_dir)


# This mocks subprocess.Popen for interactive clang sessions
@contextlib.contextmanager
def mock_interactive_clang(cmdline, stderr, stdout):
  del stderr
  del stdout
  # do basic argument parsing
  fname = None
  prev_arg = None
  out_path = None
  for arg in cmdline:
    if prev_arg and prev_arg == '-o':
      out_path = arg
    if arg.startswith('--interactive='):
      fname = arg[len('--interactive='):]
      break
    prev_arg = arg

  assert out_path

  class MockProcess:

    def wait(self, timeout):
      del timeout
      # Need to write the module because the environment checks that it exists
      with open(out_path, 'wt', encoding='utf8') as f:
        f.write('Compiled module')

    def kill(self):
      pass

  if not fname:
    yield MockProcess()
    return
  # Create the fds for the pipes
  # (the env doesn't create the files, it assumes they are opened by clang)
  with io.FileIO(fname + '.out', 'rb+') as f_out:
    # Write the header describing the features/rewards
    f_out.write(
        log_reader_test.json_to_bytes({
            'features': [{
                'name': 'times_called',
                'port': 0,
                'shape': [1],
                'type': 'int64_t',
            },],
            'score': {
                'name': 'reward',
                'port': 0,
                'shape': [1],
                'type': 'float',
            },
        }))
    log_reader_test.write_nl(f_out)

    class MockInteractiveProcess(MockProcess):
      """Mock clang interactive process that writes the log."""

      def __init__(self):
        self._counter = 0

      # We poll the process at every call to get_observation to ensure the
      # clang process is still alive. So here, each time poll() is called,
      # write a new context
      def poll(self):
        if self._counter >= _NUM_STEPS:
          f_out.close()
          return None
        log_reader_test.write_context_marker(f_out, f'context_{self._counter}')
        log_reader_test.write_observation_marker(f_out, 0)
        log_reader_test.write_buff(f_out, [self._counter], ctypes.c_int64)
        log_reader_test.write_nl(f_out)
        log_reader_test.write_outcome_marker(f_out, 0)
        log_reader_test.write_buff(f_out, [3.14], ctypes.c_float)
        log_reader_test.write_nl(f_out)
        self._counter += 1
        return None

    yield MockInteractiveProcess()


class ClangSessionTest(tf.test.TestCase):

  @mock.patch('subprocess.Popen')
  def test_clang_session(self, mock_popen):
    mock_task = mock_task_factory('')
    with env.clang_session(
        _CLANG_PATH, _MOCK_MODULE, mock_task_factory,
        interactive=False) as clang_session:
      cmdline = ([_CLANG_PATH] + list(_MOCK_MODULE.orig_options) +
                 mock_task.get_clang_args() +
                 ['-o', clang_session.compiled_module_path])
      mock_popen.assert_called_once_with(
          cmdline, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

  @mock.patch('subprocess.Popen')
  def test_interactive_clang_session(self, mock_popen):
    mock_popen.side_effect = mock_interactive_clang

    with env.clang_session(
        _CLANG_PATH, _MOCK_MODULE, mock_task_factory,
        interactive=True) as clang_session:
      for idx in range(_NUM_STEPS):
        obs = clang_session.get_observation()
        self.assertEqual(
            obs[env.OBS_KEY]['times_called'],
            np.array([idx], dtype=np.int64),
        )
        self.assertEqual(obs[env.CONTEXT_KEY], f'context_{idx}')
      mock_popen.assert_called_once()


class MLGOEnvironmentTest(tf.test.TestCase):

  @mock.patch('subprocess.Popen')
  def test_env(self, mock_popen):
    mock_popen.side_effect = mock_interactive_clang

    test_env = env.MLGOEnvironmentBase(
        clang_path=_CLANG_PATH,
        task_factory=mock_task_factory,
        obs_spec={},
        action_spec={},
    )

    for env_itr in range(3):
      del env_itr
      step = test_env.reset(_MOCK_MODULE)
      self.assertEqual(step[env.STEP_TYPE_KEY], env.FIRST_STEP_STR)

      for step_itr in range(_NUM_STEPS - 1):
        del step_itr
        step = test_env.step(np.array([1], dtype=np.int64))
        self.assertEqual(step[env.STEP_TYPE_KEY], env.MID_STEP_STR)

      step = test_env.step(np.array([1], dtype=np.int64))
      self.assertEqual(step[env.STEP_TYPE_KEY], env.LAST_STEP_STR)


if __name__ == '__main__':
  tf.test.main()
