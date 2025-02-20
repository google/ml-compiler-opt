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
import multiprocessing
import time
from unittest import mock
import subprocess
import os
import tempfile
from absl.testing import flagsaver
from absl.testing import parameterized

import tensorflow as tf
import numpy as np

from compiler_opt.rl import env, log_reader
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

  def get_cmdline(self, clang_path: str, base_args: list[str],
                  interactive_base_path: str | None,
                  working_dir: str) -> list[str]:
    if interactive_base_path:
      interactive_args = [
          f'--interactive={interactive_base_path}',
      ]
    else:
      interactive_args = []
    return [clang_path] + base_args + interactive_args

  def get_module_scores(self, working_dir: str) -> dict[str, float]:
    return {'default': 47}


# This mocks subprocess.Popen for interactive clang sessions
@contextlib.contextmanager
def mock_interactive_clang(cmdline, stderr, stdout):
  del stderr
  del stdout
  # do basic argument parsing
  fname = None
  for arg in cmdline:
    if arg.startswith('--interactive='):
      fname = arg[len('--interactive='):]
      break

  class MockProcess:

    def wait(self, timeout):
      pass

    def kill(self):
      pass

  if not fname:
    yield MockProcess()
    return
  # Create the fds for the pipes
  # (the env doesn't create the files, it assumes they are opened by clang)
  with io.FileIO(fname + '.out', 'wb+') as f_out:
    with io.FileIO(fname + '.in', 'rb+') as f_in:
      del f_in
      writer = log_reader_test.LogTestExampleBuilder(opened_file=f_out)
      # Write the header describing the features/rewards
      writer.write_header({
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
      })
      writer.write_newline()

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
          example_writer = log_reader_test.LogTestExampleBuilder(
              opened_file=f_out)
          example_writer.write_context_marker(f'context_{self._counter}')
          example_writer.write_observation_marker(0)
          example_writer.write_buff([self._counter], ctypes.c_int64)
          example_writer.write_newline()
          example_writer.write_outcome_marker(0)
          example_writer.write_buff([3.14], ctypes.c_float)
          example_writer.write_newline()
          self._counter += 1
          return None

      yield MockInteractiveProcess()


class ClangSessionTest(tf.test.TestCase):

  @mock.patch('subprocess.Popen')
  def test_clang_session(self, mock_popen):
    mock_task = MockTask()
    with env.clang_session(
        _CLANG_PATH, _MOCK_MODULE, MockTask,
        interactive=False) as clang_session:
      del clang_session
      cmdline = mock_task.get_cmdline(_CLANG_PATH,
                                      list(_MOCK_MODULE.orig_options), None,
                                      '/tmp/mock/tmp/file')
      mock_popen.assert_called_once_with(
          cmdline, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

  @mock.patch('subprocess.Popen')
  def test_interactive_clang_session(self, mock_popen):
    mock_popen.side_effect = mock_interactive_clang

    with env.clang_session(
        _CLANG_PATH, _MOCK_MODULE, MockTask, interactive=True) as clang_session:
      for idx in range(_NUM_STEPS):
        obs = clang_session.get_observation()
        self.assertEqual(
            obs.obs['times_called'],
            np.array([idx], dtype=np.int64),
        )
        self.assertEqual(obs.context, f'context_{idx}')
      mock_popen.assert_called_once()

  @mock.patch('subprocess.Popen')
  def test_interactive_clang_temp_dir(self, mock_popen):
    mock_popen.side_effect = mock_interactive_clang
    working_dir = None

    with env.clang_session(
        _CLANG_PATH, _MOCK_MODULE, MockTask, interactive=True) as clang_session:
      for _ in range(_NUM_STEPS):
        obs = clang_session.get_observation()
        working_dir = obs.working_dir
        self.assertEqual(os.path.exists(working_dir), True)
    self.assertEqual(os.path.exists(working_dir), False)

    with tempfile.TemporaryDirectory() as td:
      with flagsaver.flagsaver(
          (env.compilation_runner._EXPLICIT_TEMPS_DIR, td)):  # pylint: disable=protected-access
        with env.clang_session(
            _CLANG_PATH, _MOCK_MODULE, MockTask,
            interactive=True) as clang_session:
          for _ in range(_NUM_STEPS):
            obs = clang_session.get_observation()
            working_dir = obs.working_dir
            self.assertEqual(os.path.exists(working_dir), True)
        self.assertEqual(os.path.exists(working_dir), True)


class PipelineCommsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'write',
          'method': 'open_write_pipe'
      }, {
          'testcase_name': 'read',
          'method': 'open_read_pipe'
      })
  def test_pipe_timeout_open(self, method):
    slept = multiprocessing.Event()

    def _sleep():
      time.sleep(3600)
      slept.set()

    with tempfile.TemporaryDirectory() as td:
      fname = os.path.join(td, 'something')
      os.mkfifo(fname, 0o666)
      proc = multiprocessing.Process(target=_sleep)
      proc.start()
      with self.assertRaises(TimeoutError):
        with getattr(env, method)(fname, timeout=5):
          self.fail()
      proc.kill()
      proc.join()
      self.assertFalse(slept.is_set())

  @parameterized.named_parameters({
      'testcase_name': 'read',
      'method': 'read'
  }, {
      'testcase_name': 'readline',
      'method': 'readline'
  })
  def test_read_pipe_timeout_read(self, method):
    slept = multiprocessing.Event()

    def _open_then_sleep(fname):
      with open(fname, 'wb'):
        time.sleep(3600)
        slept.set()

    with tempfile.TemporaryDirectory() as td:
      fname = os.path.join(td, 'something')
      os.mkfifo(fname, 0o666)
      proc = multiprocessing.Process(target=_open_then_sleep, args=(fname,))
      proc.start()
      with env.open_read_pipe(fname, timeout=5) as read_pipe:
        with self.assertRaises(TimeoutError):
          getattr(read_pipe, method)()
      proc.kill()
      proc.join()
      self.assertFalse(slept.is_set())

  def test_write_pipeline_timeout_open(self):
    slept = multiprocessing.Event()

    def _sleep():
      time.sleep(3600)
      slept.set()

    with tempfile.TemporaryDirectory() as td:
      fname = os.path.join(td, 'something')
      os.mkfifo(fname, 0o666)
      proc = multiprocessing.Process(target=_sleep)
      proc.start()
      with self.assertRaises(TimeoutError):
        with env.open_write_pipe(fname, timeout=5):
          self.fail()
      proc.kill()
      proc.join()
      self.assertFalse(slept.is_set())

  def test_process_fails_to_open_writer(self):
    slept = multiprocessing.Event()

    def _sleep():
      time.sleep(3600)
      slept.set()

    with tempfile.TemporaryDirectory() as td:
      reader = os.path.join(td, 'reader')
      writer = os.path.join(td, 'writer')
      os.mkfifo(reader, 0o666)
      os.mkfifo(writer, 0o666)
      proc = multiprocessing.Process(target=_sleep)
      proc.start()
      with self.assertRaises(TimeoutError):
        with env.interactive_session(
            reader_name=reader, writer_name=writer, timeout=5):
          self.fail()
      self.assertFalse(slept.is_set())
      proc.kill()
      proc.join()

  def test_process_fails_to_answer(self):
    post_sleep_event = multiprocessing.Event()
    opened_event = multiprocessing.Event()

    with tempfile.TemporaryDirectory() as td:
      reader = os.path.join(td, 'reader')
      writer = os.path.join(td, 'writer')
      os.mkfifo(reader, 0o666)
      os.mkfifo(writer, 0o666)

      def _the_process():
        with open(writer, 'rb'):
          opened_event.set()
          pass
        time.sleep(3600)
        post_sleep_event.set()

      proc = multiprocessing.Process(target=_the_process)
      proc.start()
      with self.assertRaises(TimeoutError):
        with env.interactive_session(
            reader_name=reader, writer_name=writer, timeout=5):
          pass
      self.assertTrue(opened_event.is_set())
      self.assertFalse(post_sleep_event.is_set())
      proc.kill()
      proc.join()

  def test_process_quits_midway(self):
    post_sleep_event = multiprocessing.Event()
    opened_event = multiprocessing.Event()
    wrote_event = multiprocessing.Event()

    with tempfile.TemporaryDirectory() as td:
      reader = os.path.join(td, 'reader')
      writer = os.path.join(td, 'writer')
      os.mkfifo(reader, 0o666)
      os.mkfifo(writer, 0o666)

      def _the_process():
        with open(writer, 'rb'):
          with open(reader, 'wb') as out:
            opened_event.set()
            w = log_reader_test.LogTestExampleBuilder(opened_file=out)
            w.write_header({
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
                }
            })
            w.write_newline()
            w.write_context_marker('hello')
            w.write_observation_marker(0)
            w.write_buff([1], ctypes.c_int16)
            out.flush()
            wrote_event.set()
            time.sleep(3600)
        post_sleep_event.set()

      proc = multiprocessing.Process(target=_the_process)
      proc.start()
      with env.interactive_session(
          reader_name=reader, writer_name=writer, timeout=10) as (read_pipe, _):
        with self.assertRaises(IOError):
          for _ in log_reader.read_log_from_file(read_pipe):
            self.fail()

      self.assertTrue(opened_event.is_set())
      self.assertTrue(wrote_event.is_set())
      self.assertFalse(post_sleep_event.is_set())
      proc.kill()
      proc.join()

  def test_process_stops_talking_back(self):
    post_sleep_event = multiprocessing.Event()
    opened_event = multiprocessing.Event()
    wrote_event = multiprocessing.Event()

    with tempfile.TemporaryDirectory() as td:
      reader = os.path.join(td, 'reader')
      writer = os.path.join(td, 'writer')
      os.mkfifo(reader, 0o666)
      os.mkfifo(writer, 0o666)

      def _the_process():
        with open(writer, 'rb'):
          with open(reader, 'wb') as out:
            opened_event.set()
            w = log_reader_test.LogTestExampleBuilder(opened_file=out)
            w.write_header({
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
                }
            })
            w.write_newline()
            w.write_context_marker('hello')
            w.write_observation_marker(0)
            w.write_buff([1], ctypes.c_int64)
            w.write_newline()
            w.write_outcome_marker(0)
            w.write_buff([3.14], ctypes.c_float)
            w.write_newline()
            out.flush()
            wrote_event.set()
            time.sleep(3600)
        post_sleep_event.set()

      proc = multiprocessing.Process(target=_the_process)
      proc.start()
      read_count = 0
      with self.assertRaises(TimeoutError):
        with env.interactive_session(
            reader_name=reader, writer_name=writer,
            timeout=10) as (read_pipe, _):
          for obs in log_reader.read_log_from_file(read_pipe):
            self.assertIsInstance(obs, log_reader.ObservationRecord)
            read_count += 1
      self.assertEqual(read_count, 1)
      self.assertTrue(opened_event.is_set())
      self.assertTrue(wrote_event.is_set())
      self.assertFalse(post_sleep_event.is_set())

      proc.kill()
      proc.join()


class MLGOEnvironmentTest(tf.test.TestCase):

  @mock.patch('subprocess.Popen')
  def test_env(self, mock_popen):
    mock_popen.side_effect = mock_interactive_clang

    test_env = env.MLGOEnvironmentBase(
        clang_path=_CLANG_PATH,
        task_type=MockTask,
        obs_spec={},
        action_spec={},
    )

    for env_itr in range(3):
      del env_itr
      step = test_env.reset(_MOCK_MODULE)
      self.assertEqual(step.step_type, env.StepType.FIRST)

      for step_itr in range(_NUM_STEPS - 1):
        del step_itr
        step = test_env.step(np.array([1], dtype=np.int64))
        self.assertEqual(step.step_type, env.StepType.MID)

      step = test_env.step(np.array([1], dtype=np.int64))
      self.assertEqual(step.step_type, env.StepType.LAST)
      self.assertNotEqual(test_env._iclang, test_env._clang)  # pylint: disable=protected-access

  @mock.patch('subprocess.Popen')
  def test_env_interactive_only(self, mock_popen):
    mock_popen.side_effect = mock_interactive_clang

    test_env = env.MLGOEnvironmentBase(
        clang_path=_CLANG_PATH,
        task_type=MockTask,
        obs_spec={},
        action_spec={},
        interactive_only=True,
    )

    for env_itr in range(3):
      del env_itr
      step = test_env.reset(_MOCK_MODULE)
      self.assertEqual(step.step_type, env.StepType.FIRST)

      for step_itr in range(_NUM_STEPS - 1):
        del step_itr
        step = test_env.step(np.array([1], dtype=np.int64))
        self.assertEqual(step.step_type, env.StepType.MID)

      step = test_env.step(np.array([1], dtype=np.int64))
      self.assertEqual(step.step_type, env.StepType.LAST)
      self.assertEqual(test_env._iclang, test_env._clang)  # pylint: disable=protected-access


if __name__ == '__main__':
  tf.test.main()
