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
"""Module for running cancellable processes."""

import os
import shlex
import signal
import subprocess
import threading

from absl import flags
from absl import logging

_QUIET = flags.DEFINE_bool(
    'quiet', True, 'Whether or not to compile quietly (hiding info logging)')


class ProcessKilledError(Exception):

  def __init__(self):
    Exception.__init__(self)


def _kill_process_ignore_exceptions(p: 'subprocess.Popen[bytes]'):  # pylint: disable=useless-return
  # kill the process and ignore exceptions. Exceptions would be thrown if the
  # process has already been killed/finished (which is inherently in a race
  # condition with us killing it)
  try:
    p.kill()
    p.wait()
  finally:
    return  # pylint: disable=lost-exception,return-in-finally


class WorkerCancellationManager:
  """A thread-safe object that can be used to signal cancellation.

  This allows killing long-running processes promptly, and thus efficiently
  managing resources.
  """

  def __init__(self, timeout: float | None = None):
    # the queue is filled only by workers, and drained only by the single
    # consumer. we use _done to manage access to the queue. We can then assume
    # empty() is accurate and get() never blocks.
    self._processes = set()
    self._done = False
    self._paused = False
    self._lock = threading.Lock()
    self._timeout = timeout

  def enable(self):
    with self._lock:
      self._done = False

  def register_process(self, p: 'subprocess.Popen[bytes]'):
    """Register a process for potential cancellation."""
    with self._lock:
      if not self._done:
        self._processes.add(p)
        return
    _kill_process_ignore_exceptions(p)

  def kill_all_processes(self):
    """Cancel any pending work."""
    with self._lock:
      self._done = True
    for p in self._processes:
      _kill_process_ignore_exceptions(p)

  def pause_all_processes(self):
    with self._lock:
      if self._paused:
        return
      self._paused = True

      for p in self._processes:
        # used to send the STOP signal; does not actually kill the process
        os.kill(p.pid, signal.SIGSTOP)

  def resume_all_processes(self):
    with self._lock:
      if not self._paused:
        return
      self._paused = False

      for p in self._processes:
        # used to send the CONTINUE signal; does not actually kill the process
        os.kill(p.pid, signal.SIGCONT)

  def unregister_process(self, p: 'subprocess.Popen[bytes]'):
    with self._lock:
      if p in self._processes:
        self._processes.remove(p)

  def __del__(self):
    if len(self._processes) > 0:
      raise RuntimeError('Cancellation manager deleted while containing items.')

  def start_cancellable_process(
      self,
      cmdline: list[str],
      **kwargs,
  ) -> bytes | str | None:
    """Start a cancellable process.

    Args:
      cmdline: the process executable and command line
      **kwargs: keyword arguments to subprocess.Popen (e.g. stdout, env)

    Returns:
      stdout if stdout=subprocess.PIPE was requested, else None.
    Raises:
      CalledProcessError: if the process encounters an error.
      TimeoutExpired: if the process times out.
      ProcessKilledError: if the process was killed via the cancellation token.
    """
    env = kwargs.pop('env', None)
    command_env = env.copy() if env is not None else os.environ.copy()
    if _QUIET.value:
      command_env['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
      logging.info(shlex.join(cmdline))
    with subprocess.Popen(
        cmdline,
        env=command_env,
        **kwargs,
    ) as p:
      self.register_process(p)

      try:
        retcode = p.wait(timeout=self._timeout)
      except subprocess.TimeoutExpired as e:
        logging.info('Command hit timeout: %s', shlex.join(cmdline))
        _kill_process_ignore_exceptions(p)
        raise e
      finally:
        self.unregister_process(p)

      if retcode != 0:
        if retcode == -9:
          raise ProcessKilledError()
        logging.info(
            'Command returned code %d: %s', retcode, shlex.join(cmdline)
        )
        raise subprocess.CalledProcessError(retcode, cmdline)
      else:
        if p.stdout:
          ret = p.stdout.read()
          p.stdout.close()
          return ret
