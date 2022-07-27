#!/usr/bin/env python3
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
"""Pre-submit checks, equivalent to GitHub CI runs."""
import functools
import pathlib
import signal
import subprocess
import sys
import threading

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

_READ_ONLY = flags.DEFINE_boolean(
    'read_only', False, 'Do not modify any files.', short_name='ro')
# -v is used by absl
_VERBOSE = flags.DEFINE_boolean(
    'verbose', False, 'Print stdout.', short_name='vv')
_VVERBOSE = flags.DEFINE_boolean(
    'vverbose', False, 'Print stdout and stderr.', short_name='vvv')
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR_STR = str(ROOT_DIR)
OS_LOCK = threading.Lock()
RETCODE = 0


def job_pytype():
  p = subprocess.run(['pytype', '-j', 'auto', ROOT_DIR],
                     capture_output=True,
                     check=False,
                     universal_newlines=True)
  return 'pytype', p.returncode, p.stdout, p.stderr


def job_pylint():
  p = subprocess.run([
      'pylint', '--rcfile', ROOT_DIR / '.pylintrc', '--recursive', 'yes',
      ROOT_DIR_STR
  ],
                     capture_output=True,
                     check=False,
                     universal_newlines=True)
  return 'pylint', p.returncode, p.stdout, p.stderr


def job_pytest():
  p = subprocess.run(['pytest', ROOT_DIR_STR],
                     capture_output=True,
                     check=False,
                     universal_newlines=True)
  return 'pytest', p.returncode, p.stdout, p.stderr


def job_yapf(read_only):
  p = subprocess.run(['yapf', '-drp' if read_only else '-irp', ROOT_DIR_STR],
                     capture_output=True,
                     check=False,
                     universal_newlines=True)
  return 'yapf-ro' if read_only else 'yapf', p.returncode, p.stdout, p.stderr


def job_check_license():
  p = subprocess.run([ROOT_DIR / 'check-license.sh'],
                     capture_output=True,
                     check=False,
                     universal_newlines=True,
                     cwd=ROOT_DIR_STR)
  return 'check_license', p.returncode, p.stdout, p.stderr


def callback(job, verbosity):
  name, retcode, stdout, stderr = job()
  with OS_LOCK:
    global RETCODE
    RETCODE += abs(retcode)
    if retcode == 0:
      logging.info('\033[1m%s: \033[92mOK\033[0m', name)
    else:
      logging.info('\033[1m%s: \033[91mFAIL\033[0m', name)
      verbosity += 2
    if verbosity > 0:
      logging.info('stdout:\n%s', stdout)
    if verbosity > 1:
      logging.error('stderr:\n%s', stderr)


PARALLEL_JOBS = [job_pytest, job_pylint, job_pytype, job_check_license]


def main(argv):
  if len(argv) != 1:
    raise ValueError('Too many arguments!')
  verbosity = _VERBOSE.value + _VVERBOSE.value * 2

  # Format everything before running tests
  yapf = threading.Thread(
      target=callback,
      args=(functools.partial(job_yapf, _READ_ONLY.value), verbosity),
      daemon=True)
  yapf.start()
  yapf.join()

  threads = [
      threading.Thread(target=callback, args=(job, verbosity), daemon=True)
      for job in PARALLEL_JOBS
  ]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  return RETCODE


if __name__ == '__main__':

  # This prevents a stacktrace printing on Ctrl-C
  def signal_handler(*_):
    sys.exit(1)

  signal.signal(signal.SIGINT, signal_handler)
  app.run(main)
