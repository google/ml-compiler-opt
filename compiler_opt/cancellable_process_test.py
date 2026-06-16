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
"""Tests for compiler_opt.cancellable_process."""

import os
import subprocess
import threading
import time

from absl import flags
import tensorflow as tf

from compiler_opt import cancellable_process


class CancellableProcessTest(tf.test.TestCase):

  def test_start_subprocess_output(self):
    cm = cancellable_process.WorkerCancellationManager(timeout=100)
    output_str = cm.start_cancellable_process(['ls', '-l'],
                                              stdout=subprocess.PIPE,
                                              text=True)
    if not output_str:
      self.fail('output should have been non-empty')
    self.assertNotEmpty(output_str)

  def test_timeout_kills_process(self):
    sentinel_file = os.path.join(flags.FLAGS.test_tmpdir,
                                 'test_timeout_kills_test_file')
    if os.path.exists(sentinel_file):
      os.remove(sentinel_file)
    cm = cancellable_process.WorkerCancellationManager(timeout=0.5)
    with self.assertRaises(subprocess.TimeoutExpired):
      cm.start_cancellable_process(
          ['bash', '-c', 'sleep 1s ; touch ' + sentinel_file])
    time.sleep(2)
    self.assertFalse(os.path.exists(sentinel_file))

  def test_pause_resume(self):
    cm = cancellable_process.WorkerCancellationManager(timeout=30)
    start_time = time.time()

    def stop_and_start():
      time.sleep(0.25)
      cm.pause_all_processes()
      time.sleep(1)
      cm.resume_all_processes()

    threading.Thread(target=stop_and_start).start()
    cm.start_cancellable_process(['sleep', '0.5'])
    # should be at least 1 second due to the pause.
    self.assertGreater(time.time() - start_time, 1)


if __name__ == '__main__':
  tf.test.main()
