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
"""Module for running compilation and collect training data."""

import concurrent
import dataclasses
import json
import multiprocessing
import subprocess
import threading
from typing import Dict, List, Optional, Tuple

from absl import flags
import tensorflow as tf

from compiler_opt.rl import constant
from compiler_opt.rl import corpus

_COMPILATION_TIMEOUT = flags.DEFINE_integer(
    'compilation_timeout', 60,
    'Max duration (in seconds) after which we cancel any compilation job.')


def _calculate_reward(policy: float, baseline: float) -> float:
  # This assumption allows us to imply baseline + constant.DELTA > 0.
  assert baseline >= 0
  return 1 - (policy + constant.DELTA) / (baseline + constant.DELTA)


@dataclasses.dataclass
class RewardStat:
  default_reward: float
  moving_average_reward: float


class DataClassJSONEncoder(json.JSONEncoder):

  def default(self, o):
    if dataclasses.is_dataclass(o):
      return dataclasses.asdict(o)
    return super().default(o)


def _overwrite_trajectory_reward(sequence_example: tf.train.SequenceExample,
                                 reward: float) -> tf.train.SequenceExample:
  """Overwrite the reward in the trace (sequence_example) with the given one.


  Args:
    sequence_example: A tf.SequenceExample proto describing compilation trace.
    reward: The reward to overwrite with.

  Returns:
    The tf.SequenceExample proto after post-processing.
  """
  sequence_length = len(
      next(iter(sequence_example.feature_lists.feature_list.values())).feature)

  reward_list = sequence_example.feature_lists.feature_list['reward']
  for _ in range(sequence_length):
    added_feature = reward_list.feature.add()
    added_feature.float_list.value.append(reward)

  return sequence_example


class ProcessKilledError(Exception):

  def __init__(self):
    Exception.__init__(self)


class ProcessCancellationToken:

  def __init__(self):
    self._event = multiprocessing.Manager().Event()

  def signal(self):
    self._event.set()

  def wait(self):
    self._event.wait()


def kill_process_ignore_exceptions(p: 'subprocess.Popen[bytes]'):
  # kill the process and ignore exceptions. Exceptions would be thrown if the
  # process has already been killed/finished (which is inherently in a race
  # condition with us killing it)
  try:
    p.kill()
    p.wait()
  finally:
    return  # pylint: disable=lost-exception


class WorkerCancellationManager:
  """A thread-safe object that can be used to signal cancellation.

  This allows killing long-running processes promptly, and thus efficiently
  managing resources.
  """

  def __init__(self):
    # the queue is filled only by workers, and drained only by the single
    # consumer. we use _done to manage access to the queue. We can then assume
    # empty() is accurate and get() never blocks.
    self._processes = set()
    self._done = False
    self._lock = threading.Lock()

  def register_process(self, p: 'subprocess.Popen[bytes]'):
    """Register a process for potential cancellation."""
    with self._lock:
      if not self._done:
        self._processes.add(p)
        return
    kill_process_ignore_exceptions(p)

  def signal(self):
    """Cancel any pending work."""
    with self._lock:
      self._done = True
    for p in self._processes:
      kill_process_ignore_exceptions(p)

  def unregister_process(self, p: 'subprocess.Popen[bytes]'):
    with self._lock:
      if not self._done:
        self._processes.remove(p)


def start_cancellable_process(
    cmdline: List[str],
    timeout: float,
    cancellation_manager: Optional[WorkerCancellationManager],
    want_output: bool = False) -> Optional[bytes]:
  """Start a cancellable process.

  Args:
    cmdline: the process executable and command line
    timeout: process execution timeout
    cancellation_manager: kill any running process if signaled to do so
    want_output: if True, return a buffer containing stdout

  Returns:
    stdout
  Raises:
    CalledProcessError: if the process encounters an error.
    TimeoutExpired: if the process times out.
    ProcessKilledError: if the process was killed via the cancellation token.
  """
  with subprocess.Popen(
      cmdline, stdout=(subprocess.PIPE if want_output else None)) as p:
    if cancellation_manager:
      cancellation_manager.register_process(p)

    try:
      retcode = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
      kill_process_ignore_exceptions(p)
      raise e
    finally:
      if cancellation_manager:
        cancellation_manager.unregister_process(p)
    if retcode != 0:
      raise ProcessKilledError(
      ) if retcode == -9 else subprocess.CalledProcessError(retcode, cmdline)
    else:
      if want_output:
        ret: bytes = p.stdout.read()
        p.stdout.close()
        return ret


@dataclasses.dataclass(frozen=True)
class CompilationResult:
  """Result of a call to CompilationRunner.collect_data.

  sequence_examples: a list of tf.train.SequenceExample protos, init-only
                     variables.
  serialized_sequence_examples: a list of tf.train.SequenceExample serialized
                                protos, derived from sequence_examples.
  length: total length of all sequence examples, derived from sequence_examples.
  reward_stats: a dictionary from keys (e.g. function names) to a RewardStat.
  rewards: a list of reward values.
  keys: a list of keys.

  The object must observe the following invariants:
  1) The entries of sequence_examples, rewards, and keys correspond to eachoter
  at the same index.

  2) The keys in reward stats are those in the keys field.
  """
  sequence_examples: dataclasses.InitVar[List[tf.train.SequenceExample]]
  serialized_sequence_examples: List[str] = dataclasses.field(init=False)
  length: int = dataclasses.field(init=False)
  reward_stats: Dict[str, RewardStat]
  rewards: List[float]
  keys: List[str]

  def __post_init__(self, sequence_examples: List[tf.train.SequenceExample]):
    object.__setattr__(self, 'serialized_sequence_examples',
                       [x.SerializeToString() for x in sequence_examples])
    lengths = [
        len(next(iter(x.feature_lists.feature_list.values())).feature)
        for x in sequence_examples
    ]
    object.__setattr__(self, 'length', sum(lengths))

    assert (len(self.serialized_sequence_examples) == len(self.rewards) ==
            (len(self.keys)))
    assert set(self.keys) == set(self.reward_stats.keys())
    assert not hasattr(self, 'sequence_examples')


class CompilationRunner:
  """Base class for collecting compilation data."""

  _POOL: concurrent.futures.ThreadPoolExecutor = None

  @staticmethod
  def init_pool():
    """Worker process initialization."""
    CompilationRunner._POOL = concurrent.futures.ThreadPoolExecutor()

  @staticmethod
  def _get_pool():
    """Internal API for fetching the cancellation token waiting pool."""
    assert CompilationRunner._POOL
    return CompilationRunner._POOL

  def __init__(self,
               clang_path: Optional[str] = None,
               launcher_path: Optional[str] = None,
               moving_average_decay_rate: float = 1):
    """Initialization of CompilationRunner class.

    Args:
      clang_path: path to the clang binary.
      launcher_path: path to the launcher binary.
      moving_average_decay_rate: moving average decay rate during training.
    """
    self._clang_path = clang_path
    self._launcher_path = launcher_path
    self._moving_average_decay_rate = moving_average_decay_rate
    self._compilation_timeout = _COMPILATION_TIMEOUT.value

  def _get_cancellation_manager(
      self, cancellation_token: Optional[ProcessCancellationToken]
  ) -> Optional[WorkerCancellationManager]:
    """Convert the ProcessCancellationToken into a WorkerCancellationManager.

    The conversion also registers the ProcessCancellationToken wait() on a
    thread which will call the WorkerCancellationManager upon completion.
    Since the token is always signaled, the thread always completes its work.

    Args:
      cancellation_token: the ProcessCancellationToken to convert.

    Returns:
      a WorkerCancellationManager, if a ProcessCancellationToken was given.
    """
    if not cancellation_token:
      return None
    ret = WorkerCancellationManager()

    def signaler():
      cancellation_token.wait()
      ret.signal()

    CompilationRunner._get_pool().submit(signaler)
    return ret

  def collect_data(
      self,
      module_spec: corpus.ModuleSpec,
      tf_policy_path: str,
      reward_stat: Optional[Dict[str, RewardStat]],
      cancellation_token: Optional[ProcessCancellationToken] = None
  ) -> CompilationResult:
    """Collect data for the given IR file and policy.

    Args:
      module_spec: a ModuleSpec.
      tf_policy_path: path to the tensorflow policy.
      reward_stat: reward stat of this module, None if unknown.
      cancellation_token: a CancellationToken through which workers may be
      signaled early termination

    Returns:
      A CompilationResult. In particular:
        reward_stat is the updated reward stat of this module;
        rewards is rewards under the current ml policy.

    Raises:
      subprocess.CalledProcessError if process fails.
      compilation_runner.ProcessKilledException is passed through.
      ValueError if example under default policy and ml policy does not match.
    """
    cancellation_manager = self._get_cancellation_manager(cancellation_token)

    if reward_stat is None:
      default_result = self._compile_fn(
          module_spec,
          tf_policy_path='',
          reward_only=bool(tf_policy_path),
          cancellation_manager=cancellation_manager)
      reward_stat = {
          k: RewardStat(v[1], v[1]) for (k, v) in default_result.items()
      }

    if tf_policy_path:
      policy_result = self._compile_fn(
          module_spec,
          tf_policy_path,
          reward_only=False,
          cancellation_manager=cancellation_manager)
    else:
      policy_result = default_result

    sequence_example_list = []
    rewards = []
    keys = []
    for k, v in policy_result.items():
      sequence_example = v[0]
      policy_reward = v[1]
      if k not in reward_stat:
        raise ValueError(
            (f'Example {k} does not exist under default policy for '
             f'module {module_spec.name}'))
      default_reward = reward_stat[k].default_reward
      moving_average_reward = reward_stat[k].moving_average_reward
      sequence_example = _overwrite_trajectory_reward(
          sequence_example=sequence_example,
          reward=_calculate_reward(
              policy=policy_reward, baseline=moving_average_reward))
      sequence_example_list.append(sequence_example)
      reward_stat[k].moving_average_reward = (
          moving_average_reward * self._moving_average_decay_rate +
          policy_reward * (1 - self._moving_average_decay_rate))
      rewards.append(
          _calculate_reward(policy=policy_reward, baseline=default_reward))
      keys.append(k)

    return CompilationResult(
        sequence_examples=sequence_example_list,
        reward_stats=reward_stat,
        rewards=rewards,
        keys=keys)

  def _compile_fn(
      self, module_spec: corpus.ModuleSpec, tf_policy_path: str,
      reward_only: bool,
      cancellation_manager: Optional[WorkerCancellationManager]
  ) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:
    """Compiles for the given IR file under the given policy.

    Args:
      module_spec: a ModuleSpec.
      tf_policy_path: path to TF policy directory on local disk.
      reward_only: whether only return reward.
      cancellation_manager: a WorkerCancellationManager to handle early
        termination

    Returns:
      A dict mapping from example identifier to tuple containing:
        sequence_example: A tf.SequenceExample proto describing compilation
        trace, None if reward_only == True.
        reward: reward under the policy.

    Raises:
      subprocess.CalledProcessError if process fails.
      ProcessKilledError if the process was killed
    """
    raise NotImplementedError('Not implemented compile fn.')
