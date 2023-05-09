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

import abc
import dataclasses
import os
import shlex
import signal
import subprocess
import tempfile
import threading
from typing import Callable, Dict, List, Optional, Tuple

from absl import flags
from absl import logging
import tensorflow as tf

from compiler_opt.distributed.worker import Worker
from compiler_opt.distributed.worker import WorkerFuture
from compiler_opt.rl import constant
from compiler_opt.rl import corpus
from compiler_opt.rl import policy_saver

COMPILATION_TIMEOUT = flags.DEFINE_integer(
    'compilation_timeout', 60,
    'Max duration (in seconds) after which we cancel any compilation job.')
_QUIET = flags.DEFINE_bool(
    'quiet', True, 'Whether or not to compile quietly (hiding info logging)')
_KEEP_TEMPS = flags.DEFINE_string(
    'keep_temps', None,
    'Put temporary files into given directory and keep them past exit.')


def _calculate_reward(policy: float, baseline: float) -> float:
  # This assumption allows us to imply baseline + constant.DELTA > 0.
  assert baseline >= 0
  return 1 - (policy + constant.DELTA) / (baseline + constant.DELTA)


@dataclasses.dataclass
class RewardStat:
  default_reward: float
  moving_average_reward: float


class NonTemporaryDirectory:
  """Behaves like `tempfile.TemporaryDirectory` but does not clean up the
  directory.  When python 3.12 is available this class can be replaced with
  `TemporaryDirectory(..., delete=False)`"""

  def __init__(
      self,
      suffix: Optional[str] = None,
      prefix: Optional[str] = None,
      dir: Optional[str] = None,  # pylint: disable=redefined-builtin
      ignore_cleanup_errors: bool = False):
    _ = ignore_cleanup_errors  # unused
    self.name = tempfile.mkdtemp(suffix, prefix, dir)

  def __repr__(self):
    return f'<{self.__class__.__name__} {self.name!r}>'

  def __enter__(self):
    return self.name

  def __exit__(self, exc, value, tb):
    pass


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
    self._paused = False
    self._lock = threading.Lock()

  def enable(self):
    with self._lock:
      self._done = False

  def register_process(self, p: 'subprocess.Popen[bytes]'):
    """Register a process for potential cancellation."""
    with self._lock:
      if not self._done:
        self._processes.add(p)
        return
    kill_process_ignore_exceptions(p)

  def kill_all_processes(self):
    """Cancel any pending work."""
    with self._lock:
      self._done = True
    for p in self._processes:
      kill_process_ignore_exceptions(p)

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
  command_env = os.environ.copy()
  # Disable tensorflow info messages during data collection
  if _QUIET.value:
    command_env['TF_CPP_MIN_LOG_LEVEL'] = '1'
  else:
    logging.info(shlex.join(cmdline))
  with subprocess.Popen(
      cmdline,
      env=command_env,
      stdout=(subprocess.PIPE if want_output else None)) as p:
    if cancellation_manager:
      cancellation_manager.register_process(p)

    try:
      retcode = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
      logging.info('Command hit timeout: %s', shlex.join(cmdline))
      kill_process_ignore_exceptions(p)
      raise e
    finally:
      if cancellation_manager:
        cancellation_manager.unregister_process(p)

    if retcode != 0:
      if retcode == -9:
        raise ProcessKilledError()
      logging.info('Command returned code %d: %s', retcode, shlex.join(cmdline))
      raise subprocess.CalledProcessError(retcode, cmdline)
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
  policy_rewards: a list of reward values under policy.
  keys: a list of keys.

  The object must observe the following invariants:
  1) The entries of sequence_examples, rewards, policy_rewards and keys
  correspond to each other at the same index.

  2) The keys in reward stats are those in the keys field.
  """
  sequence_examples: dataclasses.InitVar[List[tf.train.SequenceExample]]
  serialized_sequence_examples: List[str] = dataclasses.field(init=False)
  length: int = dataclasses.field(init=False)
  reward_stats: Dict[str, RewardStat]
  rewards: List[float]
  policy_rewards: List[float]
  keys: List[str]

  # The id of the model used to generate this compilation result
  model_id: Optional[int]

  def __post_init__(self, sequence_examples: List[tf.train.SequenceExample]):
    object.__setattr__(self, 'serialized_sequence_examples',
                       [x.SerializeToString() for x in sequence_examples])
    lengths = [
        len(next(iter(x.feature_lists.feature_list.values())).feature)
        for x in sequence_examples
    ]
    object.__setattr__(self, 'length', sum(lengths))

    assert (len(self.serialized_sequence_examples) == len(self.rewards) == len(
        self.policy_rewards) == len(self.keys))
    assert set(self.keys) == set(self.reward_stats.keys())
    assert not hasattr(self, 'sequence_examples')


class CompilationRunnerStub(metaclass=abc.ABCMeta):
  """The interface of a stub to CompilationRunner, for type checkers."""

  @abc.abstractmethod
  def collect_data(
      self,
      loaded_module_spec: corpus.LoadedModuleSpec,
      policy: Optional[policy_saver.Policy] = None,
      reward_stat: Optional[Dict[str, RewardStat]] = None,
      model_id: Optional[int] = None) -> WorkerFuture[CompilationResult]:
    raise NotImplementedError()

  @abc.abstractmethod
  def cancel_all_work(self) -> WorkerFuture:
    raise NotImplementedError()

  @abc.abstractmethod
  def enable(self) -> WorkerFuture:
    raise NotImplementedError()


class CompilationResultObserver(metaclass=abc.ABCMeta):
  """Abstract base class used to observe compilation results.

  This is indended for users who need to observe compilations while they are in
  the distributed worker pool, rather than after they have been coalesced in
  the collection script.
  """

  @abc.abstractmethod
  def observe(self, result: CompilationResult) -> None:
    """Observe a compilation result.

    Note that this will be executed on the worker in the pool rather than on
    the coordinator.

    Args:
      result: the compilation result to observe
    """
    pass


class CompilationRunner(Worker):
  """Base class for collecting compilation data."""

  @classmethod
  def is_priority_method(cls, method_name: str) -> bool:
    return method_name in {
        'cancel_all_work', 'enable', 'pause_all_work', 'resume_all_work'
    }

  def __init__(self,
               clang_path: Optional[str] = None,
               launcher_path: Optional[str] = None,
               moving_average_decay_rate: float = 1,
               create_observer_fns: Optional[List[Callable[
                   [], CompilationResultObserver]]] = None):
    """Initialization of CompilationRunner class.

    Args:
      clang_path: path to the clang binary.
      launcher_path: path to the launcher binary.
      moving_average_decay_rate: moving average decay rate during training.
      create_observer_fns: list of callables which are used to create
        CompilationResultObserver objects. We pass the create callables instead
        of the actual objects because the callables are "just code" and likely
        able to be pickled/sent to the workers, whereas the observers might
        contain non-picklable attributes, such as server connections. It is
        typed as Optional[List[]] due to W0102 dangerous-default-value.
    """
    self._clang_path = clang_path
    self._launcher_path = launcher_path
    self._moving_average_decay_rate = moving_average_decay_rate
    # Avoid reading the flag during the first interpretation of this module.
    self._compilation_timeout = COMPILATION_TIMEOUT.value
    self._cancellation_manager = WorkerCancellationManager()
    self._observers = ([f() for f in create_observer_fns]
                       if create_observer_fns else [])

  # re-allow the cancellation manager accept work.
  def enable(self):
    self._cancellation_manager.enable()

  def cancel_all_work(self):
    self._cancellation_manager.kill_all_processes()

  def pause_all_work(self):
    self._cancellation_manager.pause_all_processes()

  def resume_all_work(self):
    self._cancellation_manager.resume_all_processes()

  def collect_data(self,
                   loaded_module_spec: corpus.LoadedModuleSpec,
                   policy: Optional[policy_saver.Policy] = None,
                   reward_stat: Optional[Dict[str, RewardStat]] = None,
                   model_id: Optional[int] = None) -> CompilationResult:
    """Collect data for the given IR file and policy.

    Args:
      loaded_module_spec: a LoadedModuleSpec.
      policy: serialized policy.
      reward_stat: reward stat of this module, None if unknown.
      model_id: id for the model used to collect data.

    Returns:
      A CompilationResult. In particular:
        reward_stat is the updated reward stat of this module;
        rewards is rewards under the current ml policy.

    Raises:
      subprocess.CalledProcessError if process fails.
      compilation_runner.ProcessKilledException is passed through.
      ValueError if example under default policy and ml policy does not match.
    """
    if _KEEP_TEMPS.present:
      tempdir_context = NonTemporaryDirectory(dir=_KEEP_TEMPS.value)
    else:
      tempdir_context = tempfile.TemporaryDirectory()

    with tempdir_context as tempdir:
      final_cmd_line = loaded_module_spec.build_command_line(tempdir)
      # TODO(mtrofin): remove this once the compiler only generates this by
      # default
      final_cmd_line += (
          '-mllvm',
          '-tfutils-use-simplelogger',
      )
      tf_policy_path = ''
      if policy is not None:
        model_id_suffix = f'-{model_id}' if model_id is not None else ''
        tf_policy_path = os.path.join(tempdir, 'policy' + model_id_suffix)
        policy.to_filesystem(tf_policy_path)

      if reward_stat is None:
        default_result = self.compile_fn(
            final_cmd_line,
            tf_policy_path='',
            reward_only=bool(tf_policy_path),
            workdir=tempdir)
        reward_stat = {
            k: RewardStat(v[1], v[1]) for (k, v) in default_result.items()
        }

      if tf_policy_path:
        policy_result = self.compile_fn(
            final_cmd_line, tf_policy_path, reward_only=False, workdir=tempdir)
      else:
        policy_result = default_result

    sequence_example_list = []
    rewards = []
    policy_rewards = []
    keys = []
    for k, v in policy_result.items():
      sequence_example = v[0]
      policy_reward = v[1]
      if k not in reward_stat:
        raise ValueError(
            (f'Example {k} does not exist under default policy for '
             f'cmd line: {final_cmd_line}'))
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
      policy_rewards.append(policy_reward)
      keys.append(k)

    result = CompilationResult(
        sequence_examples=sequence_example_list,
        reward_stats=reward_stat,
        rewards=rewards,
        policy_rewards=policy_rewards,
        keys=keys,
        model_id=model_id)

    for observer in self._observers:
      observer.observe(result)

    return result

  def compile_fn(
      self, command_line: corpus.FullyQualifiedCmdLine, tf_policy_path: str,
      reward_only: bool,
      workdir: str) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:
    """Compiles for the given IR file under the given policy.

    Args:
      command_line: the fully qualified command line.
      tf_policy_path: path to TF policy directory on local disk.
      reward_only: whether only return reward.

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
