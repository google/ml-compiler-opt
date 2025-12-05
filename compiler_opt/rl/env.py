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
"""Gymlike environment definition for MLGO."""

from __future__ import annotations
import dataclasses
from enum import Enum

import logging
import math
import select
import subprocess
import abc
import contextlib
import io
import os
import threading
from collections.abc import Callable, Generator

import numpy as np

from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader
from compiler_opt.rl import compilation_runner


class StepType(Enum):
  FIRST = 1
  MID = 2
  LAST = 3


@dataclasses.dataclass
class TimeStep:
  obs: dict[str, np.NDArray] | None
  reward: dict[str, float] | None
  score_policy: dict[str, float] | None
  score_default: dict[str, float] | None
  context: str | None
  module_name: str
  working_dir: str
  obs_id: int | None
  step_type: StepType


_INTERACTIVE_PIPE_FILE_BASE = 'interactive-pipe-base'


class MLGOTask(metaclass=abc.ABCMeta):
  """Abstract base class for MLGO Tasks.

  A Task is an learning problem in LLVM, for example:
   - inlining-for-size
   - inlining-for-speed
   - register allocation (for speed)

  The Task type for a given problem defines how to build and score modules for
  the problem, both interactively and non-interactively.
  """

  @abc.abstractmethod
  def get_cmdline(self, clang_path: str, base_args: list[str],
                  interactive_base_path: str | None,
                  working_dir: str) -> list[str]:
    """Get the cmdline for building with this task.

    The resulting list[str] should be able to be passed to subprocess.run to
    execute clang.

    Args:
      clang_path: path to the clang executable.
      base_args: base arguments for building the module. Generally, these flags
        should not be modified and simply added to the result.
      interactive_base_path: the path to the interactive pipe base. if None,
        then don't run clang interactively.
      working_dir: directory where all artifacts from compilation should be
        written. This will be a temp directory whose lifetime is managed outside
        of the Task.

    Returns:
      The constructed command line.
    """
    pass

  @abc.abstractmethod
  def get_module_scores(self, working_dir: str) -> dict[str, float]:
    """Get the scores for each context in the module.

    This method should not be aware of whether the module was built with the
    default heuristic or a ML policy.

    Args:
      working_dir: Directory which was passed as working_dir to get_cmdline.
        Used to recover binaries/artifacts from the build

    Returns:
      A dictionary mapping [context name] -> [score].
    """
    pass


class ClangProcess:
  """Simple wrapper class around a clang process.

  This is used wrap both the clang process and the method to return the scores
  associated to the default-compiled binary.
  """

  def __init__(self, proc: subprocess.Popen,
               get_scores_fn: Callable[[], dict[str, float]], module_name: str,
               working_dir: str):
    self._proc = proc
    self._get_scores_fn = get_scores_fn
    self._module_name = module_name
    self._working_dir = working_dir

  def get_scores(self, timeout: int | None = None):
    self._proc.wait(timeout=timeout)
    return self._get_scores_fn()


class InteractiveClang(ClangProcess):
  """Wrapper around clang's interactive mode."""

  def __init__(
      self,
      proc: subprocess.Popen,
      get_scores_fn: Callable[[], dict[str, float]],
      module_name: str,
      working_dir: str,
      reader_pipe: io.BufferedReader,
      writer_pipe: io.BufferedWriter,
  ):
    super().__init__(proc, get_scores_fn, module_name, working_dir)
    self._reader_pipe = reader_pipe
    self._writer_pipe = writer_pipe
    self._obs_gen = log_reader.read_log_from_file(self._reader_pipe)

    self._is_first_obs = True

    self._terminal_obs = TimeStep(
        obs=None,
        reward=None,
        score_policy=None,
        score_default=None,
        context=None,
        module_name=module_name,
        working_dir=working_dir,
        obs_id=None,
        step_type=StepType.LAST,
    )

  def _running(self) -> bool:
    return self._proc.poll() is None

  def get_observation(self) -> TimeStep:
    if not self._running():
      return self._terminal_obs

    def _get_step_type() -> StepType:
      step_type = StepType.FIRST if self._is_first_obs else StepType.MID
      self._is_first_obs = False
      return step_type

    try:
      obs: log_reader.ObservationRecord = next(self._obs_gen)

      tv_dict = {}
      for fv in obs.feature_values:
        array = fv.to_numpy()
        tv_dict[fv.spec.name] = np.reshape(array, fv.spec.shape)
      return TimeStep(
          obs=tv_dict,
          reward={obs.context: obs.score} if obs.score else None,
          score_policy=None,
          score_default=None,
          context=obs.context,
          module_name=self._module_name,
          working_dir=self._working_dir,
          obs_id=obs.observation_id,
          step_type=_get_step_type(),
      )
    except StopIteration:
      return self._terminal_obs

  def send_action(self, action: np.ndarray) -> None:
    assert self._running()
    data = action.tobytes()
    bytes_sent = self._writer_pipe.write(data)
    # Here we use the fact that for common types, the np.dtype and ctype should
    # behave the same
    assert bytes_sent == action.dtype.itemsize * math.prod(action.shape)
    try:
      self._writer_pipe.flush()
    except BrokenPipeError:
      # The pipe can break after we send the last action
      pass


_EPS = 1e-4


def compute_relative_rewards(score_a: dict[str, float],
                             score_b: dict[str, float]) -> dict[str, float]:

  def _reward_fn(a: float, b: float) -> float:
    return 1.0 - (a + _EPS) / (b + _EPS)

  assert score_a.keys() == score_b.keys()
  return {key: _reward_fn(score_a[key], score_b[key]) for key in score_a}


@contextlib.contextmanager
def open_write_pipe(filename: str, *, timeout: float):
  """Open the write pipe or timeout.

  Assuming a fifo, the `open` will block until the other party (the process we
  communicate to) also opens the pipe. If that doesn't happen, we time out.
  Afterwards, `write` ops shouldn't block.
  """
  opened = threading.Event()
  timed_out = threading.Event()

  # start a thread that waits for `open` to unblock. If it doesn't, we open the
  # fifo ourselves just to unblock.
  def _timeout_thread():
    if opened.wait(timeout):
      logging.debug('[timeout thread] writer opened successfully')
      return
    timed_out.set()
    logging.debug('[timeout thread] writer failed to open')
    with open(filename, 'rb'):
      pass

  waiter = threading.Thread(target=_timeout_thread)
  waiter.start()
  try:
    with io.BufferedWriter(io.FileIO(filename, 'wb')) as writer_pipe:
      if not timed_out.is_set():
        opened.set()
        yield writer_pipe
  finally:
    waiter.join()
    if timed_out.is_set():
      # it's possible that the timeout thread timed out but also the other
      # process finally opened the pipe and thus the `writer_pipe` is
      # functional, but at the end we still raise TimeoutError. We accept that
      # right now.
      raise TimeoutError('write pipe open')


@contextlib.contextmanager
def open_read_pipe(filename: str, *, timeout: float):
  """Open the read pipe, with a timeout governing the open and each read.

  Just like in the writer case, assuming we're opening a fifo pipe, the open
  operation will block until the other party opens the pipe. Then, because this
  is a reader, each read operation (and variations - readline, etc) can block,
  but no more than the provided timeout.
  """

  # wrap the underlying io.RawIOBase such that we poll before attempting to read
  def _wrap_raw_io(obj: io.RawIOBase):

    def _get_polling_wrapper(wrapped_method):

      def _replacement(*args, **kwargs):
        name = wrapped_method.__name__
        logging.debug('ReaderWithTimeout is asked to %s', name)
        (r, _, _) = select.select([obj], [], [], timeout)
        if r:
          logging.debug('ReaderWithTimeout %s should be unblocked', name)
          result = wrapped_method(*args, **kwargs)
          logging.debug('ReaderWithTimeout %s completed', name)
          return result
        logging.info('ReaderWithTimeout timed out waiting to %s', name)
        raise TimeoutError('timed out reading')

      return _replacement

    obj.read = _get_polling_wrapper(obj.read)
    obj.readline = _get_polling_wrapper(obj.readline)
    obj.readinto = _get_polling_wrapper(obj.readinto)
    obj.readall = _get_polling_wrapper(obj.readall)

    return obj

  opened = threading.Event()
  timed_out = threading.Event()

  # same idea as in the writer case - unblock the `open`
  def _timeout_thread():
    if opened.wait(timeout):
      logging.debug('[timeout thread] reader opened successfully')
      return
    timed_out.set()
    logging.debug('[timeout thread] reader failed to open')
    with open(filename, 'wb'):
      pass
    logging.debug('[timeout thread] force-opened the reader')

  waiter = threading.Thread(target=_timeout_thread)
  waiter.start()
  try:
    # we must wrap the *raw* stream! wrapping the buffered stream would be
    # incorrect because calls to `read` APIs shouldn't poll (they may just
    # return from the buffer).
    with io.BufferedReader(_wrap_raw_io(io.FileIO(filename,
                                                  'rb'))) as reader_pipe:
      if not timed_out.is_set():
        opened.set()
        yield reader_pipe
  finally:
    waiter.join()
    if timed_out.is_set():
      # same as in the writer case - we could successfully keep reading but
      # still report a timeout at the end of this context.
      raise TimeoutError('read pipe open')


@contextlib.contextmanager
def interactive_session(*, reader_name: str, writer_name: str, timeout: float):
  """Start an interactive session with the started process proc.

  Blocking pipe operations - open and read - happen under a timeout.
  """

  try:
    with open_write_pipe(writer_name, timeout=timeout) as writer_pipe:
      with open_read_pipe(reader_name, timeout=timeout) as reader_pipe:
        yield (reader_pipe, writer_pipe)
  finally:
    pass


@contextlib.contextmanager
def clang_session(
    clang_path: str,
    module: corpus.LoadedModuleSpec,
    task_type: type[MLGOTask],
    *,
    explicit_temps_dir: str | None = None,
    interactive: bool,
):
  """Context manager for clang session.

  We need to manage the context so resources like tempfiles and pipes have
  their lifetimes managed appropriately.

  Args:
    clang_path: The clang binary to use for the InteractiveClang session.
    module: The module to compile with clang.
    task_type: Type of the MLGOTask to use.
    explicit_temps_dir: Put temporary files into given directory and keep them
      past exit when compilining
    interactive: Whether to use an interactive or default clang instance

  Yields:
    Either the constructed InteractiveClang or DefaultClang object.
  """
  tempdir_context = compilation_runner.get_workdir_context(
      explicit_temps_dir=explicit_temps_dir)
  with tempdir_context as td:
    task_working_dir = os.path.join(td, '__task_working_dir__')
    os.mkdir(task_working_dir)
    task = task_type()

    base_args = list(module.build_command_line(td))
    interactive_base = os.path.join(
        td, _INTERACTIVE_PIPE_FILE_BASE) if interactive else None
    cmdline = task.get_cmdline(clang_path, base_args, interactive_base,
                               task_working_dir)

    def _get_scores() -> dict[str, float]:
      return task.get_module_scores(task_working_dir)

    writer_name = os.path.join(td, _INTERACTIVE_PIPE_FILE_BASE + '.in')
    reader_name = os.path.join(td, _INTERACTIVE_PIPE_FILE_BASE + '.out')
    if interactive:
      os.mkfifo(reader_name, 0o666)
      os.mkfifo(writer_name, 0o666)
    with subprocess.Popen(
        cmdline, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
      try:
        if interactive:
          with interactive_session(
              writer_name=writer_name,
              reader_name=reader_name,
              timeout=compilation_runner.COMPILATION_TIMEOUT.value) as (
                  reader_pipe, writer_pipe):
            yield InteractiveClang(
                proc,
                _get_scores,
                module.name,
                task_working_dir,
                reader_pipe,
                writer_pipe,
            )
        else:
          yield ClangProcess(
              proc,
              _get_scores,
              module.name,
              task_working_dir,
          )

      finally:
        proc.kill()


def _get_clang_generator(
    clang_path: str,
    task_type: type[MLGOTask],
    explicit_temps_dir: str | None = None,
    interactive_only: bool = False,
) -> Generator[tuple[ClangProcess, InteractiveClang] | None,
               corpus.LoadedModuleSpec | None, None]:
  """Returns a tuple of generators for creating InteractiveClang objects.

  Args:
    clang_path: Path to the clang binary to use within InteractiveClang.
    task_type: Type of the MLGO task to use.
    explicit_temps_dir: Put temporary files into given directory and keep them
      past exit when compilining
    interactive_only: If set to true the returned tuple of generators is
      iclang, iclang instead of iclang, clang

  Returns:
    A generator of tuples. Each element of the tuple is created with
    clang_session. First argument of the tuple is always an interactive
    clang session. The second argument is a default clang session if
    interactive_only is False and otherwise the exact same interactive
    clang session object as the first argument.
  """
  while True:
    # The following line should be type-hinted as follows:
    #   module: corpus.LoadedModuleSpec = yield
    # However, this triggers a yapf crash. See:
    #   https://github.com/google/yapf/issues/1092
    module = yield
    with clang_session(
        clang_path,
        module,
        task_type,
        explicit_temps_dir=explicit_temps_dir,
        interactive=True) as iclang:
      if interactive_only:
        yield iclang, iclang
      else:
        with clang_session(
            clang_path,
            module,
            task_type,
            explicit_temps_dir=explicit_temps_dir,
            interactive=False) as clang:
          yield iclang, clang


class MLGOEnvironmentBase:
  """Base implementation for all MLGO environments.

  Depending on the RL framework, one may want different implementations of an
  environment (tf_agents: PyEnvironment, jax: dm-env, etc). This class
  implements the core methods that are needed to then implement any of these
  other environments as well.
  """

  def __init__(
      self,
      *,
      clang_path: str,
      task_type: type[MLGOTask],
      obs_spec,
      action_spec,
      explicit_temps_dir: str | None = None,
      interactive_only: bool = False,
  ):
    self._clang_generator = _get_clang_generator(
        clang_path,
        task_type,
        explicit_temps_dir=explicit_temps_dir,
        interactive_only=interactive_only)
    self._obs_spec = obs_spec
    self._action_spec = action_spec

    self._iclang: InteractiveClang | None = None
    self._clang: ClangProcess | None = None

  @property
  def obs_spec(self):
    return self._obs_spec

  @property
  def action_spec(self):
    return self._action_spec

  def observation(self):
    return self._last_obs

  def _get_observation(self) -> TimeStep:
    self._last_obs = self._iclang.get_observation()
    if self._last_obs.step_type == StepType.LAST:
      self._last_obs.score_policy = self._iclang.get_scores()
      self._last_obs.score_default = self._clang.get_scores()
      self._last_obs.reward = compute_relative_rewards(
          self._last_obs.score_policy, self._last_obs.score_default)
    return self._last_obs

  def reset(self, module: corpus.LoadedModuleSpec):
    # On the first call to reset(...), sending None starts the coroutine.
    # On subsequent calls, this resumes execution after
    # yielding the clang pair, which terminates the session pauses execution in
    # the coroutine where it awaits a module
    self._clang_generator.send(None)
    # pytype: disable=attribute-error
    self._iclang, self._clang = self._clang_generator.send(module)
    # pytype: enable=attribute-error
    return self._get_observation()

  def step(self, action: np.ndarray):
    self._iclang.send_action(action)
    return self._get_observation()
