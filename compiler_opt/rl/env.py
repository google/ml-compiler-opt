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
"""Gymlike environment definition for MLGO."""

from __future__ import annotations
import dataclasses
from enum import Enum

import math
import subprocess
import abc
import contextlib
import io
import os
import tempfile
from typing import Callable, Generator, List, Optional, Tuple, Type

import numpy as np

from compiler_opt.rl import corpus
from compiler_opt.rl import log_reader


class StepType(Enum):
  FIRST = 1
  MID = 2
  LAST = 3


@dataclasses.dataclass
class TimeStep:
  obs: Optional[dict[str, np.NDArray]]
  reward: Optional[dict[str, float]]
  score_policy: Optional[dict[str, float]]
  score_default: Optional[dict[str, float]]
  context: Optional[str]
  module_name: str
  obs_id: Optional[int]
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
  def get_cmdline(self, clang_path: str, base_args: List[str],
                  interactive_base_path: Optional[str],
                  working_dir: str) -> List[str]:
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
               get_scores_fn: Callable[[], dict[str, float]], module_name):
    self._proc = proc
    self._get_scores_fn = get_scores_fn
    self._module_name = module_name

  def get_scores(self, timeout: Optional[int] = None):
    self._proc.wait(timeout=timeout)
    return self._get_scores_fn()


class InteractiveClang(ClangProcess):
  """Wrapper around clang's interactive mode."""

  def __init__(
      self,
      proc: subprocess.Popen,
      get_scores_fn: Callable[[], dict[str, float]],
      module_name: str,
      reader_pipe: io.BufferedReader,
      writer_pipe: io.BufferedWriter,
  ):
    super().__init__(proc, get_scores_fn, module_name)
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
        tv_dict[fv.spec.name] = np.reshape(array, newshape=fv.spec.shape)
      return TimeStep(
          obs=tv_dict,
          reward={obs.context: obs.score} if obs.score else None,
          score_policy=None,
          score_default=None,
          context=obs.context,
          module_name=self._module_name,
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
def clang_session(
    clang_path: str,
    module: corpus.LoadedModuleSpec,
    task_type: Type[MLGOTask],
    *,
    interactive: bool,
):
  """Context manager for clang session.

  We need to manage the context so resources like tempfiles and pipes have
  their lifetimes managed appropriately.

  Args:
    clang_path: The clang binary to use for the InteractiveClang session.
    module: The module to compile with clang.
    task_type: Type of the MLGOTask to use.
    interactive: Whether to use an interactive or default clang instance

  Yields:
    Either the constructed InteractiveClang or DefaultClang object.
  """
  with tempfile.TemporaryDirectory() as td:
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
          with io.BufferedWriter(io.FileIO(writer_name, 'wb')) as writer_pipe:
            with io.BufferedReader(io.FileIO(reader_name, 'rb')) as reader_pipe:
              yield InteractiveClang(
                  proc,
                  _get_scores,
                  module.name,
                  reader_pipe,
                  writer_pipe,
              )
        else:
          yield ClangProcess(
              proc,
              _get_scores,
              module.name,
          )

      finally:
        proc.kill()


def _get_clang_generator(
    clang_path: str,
    task_type: Type[MLGOTask],
) -> Generator[Optional[Tuple[ClangProcess, InteractiveClang]],
               Optional[corpus.LoadedModuleSpec], None]:
  """Returns a generator for creating InteractiveClang objects.

  TODO: fix this docstring

  Args:
    clang_path: Path to the clang binary to use within InteractiveClang.
    task_type: Type of the MLGO task to use.

  Returns:
    The generator for InteractiveClang objects.
  """
  while True:
    # The following line should be type-hinted as follows:
    #   module: corpus.LoadedModuleSpec = yield
    # However, this triggers a yapf crash. See:
    #   https://github.com/google/yapf/issues/1092
    module = yield
    with clang_session(
        clang_path, module, task_type, interactive=True) as iclang:
      with clang_session(
          clang_path, module, task_type, interactive=False) as clang:
        yield iclang, clang


class MLGOEnvironmentBase:
  """Base implementation for all MLGO environments.

  Depending on the RL framework, one may want different implementations of an
  enviroment (tf_agents: PyEnvironment, jax: dm-env, etc). This class
  implements the core methods that are needed to then implement any of these
  other environments as well.
  """

  def __init__(
      self,
      *,
      clang_path: str,
      task_type: Type[MLGOTask],
      obs_spec,
      action_spec,
  ):
    self._clang_generator = _get_clang_generator(clang_path, task_type)
    self._obs_spec = obs_spec
    self._action_spec = action_spec

    self._iclang: Optional[InteractiveClang] = None
    self._clang: Optional[ClangProcess] = None

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
    return self._get_observation()

  def step(self, action: np.ndarray):
    self._iclang.send_action(action)
    return self._get_observation()
