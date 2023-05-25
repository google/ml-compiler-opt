"""Module for collect data of the LR encoder."""

import os
import tempfile
from typing import Dict, Tuple

import gin
import tensorflow as tf

from google3.third_party.ml_compiler_opt.compiler_opt.rl import compilation_runner
from google3.third_party.ml_compiler_opt.compiler_opt.rl import corpus
from google3.third_party.ml_compiler_opt.compiler_opt.rl import log_reader


@gin.configurable(module='runners')
class LREncoderRunner(compilation_runner.CompilationRunner):
  """Class for collecting data for the LR encoder."""

  def compile_fn(
      self,
      command_line: corpus.FullyQualifiedCmdLine,
      tf_policy_path: str,
      reward_only: bool,
  ) -> Dict[str, Tuple[tf.train.SequenceExample, float]]:
    """Run compilation for the given IR file under the given policy.

    Args:
      command_line: the fully qualified command line.
      tf_policy_path: path to TF policy direcoty on local disk.
      reward_only: whether only return reward.

    Returns:
      A dict mapping from example identifier to tuple containing:
        sequence_example: A tf.SequenceExample proto describing compilation
          trace, None if reward_only == True.
        reward: reward of register allocation.

    Raises:
      subprocess.CalledProcessError: if process fails.
      compilation_runner.ProcessKilledError: (which it must pass through) on
        cancelled work.
      RuntimeError: if llvm-size produces unexpected output.
    """
    assert not tf_policy_path

    working_dir = tempfile.mkdtemp()

    log_path = os.path.join(working_dir, 'log')
    output_native_path = os.path.join(working_dir, 'native')

    result = {}
    try:
      cmdline = []
      if self._launcher_path:
        cmdline.append(self._launcher_path)
      cmdline.extend(
          [self._clang_path]
          + list(command_line)
          + [
              '-mllvm',
              '-regalloc-enable-advisor=development',
              '-mllvm',
              '-regalloc-lr-encoder-training-log=' + log_path,
              '-mllvm',
              '-regalloc-training-log=/dev/null',
              '-o',
              output_native_path,
          ]
      )

      compilation_runner.start_cancellable_process(
          cmdline, self._compilation_timeout, self._cancellation_manager
      )

      if not os.path.exists(log_path):
        return {}

      # TODO(#202)
      log_result = log_reader.read_log_as_sequence_examples(log_path)

      for fct_name, trajectory in log_result.items():
        if not trajectory.HasField('feature_lists'):
          continue
        # score = (
        #    trajectory.feature_lists.feature_list['reward']
        #    .feature[-1]
        #    .float_list.value[0]
        # )
        result[fct_name] = (trajectory, 1.0)

    finally:
      tf.io.gfile.rmtree(working_dir)

    return result
