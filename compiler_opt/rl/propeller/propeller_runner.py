"""Propeller Runner for RL."""

import os


import gin
import tensorflow as tf

from .. import compilation_runner
from .. import corpus
from .. import log_reader

_DEFAULT_IDENTIFIER = 'default'


@gin.configurable(module='runners')
class PropellerRunner(compilation_runner.CompilationRunner):
  """Runner for Propeller Optimization."""

  def __init__(
      self,
      propeller_prof_gen_path,
      perf_profile_path,
      initial_clang_path,
      cc_profile_path,
      ld_profile_path,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._propeller_prof_gen_path = propeller_prof_gen_path
    # perf_profile_path can be a file or a directory.
    # If directory, we look for {module_name}.perf
    self._perf_profile_path = perf_profile_path
    self._initial_clang_path = initial_clang_path
    self._cc_profile_path = cc_profile_path
    self._ld_profile_path = ld_profile_path

  def compile_fn(
      self,
      command_line: corpus.FullyQualifiedCmdLine,
      tf_policy_path: str,
      reward_only: bool,
      workdir: str,
      module_name: str | None = None,
  ) -> dict[str, tuple[tf.train.SequenceExample, float]]:

    results = {}

    if not module_name:
      print('Error: module_name must be provided.')
      return {}

    current_perf_profile = os.path.join(
        self._perf_profile_path, f'{module_name}.perf'
    )

    # current_perf_profile = os.path.join(self._perf_profile_path, f'perf.data')

    if not os.path.exists(current_perf_profile):
      print(f'Error: {current_perf_profile} does not exist.')
      return {}

    print(f'Processing {module_name}')

    # Unique log path for this module
    log_path = os.path.join(
        workdir,
        f'log_{module_name}.log',
    )

    cmd = [
        self._propeller_prof_gen_path,
        f'--profile={current_perf_profile}',
        f'--binary={self._initial_clang_path}',
        f'--cc_profile={self._cc_profile_path}',
        f'--ld_profile={self._ld_profile_path}',
        '--alsologtostderr',
    ]

    propeller_options = [
        'inter_function_reordering: true',
        'split_all_basic_blocks: true',
    ]

    if tf_policy_path:
      propeller_options.append(f"policy_path: '{tf_policy_path}'")

    # Always generate logs to discover keys
    cmd.append(f'--training_log={log_path}')

    cmd.append('--use_ml')

    options_str = f"code_layout_params {{ {', '.join(propeller_options)} }}"
    cmd.append(f'--propeller_options={options_str}')

    # Run the tool
    try:
      compilation_runner.start_cancellable_process(
          cmd,
          timeout=self._compilation_timeout,
          cancellation_manager=self._cancellation_manager,
      )
    except Exception as e:
      print(f'Error running Propeller for {module_name}: {e}')
      return {}

    # Calculate Reward (Placeholder)
    reward = 0.0

    # Read the generated trace
    if not os.path.exists(log_path):
      return {}

    log_result = log_reader.read_log_as_sequence_examples(log_path)
    if not log_result:
      return {}

    for func_name, sequence_example in log_result.items():
      key = f'{module_name}/{func_name}'
      if reward_only:
        results[key] = (None, reward)
      else:
        results[key] = (sequence_example, reward)

    return results
