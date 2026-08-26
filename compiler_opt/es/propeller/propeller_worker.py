"""Worker for propeller for code layout evaluation (Open Source)."""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

import gin
import numpy as np

from ...distributed import worker
from ...es import policy_utils


def _run_command(cmd: list[str], cwd: Optional[str] = None) -> tuple[int, str, str]:
  """Runs a local command."""
  logging.info('Running local command: %s', ' '.join(cmd))
  try:
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
        errors='ignore',
    )
    return process.returncode, process.stdout, process.stderr
  except Exception as e:  # pylint: disable=broad-except
    logging.exception('Local command failed: %s', e)
    return -1, '', str(e)


def _run_remote_command(host: str, cmd_list: list[str]) -> tuple[int, str, str]:
  """Runs a remote command via standard SSH."""
  if host == 'local':
    return _run_command(cmd_list)

  full_cmd = ['ssh', host] + cmd_list
  logging.info('Running remote command on [%s]: %s', host, ' '.join(cmd_list))
  try:
    process = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        check=False,
        errors='ignore',
    )
    return process.returncode, process.stdout, process.stderr
  except Exception as e:  # pylint: disable=broad-except
    logging.exception('Remote command failed: %s', e)
    return -1, '', str(e)


def _copy_to_host(host: str, src: str, dest: str) -> bool:
  """Copies a file to a remote host via standard SCP."""
  if not os.path.exists(src):
    logging.error('Local file missing: %s', src)
    return False

  if host == 'local':
    try:
      os.makedirs(dest, exist_ok=True)
      shutil.copy(src, dest)
      return True
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Local copy failed: %s', e)
      return False

  scp_cmd = ['scp', '-r', src, f'{host}:{dest}/']
  logging.info('Copying file to [%s]: %s -> %s', host, src, dest)
  try:
    subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
    return True
  except Exception as e:  # pylint: disable=broad-except
    logging.exception('SCP failed: %s', e)
    return False


@gin.configurable
class PropellerWorker(worker.Worker):
  """A generic worker that evaluates Propeller policies via SSH/SCP."""

  def __init__(
      self,
      *,
      initial_clang_path: str,
      perf_profile_path: str,
      propeller_prof_gen_path: str,
      remote_workdir: str,
      link_cmd: list[str],
      benchmark_cmd: list[str],
      base_policy_path: Optional[str] = None,
  ):
    """Initializes the PropellerWorker.

    Args:
      initial_clang_path: Path to the initial Clang binary (local).
      perf_profile_path: Path to the input perf profile (local).
      propeller_prof_gen_path: Path to generate_propeller_profiles binary
        (local).
      remote_workdir: Directory on the remote host to use for execution.
      link_cmd: Command to run on remote host to link Clang.
      benchmark_cmd: Command to run on remote host to benchmark Clang.
      base_policy_path: Optional path to base policy for TFLite conversion.
    """
    self._initial_clang_path = initial_clang_path
    self._perf_profile_path = perf_profile_path
    self._propeller_prof_gen_path = propeller_prof_gen_path
    self._remote_workdir = remote_workdir
    self._link_cmd = link_cmd
    self._benchmark_cmd = benchmark_cmd
    self._base_policy_path = base_policy_path
    self._eval_counter = 0

  def compile(
      self,
      policy: Optional[bytes],
      modules: list,  # Ignored, but kept for interface compatibility
      gcp_host: Optional[str] = None,
      perf_profile_path: Optional[str] = None,
      perturbation_index: Optional[int] = 0,
  ) -> Optional[float]:
    # pylint: disable=unused-argument
    """Evaluates a policy by generating profiles, linking, and benchmarking."""
    host = gcp_host
    if not host:
      raise ValueError("gcp_host must be provided")

    self._eval_counter += 1
    eval_id = f'{host}-{self._eval_counter}'
    logging.info('[%s] Starting evaluation %s', host, eval_id)

    # Create a local temp dir for profile generation
    with tempfile.TemporaryDirectory(prefix='propeller_worker_') as tmp_dir:
      clang_input_path = self._initial_clang_path
      perf_data_path = perf_profile_path or self._perf_profile_path

      tflite_policy_dir = None
      if policy is not None:
        if not self._base_policy_path:
          raise ValueError('base_policy_path is required when policy is provided')
        local_policy_dir = os.path.join(tmp_dir, 'policy')
        tflite_policy_dir = policy_utils.convert_to_tflite(
            policy, local_policy_dir, self._base_policy_path
        )
        logging.info('Saved TFLite model to %s', tflite_policy_dir)

      # Output paths for generated profiles (local temp dir)
      cc_profile_path = os.path.join(tmp_dir, 'cc_profile.txt')
      ld_profile_path = os.path.join(tmp_dir, 'ld_profile.txt')

      prof_gen_cmd = [
          self._propeller_prof_gen_path,
          f'--binary={clang_input_path}',
          f'--profile={perf_data_path}',
          f'--cc_profile={cc_profile_path}',
          f'--ld_profile={ld_profile_path}',
          '--alsologtostderr',
      ]

      if tflite_policy_dir and policy is not None:
        options_str = (
            'code_layout_params {'
            ' inter_function_reordering: true, split_all_basic_blocks: true,'
            f" policy_path: '{tflite_policy_dir}'"
            '}'
        )
        prof_gen_cmd.append(f'--propeller_options="{options_str}"')
        prof_gen_cmd.append('--use_ml')
      else:
        options_str = (
            'code_layout_params { inter_function_reordering: false,'
            ' split_all_basic_blocks: true }'
        )
        prof_gen_cmd.append(f'--propeller_options="{options_str}"')

      # Run profile generation locally
      return_code, stdout, stderr = _run_command(prof_gen_cmd)
      if return_code != 0:
        logging.error('Profile generation failed')
        logging.error('STDOUT:\n%s', stdout)
        logging.error('STDERR:\n%s', stderr)
        return None

      logging.info('Profile generation succeeded.')

      # Copy profiles to remote host
      # We copy them to the configured remote_workdir

      # Ensure remote workdir exists
      _run_remote_command(host, ['mkdir', '-p', self._remote_workdir])

      if not _copy_to_host(host, cc_profile_path, self._remote_workdir):
        logging.error('Failed to copy cc_profile.txt to host')
        return None
      if not _copy_to_host(host, ld_profile_path, self._remote_workdir):
        logging.error('Failed to copy ld_profile.txt to host')
        return None

      # Run link command on remote host
      logging.info('Running link command on %s...', host)
      return_code, stdout, stderr = _run_remote_command(
          host, self._link_cmd
      )
      if return_code != 0:
        logging.error('Failed to link Clang on remote host')
        logging.error('STDOUT:\n%s', stdout)
        logging.error('STDERR:\n%s', stderr)
        return None
      logging.info('Link command succeeded.')

      # Run benchmark on remote host
      logging.info('Running benchmark on %s...', host)
      return_code, stdout, stderr = _run_remote_command(
          host, self._benchmark_cmd
      )
      if return_code != 0:
        logging.error('Failed to run benchmark on remote host')
        logging.error('STDOUT:\n%s', stdout)
        logging.error('STDERR:\n%s', stderr)
        return None
      logging.info('Benchmark command succeeded.')

      # Parse output for cycles (similar to verified worker)
      matches = re.findall(r'(\d[\d,]*)\s+cycles[:\w]*', stdout + '\n' + stderr)
      if matches:
        cycles_list = [float(m.replace(',', '')) for m in matches]
        # If we have multiple runs, discard first as warm-up and take median
        if len(cycles_list) > 1:
          warmed_cycles_list = cycles_list[1:]
          measured_cycles = float(np.median(warmed_cycles_list))
        else:
          measured_cycles = cycles_list[0]
        logging.info('[%s] Measured cycles: %f', host, measured_cycles)
        return measured_cycles
      else:
        logging.error('Failed to parse cycles from benchmark output')
        logging.error('Benchmark output was:\n%s', stdout + '\n' + stderr)
        return None
