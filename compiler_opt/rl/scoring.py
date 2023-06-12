import dataclasses
import os
import subprocess
import tempfile
from typing import Callable, Dict, List

from compiler_opt.rl import corpus, env


def _extract_fn_from_module(module_ir: bytes, fn: str,
                            llvm_extract_path: str) -> bytes:
  cmdline = [llvm_extract_path, '-func', fn, '-', '-o', '-']
  proc = subprocess.run(
      cmdline, input=module_ir, stdout=subprocess.PIPE, check=True)
  return proc.stdout


def for_each_extracted_fn_from_module(module: corpus.LoadedModuleSpec,
                                      fns: List[str], foreach_fn: Callable[
                                          [corpus.LoadedModuleSpec, str], None],
                                      llvm_extract_path: str) -> None:
  for fn in fns:
    extracted_fn_bc = _extract_fn_from_module(module.loaded_ir, fn,
                                              llvm_extract_path)
    extracted_fn_module = dataclasses.replace(
        module, name='extracted_fn_module', loaded_ir=extracted_fn_bc)
    foreach_fn(extracted_fn_module, fn)


def compute_per_fn_score(module: corpus.LoadedModuleSpec, fns: List[str],
                         score_module_fn: Callable[[corpus.LoadedModuleSpec],
                                                   float],
                         llvm_extract_path: str) -> Dict[str, float]:
  scores = {}

  def _foreach_fn(fn_module: corpus.LoadedModuleSpec, fn_name: str):
    scores[fn_name] = score_module_fn(fn_module)

  for_each_extracted_fn_from_module(
      module=module,
      fns=fns,
      foreach_fn=_foreach_fn,
      llvm_extract_path=llvm_extract_path)
  return scores


def get_compiled_module_size(compiled_module_path: str,
                             llvm_size_path: str) -> int:
  """Compute the size of a compiled module using llvm-size.

  Args:
    compiled_module_path: Path to the compiled module.
    llvm_size_path: Path to a llvm-size executable.

  Returns:
    The size, in bytes, of the object file.
  """
  cmdline = [llvm_size_path, compiled_module_path]
  completed_proc = subprocess.run(cmdline, capture_output=True, check=True)
  if not completed_proc.stdout:
    raise RuntimeError(f'Empty llvm-size output: {" ".join(cmdline)}')
  output = completed_proc.stdout.decode('utf-8')
  tmp = output.split('\n')
  if len(tmp) != 3:
    raise RuntimeError(f'Wrong llvm-size output {output}')
  tmp = tmp[1].split('\t')
  native_size = int(tmp[0])
  return native_size


def compile_module_and_get_size(module: corpus.LoadedModuleSpec,
                                clang_path: str, llvm_size_path: str) -> float:

  def _score_fn(compiled_module_path: str) -> Dict[str, float]:
    return {
        'default':
            get_compiled_module_size(
                compiled_module_path=compiled_module_path,
                llvm_size_path=llvm_size_path)
    }

  task_type = env.get_simple_compile_and_score_task_type(score_fn=_score_fn)
  with env.clang_session(
      clang_path=clang_path,
      module=module,
      task_type=task_type,
      interactive=False) as clang_session:
    return clang_session.get_scores()['default']


def for_each_fn_compile_and_get_size(
    module: corpus.LoadedModuleSpec, fns: List[str], clang_path: str,
    llvm_size_path: str, llvm_extract_path: str) -> Dict[str, float]:

  def _score_fn(fn_module: corpus.LoadedModuleSpec) -> float:
    return compile_module_and_get_size(
        module=fn_module, clang_path=clang_path, llvm_size_path=llvm_size_path)

  return compute_per_fn_score(
      module=module,
      fns=fns,
      score_module_fn=_score_fn,
      llvm_extract_path=llvm_extract_path)
