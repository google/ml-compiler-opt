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
"""Utility functions to set cpu affinities when operating main and subprocesses
simultaneously."""
import gin
import psutil
import itertools

_NR_CPUS = psutil.cpu_count()

_CPU_CONFIG = {  # List of CPU numbers in cache-sharing order.
    # 'google-epyc' assumes logical core 0 and N/2 are the same physical core.
    # Also, L3 cache is assumed to be shared between consecutive core numbers.
    'google-epyc':
        list(
            itertools.chain(
                *zip(range(_NR_CPUS // 2), range(_NR_CPUS // 2, _NR_CPUS))))
}


@gin.configurable
def set_and_get(is_main_process: bool,
                max_cpus=_NR_CPUS,
                min_main_cpu: int = 32,
                arch: str = 'google-epyc'):
  """
  Sets the cpu affinity of the current process to appropriate values, and
  returns the list of cpus the process is set to use.
  Args:
    is_main_process: whether the caller is the main process.
    max_cpus: maximal number of cpus to use
    min_main_cpu: number of cpus to assign to the main process.
    arch: the system type, used to infer the cpu cache architecture.
  """
  config = _CPU_CONFIG[arch][:max_cpus]
  if is_main_process:
    cpus = config[:min_main_cpu]
  else:
    cpus = config[min_main_cpu:]
  if len(cpus) == 0:
    raise ValueError('Attempting to set cpu affinity of process to nothing.')
  psutil.Process().cpu_affinity(cpus)
  return list(cpus)
