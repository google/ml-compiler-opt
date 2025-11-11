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
"""Qualify a proposed IR benchmark, given the original IR"""

import math

from absl import flags

import toolchain

_EPSILON = flags.DEFINE_float(
    "qualifier_epsilon",
    0.05,
    "percentage variation of each dimension of the original IR stats within"
    " which the proposed IR is acceptable",
)

_EXTRA_STATS = flags.DEFINE_integer(
    "qualifier_extra_stats",
    10,
    "number of stats that the benchmark IR can have that don't exist in the"
    " original",
)


def _get_nr_branch_weights(ir_path: str) -> int:
  total = 0
  with open(ir_path) as f:
    for line in f:
      if """!"branch_weights""" in line:
        total += 1
  return total


async def _get_stats(
    builder: toolchain.BuildEnv,
    orig_ir_function_path: str,
    proposed_function_path: str,
):
  orig_stats = await builder.get_ir_stats(orig_ir_function_path)
  orig_stats["branch_weights"] = _get_nr_branch_weights(orig_ir_function_path)
  bm_stats = await builder.get_ir_stats(proposed_function_path)
  bm_stats["branch_weights"] = _get_nr_branch_weights(proposed_function_path)
  return (orig_stats, bm_stats)


async def get_distance(
    builder: toolchain.BuildEnv,
    orig_ir_function_path: str,
    proposed_function_path: str,
):
  (orig_stats, bm_stats) = await _get_stats(builder, orig_ir_function_path,
                                            proposed_function_path)
  s = 0
  all_keys = set(orig_stats.keys()).union(bm_stats.keys())
  for k in all_keys:
    bm_v = 0 if k not in bm_stats else bm_stats[k]
    orig_v = 0 if k not in orig_stats else orig_stats[k]
    s += math.pow((bm_v - orig_v), 2)
  return math.pow(s, 0.5)


async def compare(
    builder: toolchain.BuildEnv,
    orig_ir_function_path: str,
    proposed_function_path: str,
) -> bool:
  """Compare the stats of given and generated IR."""
  (orig_stats, bm_stats) = await _get_stats(builder, orig_ir_function_path,
                                            proposed_function_path)

  extras = _EXTRA_STATS.value
  for k, v in orig_stats.items():
    if k not in bm_stats:
      if extras == 0:
        return False
      extras -= 1
      continue
    low = (1 - _EPSILON.value) * v
    hi = (1 + _EPSILON.value) * v
    if low >= bm_stats[k] or hi <= bm_stats[k]:
      return False
  return True
