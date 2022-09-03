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
"""Analysis for benchmark results.json."""

import collections
import math
import statistics

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from absl import logging

# For each benchmark, and for each counter, capture the recorded values.
PerBenchmarkResults = Dict[str, Dict[str, List[float]]]

# Benchmark data, as captured by the benchmark json output: a dictionary from
# benchmark names to a list of run results. Each run result is a dictionary of
# key-value pairs, e.g. counter name - value.
BenchmarkRunResults = Dict[str, List[Dict[str, Any]]]

# A comparison per benchmark, per counter, capturing the geomean and the stdev
# of the base and experiment values.
ABComparison = Dict[str, Dict[str, Tuple[float, float, float]]]


def _geomean(data: List[float]):
  return math.exp(sum(math.log(x) for x in data) / len(data))


def _stdev(data: List[float]):
  assert data
  return 0.0 if len(data) == 1 else statistics.stdev(data)


class BenchmarkReport:
  """The counter values collected for benchmarks in a benchmark suite."""

  def __init__(self, suite_name: str, json_data: BenchmarkRunResults,
               counter_names: Iterable[str]):
    self._suite_name = suite_name
    self._load_values(json_data, counter_names)

  def suite_name(self):
    return self._suite_name

  def values(self):
    return self._values

  def names(self):
    return self._names

  def counters(self):
    return self._counters

  def raw_measurements(self):
    return self._raw_measurements

  def counter_means(self, benchmark: str, counter: str) -> Tuple[float, float]:
    if counter not in self.counters():
      raise ValueError('unknown counter')
    if benchmark not in self.names():
      raise ValueError('unknown benchmark')
    return (_geomean(self._values[benchmark][counter]),
            _stdev(self._values[benchmark][counter]))

  def zero_counters(self):
    ret = set()
    for name in self.names():
      for counter in self.values()[name]:
        if 0.0 in self.values()[name][counter]:
          ret.add((name, counter))
    return frozenset(ret)

  def large_variation_counters(self, variation: float):
    ret = set()
    for name in self.names():
      for counter in self.values()[name]:
        vals = self.values()[name][counter]
        swing = _stdev(vals) / _geomean(vals)
        if swing > variation:
          ret.add((name, counter, swing))
    return frozenset(ret)

  def _load_values(self, data: BenchmarkRunResults,
                   names: Iterable[str]) -> None:
    """Organize json values per-benchmark, per counter.

    Args:
        data: json data
        names: perf counter names
    Returns:
        benchmark data organized per-benchmark, per-counter name.
    """
    runs = data['benchmarks']
    self._values = collections.defaultdict(
        lambda: collections.defaultdict(list))
    self._raw_measurements = collections.defaultdict(
        lambda: collections.defaultdict(list))
    self._counters = set()
    self._names = set()

    for r in runs:
      benchmark_name = r['name']
      for counter in names:
        value = float(r[counter])
        iters = float(r['iterations'])
        self._raw_measurements[benchmark_name][counter].append(value * iters)
        self._values[benchmark_name][counter].append(value)
        self._counters.add(counter)
        self._names.add(benchmark_name)
    self._counters = frozenset(self._counters)
    self._names = frozenset(self._names)


class BenchmarkComparison:
  """Analysis of 2 benchmark runs."""

  def __init__(self, base_report: BenchmarkReport, exp_report: BenchmarkReport):
    base_names_set = set(base_report.names())
    exp_names_set = set(exp_report.names())
    if base_report.suite_name() != exp_report.suite_name():
      raise ValueError('cannot compare different suites')
    if base_names_set != exp_names_set:
      diff_base_exp = base_names_set.difference(exp_names_set)
      diff_exp_base = exp_names_set.difference(base_names_set)
      diff_set = diff_base_exp.union(diff_exp_base)
      logging.info('The following tests differ between the test suites: %s',
                   diff_set)
      raise ValueError('suite runs have different benchmark names')
    if set(base_report.counters()) != set(exp_report.counters()):
      raise ValueError(
          'counter names are different between base and experiment')

    self._base = base_report
    self._exp = exp_report

  def suite_name(self):
    return self._base.suite_name()

  def summarize(self) -> ABComparison:
    """Summarize the results from two runs (base/experiment).

    Returns:
      A per benchmark, per counter summary of the improvement/regression
      between the 2 runs, in percents.
    """
    base_results = self._base.values()
    exp_results = self._exp.values()

    ret = {}
    for bname in base_results:
      ret[bname] = {}
      for counter in base_results[bname]:
        base_vals = base_results[bname][counter]
        exp_vals = exp_results[bname][counter]
        base_geomean = _geomean(base_vals)
        exp_geomean = _geomean(exp_vals)
        improvement = 1 - exp_geomean / base_geomean
        base_stdev = _stdev(base_vals)
        exp_stdev = _stdev(exp_vals)
        ret[bname][counter] = (improvement, base_stdev / base_geomean,
                               exp_stdev / exp_geomean)
    return ret

  def names(self):
    return self._base.names()

  def counters(self):
    return self._base.counters()

  def total_improvement(self, counter: str):
    assert counter in self.counters()
    logsum = 0
    # we look at the geomean of the improvement for each benchmark
    for bname in self.names():
      b_geomean, _ = self._base.counter_means(bname, counter)
      e_geomean, _ = self._exp.counter_means(bname, counter)
      logsum += math.log(e_geomean / b_geomean)
    return 1.0 - math.exp(logsum / len(self.names()))
