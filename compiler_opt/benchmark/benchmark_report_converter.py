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
r"""Convert benchmark results.json to csv.

To run:
python3 \
compiler_opt/benchmark/benchmark_report_converter.py \
  --base=/tmp/base_report.json \
  --exp=/tmp/exp_report.json \
  --counters=INSTRUCTIONS \
  --counters=CYCLES \
  --output=/tmp/summary.csv

optionally, add --suite_name=<name of benchmark>, if batch-processing multiple
benchmarks' reports.

Assuming /tmp/{base|exp}_report.json were produced from benchmark runs, which
were asked to collect the counters named INSTRUCTIONS and CYCLES.
"""

import csv
import json

from typing import Sequence

from absl import app
from absl import flags

import tensorflow.compat.v2 as tf

from compiler_opt.benchmark import benchmark_report

flags.DEFINE_string('suite_name', 'benchmark_suite',
                    'The name of the benchmark suite (for reporting).')
flags.DEFINE_string('base', None,
                    'JSON report produced by the base benchmark run.')
flags.DEFINE_string('exp', None,
                    'JSON report produced by the experiment benchmark run.')
flags.DEFINE_string('output', 'reports.csv', 'CSV output')
flags.DEFINE_multi_string(
    'counters', None,
    'Counter names. Should match exactly the names used when running the'
    'benchmark.')

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  with tf.io.gfile.GFile(FLAGS.base, 'r') as b:
    with tf.io.gfile.GFile(FLAGS.exp, 'r') as e:
      b = benchmark_report.BenchmarkReport(FLAGS.suite_name, json.load(b),
                                           FLAGS.counters)
      e = benchmark_report.BenchmarkReport(FLAGS.suite_name, json.load(e),
                                           FLAGS.counters)
      comparison = benchmark_report.BenchmarkComparison(b, e)
      summary = comparison.summarize()
  with tf.io.gfile.GFile(FLAGS.output, 'w+') as o:
    co = csv.writer(o)
    # Pylint suggest using items for some reason, but the key is still needed
    # pylint: disable=consider-using-dict-items
    for bm in summary:
      for c in summary[bm]:
        co.writerow([bm, c] + list(summary[bm][c]))


if __name__ == '__main__':
  app.run(main)
