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
"""Tests for compiler_opt.tools.benchmark_report_converter."""

from absl.testing import absltest

from compiler_opt.benchmark import benchmark_report

base_data = {
    'benchmarks': [
        {
            'PerfCounter_0': 10,
            'PerfCounter_1': 20,
            'iterations': 10,
            'name': 'BM_A',
        },
        {
            'PerfCounter_0': 11,
            'PerfCounter_1': 19,
            'iterations': 11,
            'name': 'BM_A',
        },
        {
            'PerfCounter_0': 60,
            'PerfCounter_1': 50,
            'iterations': 15,
            'name': 'BM_B',
        },
    ]
}

exp_data = {
    'benchmarks': [
        {
            'PerfCounter_0': 9,
            'PerfCounter_1': 11,
            'iterations': 11,
            'name': 'BM_A',
        },
        {
            'PerfCounter_0': 8,
            'PerfCounter_1': 10,
            'iterations': 8,
            'name': 'BM_A',
        },
        {
            'PerfCounter_0': 62,
            'PerfCounter_1': 54,
            'iterations': 14,
            'name': 'BM_B',
        },
    ]
}


class BenchmarkReportConverterTest(absltest.TestCase):

  def test_loading(self):
    report = benchmark_report.BenchmarkReport(
        'foo', base_data, ['PerfCounter_0', 'PerfCounter_1'])
    self.assertEqual(
        report.values(), {
            'BM_A': {
                'PerfCounter_0': [10, 11],
                'PerfCounter_1': [20, 19]
            },
            'BM_B': {
                'PerfCounter_0': [60],
                'PerfCounter_1': [50],
            }
        })
    self.assertSetEqual(report.names(), set(['BM_A', 'BM_B']))
    self.assertSetEqual(report.counters(),
                        set(['PerfCounter_0', 'PerfCounter_1']))
    self.assertEqual(
        report.counter_means('BM_A', 'PerfCounter_0'),
        (10.488088481701517, 0.7071067811865476))

  def test_summarize_results(self):
    b_values = benchmark_report.BenchmarkReport(
        'foo', base_data, ['PerfCounter_0', 'PerfCounter_1'])
    e_values = benchmark_report.BenchmarkReport(
        'foo', exp_data, ['PerfCounter_0', 'PerfCounter_1'])
    summary = benchmark_report.BenchmarkComparison(b_values, e_values)
    self.assertDictEqual(
        summary.summarize(), {
            'BM_A': {
                'PerfCounter_0': (0.19096016504410973, 0.0674199862463242,
                                  0.08333333333333334),
                'PerfCounter_1':
                    (0.4619724131510293, 0.0362738125055006, 0.0674199862463242)
            },
            'BM_B': {
                'PerfCounter_0': (-0.03333333333333366, 0.0, 0.0),
                'PerfCounter_1': (-0.0800000000000003, 0.0, 0.0)
            }
        })
    self.assertEqual(
        summary.total_improvement('PerfCounter_0'), 0.08566536243319522)


if __name__ == '__main__':
  absltest.main()
