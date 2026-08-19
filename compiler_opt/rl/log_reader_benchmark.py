# Copyright 2026 Google LLC
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
"""Benchmark for compiler_opt.rl.log_reader.

Generates a synthetic log in the "simple log format" consumed by
read_log_as_sequence_examples: a JSON header followed by raw binary tensor
buffers, with one float32 and one int64 feature tensor per observation and a
float32 score tensor, mirroring the feature mix used by the inlining and
regalloc problem configs.

The end-to-end runtime of read_log_as_sequence_examples is measured for the
current _add_feature implementation and for a reference copy of the previous
one. The two variants are measured in interleaved order, and the serialized
output of the two is checked to be byte-for-byte identical. Timings are
machine-dependent; results are reported as median, mean, p95 and a 95%
confidence interval.
"""

import ctypes
import json
import os
import statistics
import tempfile
import time

from absl import app
from absl import flags
import tensorflow as tf

from compiler_opt.rl import log_reader

flags.DEFINE_integer("observations", 8, "Number of observations to log.")
flags.DEFINE_integer("elems", 2_100_000, "Elements per feature tensor.")
flags.DEFINE_integer("samples", 10, "Interleaved samples per variant.")

FLAGS = flags.FLAGS


def _write_log(fname: str, observations: int, elems: int) -> None:
    """Writes a synthetic log file in the simple log format."""
    header = {
        "features": [
            {
                "name": "feature_f32",
                "port": 0,
                "shape": [elems],
                "type": "float",
            },
            {
                "name": "feature_i64",
                "port": 0,
                "shape": [elems],
                "type": "int64_t",
            },
        ],
        "score": {"name": "reward", "port": 0, "shape": [1], "type": "float"},
    }
    f32_bytes = b"\x00" * (ctypes.sizeof(ctypes.c_float) * elems)
    i64_bytes = b"\x00" * (ctypes.sizeof(ctypes.c_int64) * elems)
    with open(fname, "wb") as f:
        f.write(json.dumps(header).encode("utf-8"))
        f.write(b"\n")
        for _ in range(observations):
            f.write(b'{"context": "context_0"}\n')
            f.write(b'{"observation": 0}\n')
            f.write(f32_bytes)
            f.write(i64_bytes)
            f.write(b"\n")
            f.write(b'{"outcome": 0}\n')
            f.write(b"\x00" * ctypes.sizeof(ctypes.c_float))
            f.write(b"\n")


def _add_feature_original(
    se: tf.train.SequenceExample, spec: tf.TensorSpec, value: log_reader.LogReaderTensorValue
):
    """Reference copy of the previous _add_feature implementation."""
    f = se.feature_lists.feature_list[spec.name].feature.add()
    if spec.dtype not in log_reader._dtype_to_ctype:
        raise ValueError(f"Unsupported dtype: f{spec.dtype}")
    if spec.dtype in [tf.float32, tf.float64]:
        lst = f.float_list.value
    else:
        lst = f.int64_list.value
    lst.extend(value)


def _time_parse(fname: str, add_feature) -> tuple[float, dict[str, tf.train.SequenceExample]]:
    """Times read_log_as_sequence_examples with the given _add_feature."""
    log_reader._add_feature = add_feature  # pylint: disable=protected-access
    start = time.perf_counter()
    result = log_reader.read_log_as_sequence_examples(fname)
    elapsed = time.perf_counter() - start
    return elapsed, result


def _stats(samples: list[float]) -> tuple[float, float, float, tuple[float, float]]:
    """Returns (median, mean, p95, 95% CI)."""
    median = statistics.median(samples)
    mean = statistics.mean(samples)
    sorted_samples = sorted(samples)
    p95 = sorted_samples[min(len(sorted_samples) - 1, int(0.95 * len(sorted_samples)))]
    half_width = 1.96 * statistics.stdev(samples) / len(samples) ** 0.5
    return median, mean, p95, (median - half_width, median + half_width)


def main(_):
    logfile = tempfile.NamedTemporaryFile(delete=False).name
    try:
        _write_log(logfile, FLAGS.observations, FLAGS.elems)
        original_times = []
        current_times = []
        serialized_bytes = None
        for _ in range(FLAGS.samples):
            t_orig, se_orig = _time_parse(logfile, _add_feature_original)
            t_cur, se_cur = _time_parse(logfile, log_reader._add_feature)
            original_times.append(t_orig)
            current_times.append(t_cur)
            for key in se_orig:
                if serialized_bytes is None:
                    serialized_bytes = len(se_orig[key].SerializeToString())
                assert se_orig[key].SerializeToString() == se_cur[key].SerializeToString()
        m_orig, mean_orig, p95_orig, ci_orig = _stats(original_times)
        m_cur, mean_cur, p95_cur, ci_cur = _stats(current_times)
        print(
            f"read_log_as_sequence_examples: observations={FLAGS.observations}, "
            f"elems/feature={FLAGS.elems}, samples={FLAGS.samples}"
        )
        print(
            f"  original: median={m_orig:.3f}s mean={mean_orig:.3f}s "
            f"p95={p95_orig:.3f}s 95% CI=({ci_orig[0]:.3f},{ci_orig[1]:.3f})"
        )
        print(
            f"    numpy: median={m_cur:.3f}s mean={mean_cur:.3f}s "
            f"p95={p95_cur:.3f}s 95% CI=({ci_cur[0]:.3f},{ci_cur[1]:.3f})"
        )
        print(
            f"speedup (median): {m_orig / m_cur:.2f}x ({100 * (m_orig - m_cur) / m_orig:.1f}% faster)"
        )
        print(f"serialized output identical ({serialized_bytes} bytes per context)")
    finally:
        os.unlink(logfile)


if __name__ == "__main__":
    app.run(main)
