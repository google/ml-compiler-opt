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
"""Tool driving the toolchain BuildEnv

Intended for testing, this allows performing the actions supported by BuildEnv
one at a time
"""

import asyncio
from collections.abc import Sequence
import json
import tempfile

from absl import app
from absl import flags

from compiler_opt.tools.profgen import qualifier
from compiler_opt.tools.profgen import toolchain
import tensorflow as tf

gfile = tf.io.gfile

_ACTION = flags.DEFINE_enum(
    "action",
    default=None,
    enum_values=[
        "verify",
        "extract_profile",
        "extract_stats",
        "compare",
        "distance",
    ],
    help="""which toolchain action to perform:
    - extract_profile:  the `input` should be a self-contained C++ program.
    This option builds (-O0, i.e. default opt level) it with IR instrumentation
    and runs it. After that it recompiles it with that profile and extracts the
    IR for everything in the original input C++ file, except the `main`
    function. That IR is placed in the `output` directory.
    - extract_stats: uses llc -O2 -stats on the given `input`, which should be IR.
    - verify: checks that the given `input`, a C++ file, can be built to a
    standalone executable. Raises exeeption if not.
    - compare: compares 2 IR files, `input` and `original`, for similarity.
    - distance: produces a measure of distance between an `input` and `original`
    IR.
    """,
)

_INPUT = flags.DEFINE_string(
    "input", default=None, help="input file (cpp or ir, depends on the action)")
_ORIGINAL = flags.DEFINE_string(
    "original",
    default=None,
    help="original IR - used for the 'compare' action only",
)

_TOOLCHAIN = flags.DEFINE_string(
    "toolchain_dir",
    default=None,
    help="path to location containing toolchain binaries",
)

_TOOLCHAIN_VERSION = flags.DEFINE_string(
    "toolchain_version",
    default="",
    help="suffix to add to tool names. Do not include '-'",
)

_OUTPUT = flags.DEFINE_string(
    "output", default=None, help="where to place any output file")


async def async_main():
  with tempfile.TemporaryDirectory() as tempdir:
    tc = toolchain.BuildEnv(
        toolchain_path=_TOOLCHAIN.value,
        workingdir=tempdir,
        tool_suffix="" if not _TOOLCHAIN_VERSION.value else "-" +
        _TOOLCHAIN_VERSION.value,
    )
    match _ACTION.value:
      case "verify":
        await tc.verify_cpp(_INPUT.value)
        print("succeeded")
      case "extract_profile":
        result = await tc.generate_compiled_ir_with_profile(_INPUT.value)
        gfile.Copy(result, _OUTPUT.value)
      case "extract_stats":
        result = await tc.get_ir_stats(_INPUT.value)
        print(json.dumps(result))
      case "compare":
        print(await qualifier.compare(tc, _ORIGINAL.value, _INPUT.value))
      case "distance":
        print(await qualifier.get_distance(tc, _ORIGINAL.value, _INPUT.value))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  asyncio.run(async_main())


if __name__ == "__main__":
  flags.mark_flag_as_required("action")
  flags.mark_flag_as_required("input")
  flags.mark_flag_as_required("toolchain_dir")

  app.run(main)
