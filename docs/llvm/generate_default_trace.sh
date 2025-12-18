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

#!/bin/bash

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=/work/ml-compiler-opt:$PYTHONPATH \
  python /work/ml-compiler-opt/compiler_opt/tools/generate_default_trace.py \
  --data_path /work/corpus/modules \
  --gin_files /work/ml-compiler-opt/compiler_opt/rl/inlining/gin_configs/common.gin \
  --gin_bindings clang_path="'/work/llvm-train/bin/clang'" \
  --gin_bindings llvm_size_path="'/work/llvm-train/bin/llvm-size'" \
  --compilation_timeout=600 \
  --output_path /work/corpus/default_trace
