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

mkdir -p /work/llvm-corpus
cmake -B /work/llvm-corpus \
  -GNinja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=clang \
  -DCMAKE_C_COMPILER=/work/llvm-train/bin/clang \
  -DCMAKE_CXX_COMPILER=/work/llvm-train/bin/clang++ \
  -DCMAKE_C_FLAGS="-Xclang=-fembed-bitcode=all" \
  -DCMAKE_CXX_FLAGS="-Xclang=-fembed-bitcode=all" \
  /work/llvm-project/llvm

ninja -C /work/llvm-corpus