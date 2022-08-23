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
set -e
cd /
git clone --depth 1 https://github.com/llvm/llvm-project
mkdir /llvm-build
cd /llvm-build
apt-get install -y lld
cmake -G Ninja \
  -DLLVM_ENABLE_LTO=OFF \
  -DLINUX_x86_64-unknown-linux-gnu_SYSROOT=/fuchsia-sysroot/linux-x64 \
  -DLINUX_aarch64-unknown-linux-gnu_SYSROOT=/fuchsia-sysroot/linux-arm64 \
  -DFUCHSIA_SDK=/fuchsia-idk \
  -DCMAKE_INSTALL_PREFIX= \
  -DTENSORFLOW_C_LIB_PATH=/tmp/tensorflow \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -C /llvm-project/clang/cmake/caches/Fuchsia-stage2.cmake \
  /llvm-project/llvm
ninja distribution
DESTDIR=/llvm-install ninja install-distribution-stripped
cd /fuchsia
python3 scripts/clang/generate_runtimes.py \
    --clang-prefix=/llvm-install \
    --sdk-dir=/fuchsia-idk \
    --build-id-dir=/llvm-install/lib/.build-id > /llvm-install/lib/runtime.json
