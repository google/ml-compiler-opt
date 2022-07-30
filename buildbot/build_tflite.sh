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
rm -rf /src

neon_hash=cef9501d1d1c47223466bf2b8cd43f3368c37773
abseil_hash=dc370a82467cb35066475537b797197aee3e5164
cpuinfo_hash=beb46ca0319882f262e682dd596880c92830687f
eigen_hash=7896c7dc6bd1bd34dd9636bdd3426e3c28e6a246
flatbuffers_hash=6e2791640e789459078eece008d6200c18dda5da
protobuf_hash=a321b050f5d72f726d29d9f542bfaf2636b9138f
ruy_hash=72155b3185246e9143f4c6a3a7f283d2ebba8524
tensorflow_hash=4d5b6bee153d25a59f38ae9d743a5dc708471950

# cpuinfo
git clone https://github.com/pytorch/cpuinfo /src/cpuinfo && cd /src/cpuinfo
git checkout "${cpuinfo_hash}"
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX= -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
ninja
DESTDIR=/src/cpuinfo/output ninja install

# ruy
git clone https://github.com/google/ruy /src/ruy && cd /src/ruy
git checkout "${ruy_hash}"
mkdir build && cd build
cmake -GNinja -DRUY_MINIMAL_BUILD=ON -DCMAKE_INSTALL_PREFIX= -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dcpuinfo_DIR="/src/cpuinfo/output/share/cpuinfo" -DRUY_ENABLE_INSTALL=ON -DRUY_FIND_CPUINFO=ON ..
ninja
DESTDIR=/src/ruy/output ninja install

# absl
git clone https://github.com/abseil/abseil-cpp /src/abseil-cpp && cd /src/abseil-cpp
git checkout "${abseil_hash}"
mkdir build && cd build
cmake -GNinja -DABSL_ENABLE_INSTALL=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX= ..
ninja
DESTDIR=/src/abseil-cpp/output ninja install

# eigen
git clone https://gitlab.com/libeigen/eigen /src/eigen && cd /src/eigen
git checkout "${eigen_hash}"
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX= -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
ninja
DESTDIR=/src/eigen/output ninja install

# ARM_NEON_2_x86_SSE
git clone https://github.com/intel/ARM_NEON_2_x86_SSE /src/ARM_NEON_2_x86_SSE && cd /src/ARM_NEON_2_x86_SSE
git checkout "${neon_hash}"
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX= -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
ninja
DESTDIR=/src/ARM_NEON_2_x86_SSE/output ninja install

# flatbuffers
git clone https://github.com/google/flatbuffers /src/flatbuffers && cd /src/flatbuffers
git checkout "${flatbuffers_hash}"
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX= -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
ninja
DESTDIR=/src/flatbuffers/output ninja install

# tflite
git clone https://github.com/tensorflow/tensorflow /src/tensorflow && cd /src/tensorflow
git checkout "${tensorflow_hash}"
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX= -DTFLITE_ENABLE_XNNPACK=OFF -Dcpuinfo_DIR="/src/cpuinfo/output/share/cpuinfo" -Druy_DIR="/src/ruy/output/lib/cmake/ruy" -Dabsl_DIR="/src/abseil-cpp/output/lib/cmake/absl" -DABSL_ENABLE_INSTALL=ON -DRUY_ENABLE_INSTALL=ON ../tensorflow/lite
ninja
DESTDIR=/src/tensorflow/output ninja install

# protobuf
git clone https://github.com/protocolbuffers/protobuf /src/protobuf && cd /src/protobuf
git checkout "${protobuf_hash}"
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX= -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF ../cmake
ninja
DESTDIR=/src/protobuf/output ninja install
