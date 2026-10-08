#!/bin/bash
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


set -e

readonly CPUINFO_REPOSITORY="https://github.com/pytorch/cpuinfo"
readonly CPUINFO_TAG="05332fd802d9109a2a151ec32154b107c1e5caf9"

readonly RUY_REPOSITORY="https://github.com/google/ruy"
readonly RUY_TAG="c08ec529fc91722bde519628d9449258082eb847"

readonly ABSEIL_REPOSITORY="https://github.com/abseil/abseil-cpp"
readonly ABSEIL_TAG="1278ee9bd9bd4916181521fac96d6fa1100e38e6"

readonly EIGEN_REPOSITORY="https://gitlab.com/libeigen/eigen"
readonly EIGEN_TAG="d791d48859c6fc7850c9fd5270d2b236c818068d"

readonly NEON_2_SSE_REPOSITORY="https://github.com/intel/ARM_NEON_2_x86_SSE"
readonly NEON_2_SSE_TAG="6315d3c5007e6c209eb77abae4deece4978d8dbc"

readonly FLATBUFFERS_REPOSITORY="https://github.com/google/flatbuffers"
readonly FLATBUFFERS_TAG="fb9afbafc7dfe226b9db54d4923bfb8839635274"

readonly GEMMLOWP_REPOSITORY="https://github.com/google/gemmlowp"
readonly GEMMLOWP_TAG="16e8662c34917be0065110bfcd9cc27d30f52fdf"

readonly ML_DTYPES_REPOSITORY="https://github.com/jax-ml/ml_dtypes"
readonly ML_DTYPES_TAG="f739b2f0256d543e68caf214cd54b367302fbf68"

readonly TENSORFLOW_REPOSITORY="https://github.com/tensorflow/tensorflow"
readonly TENSORFLOW_TAG="bbed5f60cd2efbbc3f1032a4e7f9fe393a1760af"

function clone_and_checkout {
  local repo="$1"
  local tag="$2"
  local dest="$3"
  local sleep_seconds=10
  for i in $(seq 1 5); do
    rm -rf "${dest}"
    if git clone --filter=tree:0 --no-checkout "${repo}" "${dest}" && \
       git -C "${dest}" checkout "${tag}"; then
      return 0
    fi
    sleep "${sleep_seconds}"
    sleep_seconds=$((sleep_seconds + 10))
  done
  return 1
}

# cpuinfo
clone_and_checkout ${CPUINFO_REPOSITORY} ${CPUINFO_TAG} cpuinfo/src/cpuinfo
cmake -GNinja -S cpuinfo/src/cpuinfo -B cpuinfo/src/cpuinfo-build \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/cpuinfo \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -DCPUINFO_BUILD_UNIT_TESTS:BOOL=OFF \
  -DCPUINFO_BUILD_MOCK_TESTS:BOOL=OFF \
  -DCPUINFO_BUILD_BENCHMARKS:BOOL=OFF
ninja -C cpuinfo/src/cpuinfo-build install

# ruy
clone_and_checkout ${RUY_REPOSITORY} ${RUY_TAG} ruy/src/ruy
cmake -GNinja -S ruy/src/ruy -B ruy/src/ruy-build \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/ruy \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -Dcpuinfo_DIR:PATH=${PWD}/cpuinfo/share/cpuinfo \
  -DRUY_MINIMAL_BUILD:BOOL=ON \
  -DRUY_ENABLE_INSTALL:BOOL=ON \
  -DRUY_FIND_CPUINFO:BOOL=ON
ninja -C ruy/src/ruy-build install

# absl
clone_and_checkout ${ABSEIL_REPOSITORY} ${ABSEIL_TAG} abseil-cpp/src/abseil-cpp
cmake -GNinja -S abseil-cpp/src/abseil-cpp -B abseil-cpp/src/abseil-cpp-build \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/abseil-cpp \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -DABSL_BUILD_TESTING:BOOL=OFF \
  -DABSL_ENABLE_INSTALL:BOOL=ON
ninja -C abseil-cpp/src/abseil-cpp-build install

# eigen
clone_and_checkout ${EIGEN_REPOSITORY} ${EIGEN_TAG} eigen/src/eigen
cmake -GNinja -S eigen/src/eigen -B eigen/src/eigen-build \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/eigen \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -DEIGEN_BUILD_DOC:BOOL=OFF \
  -DEIGEN_BUILD_TESTING:BOOL=OFF
ninja -C eigen/src/eigen-build install

# ARM_NEON_2_x86_SSE
clone_and_checkout ${NEON_2_SSE_REPOSITORY} ${NEON_2_SSE_TAG} ARM_NEON_2_x86_SSE/src/ARM_NEON_2_x86_SSE
cmake -GNinja -S ARM_NEON_2_x86_SSE/src/ARM_NEON_2_x86_SSE -B ARM_NEON_2_x86_SSE/src/ARM_NEON_2_x86_SSE-build \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/ARM_NEON_2_x86_SSE \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
ninja -C ARM_NEON_2_x86_SSE/src/ARM_NEON_2_x86_SSE-build install

# flatbuffers
clone_and_checkout ${FLATBUFFERS_REPOSITORY} ${FLATBUFFERS_TAG} flatbuffers/src/flatbuffers
cmake -GNinja -S flatbuffers/src/flatbuffers -B flatbuffers/src/flatbuffers-build \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/flatbuffers \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -DFLATBUFFERS_BUILD_TESTS:BOOL=OFF
ninja -C flatbuffers/src/flatbuffers-build install

# gemmlowp
clone_and_checkout ${GEMMLOWP_REPOSITORY} ${GEMMLOWP_TAG} gemmlowp/src/gemmlowp
cmake -GNinja -S gemmlowp/src/gemmlowp/contrib -B gemmlowp/src/gemmlowp-build \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/gemmlowp \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON   \
  -DCMAKE_INSTALL_LIBDIR=lib
ninja -C gemmlowp/src/gemmlowp-build install

# ml_dtypes
clone_and_checkout ${ML_DTYPES_REPOSITORY} ${ML_DTYPES_TAG} ml_dtypes/src/ml_dtypes

# tflite
clone_and_checkout ${TENSORFLOW_REPOSITORY} ${TENSORFLOW_TAG} tensorflow/src/tensorflow
cmake -GNinja -S tensorflow/src/tensorflow/tensorflow/lite -B tensorflow/src/tensorflow-build \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/tensorflow \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DTFLITE_ENABLE_INSTALL:BOOL=ON \
  -DTFLITE_ENABLE_XNNPACK:BOOL=OFF \
  -Dcpuinfo_DIR:PATH=${PWD}/cpuinfo/share/cpuinfo \
  -Druy_DIR:PATH=${PWD}/ruy/lib/cmake/ruy \
  -Dabsl_DIR:PATH=${PWD}/abseil-cpp/lib/cmake/absl \
  -DEigen3_DIR:PATH=${PWD}/eigen/share/eigen3/cmake \
  -Dgemmlowp_DIR:PATH=${PWD}/gemmlowp/lib/cmake/gemmlowp \
  -DNEON_2_SSE_DIR:PATH=${PWD}/ARM_NEON_2_x86_SSE/lib/cmake/NEON_2_SSE \
  -DFlatBuffers_DIR:PATH=${PWD}/flatbuffers/lib/cmake/flatbuffers \
  -DML_DTYPES_SOURCE_DIR:PATH=${PWD}/ml_dtypes/src/ml_dtypes
ninja -C tensorflow/src/tensorflow-build install

# CMake cache file
cat <<EOF >>tflite.cmake
set(cpuinfo_DIR "${PWD}/cpuinfo/share/cpuinfo" CACHE PATH "")
set(ruy_DIR "${PWD}/ruy/lib/cmake/ruy" CACHE PATH "")
set(absl_DIR "${PWD}/abseil-cpp/lib/cmake/absl" CACHE PATH "")
set(Eigen3_DIR "${PWD}/eigen/share/eigen3/cmake" CACHE PATH "")
set(NEON_2_SSE_DIR "${PWD}/ARM_NEON_2_x86_SSE/lib/cmake/NEON_2_SSE" CACHE PATH "")
set(gemmlowp_DIR "${PWD}/gemmlowp/lib/cmake/gemmlowp" CACHE PATH "")
set(FlatBuffers_DIR "${PWD}/flatbuffers/lib/cmake/flatbuffers" CACHE PATH "")
set(tensorflow-lite_DIR "${PWD}/tensorflow/lib/cmake/tensorflow-lite" CACHE PATH "")
set(TENSORFLOW_SRC_DIR "${PWD}/tensorflow/src/tensorflow" CACHE PATH "")
set(LLVM_HAVE_TFLITE ON CACHE BOOL "")
EOF
