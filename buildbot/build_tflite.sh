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

if [[ -z "${TFLITE_SYSROOT}" ]]; then
  EXTRA_CMAKE_FLAGS=" "
else
  EXTRA_CMAKE_FLAGS="-DCMAKE_SYSROOT=${TFLITE_SYSROOT} "
fi

echo extra cmake flags: ${EXTRA_CMAKE_FLAGS}

readonly CPUINFO_REPOSITORY="https://github.com/pytorch/cpuinfo"
readonly CPUINFO_TAG="ef634603954d88d2643d5809011288b890ac126e"

readonly RUY_REPOSITORY="https://github.com/google/ruy"
readonly RUY_TAG="3286a34cc8de6149ac6844107dfdffac91531e72"

readonly ABSEIL_REPOSITORY="https://github.com/abseil/abseil-cpp"
readonly ABSEIL_TAG="fb3621f4f897824c0dbe0615fa94543df6192f30"

readonly EIGEN_REPOSITORY="https://gitlab.com/libeigen/eigen"
readonly EIGEN_TAG="aa6964bf3a34fd607837dd8123bc42465185c4f8"

readonly NEON_2_SSE_REPOSITORY="https://github.com/intel/ARM_NEON_2_x86_SSE"
readonly NEON_2_SSE_TAG="a15b489e1222b2087007546b4912e21293ea86ff"

readonly FLATBUFFERS_REPOSITORY="https://github.com/google/flatbuffers"
readonly FLATBUFFERS_TAG="7d6d99c6befa635780a4e944d37ebfd58e68a108"

readonly GEMMLOWP_REPOSITORY="https://github.com/google/gemmlowp"
readonly GEMMLOWP_TAG="16e8662c34917be0065110bfcd9cc27d30f52fdf"

readonly ML_DTYPES_REPOSITORY="https://github.com/jax-ml/ml_dtypes"
readonly ML_DTYPES_TAG="2ca30a2b3c0744625ae3d6988f5596740080bbd0"

readonly PTHREADPOOL_REPOSITORY="https://github.com/Maratyszcza/pthreadpool"
readonly PTHREADPOOL_TAG="4fe0e1e183925bf8cfa6aae24237e724a96479b8"

readonly TENSORFLOW_REPOSITORY="https://github.com/tensorflow/tensorflow"
readonly TENSORFLOW_TAG="v2.16.1"

# cpuinfo
git clone --filter=tree:0 --no-checkout ${CPUINFO_REPOSITORY} cpuinfo/src/cpuinfo
git -C cpuinfo/src/cpuinfo checkout ${CPUINFO_TAG}
cmake -GNinja -S cpuinfo/src/cpuinfo -B cpuinfo/src/cpuinfo-build \
  ${EXTRA_CMAKE_FLAGS} \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/cpuinfo \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -DCPUINFO_BUILD_UNIT_TESTS:BOOL=OFF \
  -DCPUINFO_BUILD_MOCK_TESTS:BOOL=OFF \
  -DCPUINFO_BUILD_BENCHMARKS:BOOL=OFF
ninja -C cpuinfo/src/cpuinfo-build install

# ruy
git clone --filter=tree:0 --no-checkout ${RUY_REPOSITORY} ruy/src/ruy
git -C ruy/src/ruy checkout ${RUY_TAG}
cmake -GNinja -S ruy/src/ruy -B ruy/src/ruy-build \
  ${EXTRA_CMAKE_FLAGS} \
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
git clone --filter=tree:0 --no-checkout ${ABSEIL_REPOSITORY} abseil-cpp/src/abseil-cpp
git -C abseil-cpp/src/abseil-cpp checkout ${ABSEIL_TAG}
cmake -GNinja -S abseil-cpp/src/abseil-cpp -B abseil-cpp/src/abseil-cpp-build \
  ${EXTRA_CMAKE_FLAGS} \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/abseil-cpp \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -DABSL_BUILD_TESTING:BOOL=OFF \
  -DABSL_ENABLE_INSTALL:BOOL=ON
ninja -C abseil-cpp/src/abseil-cpp-build install

# eigen
git clone --filter=tree:0 --no-checkout ${EIGEN_REPOSITORY} eigen/src/eigen
git -C eigen/src/eigen checkout ${EIGEN_TAG}
cmake -GNinja -S eigen/src/eigen -B eigen/src/eigen-build \
  ${EXTRA_CMAKE_FLAGS} \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/eigen \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -DEIGEN_BUILD_DOC:BOOL=OFF \
  -DEIGEN_BUILD_TESTING:BOOL=OFF
ninja -C eigen/src/eigen-build install

# ARM_NEON_2_x86_SSE
git clone --filter=tree:0 --no-checkout ${NEON_2_SSE_REPOSITORY} ARM_NEON_2_x86_SSE/src/ARM_NEON_2_x86_SSE
git -C ARM_NEON_2_x86_SSE/src/ARM_NEON_2_x86_SSE checkout ${NEON_2_SSE_TAG}
cmake -GNinja -S ARM_NEON_2_x86_SSE/src/ARM_NEON_2_x86_SSE -B ARM_NEON_2_x86_SSE/src/ARM_NEON_2_x86_SSE-build \
  ${EXTRA_CMAKE_FLAGS} \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/ARM_NEON_2_x86_SSE \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
ninja -C ARM_NEON_2_x86_SSE/src/ARM_NEON_2_x86_SSE-build install

# flatbuffers
git clone --filter=tree:0 --no-checkout ${FLATBUFFERS_REPOSITORY} flatbuffers/src/flatbuffers
git -C flatbuffers/src/flatbuffers checkout ${FLATBUFFERS_TAG}
cmake -GNinja -S flatbuffers/src/flatbuffers -B flatbuffers/src/flatbuffers-build \
  ${EXTRA_CMAKE_FLAGS} \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/flatbuffers \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -DFLATBUFFERS_BUILD_TESTS:BOOL=OFF
ninja -C flatbuffers/src/flatbuffers-build install

# gemmlowp
git clone --filter=tree:0 --no-checkout ${GEMMLOWP_REPOSITORY} gemmlowp/src/gemmlowp
git -C gemmlowp/src/gemmlowp checkout ${GEMMLOWP_TAG}
cmake -GNinja -S gemmlowp/src/gemmlowp/contrib -B gemmlowp/src/gemmlowp-build \
  ${EXTRA_CMAKE_FLAGS} \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/gemmlowp \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
ninja -C gemmlowp/src/gemmlowp-build install

# ml_dtypes
git clone --filter=tree:0 --no-checkout ${ML_DTYPES_REPOSITORY} ml_dtypes/src/ml_dtypes
git -C ml_dtypes/src/ml_dtypes checkout ${ML_DTYPES_TAG}

# pthreadpool
# NOTE: currently a hack, because pthreadpool doesn't support find_package.
# we install in the default install dir, which really means the script must be run
# under sudo. Works for buildbots. Not ideal elsewhere.
if [[ -z "${TFLITE_SYSROOT}" ]]; then
  PTHREADPOOL_INSTALL_PREFIX="/usr/local"
else
  PTHREADPOOL_INSTALL_PREFIX="${TFLITE_SYSROOT}"
fi
git clone --filter=tree:0 --no-checkout ${PTHREADPOOL_REPOSITORY} pthreadpool/src/pthreadpool
git -C pthreadpool/src/pthreadpool checkout ${PTHREADPOOL_TAG}
cmake -GNinja -S pthreadpool/src/pthreadpool -B pthreadpool/src/pthreadpool-build \
  ${EXTRA_CMAKE_FLAGS} \
  -DCMAKE_INSTALL_PREFIX:PATH=${PTHREADPOOL_INSTALL_PREFIX} \
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
  -DPTHREADPOOL_BUILD_TESTS:BOOL=OFF \
  -DPTHREADPOOL_BUILD_BENCHMARKS:BOOL=OFF
ninja -C pthreadpool/src/pthreadpool-build install

# tflite
git clone --filter=tree:0 --no-checkout ${TENSORFLOW_REPOSITORY} tensorflow/src/tensorflow
git -C tensorflow/src/tensorflow checkout ${TENSORFLOW_TAG}
cmake -GNinja -S tensorflow/src/tensorflow/tensorflow/lite -B tensorflow/src/tensorflow-build \
  ${EXTRA_CMAKE_FLAGS} \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/tensorflow \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG:BOOL=ON \
  -DTFLITE_ENABLE_INSTALL:BOOL=ON \
  -DTFLITE_ENABLE_XNNPACK:BOOL=OFF \
  -DSYSTEM_PTHREADPOOL=ON \
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
