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

FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LLVM_MAJOR_VERSION=22

RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      automake \
      binutils-dev \
      ca-certificates \
      ccache \
      clang-${LLVM_MAJOR_VERSION} \
      cmake \
      cmake-data \
      curl \
      dos2unix \
      g++-multilib \
      gawk \
      gcc-multilib \
      git \
      gnupg \
      libgcrypt20-dev \
      libgss-dev \
      liblzma-dev \
      libssl-dev \
      libtool \
      libxml2-dev \
      lld-${LLVM_MAJOR_VERSION} \
      lsb-release \
      m4 \
      ninja-build \
      pkg-config \
      procps \
      software-properties-common \
      sudo \
      tmux \
      wget \
      zlib1g-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 \
      python3.11-dev \
      python3.11-distutils \
      python3.11-venv && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/clang-${LLVM_MAJOR_VERSION} /usr/bin/clang && \
    ln -sf /usr/bin/clang++-${LLVM_MAJOR_VERSION} /usr/bin/clang++ && \
    ln -sf /usr/bin/clang-${LLVM_MAJOR_VERSION} /usr/bin/cc && \
    ln -sf /usr/bin/clang++-${LLVM_MAJOR_VERSION} /usr/bin/c++ && \
    ln -sf /usr/bin/ld.lld-${LLVM_MAJOR_VERSION} /usr/bin/ld && \
    ln -sf /usr/bin/ld.lld-${LLVM_MAJOR_VERSION} /usr/bin/ld.lld && \
    rm -rf /var/lib/apt/lists/*

RUN cat <<'EOF' > /etc/ccache.conf
sloppiness = pch_defines,time_macros
depend_mode = true
run_second_cpp = true
hash_dir = true
EOF

RUN mkdir -p /tflite
WORKDIR /tflite
COPY buildbot/build_tflite.sh /tmp/build_tflite.sh
RUN CMAKE_POLICY_VERSION_MINIMUM=3.5 bash /tmp/build_tflite.sh && rm /tmp/build_tflite.sh

COPY Pipfile Pipfile.lock.3.10 Pipfile.lock.3.11 versioned_pipenv /tmp/mlgo-pipenv/
RUN python3 -m ensurepip --upgrade && \
    python3 -m pip install --no-cache-dir --break-system-packages --upgrade pip && \
    python3 -m pip install --no-cache-dir --break-system-packages pipenv buildbot-worker==2.9.0 && \
    cd /tmp/mlgo-pipenv && \
    python3 versioned_pipenv sync --extra-pip-args="--break-system-packages" --categories "packages dev-packages" --system && \
    cd / && \
    rm -rf /tmp/mlgo-pipenv /root/.cache && \
    curl -fsSL --retry 10 https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/tools/llvm-ir2vec/Bindings/requirements.txt -o /tmp/ir2vec-reqs.txt && \
    python3 -m pip install --no-cache-dir --break-system-packages -r /tmp/ir2vec-reqs.txt && \
    rm -f /tmp/ir2vec-reqs.txt && \
    python3 -c "import os, tensorflow as tf; assert os.path.isdir(os.path.join(os.path.dirname(tf.__file__), 'xla_aot_runtime_src'))"

ENV TFLITE_CMAKE=/tflite/tflite.cmake
WORKDIR /
