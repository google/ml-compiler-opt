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
      python3-pip \
      software-properties-common \
      sudo \
      tmux \
      wget \
      zlib1g-dev && \
    python3 -m pip install --no-cache-dir --break-system-packages uv && \
    UV_PYTHON_INSTALL_DIR=/opt/python uv python install 3.11 && \
    uv venv --python 3.11 --seed /opt/venv && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python && \
    ln -sf /opt/venv/bin/python3 /usr/local/bin/python3 && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip && \
    ln -sf /opt/venv/bin/pip3 /usr/local/bin/pip3 && \
    ln -sf /usr/bin/clang-${LLVM_MAJOR_VERSION} /usr/bin/clang && \
    ln -sf /usr/bin/clang++-${LLVM_MAJOR_VERSION} /usr/bin/clang++ && \
    ln -sf /usr/bin/clang-${LLVM_MAJOR_VERSION} /usr/bin/cc && \
    ln -sf /usr/bin/clang++-${LLVM_MAJOR_VERSION} /usr/bin/c++ && \
    ln -sf /usr/bin/ld.lld-${LLVM_MAJOR_VERSION} /usr/bin/ld && \
    ln -sf /usr/bin/ld.lld-${LLVM_MAJOR_VERSION} /usr/bin/ld.lld && \
    rm -rf /var/lib/apt/lists/* /root/.cache

ENV PATH="/opt/venv/bin:${PATH}"

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
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir pipenv buildbot-worker==2.9.0 && \
    cd /tmp/mlgo-pipenv && \
    python3 versioned_pipenv sync --categories "packages dev-packages" --system && \
    cd / && \
    rm -rf /tmp/mlgo-pipenv /root/.cache && \
    curl -fsSL --retry 10 https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/tools/llvm-ir2vec/Bindings/requirements.txt -o /tmp/ir2vec-reqs.txt && \
    python3 -m pip install --no-cache-dir -r /tmp/ir2vec-reqs.txt && \
    rm -f /tmp/ir2vec-reqs.txt && \
    ln -sf /opt/venv/bin/buildbot-worker /usr/local/bin/buildbot-worker && \
    ln -sf /opt/venv/bin/saved_model_cli /usr/local/bin/saved_model_cli && \
    python3 -c "import os, tensorflow as tf; assert os.path.isdir(os.path.join(os.path.dirname(tf.__file__), 'xla_aot_runtime_src'))"

ENV TFLITE_CMAKE=/tflite/tflite.cmake
WORKDIR /
