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

TF_API_DEP_PACKAGES="python3 python3-pip"
ADMIN_PACKAGES="tmux"

(
  SLEEP=0
  for i in `seq 1 5`; do
    sleep $SLEEP
    SLEEP=$(( SLEEP + 10))

    (
      set -ex

      dpkg --add-architecture i386
      echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
      dpkg --configure -a
      apt-get clean
      apt-get -qq -y update --allow-releaseinfo-change

      # Logs consume a lot of storage space.
      apt-get remove -qq -y --purge auditd puppet-agent rsync rsyslog libheif1

      apt-get install -qq -y --no-install-recommends \
        python3-distutils \
        python-is-python3 \
        $TF_API_DEP_PACKAGES \
        $ADMIN_PACKAGES \
        cmake \
        cmake-data \
        ccache \
        binutils-dev \
        ninja-build \
        pkg-config \
        gcc-multilib \
        g++-multilib \
        gawk \
        dos2unix \
        libxml2-dev \
        git \
        libtool \
        m4 \
        automake \
        libgcrypt-dev \
        liblzma-dev \
        libssl-dev \
        libgss-dev \
        python3-dev \
        wget \
        zlib1g-dev \
        lsb-release \
        software-properties-common \
        gnupg

      export LLVM_MAJOR_VERSION=22
      bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" ${LLVM_MAJOR_VERSION}
      ln -sf /usr/bin/clang-${LLVM_MAJOR_VERSION} /usr/bin/cc
      ln -sf /usr/bin/clang++-${LLVM_MAJOR_VERSION} /usr/bin/c++
      ln -sf /usr/bin/ld.lld-${LLVM_MAJOR_VERSION} /usr/bin/ld

    ) && exit 0
  done
  exit 1
) || on_error "Failed to install required packages."

# BEGIN stopgap until zorg-side LLVM_CCACHE_BUILD propagates
cat << 'EOF' > /etc/ccache.conf
sloppiness = pch_defines,time_macros
depend_mode = true
run_second_cpp = true
hash_dir = true
EOF
# END

# Install IR2Vec Python binding dependencies.
# Installs from llvm-project main; canonical source is
# llvm/tools/llvm-ir2vec/Bindings/requirements.txt
curl -L --retry 10 https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/tools/llvm-ir2vec/Bindings/requirements.txt > /tmp/ir2vec-reqs.txt
python3 -m pip install --break-system-packages -r /tmp/ir2vec-reqs.txt buildbot-worker==2.9.0

( start_buildbot_worker "BuildBotReproduceLocallyRel.md" ) || on_error "Failed to start buildbot worker on host."
