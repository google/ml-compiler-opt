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

# Script to configure GCE instance to run LLVM ml-driven optimization build bots.

# NOTE: GCE can wait up to 20 hours before reloading this file.
# If some instance needs changes sooner just shutdown the instance 
# with GCE UI or "sudo shutdown now" over ssh. GCE will recreate
# the instance and reload the script.

function on_error {
  echo $1
  # FIXME: ON_ERROR should shutdown. Echo-ing for now, for experimentation
  # shutdown now
}

SERVER_PORT=${SERVER_PORT:-9990}
BOT_DIR=/b

mount -t tmpfs tmpfs /tmp
mkdir -p $BOT_DIR
mount -t tmpfs tmpfs -o size=80% $BOT_DIR

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
      apt-get remove -qq -y --purge auditd puppet-agent

      apt-get install -qq -y \
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
        rsync \
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

      bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" 20
      ln -sf /usr/bin/clang-20 /usr/bin/cc
      ln -sf /usr/bin/clang++-20 /usr/bin/c++
      ln -sf /usr/bin/ld.lld-20 /usr/bin/ld

    ) && exit 0
  done
  exit 1
) || on_error "Failed to install required packages."

userdel buildbot
groupadd buildbot
useradd buildbot -g buildbot -m -d /b/home
rm -rf /var/lib/buildbot
ln -s /b/home /var/lib/buildbot
chmod 777 /var/lib/buildbot

if [[ "${HOSTNAME}" == ml-opt-dev* ]]
then
  echo "Building TFLite"
  rm -rf /tmp/tflitebuild
  mkdir -p /tmp/tflitebuild
  pushd /tmp/tflitebuild
  curl https://raw.githubusercontent.com/google/ml-compiler-opt/main/buildbot/build_tflite.sh | bash -s
  popd
  echo "Done building TFLite"
else
  echo "NOT building TFLite - this is a release only bot."
fi

pushd /tmp
sudo -u buildbot git clone https://github.com/google/ml-compiler-opt || on_error "failed to clone ml-compiler-opt repo"
pushd ml-compiler-opt

# install the tf pip package for the AOT ("release" scenario) and for test model builds.
sudo -u buildbot python3 -m pip install --break-system-packages pipenv
echo installed pipenv
sudo -u buildbot python3 versioned_pipenv sync --extra-pip-args="--break-system-packages" --categories "packages dev-packages" --system
echo used pipenv
popd
popd

python3 -m pip install --break-system-packages buildbot-worker==2.9.0
echo installed buildbot worker

TF_PIP=$(sudo -u buildbot python3 -c "import tensorflow as tf; import os; print(os.path.dirname(tf.__file__))")

# temp location until zorg updates
sudo -u buildbot ln -s ${TF_PIP}/../../ /var/lib/buildbot/.local/lib/python3.7

# location we want
sudo -u buildbot ln -s ${TF_PIP}/../../ /tmp/tf-aot

export TENSORFLOW_AOT_PATH="${TF_PIP}"

if [ -d "${TENSORFLOW_AOT_PATH}/xla_aot_runtime_src" ]
then
  echo "TENSORFLOW_AOT_PATH=${TENSORFLOW_AOT_PATH}"
else
  on_error "TENSORFLOW_AOT_PATH not found"
fi

# install the tf C API library ("development" scenario).
rm -rf /tmp/tensorflow
mkdir /tmp/tensorflow
export TENSORFLOW_API_PATH=/tmp/tensorflow
wget --quiet 	https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz \
  || on_error "failed to download tensorflow C library"
tar xfz libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz -C "${TENSORFLOW_API_PATH}" || echo "failed to unarchive tensorflow C library"

if [ -f "${TENSORFLOW_API_PATH}/lib/libtensorflow.so" ]
then
  echo "TENSORFLOW_API_PATH=${TENSORFLOW_API_PATH}"
else
  on_error "TENSORFLOW_API_PATH not found"
fi

chown buildbot:buildbot $BOT_DIR
chown buildbot:buildbot $TENSORFLOW_API_PATH

rm -f /b/buildbot.tac


WORKER_NAME="$(hostname)"
WORKER_PASSWORD="$(gsutil cat gs://ml-compiler-opt-buildbot/buildbot_password)"
SERVICE_NAME=buildbot-worker@b.service
[[ -d /var/lib/buildbot/workers/b ]] || ln -s $BOT_DIR /var/lib/buildbot/workers/b

while pkill buildbot-worker; do sleep 5; done;

rm -rf ${BOT_DIR}/buildbot.tac ${BOT_DIR}/twistd.log
echo "Starting build worker ${WORKER_NAME}"
sudo -u buildbot buildbot-worker create-worker -f --allow-shutdown=signal $BOT_DIR lab.llvm.org:$SERVER_PORT \
   "${WORKER_NAME}" "${WORKER_PASSWORD}"

echo "Mircea Trofin <mtrofin@google.com>" > $BOT_DIR/info/admin

{
  echo "How to reproduce locally: https://github.com/google/ml-compiler-opt/wiki/BuildBotReproduceLocally"
  echo
  uname -a | head -n1
  date
  cmake --version | head -n1
  g++ --version | head -n1
  ld --version | head -n1
  lscpu
} > $BOT_DIR/info/host


chown -R buildbot:buildbot $BOT_DIR
sudo -u buildbot buildbot-worker start $BOT_DIR

sleep 30
cat $BOT_DIR/twistd.log
grep "worker is ready" $BOT_DIR/twistd.log || on_error "build worker not ready"
