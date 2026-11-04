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

echo "${HOSTNAME} using server port ${SERVER_PORT}"

BOT_DIR=/b

mount -t tmpfs tmpfs /tmp
mkdir -p $BOT_DIR
mount -t tmpfs tmpfs -o size=80% $BOT_DIR

WORKER_NAME="$(hostname)"
WORKER_PASSWORD="$(gcloud storage cat gs://ml-compiler-opt-buildbot/buildbot_password)"

function start_buildbot_worker {
  set -ex
  local repro_doc="${1:?Missing reproduction document name}"

  userdel buildbot 2>/dev/null || true
  groupadd -f buildbot
  useradd buildbot -g buildbot -m -d "${BOT_DIR}/home"
  rm -rf /var/lib/buildbot
  ln -s "${BOT_DIR}/home" /var/lib/buildbot
  chmod 777 /var/lib/buildbot

  chown buildbot:buildbot "${BOT_DIR}"
  rm -f "${BOT_DIR}/buildbot.tac"

  mkdir -p /var/lib/buildbot/workers
  [[ -d /var/lib/buildbot/workers/b ]] || ln -s "${BOT_DIR}" /var/lib/buildbot/workers/b

  while pkill buildbot-worker; do sleep 5; done

  rm -rf "${BOT_DIR}/buildbot.tac" "${BOT_DIR}/twistd.log"
  echo "Starting build worker ${WORKER_NAME}"
  sudo -i -u buildbot buildbot-worker create-worker -f --allow-shutdown=signal "${BOT_DIR}" "lab.llvm.org:${SERVER_PORT}" \
    "${WORKER_NAME}" "${WORKER_PASSWORD}"

  echo "Mircea Trofin <mtrofin@google.com>" > "${BOT_DIR}/info/admin"

  {
    echo "How to reproduce locally: https://github.com/google/ml-compiler-opt/blob/main/buildbot/${repro_doc}"
    echo
    uname -a | head -n1
    date
    cmake --version | head -n1
    c++ --version | head -n1
    ld --version | head -n1
    lscpu
  } > "${BOT_DIR}/info/host"

  chown -R buildbot:buildbot "${BOT_DIR}"
  sudo -i -u buildbot buildbot-worker start "${BOT_DIR}"
}

function run_bot_script {
  local script_name="$1"
  local script_dir=""
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
  if [[ -n "${script_dir}" && -f "${script_dir}/${script_name}" ]]; then
    source "${script_dir}/${script_name}"
  else
    local tmp_script="/tmp/${script_name}"
    curl -fsSL --retry 10 "https://raw.githubusercontent.com/google/ml-compiler-opt/main/buildbot/${script_name}" -o "${tmp_script}" \
      || { on_error "Failed to download ${script_name}."; exit 1; }
    source "${tmp_script}"
  fi
}

if [[ "${HOSTNAME}" == ml-opt-dev* ]]
then
  run_bot_script "buildbot_init_dev.sh"
elif [[ "${HOSTNAME}" == ml-opt-rel* ]]
then
  run_bot_script "buildbot_init_rel.sh"
else
  on_error "Unrecognized bot hostname: ${HOSTNAME}"
  exit 1
fi

sleep 30
cat $BOT_DIR/twistd.log
grep "worker is ready" $BOT_DIR/twistd.log || on_error "build worker not ready"
