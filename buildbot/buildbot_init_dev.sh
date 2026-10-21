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

mkdir -p /var/lib/docker
mount -t tmpfs tmpfs -o size=50% /var/lib/docker

(
  SLEEP=0
  for i in `seq 1 5`; do
    sleep $SLEEP
    SLEEP=$(( SLEEP + 10))

    (
      set -ex
      echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
      dpkg --configure -a
      apt-get clean
      apt-get -qq -y update --allow-releaseinfo-change

      # Logs consume a lot of storage space.
      apt-get remove -qq -y --purge auditd puppet-agent rsync rsyslog

      apt-get install -qq -y docker.io tmux
    ) && exit 0
  done
  exit 1
) || on_error "Failed to install Docker."

DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/google/ml-compiler-opt/mlgo-buildbot-deps:latest}"
(
  SLEEP=0
  for i in `seq 1 5`; do
    sleep $SLEEP
    SLEEP=$(( SLEEP + 10))
    docker pull "${DOCKER_IMAGE}" && exit 0
  done
  exit 1
) || on_error "Failed to pull Docker image ${DOCKER_IMAGE}."

docker rm -f mlgo-buildbot 2>/dev/null || true
docker run -d \
  --name mlgo-buildbot \
  --restart unless-stopped \
  --network host \
  --hostname "${WORKER_NAME}" \
  -v "${BOT_DIR}:${BOT_DIR}" \
  "${DOCKER_IMAGE}" \
  sleep infinity || on_error "Failed to start mlgo-buildbot container."

docker exec -i \
  -e BOT_DIR="${BOT_DIR}" \
  -e SERVER_PORT="${SERVER_PORT}" \
  -e WORKER_NAME="${WORKER_NAME}" \
  -e WORKER_PASSWORD="${WORKER_PASSWORD}" \
  mlgo-buildbot bash -c "$(declare -f start_buildbot_worker); start_buildbot_worker BuildBotReproduceLocallyDev.md" \
  || on_error "Failed to start buildbot worker in container."
