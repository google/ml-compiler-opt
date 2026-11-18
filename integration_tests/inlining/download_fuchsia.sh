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
apt-get install -y curl file unzip
curl -s "https://fuchsia.googlesource.com/fuchsia/+/HEAD/scripts/bootstrap?format=TEXT" | base64 --decode | bash
export PATH=/fuchsia/.jiri_root/bin:$PATH
cipd install fuchsia/sdk/core/linux-amd64 latest -root /fuchsia-idk
cipd install fuchsia/sysroot/linux-arm64 latest -root /fuchsia-sysroot/linux-arm64
cipd install fuchsia/sysroot/linux-amd64 latest -root /fuchsia-sysroot/linux-x64
