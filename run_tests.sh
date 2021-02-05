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
set -x

TEST_TMP=$(mktemp -d)

VENV_PATH=${TEST_TMP}/virtualenv
virtualenv "${VENV_PATH}" -p python3 --system-site-packages
source "${VENV_PATH}"/bin/activate

# Download pre-requisite packages.
pip3 install -r requirements.txt

PYTHONPATH="${PYTHONPATH}:$(dirname "$0")"

for file in $(find . -name '*_test.py')
do
  python3 "${file}"
done

deactivate

rm -rf "${TEST_TMP}"
