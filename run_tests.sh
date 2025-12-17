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

# Download package management tooling
pip3 install pipenv

# Download pre-requisite packages.
PIPENV_VENV_IN_PROJECT=1
./versioned_pipenv sync --dev

# Run tests
TF_CPP_MIN_LOG_LEVE=3
PYTHONPATH="${PYTHONPATH}:$(dirname "$0")"
./versioned_pipenv run python3 -m pytest
