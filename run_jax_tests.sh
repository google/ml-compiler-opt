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
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"

TEST_TARGETS=(
  "compiler_opt/es/blackbox_optimizers_test.py"
  "compiler_opt/es/gradient_ascent_optimization_algorithms_test.py"
  "compiler_opt/es/perturbations_test.py"
  "compiler_opt/distributed/worker_test.py"
  "compiler_opt/distributed/buffered_scheduler_test.py"
  "compiler_opt/distributed/local/local_worker_manager_test.py"
)

python3 -m pytest "${TEST_TARGETS[@]}"
