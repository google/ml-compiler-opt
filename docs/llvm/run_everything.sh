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

SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"

SCRIPT_DIR="$(cd -- "$SCRIPT_DIR" && pwd)"
if [[ -z "$SCRIPT_DIR" ]] ; then
    exit 1
fi

"${SCRIPT_DIR}/init.sh"
"${SCRIPT_DIR}/build_clang_for_training.sh"
"${SCRIPT_DIR}/build_clang_for_corpus.sh"
"${SCRIPT_DIR}/extract_corpus.sh"
"${SCRIPT_DIR}/generate_default_trace.sh"
"${SCRIPT_DIR}/generate_vocab.sh"
"${SCRIPT_DIR}/train_with_es.sh"