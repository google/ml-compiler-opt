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

cd /work/ml-compiler-opt

TF_CPP_MIN_LOG_LEVEL=3 GINPATH=/work/ml-compiler-opt PYTHONPATH=/work/ml-compiler-opt:$PYTHONPATH \
  python /work/ml-compiler-opt/compiler_opt/rl/train_bc.py \
  --root_dir /work/corpus/bc_model \
  --gin_files /work/ml-compiler-opt/compiler_opt/rl/inlining/gin_configs/behavioral_cloning_nn_agent.gin \
  --gin_bindings inlining.config.get_observation_processing_layer_creator.quantile_file_dir="'/work/corpus/vocab'" \
  --data_path /work/corpus/default_trace
