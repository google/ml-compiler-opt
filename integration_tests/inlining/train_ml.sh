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
cd /ml-compiler-opt
export PYTHONPATH=$PYTHONPATH:.
mkdir /corpus
python3 compiler_opt/tools/extract_ir.py \
  --cmd_filter="^-O2|-Os|-Oz$" \
  --input=/fuchsia/out/default/compile_commands.json \
  --input_type=json \
  --llvm_objcopy_path=/llvm-build/bin/llvm-objcopy \
  --output_dir=/corpus
python3 compiler_opt/tools/generate_default_trace.py \
    --data_path=/corpus \
    --output_path=/default_trace \
    --gin_files=compiler_opt/rl/inlining/gin_configs/common.gin \
    --gin_bindings=config_registry.get_configuration.implementation=@configs.InliningConfig \
    --gin_bindings=clang_path="'/llvm-build/bin/clang'" \
    --gin_bindings=llvm_size_path="'/llvm-build/bin/llvm-size'" \
    --sampling_rate=0.2
rm -rf compiler_opt/rl/inlining/vocab
python3 compiler_opt/tools/sparse_bucket_generator.py \
    --gin_files=compiler_opt/rl/inlining/gin_configs/common.gin \
    --input=/default_trace \
    --output_dir=compiler_opt/rl/inlining/vocab
# Only train bc agent for 100 iterations. We're verifying the tooling is
# working correctly, not the model performance.
sed -i 's/train_eval.num_iterations=10000/train_eval.num_iterations=100/' \
    compiler_opt/rl/inlining/gin_configs/behavioral_cloning_nn_agent.gin
python3 compiler_opt/rl/train_bc.py \
    --root_dir=/warmstart \
    --data_path=/default_trace \
    --gin_files=compiler_opt/rl/regalloc/gin_configs/behavioral_cloning_nn_agent.gin
# same for the rl training. Only do one policy iteration.
sed -i 's/train_eval.num_policy_iterations=3000/train_eval.num_policy_iterations=1/' \
    compiler_opt/rl/inlining/gin_configs/ppo_nn_agent.gin
python3 compiler_opt/rl/train_locally.py \
  --root_dir=/output_model \
  --data_path=/corpus \
  --gin_bindings=clang_path="'/llvm-build/bin/clang'" \
  --gin_bindings=llvm_size_path="'/llvm-build/bin/llvm-size'" \
  --num_modules=100 \
  --gin_files=compiler_opt/rl/inlining/gin_configs/ppo_nn_agent.gin \
  --gin_bindings=train_eval.warmstart_policy_dir=\"/warmstart/saved_policy\"