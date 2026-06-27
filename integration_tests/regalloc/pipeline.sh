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
# clone and build llvm
cd /
git clone https://github.com/llvm/llvm-project
mkdir /llvm-build
cd /llvm-build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORFLOW_C_LIB_PATH=/tmp/tensorflow \
    -DTENSORFLOW_AOT_PATH=$(python3 -c "import tensorflow; import os; print(os.path.dirname(tensorflow.__file__))") \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DLLVM_ENABLE_PROJECTS="clang" \
    /llvm-project/llvm
cmake --build .

# clone and build chromium
cd /
git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
export PATH="$PATH:/depot_tools"
mkdir /chromium
cd /chromium
fetch --nohooks --no-history chromium
sed -i 's/"custom_vars": {},/"custom_vars": { "checkout_pgo_profiles" : True },/' .gclient
cd src
git apply /ml-compiler-opt/experimental/chromium-bitcode-embedding.patch
apt-get install -y sudo
sed -i 's/${dev_list} snapcraft/${dev_list}/' ./build/install-build-deps.sh
./build/install-build-deps.sh
gclient runhooks
gn gen out/Release --export-compile-commands --args="\
  is_official_build=true \
  use_thin_lto=false \
  is_cfi=false \
  clang_embed_bitcode=true \
  is_debug=false \
  symbol_level=0 \
  enable_nacl=false"
# tensorflow and chromium protobuf versions are incompatible
# install protobuf4 here so that the chromium build will complete
pip3 install protobuf==4.21.5
autoninja -C out/Release

# test training ml
cd /ml-compiler-opt
export PYTHONPATH=$PYTHONPATH:.
mkdir /corpus
# make sure protobuf is now at the tensorflow version requirement
pip3 install -r requirements.txt
python3 compiler_opt/tools/extract_ir.py \
    --cmd_filter="^-O2|-O3$" \
    --input=/chromium/src/out/Release/compile_commands.json \
    --input_type=json \
    --llvm_objcopy_path=/llvm-build/bin/llvm-objcopy \
    --output_dir=/corpus
python3 compiler_opt/tools/generate_default_trace.py \
    --data_path=/corpus \
    --output_path=/default_trace \
    --gin_files=compiler_opt/rl/regalloc/gin_configs/common.gin \
    --gin_bindings=config_registry.get_configuration.implementation=@configs.RegallocEvictionConfig \
    --gin_bindings=clang_path="'/llvm-build/bin/clang'" \
    --sampling_rate=0.2
rm -rf ./compiler_opt/rl/regalloc/vocab
python3 \
    compiler_opt/tools/sparse_bucket_generator.py \
    --input=/default_trace \
    --output_dir=./compiler_opt/rl/regalloc/vocab
# Only train bc agent for 100 iterations. We're verifying the tooling is
# working correctly, not the model performance.
sed -i 's/train_eval.num_iterations=10000/train_eval.num_iterations=100/' \
    compiler_opt/rl/regalloc/gin_configs/behavioral_cloning_nn_agent.gin
python3 compiler_opt/rl/train_bc.py \
    --root_dir=/warmstart \
    --data_path=/default_trace \
    --gin_files=compiler_opt/rl/regalloc/gin_configs/behavioral_cloning_nn_agent.gin
# same for the rl training. Only do one policy iteration.
sed -i 's/train_eval.num_policy_iterations=3000/train_eval.num_policy_iterations=1/' \
    compiler_opt/rl/regalloc/gin_configs/ppo_nn_agent.gin
python3 compiler_opt/rl/train_locally.py \
    --root_dir=/output_model \
    --data_path=/corpus \
    --gin_bindings=clang_path="'/llvm-build/bin/clang'" \
    --gin_files=compiler_opt/rl/regalloc/gin_configs/ppo_nn_agent.gin \
    --gin_bindings=train_eval.warmstart_policy_dir=\"/warmstart/saved_policy\"
