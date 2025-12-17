#!/bin/bash

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=/work/ml-compiler-opt:$PYTHONPATH \
  python /work/ml-compiler-opt/compiler_opt/tools/generate_default_trace.py \
  --data_path /work/corpus/modules \
  --gin_files /work/ml-compiler-opt/compiler_opt/rl/inlining/gin_configs/common.gin \
  --gin_bindings clang_path="'/work/llvm-train/bin/clang'" \
  --gin_bindings llvm_size_path="'/work/llvm-train/bin/llvm-size'" \
  --compilation_timeout=600 \
  --output_path /work/corpus/default_trace