#!/bin/bash

mkdir -p /work/llvm-corpus
cmake -B /work/llvm-corpus \
  -GNinja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=clang \
  -DCMAKE_C_COMPILER=/work/llvm-train/bin/clang \
  -DCMAKE_CXX_COMPILER=/work/llvm-train/bin/clang++ \
  -DCMAKE_C_FLAGS="-Xclang=-fembed-bitcode=all" \
  -DCMAKE_CXX_FLAGS="-Xclang=-fembed-bitcode=all" \
  /work/llvm-project/llvm

ninja -C /work/llvm-corpus