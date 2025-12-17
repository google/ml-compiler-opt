#!/bin/bash

mkdir -p /work/llvm-train
cmake -B /work/llvm-train \
  -GNinja \
  -C /work/tflite/tflite.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=clang \
  /work/llvm-project/llvm

ninja -C /work/llvm-train