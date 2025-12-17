#!/bin/bash

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=/work/ml-compiler-opt:$PYTHONPATH \
  python /work/ml-compiler-opt/compiler_opt/tools/generate_vocab.py \
  --gin_files /work/ml-compiler-opt/compiler_opt/rl/inlining/gin_configs/common.gin \
  --input /work/corpus/default_trace \
  --output_dir /work/corpus/vocab