#!/bin/bash

PYTHONPATH=/work/llvm-project/llvm/utils/mlgo-utils:$PYTHONPATH \
  python /work/llvm-project/llvm/utils/mlgo-utils/extract_ir.py \
  --input /work/llvm-corpus/compile_commands.json \
  --input_type json \
  --output_dir /work/corpus/modules \
  --llvm_objcopy_path /work/llvm-train/bin/llvm-objcopy \
  --obj_base_dir /work/llvm-corpus