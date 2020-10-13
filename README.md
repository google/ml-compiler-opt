# Infrastructure for ML - Driven Optimizations in LLVM

This repository contains tools for ML-driven optimizations in LLVM.

To train inlining model:

1.  Build "llvm/llvm-project/llvm/{opt|llc|llvm-size}".

2.  Prepare IR files.

3.  Run the following command line.

```
OUTPUT_DIR=$HOME/llvm_inlining && \
rm -rf $OUTPUT_DIR && \
python3 compiler_opt/rl/train_locally.py \
  --root_dir=$OUTPUT_DIR \
  --data_path=/path/to/IRFiles \
  --opt_path=/path/to/opt \
  --llc_path=/path/to/llc \
  --llvm_size_path=/path/to/llvm-size \
  --num_workers=50 \
  --num_modules=100 \
  --gin_files=compiler_opt/rl/gin_configs/ppo_nn_agent.gin
```
