#!/bin/bash

TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=/work/ml-compiler-opt:$PYTHONPATH \
  python /work/ml-compiler-opt/compiler_opt/es/es_trainer.py \
  --gin_files /work/ml-compiler-opt/compiler_opt/es/inlining/gin_configs/inlining.gin \
  --gin_files /work/ml-compiler-opt/compiler_opt/es/inlining/gin_configs/blackbox_learner.gin \
  --train_corpora /work/corpus/modules \
  --gin_bindings clang_path="'/work/llvm-train/bin/clang'" \
  --gin_bindings llvm_size_path="'/work/llvm-train/bin/llvm-size'" \
  --gin_bindings inlining.config.get_observation_processing_layer_creator.quantile_file_dir="'/work/corpus/vocab'" \
  --output_path /work/corpus/trained_model_es