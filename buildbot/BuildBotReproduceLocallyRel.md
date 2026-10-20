# Reproducing `ml-opt-rel-x86-64` Locally

The `ml-opt-rel-x86-64` buildbot runs on a host configured by
[`buildbot/buildbot_init_rel.sh`](buildbot_init_rel.sh) and executes the
annotated build script
[`zorg/buildbot/builders/annotated/mlgo-emitc.sh`](https://github.com/llvm/llvm-zorg/blob/main/zorg/buildbot/builders/annotated/mlgo-emitc.sh).
It compiles embedded MLGO models with MLIR EmitC and does not require TensorFlow
or TFLite.

## Steps

1. Install the standard LLVM build prerequisites (Clang, LLD, CMake, Ninja) and
   the IR2Vec Python binding dependencies as shown in
   [`buildbot/buildbot_init_rel.sh`](buildbot_init_rel.sh):

   ```bash
   python3 -m pip install -r /path/to/llvm-project/llvm/tools/llvm-ir2vec/Bindings/requirements.txt
   ```

2. Run the annotated script against your `llvm-project` checkout:

   ```bash
   LLVM_ROOT=/path/to/llvm-project \
     /path/to/llvm-zorg/zorg/buildbot/builders/annotated/mlgo-emitc.sh
   ```

   Alternatively, run the `cmake` and `ninja` invocations from the buildbot log
   directly (`mlir-opt` and `mlir-translate` first, followed by the LLVM build
   with `-DLLVM_MLGO_MODELS=...` and `ninja check-llvm`).
