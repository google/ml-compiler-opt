# Reproducing `ml-opt-dev-x86-64` and `ml-opt-devrel-x86-64` Locally

The `ml-opt-dev-x86-64` and `ml-opt-devrel-x86-64` buildbots run inside the
`ghcr.io/google/ml-compiler-opt/mlgo-buildbot-deps:latest` container built from
[`buildbot/Dockerfile`](Dockerfile) and execute the annotated build script
[`zorg/buildbot/builders/annotated/mlgo-dev.sh`](https://github.com/llvm/llvm-zorg/blob/main/zorg/buildbot/builders/annotated/mlgo-dev.sh).

## Steps

1. Pull the container image (or build [`buildbot/Dockerfile`](Dockerfile) from
   the root of `ml-compiler-opt`):

   ```bash
   docker pull ghcr.io/google/ml-compiler-opt/mlgo-buildbot-deps:latest
   ```

2. Run the annotated script inside the container with your local `llvm-project`
   and `llvm-zorg` checkouts mounted:

   For `ml-opt-dev-x86-64`:

   ```bash
   docker run --rm -it \
     -v /path/to/llvm-project:/work/llvm-project \
     -v /path/to/llvm-zorg:/work/llvm-zorg \
     -e LLVM_ROOT=/work/llvm-project \
     -e BUILD_MLGO=/work/build-mlgo \
     ghcr.io/google/ml-compiler-opt/mlgo-buildbot-deps:latest \
     /work/llvm-zorg/zorg/buildbot/builders/annotated/mlgo-dev.sh
   ```

   For `ml-opt-devrel-x86-64`, pass `--enable-tf-aot`:

   ```bash
   docker run --rm -it \
     -v /path/to/llvm-project:/work/llvm-project \
     -v /path/to/llvm-zorg:/work/llvm-zorg \
     -e LLVM_ROOT=/work/llvm-project \
     -e BUILD_MLGO=/work/build-mlgo \
     ghcr.io/google/ml-compiler-opt/mlgo-buildbot-deps:latest \
     /work/llvm-zorg/zorg/buildbot/builders/annotated/mlgo-dev.sh --enable-tf-aot
   ```

   Alternatively, start an interactive shell in the container and run the
   `cmake` and `ninja` commands shown in the buildbot log directly against
   `/tflite/tflite.cmake`.
