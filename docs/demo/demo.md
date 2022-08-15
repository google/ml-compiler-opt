## Overview

In this demo, we look at how to:

  * collect a training corpus for training a `-Oz` inliner policy
  * perform that training
  * use the trained policy to build a 'release' clang which embeds the policy

For the first two steps, we use a 'development' mode clang which depends on the
Tensorflow C API dynamic library and allows swapping policies from the command
line - which is necessary to enable the reinforcement training loop iterate and
improve over policies.

The 'release' mode compiler does not have this option. The policy is fixed and
embedded in the compiler, and a user opts in to using it via a command line flag
(`-mllvm -enable-ml-inliner=release`). Also the compiler has no runtime
tensorflow dependencies.

For corpus extraction, any project that is built with clang and uses a build
system that produces a compilation database json file (like
`clang -DCMAKE_EXPORT_COMPILE_COMMANDS`) would work. The larger the number of
modules, the better (for training). For this demo, we're using the Fuchsia
project.

## Preliminaries
We assume all repositories are installed under `$HOME`, e.g. you should have
(after the next step) a `~/fuchsia`, `~/ml-compiler-opt`, etc.

## Get repositories

### LLVM

Follow the instructions available at https://llvm.org/docs/GettingStarted.html.
In most cases, it should be as simple as:

```shell
cd ~ && git clone https://github.com/llvm/llvm-project.git
```

Typical prerequisites:

```shell
sudo apt-get install cmake ninja-build lld
```


```shell
export LLVM_SRCDIR=~/llvm-project
export LLVM_INSTALLDIR=~/llvm-install
```

### Fuchsia

See instructions at https://fuchsia.dev/fuchsia-src/get-started/get_fuchsia_source. Make sure `PATH` is set up appropriately.

```shell
export IDK_DIR=~/fuchsia-idk
export SYSROOT_DIR=~/fuchsia-sysroot
export FUCHSIA_SRCDIR=~/fuchsia
```

We also need:
* [IDK](https://fuchsia.googlesource.com/fuchsia/+/master/docs/development/build/toolchain.md#fuchsia-idk)
* [sysroot](https://fuchsia.googlesource.com/fuchsia/+/master/docs/development/build/toolchain.md#sysroot-for-linux)

This should amount to:

```shell
cipd install fuchsia/sdk/core/linux-amd64 latest -root ${IDK_DIR}
cipd install fuchsia/sysroot/linux-arm64 latest -root ${SYSROOT_DIR}/linux-arm64
cipd install fuchsia/sysroot/linux-amd64 latest -root ${SYSROOT_DIR}/linux-x64
```

## Set up the correct package versions

We need to make sure the git revision of llvm is one that works with the version
of the Fuchsia tree. Note that Fuchsia frequently updates their llvm dependency,
which should mean you could try working at HEAD for both, but the method here is
safe.

*Optionally, you can get Fuchsia to the version we used we wrote this guide:*

```
cd ${FUCHSIA_SRCDIR}
jiri update -gc ~/ml-compiler-opt/docs/demo/fuchsia.xml
```

Either way - i.e. you can skip the step above to be at Fuchsia HEAD - to get the
git hash at which we know this version of Fuchsia will build, do:

```
cd ${FUCHSIA_SRCDIR}
jiri package 'fuchsia/third_party/clang/${platform}'
```

The output will contain a `git_revision` line, for example `fa4c3f70ff0768a270b0620dc6d158ed1205ec4e`. Copy that hash and then (using the
example):

```
cd ${LLVM_SRCDIR}
git checkout fa4c3f70ff0768a270b0620dc6d158ed1205ec4e
```


## ML Training

This is this repository.

### Tensorflow dependencies

See also [the build bot script](../../buildbot/buildbot_init.sh)

**NOTE:** The versioning of Tensorflow is pretty important in order to get a
functioning build with the demo setup. Building with a Tensorflow pip
package above v2.7.0 will cause the build to fail as well as building with
a libtensorflow version above v1.15.x. Versions of the Tensorflow pip package
under v2.8.0 don't have any available wheels for Python 3.10, so if you
are on a more recent distro (eg Ubuntu 22.04) that has Python 3.10, you will
run into version compatibility issues between the required Tensorflow version
and the available Python version (pip will refuse to install Tensorflow) trying
to install the required python packages from `ml-compiler-opt/requirements.txt`.
To mitigate this, either use a distro with a compatible python version
(eg Ubuntu 20.04), or your preferred flavor of python virtual environment
to utilize a compatible Python version.

```shell
cd
sudo apt-get install python3-pip
python3 -m pip install --upgrade pip
python3 -m pip install --user -r ml-compiler-opt/requirements.txt

TF_PIP=$(python3 -m pip show tensorflow | grep Location | cut -d ' ' -f 2)

export TENSORFLOW_AOT_PATH="${TF_PIP}/tensorflow"

mkdir ~/tensorflow
export TENSORFLOW_C_LIB_PATH=~/tensorflow
wget --quiet https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz
tar xfz libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz -C "${TENSORFLOW_C_LIB_PATH}"
```

## Build LLVM

Build LLVM with the 'development' ML mode, and additional
Fuchsia-specific settings:

```
cd ${LLVM_SRCDIR}
mkdir build
cd build
cmake -G Ninja \
  -DLLVM_ENABLE_LTO=OFF \
  -DLINUX_x86_64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux-x64 \
  -DLINUX_aarch64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux-arm64 \
  -DFUCHSIA_SDK=${IDK_DIR} \
  -DCMAKE_INSTALL_PREFIX= \
  -DTENSORFLOW_C_LIB_PATH=${TENSORFLOW_C_LIB_PATH} \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On \
  -C ${LLVM_SRCDIR}/clang/cmake/caches/Fuchsia-stage2.cmake \
  ${LLVM_SRCDIR}/llvm

ninja distribution
DESTDIR=${LLVM_INSTALLDIR} ninja install-distribution-stripped
cd ${FUCHSIA_SRCDIR}
python3 scripts/clang/generate_runtimes.py --clang-prefix=$LLVM_INSTALLDIR --sdk-dir=$IDK_DIR --build-id-dir=$LLVM_INSTALLDIR/lib/.build-id > $LLVM_INSTALLDIR/lib/runtime.json
```

**NOTE 1**: The only flag specific to MLGO is `TENSORFLOW_C_LIB_PATH`.

**NOTE 2**: Fuchsia's `clang/cmake/caches/Fuchsia-stage2.cmake` enables the new
pass manager by default. This allows us to not need to require it explicitly at
compile time, but it is a requirement for the MLGO project (all our work assumes
the new pass manager)

**NOTE 3**: The python executable should be explicitly set if there are multiple
(particularly, newer) Python executables on the system with
`-DPython3_EXECUTABLE=/path/to/compatible/python`

## Build Fuchsia

```shell
cd ${FUCHSIA_SRCDIR}
fx set core.x64 \
  --args='clang_prefix="${LLVM_INSTALLDIR}/bin"' \
  --args=clang_embed_bitcode=true \
  --args='optimize="size"' \
  --args='clang_ml_inliner=false'
fx build
```
We set `clang_ml_inliner` to false here because by default it is set to true,
but that wouldn't yet work since we don't have a model to embed.

Fuchsia build conveniently generates a size report. Let's copy it for reference.

**Note**
The `clang_prefix` is the absolute path of $LLVM_INSTALLDIR/bin(replace it by
yours). The `--args=clang_embed_bitcode=true` option above adds the compilation
flag `-Xclang=-fembed-bitcode=all`. This can be seen in the compilation database.
The effect of this is that the object files have the llvm bytecode produced by
clang, before the optimization passes, and the clang command line, captured in
the .llvmbc and .llvmcmd sections, respectivelly. This is the mechanism by which
we extract our corpus.

Naturally, the effect of this is that the object files, and the linked binaries,
are larger. Fuchsia strips the final object; but, more importantly, the size
report captures multiple dimensions, beside the final file size - including, for
example, the size of the text section.

```shell
cp out/default/elf_sizes.json /tmp/orig_sizes.json
```

## ML training

### Extract the corpus

```shell
cd ${FUCHSIA_SRCDIR}
fx compdb
```

This produces a `compile_commands.json` compilation database, akin cmake's.

```shell
export CORPUS=$HOME/corpus
cd ~/ml-compiler-opt
python3 compiler_opt/tools/extract_ir.py \
  --cmd_filter="^-O2|-Os|-Oz$" \
  --input=$FUCHSIA_SRCDIR/out/default/compile_commands.json \
  --input_type=json \
  --llvm_objcopy_path=$LLVM_INSTALLDIR/bin/llvm-objcopy \
  --output_dir=$CORPUS
```

### Collect Trace and Generate Vocab

```shell
export DEFAULT_TRACE=$HOME/default_trace
export DEFAULT_VOCAB=compiler_opt/rl/inlining/vocab
```

Collect traces from the default heuristic, to kick off the training process.

**NOTE** the double and single quotes for the `--gin-bindings` - this is because
the last value must appear, syntactically, as a python string.

```shell
rm -rf $DEFAULT_TRACE &&
  PYTHONPATH=$PYTHONPATH:. python3 \
    compiler_opt/tools/generate_default_trace.py \
    --data_path=$CORPUS \
    --output_path=$DEFAULT_TRACE \
    --gin_files=compiler_opt/rl/inlining/gin_configs/common.gin \
    --gin_bindings=config_registry.get_configuration.implementation=@configs.InliningConfig \
    --gin_bindings=clang_path="'$LLVM_INSTALLDIR/bin/clang'" \
    --gin_bindings=llvm_size_path="'$LLVM_INSTALLDIR/bin/llvm-size'" \
    --sampling_rate=0.2
```

Generate vocab for the generated trace.
This is an optional step and should be triggered if the
set of features or the distribution of features
in the trace changes.

```shell
rm -rf $DEFAULT_VOCAB &&
  PYTHONPATH=$PYTHONPATH:. python3 \
    compiler_opt/tools/sparse_bucket_generator.py \
    --gin_files=compiler_opt/rl/inlining/gin_configs/common.gin \
    --input=$DEFAULT_TRACE \
    --output_dir=$DEFAULT_VOCAB
```

**Note**
The `sparse_bucket_generator.py` tool optionally accepts two more additional
flags `--sampling_fraction` and `--parallelism`.
`sampling_fraction` downsamples input features and `parallelism` controls the
degree of parallelism.
These flags can be tuned to reduce memory footprint and improve execution speed
of the vocab generator.

### Train a new model

```shell
export WARMSTART_OUTPUT_DIR=$HOME/warmstart
export OUTPUT_DIR=$HOME/model
```

Train a behavioral cloning model based on the above trace, that mimics default inlining behavior. This is the 'warmstart' model.

```shell
rm -rf $WARMSTART_OUTPUT_DIR && \
  PYTHONPATH=$PYTHONPATH:. python3 \
  compiler_opt/rl/train_bc.py \
  --root_dir=$WARMSTART_OUTPUT_DIR \
  --data_path=$DEFAULT_TRACE \
  --gin_files=compiler_opt/rl/inlining/gin_configs/behavioral_cloning_nn_agent.gin
```

Starting from the warmstart model, train the optimized model - **this will take
about half a day**

```shell
rm -rf $OUTPUT_DIR && \
  PYTHONPATH=$PYTHONPATH:. python3 \
  compiler_opt/rl/train_locally.py \
  --root_dir=$OUTPUT_DIR \
  --data_path=$CORPUS \
  --gin_bindings=clang_path="'$LLVM_INSTALLDIR/bin/clang'" \
  --gin_bindings=llvm_size_path="'$LLVM_INSTALLDIR/bin/llvm-size'" \
  --num_modules=100 \
  --gin_files=compiler_opt/rl/inlining/gin_configs/ppo_nn_agent.gin \
  --gin_bindings=train_eval.warmstart_policy_dir=\"$WARMSTART_OUTPUT_DIR/saved_policy\"
```

You may also start a tensorboard to monitor the training process with 

```shell
tensorboard --logdir=$OUTPUT_DIR
```

Mainly check the reward_distribution section for the model performance. It
includes the average reward and the percentile of the reward distributions
during training. Positive reward means an improvement against the heuristic,
and negative reward means a regression.

### Evaluate trained policy on a corpus (Optional)

Optionally, if you are interested in seeing how the trained policy (`$OUTPUT_DIR/saved_policy`)
performs on a given corpus (take the training corpus `$CORPUS` as an example),
the following command line generates a csv-format report with 4 columns:
module_name, identifier (`default` in inlining case), size under heuristic,
size under the trained policy at `$OUTPUT_PERFORMANCE_PATH`.

```shell
export OUTPUT_PERFORMANCE_PATH=$HOME/performance_report && \
PYTHONPATH=$PYTHONPATH:. && \
python3 compiler_opt/tools/generate_default_trace.py \
  --data_path=$CORPUS \
  --policy_path=$OUTPUT_DIR/saved_policy \
  --output_performance_path=$OUTPUT_PERFORMANCE_PATH \
  --compile_task=inlining \
  --gin_files=compiler_opt/rl/inlining/gin_configs/common.gin \
  --gin_bindings=clang_path="'$LLVM_INSTALLDIR/bin/clang'" \
  --gin_bindings=llvm_size_path"'=$LLVM_INSTALLDIR/bin/llvm-size'" \
  --sampling_rate=0.2
```

## Deploying and using the new policy

We need to build the 'release' mode of the compiler. Currently, that means
overwritting the model in `llvm/lib/Analysis/models/inliner`.

```shell
cd $LLVM_SRCDIR
rm -rf llvm/lib/Analysis/models/inliner/*
cp -rf $OUTPUT_DIR/saved_policy/* llvm/lib/Analysis/models/inliner/
```

Setup the release build:

```shell
mkdir build-release
cd build-release
cmake -G Ninja \
  -DLLVM_ENABLE_LTO=OFF \
  -DLINUX_x86_64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux-x64 \
  -DLINUX_aarch64-unknown-linux-gnu_SYSROOT=${SYSROOT_DIR}/linux-arm64 \
  -DFUCHSIA_SDK=${IDK_DIR} \
  -DCMAKE_INSTALL_PREFIX= \
  -DTENSORFLOW_AOT_PATH=${TENSORFLOW_AOT_PATH} \
  -C ${LLVM_SRCDIR}/clang/cmake/caches/Fuchsia-stage2.cmake \
  ${LLVM_SRCDIR}/llvm

export LLVM_INSTALLDIR_RELEASE=$LLVM_INSTALLDIR-release
ninja distribution
DESTDIR=${LLVM_INSTALLDIR_RELEASE} ninja install-distribution-stripped
cd ${FUCHSIA_SRCDIR}
python3 scripts/clang/generate_runtimes.py \
  --clang-prefix=$LLVM_INSTALLDIR_RELEASE \
  --sdk-dir=$IDK_DIR \
  --build-id-dir=$LLVM_INSTALLDIR_RELEASE/lib/.build-id > $LLVM_INSTALLDIR_RELEASE/lib/runtime.json
```

**NOTE 1**: If you are using LLVM-at-head instead of an exact repro, there is an
additional flag `-DLLVM_INLINER_MODEL_PATH=` that you need to set to the path to
your model. If you set the flag to `download`, then the latest compatible model
release from github will be downloaded.

**NOTE 2**: The only flag specific to MLGO is `TENSORFLOW_AOT_PATH`, which
replaces `TENSORFLOW_C_LIB_PATH` used earlier.

```shell
cd ${FUCHSIA_SRCDIR}
fx set core.x64 \
  --args='clang_prefix="${LLVM_INSTALLDIR_RELEASE}/bin"' \
  --args='optimize="size"' \
  --args=clang_ml_inliner=true
fx build
```

Fuchsia has a nice utility for comparing the sizes of binaries' text section,
however, it currently does so indiscriminately for all targets - including those
compiled with `-O3` (which are unchanged). We can filter them out to get a `-Oz`
effect:

```shell
python3 -m pip install --user tabulate
scripts/compare_elf_sizes.py \
  /tmp/orig_sizes.json \
  out/default/elf_sizes.json \
  --field code
```
