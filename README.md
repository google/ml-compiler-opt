# Infrastructure for ML - Driven Optimizations in LLVM

This repository contains tools for ML-driven optimizations in LLVM.

## Prerequisites {.numbered}

Currently, the assumption for the is:

*   Recent Ubuntu distro, e.g. 20.04
*   python 3.8.x
*   for local training, which is currently the only supported mode, we recommend
    a high-performance workstation (e.g. 96 hardware threads).

Training assumes a clang build with ML 'development-mode'. Please refer to:

*   [LLVM documentation](https://llvm.org/docs/CMake.html)
*   the build
    [bot script](https://github.com/google/ml-compiler-opt/blob/master/buildbot/buildbot_init.sh)

The model training - specific prerequisites are:

```shell
pip3 install --user -r requirements.txt
```

Where `requirements.txt` is provided in the root of the repository.

Optionally, to run tests (run_tests.sh), you also need:

```shell
sudo apt-get install virtualenv
```

Note that the same tensorflow package is also needed for building the 'release'
mode for LLVM.

## Model training {.numbered}

TODO: overview.

### Preparation {.numbered}

#### Build LLVM {.numbered}

TODO: how

#### Extract IR files {.numbered}

TODO: how

#### Set up environment {.numbered}

```shell
export CORPUS=<root directory with IR files>
export DEFAULT_TRACE=<path to where trace under default policy will be dropped>
export LLVM_DIR=<directory where LLVM build is located - i.e. we can find $LLVM_DIR/bin/clang>
export WARMSTART_OUTPUT_DIR=<directory where warmstart model will be dropped>
export OUTPUT_DIR=<directory where trained model will be dropped>
```

### Obtain a "warmstart" model {.numbered}

Execute from the root of the git repository:

#### Prepare initial traning data {.numbered}

This collects a trace of the current heuristic's behavior, which is then used to
train an initial ("warmstart") model.

```shell
PYTHONPATH=$PYTHONPATH:. python3 \
  compiler_opt/tools/generate_default_trace.py \
  --data_path=$CORPUS \
  --output_path=$DEFAULT_TRACE \
  --compile_task=inlining \
  --clang_path=$LLVM_DIR/bin/clang \
  --llvm_size_path=$LLVM_DIR/bin/llvm-size \
  --sampling_rate=0.2
```

#### Train warmstart model {.numbered}

```shell
rm -rf $WARMSTART_OUTPUT_DIR && \
  PYTHONPATH=$PYTHONPATH:. python3 \
  compiler_opt/rl/train_bc.py \
  --root_dir=$WARMSTART_OUTPUT_DIR \
  --data_path=$DEFAULT_TRACE \
  --gin_files=compiler_opt/rl/gin_configs/behavioral_cloning_nn_agent.gin
```

### Train an optimized policy model {.numbered}

#### Train new policy {.numbered}

This will take a long time ~ half a day - on a 96 thread machine.

Execute from the root of the git repository:

```shell
rm -rf $OUTPUT_DIR && \
  PYTHONPATH=$PYTHONPATH:. python3 \
  compiler_opt/rl/train_locally.py \
  --root_dir=$OUTPUT_DIR \
  --data_path=$CORPUS \
  --clang_path=$LLVM_DIR/bin/clang \
  --llvm_size_path=$LLVM_DIR/bin/llvm-size \
  --num_modules=100 \
  --gin_files=compiler_opt/rl/gin_configs/ppo_nn_agent.gin \
  --gin_bindings=train_eval.warmstart_policy_dir=\"$WARMSTART_OUTPUT_DIR/saved_policy\"
```
