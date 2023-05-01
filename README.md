# Infrastructure for MLGO - a Machine Learning Guided Compiler Optimizations Framework.

MLGO is a framework for integrating ML techniques systematically in LLVM. It
replaces human-crafted optimization heuristics in LLVM with machine learned
models. The MLGO framework currently supports two optimizations:

1.  inlining-for-size([LLVM RFC](https://lists.llvm.org/pipermail/llvm-dev/2020-April/140763.html));
2.  register-allocation-for-performance([LLVM RFC](https://lists.llvm.org/pipermail/llvm-dev/2021-November/153639.html))

The compiler components are both available in the main LLVM repository. This
repository contains the training infrastructure and related tools for MLGO.

We currently use two different ML algorithms: Policy Gradient and Evolution
Strategies to train policies. Currently, this repository only support Policy
Gradient training. The release of Evolution Strategies training is on our
roadmap.

Check out this [demo](docs/inlining-demo/demo.md) for an end-to-end demonstration of how
to train your own inlining-for-size policy from the scratch with Policy
Gradient, or check out this [demo](docs/regalloc-demo/demo.md) for a demonstration of how
to train your own regalloc-for-performance policy.

For more details about MLGO, please refer to our paper
[MLGO: a Machine Learning Guided Compiler Optimizations Framework](https://arxiv.org/abs/2101.04808).

For more details about how to contribute to the project, please refer to
[contributions](docs/contributing.md).

## Pretrained models

We occasionally release pretrained models that may be used as-is with LLVM.
Models are released as github releases, and are named as
[task]-[major-version].[minor-version].The versions are semantic: the major
version corresponds to breaking changes on the LLVM/compiler side, and the minor
version corresponds to model updates that are independent of the compiler.

When building LLVM, there is a flag `-DLLVM_INLINER_MODEL_PATH` which you may
set to the path to your inlining model. If the path is set to `download`, then
cmake will download the most recent (compatible) model from github to use. Other
values for the flag could be:

```sh
# Model is in /tmp/model, i.e. there is a file /tmp/model/saved_model.pb along
# with the rest of the tensorflow saved_model files produced from training.
-DLLVM_INLINER_MODEL_PATH=/tmp/model

# Download the most recent compatible model
-DLLVM_INLINER_MODEL_PATH=download
```

## Prerequisites

Currently, the assumptions for the system are:

*   Recent Ubuntu distro, e.g. 20.04
*   python 3.8.x/3.9.x/3.10.x
*   for local training, which is currently the only supported mode, we recommend
    a high-performance workstation (e.g. 96 hardware threads).

Training assumes a clang build with ML 'development-mode'. Please refer to:

*   [LLVM documentation](https://llvm.org/docs/CMake.html)
*   the build
    [bot script](https://github.com/google/ml-compiler-opt/blob/main/buildbot/buildbot_init.sh)

The model training - specific prerequisites are:

Pipenv:
```shell
pip3 install pipenv
```

The actual dependencies:
```shell
pipenv sync --system
```
Note that the above command will only work from the root of the repository
since it needs to have `Pipfile.lock` in the working directory at the time
of execution.

If you plan on doing development work, make sure you grab the development
and CI categories of packages as well:
```shell
pipenv sync --system --categories "dev-packages ci"
```

Optionally, to run tests (run_tests.sh), you also need:

```shell
sudo apt-get install virtualenv
```

Note that the same tensorflow package is also needed for building the 'release'
mode for LLVM.

## Docs

An end-to-end [demo](docs/inlining-demo/demo.md) using Fuchsia as a codebase from which
we extract a corpus and train a model.

[How to add a feature](docs/adding_features.md) guide.
[Extensibility model](docs/extensibility.md).
