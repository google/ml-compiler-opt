# Infrastructure for MLGO --- a Machine Learning Guided Compiler Optimizations Feamework.

MLGO is a framework for integrating ML techniques systematically in LLVM. It
replaces human-crafted optimization heuristics in LLVM with machine learned
models. Our pioneering project is on the inlining-for-size optimization in LLVM.

We currently use two different ML algorithms: Policy Gradient and Evolution
Strategies, to train the inlining-for-size model, and achieve up to 7% size
reduction, when compared to state of the art LLVM -Oz. The compiler components
are available in the main LLVM repository. This repository contains the training
infrastructure and related tools for MLGO.

Currently we only support training inlining-for-size policy with Policy
Gradient. We are working on:

1.  releasing Evolution Strategies training;
2.  more optimization problems other than inlining-for-size.

Check out this [demo](docs/demo/demo.md) for an end-to-end demonstration of how
to train your own inlining-for-size policy from the scratch with Policy
Gradient.

For more details about MLGO, please refer to our paper
[MLGO: a Machine Learning Guided Compiler Optimizations Framework](https://arxiv.org/abs/2101.04808).

## Prerequisites

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
