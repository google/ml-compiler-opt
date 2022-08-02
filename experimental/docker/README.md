# MLGO Development Dockerfiles

This folder contains dockerfiles with all the dependencies necessary to do
development work with MLGO. Note that these dockerfiles do not contain a
clone of LLVM, only system dependencies, python dependencies, and the current
repository. You are expected to clone LLVM and configure the build yourself.
To see how to do this, the [inliner demo](../docs/demo/demo.md) should have
some useful guidance.

### Building the image

To build the image, make sure that you are in the root of the ml-compiler-opt
repository and then run:
```bash
docker build -t <your tag here> -f ./docker/development.Dockerfile .
```

### Disclaimers

**WARNING**: These development dockerfiles are not guaranteed to work. We will
do a best effort to make sure these dockerfiles are working, but they might stop
working at any point in time and should absolutely not be used in production
environments.