FROM ubuntu:24.04

ARG USERID

RUN apt-get update && \
    apt-get install -y \
    cmake \
    git \
    ninja-build \
    clang-19 \
    lld-19 \
    sudo \
    adduser \
    tmux \
    bzip2 \
    curl \
    vim \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-venv

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang-19 100
RUN update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-19 100
RUN update-alternatives --install /usr/bin/ld ld /usr/bin/lld-19  100

RUN python -m ensurepip

RUN mkdir -p /work/tflite
WORKDIR /work/tflite
RUN curl https://raw.githubusercontent.com/google/ml-compiler-opt/main/buildbot/build_tflite.sh | bash -s

WORKDIR /work

RUN git clone https://github.com/llvm/llvm-project.git
RUN git clone https://github.com/google/ml-compiler-opt

WORKDIR /work/llvm-project
WORKDIR /work/ml-compiler-opt

# we want a venv to avoid clashing with py3.11 system packages.
RUN python -m venv /work/pyenv
ENV PATH="/work/pyenv/bin:$PATH"

RUN python -m pip install pipenv
RUN ./versioned_pipenv sync --system --categories "packages dev-packages ci"

