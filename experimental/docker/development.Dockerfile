FROM ubuntu:22.04
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-distutils \
    python-is-python3 \
    python3 \
    python3-pip \
    tmux \
    g++ \
    ccache \
    binutils-gold \
    binutils-dev \
    ninja-build \
    pkg-config \
    gcc-multilib \
    g++-multilib \
    gawk \
    dos2unix \
    libxml2-dev \
    rsync \
    git \
    libtool \
    m4 \
    automake \
    libgcrypt-dev \
    liblzma-dev \
    libssl-dev \
    libgss-dev \
    python3-dev \
    wget \
    zlib1g-dev \
    tcl-dev \
    libpfm4-dev \
    software-properties-common \
    cmake \
    git \
    vim \
    libpthreadpool-dev
RUN mkdir /tflite
WORKDIR /tflite
COPY buildbot/build_tflite.sh ./
RUN ./build_tflite.sh
COPY . /ml-compiler-opt
WORKDIR /ml-compiler-opt
RUN pip3 install pipenv && pipenv sync --categories "packages dev-packages ci" --system && pipenv --clear
RUN apt-get autoremove -y --purge \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /

