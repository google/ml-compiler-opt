# TODO(boomanaiden154): Refactor to 22.04 and remove custom PPAs once we aren't
# dependent upon a specific tensorflow version and thus a specific python version
# (once TFLite patch lands)
FROM ubuntu:20.04
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
    software-properties-common
RUN wget --quiet https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz && \
    mkdir /tmp/tensorflow && \
    tar xfz libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz -C /tmp/tensorflow
# install latest cmake rather than the one in the 20.04 repos as that version
# doesn't support some cmake options used by MLGO
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake
# install latest git rather than 20.04 default due to that git version
# having some issues with projects commonly used by MLGO such as chromium
RUN apt-add-repository ppa:git-core/ppa && \
    apt-get update && \
    apt-get install -y git vim
COPY . /ml-compiler-opt
RUN python3 -m pip install -r /ml-compiler-opt/requirements-dev.txt