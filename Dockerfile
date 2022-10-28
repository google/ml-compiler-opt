FROM debian:bullseye
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python-is-python3 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY . /ml-compiler-opt
RUN python3 -m pip install -r /ml-compiler-opt/requirements.txt
ENV PYTHONPATH=/ml-compiler-opt
VOLUME /external
