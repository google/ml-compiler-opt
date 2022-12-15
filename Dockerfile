FROM python:3.9

COPY . /ml-compiler-opt
WORKDIR /ml-compiler-opt
RUN pip install pipenv && pipenv sync --system && pipenv --clear

WORKDIR /ml-compiler-opt/compiler_opt/tools
ENV PYTHONPATH=/ml-compiler-opt

VOLUME /external
