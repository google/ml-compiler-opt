FROM python:3.10

WORKDIR /ml-compiler-opt
COPY . .
RUN pip install pipenv && pipenv sync --system && pipenv --clear

WORKDIR /ml-compiler-opt/compiler_opt/tools
ENV PYTHONPATH=/ml-compiler-opt

VOLUME /external
