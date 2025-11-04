FROM python:3.10

WORKDIR /ml-compiler-opt
COPY . .
RUN pip install pipenv && pipenv sync --system --categories "packages,dev-packages,ci" && pipenv --clear

WORKDIR /ml-compiler-opt/compiler_opt/tools
ENV PYTHONPATH=/ml-compiler-opt

VOLUME /external
