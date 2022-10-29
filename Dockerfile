FROM python:3.9

COPY . /ml-compiler-opt
RUN pip install --no-cache-dir -r /ml-compiler-opt/requirements.txt

WORKDIR /ml-compiler-opt/compiler_opt/tools
ENV PYTHONPATH=/ml-compiler-opt

VOLUME /external
