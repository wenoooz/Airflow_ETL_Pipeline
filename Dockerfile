FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      llvm-14 \
      llvm-14-dev \
      && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV DRJIT_LIBLLVM_PATH=/usr/lib/llvm-14/lib/libLLVM-14.so

USER airflow
RUN pip install --no-cache-dir "tensorflow==2.15.0" "sionna==0.15.0"
