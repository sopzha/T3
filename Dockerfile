FROM --platform=linux/amd64 ubuntu:24.04

# Install apt dependencies
RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
    && apt install -y \
        binutils \
        cmake \
        g++-12 \
        git \
        libllvm17 \
        libboost-system1.83.0 \
        libboost-context1.83.0 \
        libboost-filesystem1.83.0 \
        libjemalloc2 \
        libssl3 libre2-10 \
        liburing-dev \
        libtinfo6 \
        llvm-14-runtime \
        lz4 \
        make \
        python3 \
        python-is-python3 \
        python3-pip \
        python3-venv \
        gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /venv
COPY requirements.txt /app/
RUN /venv/bin/pip install --no-cache-dir -r /app/requirements.txt

ENTRYPOINT ["/venv/bin/python", "main.py"]

