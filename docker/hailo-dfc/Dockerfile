# Hailo Dataflow Compiler dev base.
# Ubuntu 20.04 + Python 3.10 — the combo the DFC is built for.
#
# Why build Python from source instead of the deadsnakes PPA: on this machine's
# network path apt accepts the deadsnakes InRelease but never fetches its Packages
# index (no request, no error), so `python3.10` is uninstallable via the PPA.
# A source build sidesteps apt entirely and gives a stock CPython 3.10.
#
# The DFC itself is NOT bundled (it's login-gated on the Hailo Developer Zone):
# download the wheel and `pip install` it inside the running container.
# GPU passthrough still works via `docker run --gpus all`.
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PIP_NO_CACHE_DIR=1

ARG PYTHON_VERSION=3.10.14

# Runtime/system libs the DFC + model zoo pull in (graphviz, opencv, ffmpeg, tk),
# plus everything needed to build CPython with all stdlib modules.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget curl git unzip pkg-config \
        build-essential \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libncurses5-dev libncursesw5-dev libffi-dev liblzma-dev \
        libgdbm-dev libgdbm-compat-dev uuid-dev tk-dev libexpat1-dev \
        graphviz libgraphviz-dev \
        libgl1-mesa-glx libglib2.0-0 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Build and install CPython 3.10 to /usr/local (altinstall -> python3.10).
RUN cd /tmp && \
    wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" && \
    tar xf "Python-${PYTHON_VERSION}.tgz" && \
    cd "Python-${PYTHON_VERSION}" && \
    ./configure --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd /tmp && rm -rf "Python-${PYTHON_VERSION}" "Python-${PYTHON_VERSION}.tgz"

# Isolated venv so `python`/`pip` on PATH resolve to 3.10.
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace/observer

CMD ["/bin/bash"]
