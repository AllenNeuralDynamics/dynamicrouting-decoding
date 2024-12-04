# hash:sha256:1784aa4b496aeeea5ddb77f818aa89b92d00fb1957d4eba3081a0918e8785e7b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.9.2 \
    pandas[all]==2.2.3 \
    pynwb==2.8.2 \
    scikit-learn==1.5.2 \
    zarr==2.18.3
