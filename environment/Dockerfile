# hash:sha256:c1c22bdcd321907331994fc09c331b859992ad3707344b7c96828b90127ccfa8
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG API_KEY
ARG API_SECRET
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

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
    zarr==2.18.0 \
    -e git+https://github.com/AllenInstitute/dynamic_routing_analysis@5f84dd04eebbfedd18de262cd1b57e2bc67f964b#egg=dynamic-routing-analysis
