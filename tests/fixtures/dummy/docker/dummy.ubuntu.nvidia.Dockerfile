# CONTEXT {'gpu_vendor': 'NVIDIA', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
FROM $BASE_DOCKER
USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    unzip \
    jq \
    python3-pip \
    git \
    vim \
    wget \
    openmpi-bin libopenmpi-dev

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh

RUN conda --version && \
    conda init
RUN pip install --upgrade pip
RUN pip install typing-extensions
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' |  tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake

# record configuration for posterity
RUN pip list
