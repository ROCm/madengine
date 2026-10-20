# CONTEXT {'gpu_vendor': 'NVIDIA', 'guest_os': 'UBUNTU'}
# llm-d benchmark client. See dummy_llm_d.ubuntu.amd.Dockerfile — the client
# runs no GPU code, so the two variants are identical apart from this header.
ARG BASE_DOCKER=python:3.11-slim
FROM $BASE_DOCKER

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
