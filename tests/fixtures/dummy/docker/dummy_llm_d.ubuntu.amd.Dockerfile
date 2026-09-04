# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# llm-d benchmark client.
#
# This image holds no GPU code: it drives load over HTTP against an llm-d
# gateway and the inference itself happens on the model-server pods that llm-d
# manages. Keeping it a plain slim-python image makes the client pod schedulable
# on any node and keeps the build fast.
ARG BASE_DOCKER=python:3.11-slim
FROM $BASE_DOCKER

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
