# CONTEXT {'ctx_test': '1'}
ARG BASE_DOCKER=rocm/pytorch
FROM $BASE_DOCKER

ENV ctxtest=1
