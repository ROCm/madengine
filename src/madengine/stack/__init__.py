"""Compose workload images from the compatibility matrix.

ROCm framework stacks (vLLM, SGLang, Primus, ...) are built and published
independently, so the same component (torch, aiter, MoRI, ...) exists as
different builds in each. The compatibility matrix lists the builds MAD
authorizes; a workload's image_spec selects from it, and madengine turns the
selection into an image the same way for every stack:

    compose      selection + matrix -> requirement closure in install order
    dockerfile   generate the Dockerfile for a composed plan
    wheels       read wheel indexes and wheel metadata (harvest, provenance)
"""
