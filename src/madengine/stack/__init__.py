"""Cross-stack software locks for MAD images.

ROCm framework stacks (vLLM, SGLang, Primus, ...) are built and published
independently, so the same component (torch, aiter, MoRI, ...) exists as
different builds in each. A stack lock names the exact artifacts, wheel URLs or
pinned source builds, that make up one best-known stack for one platform, and
madengine turns it into an image the same way for every stack:

    wheels   read wheel indexes and wheel metadata (harvest, provenance)
    lock     load, select, override, check and diff locks
    dockerfile   generate the Dockerfile for a resolved lock
"""
