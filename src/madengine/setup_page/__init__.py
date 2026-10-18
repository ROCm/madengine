#!/usr/bin/env python3
"""
Setup page generator package for madengine.

Generates a self-contained, PyTorch-style "setup picker" HTML page that lets a
user select every relevant madengine dimension (model/tags plus the full
context-variable schema) and copy the exact ``madengine run`` command.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from .generator import collect_models, render_setup_page, generate_setup_page
from .schema import CONTEXT_SCHEMA, SECTIONS, WORKLOADS

__all__ = [
    "CONTEXT_SCHEMA",
    "SECTIONS",
    "WORKLOADS",
    "collect_models",
    "render_setup_page",
    "generate_setup_page",
]
