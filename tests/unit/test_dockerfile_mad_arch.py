#!/usr/bin/env python3
"""Tests for MAD_SYSTEM_GPU_ARCHITECTURE Dockerfile heuristics."""

import pytest

from madengine.execution.dockerfile_utils import (
    dockerfile_requires_explicit_mad_arch_build_arg,
)


@pytest.mark.parametrize(
    "content,expected",
    [
        ("ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942\nFROM ubuntu\n", False),
        ("ARG MAD_SYSTEM_GPU_ARCHITECTURE\n", True),
        ("FROM ubuntu\n", False),
        (
            "ARG MAD_SYSTEM_GPU_ARCHITECTURE\nARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942\n",
            False,
        ),
    ],
)
def test_dockerfile_requires_explicit_mad_arch(content, expected):
    assert dockerfile_requires_explicit_mad_arch_build_arg(content) is expected
