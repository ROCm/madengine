#!/usr/bin/env python3
"""
Default gpu_vendor / guest_os for build-time additional context.

Used by CLI validation and BuildOrchestrator so Dockerfile filtering (Context.filter)
sees the same keys as the validated build defaults.
"""

from typing import Any, Dict

DEFAULT_GPU_VENDOR = "AMD"
DEFAULT_GUEST_OS = "UBUNTU"


def apply_build_context_defaults(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing gpu_vendor and guest_os. Mutates and returns ctx."""
    if "gpu_vendor" not in ctx:
        ctx["gpu_vendor"] = DEFAULT_GPU_VENDOR
    if "guest_os" not in ctx:
        ctx["guest_os"] = DEFAULT_GUEST_OS
    return ctx
