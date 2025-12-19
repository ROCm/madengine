"""
madengine Utilities

Utility modules for madengine including GPU configuration resolution.
"""

from .gpu_config import GPUConfigResolver, resolve_runtime_gpus

__all__ = ["GPUConfigResolver", "resolve_runtime_gpus"]

