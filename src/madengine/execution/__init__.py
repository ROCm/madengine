"""
Execution layer for local container execution.

Provides Docker container execution capabilities for single-node local runs.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from .container_runner import ContainerRunner

__all__ = ["ContainerRunner"]

