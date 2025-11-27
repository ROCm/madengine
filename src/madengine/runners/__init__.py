#!/usr/bin/env python3
"""
MADEngine Distributed Runners Package

This package provides distributed runners for orchestrating workloads
across multiple nodes and clusters using different infrastructure types.
"""

from .base import (
    BaseDistributedRunner,
    NodeConfig,
    WorkloadSpec,
    ExecutionResult,
    DistributedResult,
)
from .factory import RunnerFactory

# Import runners (optional imports to handle missing dependencies)
try:
    from .ssh_runner import SSHDistributedRunner

    __all__ = ["SSHDistributedRunner"]
except ImportError:
    __all__ = []

try:
    from .ansible_runner import AnsibleDistributedRunner

    __all__.append("AnsibleDistributedRunner")
except ImportError:
    pass

try:
    from .k8s_runner import KubernetesDistributedRunner

    __all__.append("KubernetesDistributedRunner")
except ImportError:
    pass

# Always export base classes and factory
__all__.extend(
    [
        "BaseDistributedRunner",
        "NodeConfig",
        "WorkloadSpec",
        "ExecutionResult",
        "DistributedResult",
        "RunnerFactory",
    ]
)

__version__ = "1.0.0"
