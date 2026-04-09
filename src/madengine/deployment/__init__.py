"""
Deployment layer for distributed execution.

Provides deployment implementations for SLURM and Kubernetes clusters.
Uses Factory pattern for creating appropriate deployment instances.

Architecture:
- BaseDeployment: Abstract base class defining deployment workflow
- SlurmDeployment: SLURM cluster deployment (uses CLI commands)
- KubernetesDeployment: Kubernetes cluster deployment (uses Python library)
- DeploymentFactory: Factory for creating deployment instances

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from .base import (
    BaseDeployment,
    DeploymentConfig,
    DeploymentResult,
    DeploymentStatus,
)
from .factory import DeploymentFactory

__all__ = [
    "BaseDeployment",
    "DeploymentConfig",
    "DeploymentResult",
    "DeploymentStatus",
    "DeploymentFactory",
]

