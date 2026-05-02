#!/usr/bin/env python3
"""
Deployment Factory - Creates appropriate deployment instances.

Implements Factory pattern to dynamically create SLURM or Kubernetes
deployment instances based on target configuration.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from typing import Dict, Type

from .base import BaseDeployment, DeploymentConfig


class DeploymentFactory:
    """
    Factory for creating deployment instances.

    Supports dynamic registration and creation of deployment types.
    Currently supports: slurm, k8s/kubernetes
    """

    _deployments: Dict[str, Type[BaseDeployment]] = {}

    @classmethod
    def register(cls, deployment_type: str, deployment_class: Type[BaseDeployment]):
        """
        Register a deployment type.

        Args:
            deployment_type: Name of deployment type (e.g., "slurm", "k8s")
            deployment_class: Class implementing BaseDeployment
        """
        cls._deployments[deployment_type] = deployment_class

    @classmethod
    def create(cls, config: DeploymentConfig) -> BaseDeployment:
        """
        Create a deployment instance based on config.

        Args:
            config: Deployment configuration with target type

        Returns:
            Deployment instance for the specified target

        Raises:
            ValueError: If deployment type is not registered
        """
        deployment_class = cls._deployments.get(config.target)

        if not deployment_class:
            available = ", ".join(cls._deployments.keys())
            raise ValueError(
                f"Unknown deployment target: {config.target}. "
                f"Available: {available}"
            )

        return deployment_class(config)

    @classmethod
    def available_deployments(cls) -> list:
        """
        Get list of available deployment types.

        Returns:
            List of registered deployment type names
        """
        return list(cls._deployments.keys())


def register_default_deployments():
    """
    Register default deployment implementations.

    Called on module import to register built-in deployments.
    """
    # Always register SLURM (no optional dependencies)
    from .slurm import SlurmDeployment

    DeploymentFactory.register("slurm", SlurmDeployment)

    # Register Kubernetes if library is available
    try:
        from .kubernetes import KubernetesDeployment

        DeploymentFactory.register("k8s", KubernetesDeployment)
        DeploymentFactory.register("kubernetes", KubernetesDeployment)
    except ImportError:
        # Kubernetes library not installed, skip registration
        pass


# Auto-register on module import
register_default_deployments()

