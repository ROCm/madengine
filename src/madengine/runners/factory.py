#!/usr/bin/env python3
"""
Runner Factory for MADEngine

This module provides a factory for creating distributed runners
based on the specified runner type.
"""

import logging
from typing import Dict, Type

from madengine.runners.base import BaseDistributedRunner


class RunnerFactory:
    """Factory for creating distributed runners."""

    _runners: Dict[str, Type[BaseDistributedRunner]] = {}

    @classmethod
    def register_runner(
        cls, runner_type: str, runner_class: Type[BaseDistributedRunner]
    ):
        """Register a runner class.

        Args:
            runner_type: Type identifier for the runner
            runner_class: Runner class to register
        """
        cls._runners[runner_type] = runner_class

    @classmethod
    def create_runner(cls, runner_type: str, **kwargs) -> BaseDistributedRunner:
        """Create a runner instance.

        Args:
            runner_type: Type of runner to create
            **kwargs: Arguments to pass to runner constructor

        Returns:
            Runner instance

        Raises:
            ValueError: If runner type is not registered
        """
        if runner_type not in cls._runners:
            available_types = ", ".join(cls._runners.keys())
            raise ValueError(
                f"Unknown runner type: {runner_type}. "
                f"Available types: {available_types}"
            )

        runner_class = cls._runners[runner_type]
        return runner_class(**kwargs)

    @classmethod
    def get_available_runners(cls) -> list:
        """Get list of available runner types.

        Returns:
            List of registered runner types
        """
        return list(cls._runners.keys())


def register_default_runners():
    """Register default runners."""
    try:
        from madengine.runners.ssh_runner import SSHDistributedRunner

        RunnerFactory.register_runner("ssh", SSHDistributedRunner)
    except ImportError as e:
        logging.warning(f"SSH runner not available: {e}")

    try:
        from madengine.runners.ansible_runner import AnsibleDistributedRunner

        RunnerFactory.register_runner("ansible", AnsibleDistributedRunner)
    except ImportError as e:
        logging.warning(f"Ansible runner not available: {e}")

    try:
        from madengine.runners.k8s_runner import KubernetesDistributedRunner

        RunnerFactory.register_runner("k8s", KubernetesDistributedRunner)
        RunnerFactory.register_runner("kubernetes", KubernetesDistributedRunner)
    except ImportError as e:
        logging.warning(f"Kubernetes runner not available: {e}")

    try:
        from madengine.runners.slurm_runner import SlurmDistributedRunner

        RunnerFactory.register_runner("slurm", SlurmDistributedRunner)
    except ImportError as e:
        logging.warning(f"SLURM runner not available: {e}")


# Auto-register default runners
register_default_runners()
