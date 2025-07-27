#!/usr/bin/env python3
"""
Enhanced Runner Factory for MADEngine

This module provides an enhanced factory for creating distributed runners
with plugin discovery, validation, and better extensibility.
"""

import importlib
import logging
import os
import pkgutil
from typing import Dict, Type, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from madengine.runners.base import BaseDistributedRunner
from madengine.core.errors import MADEngineError


class RunnerRegistrationError(MADEngineError):
    """Runner registration specific errors."""
    pass


class RunnerCreationError(MADEngineError):
    """Runner creation specific errors."""
    pass


@dataclass
class RunnerMetadata:
    """Metadata for a registered runner."""
    
    runner_type: str
    runner_class: Type[BaseDistributedRunner]
    description: str
    version: str
    dependencies: List[str]
    supported_platforms: List[str]
    configuration_schema: Optional[Dict[str, Any]] = None
    
    @property
    def is_available(self) -> bool:
        """Check if runner dependencies are available."""
        try:
            for dep in self.dependencies:
                importlib.import_module(dep)
            return True
        except ImportError:
            return False


class RunnerPlugin(ABC):
    """Abstract base class for runner plugins."""
    
    @abstractmethod
    def get_metadata(self) -> RunnerMetadata:
        """Get runner metadata."""
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate runner configuration."""
        pass


class RunnerFactory:
    """Enhanced factory for creating distributed runners with plugin support."""

    _runners: Dict[str, RunnerMetadata] = {}
    _logger = logging.getLogger(__name__)

    @classmethod
    def register_runner(
        cls, 
        runner_type: str, 
        runner_class: Type[BaseDistributedRunner],
        description: str = "",
        version: str = "1.0.0",
        dependencies: Optional[List[str]] = None,
        supported_platforms: Optional[List[str]] = None,
        configuration_schema: Optional[Dict[str, Any]] = None
    ):
        """Register a runner class with metadata.

        Args:
            runner_type: Type identifier for the runner
            runner_class: Runner class to register
            description: Human-readable description
            version: Runner version
            dependencies: List of required module dependencies
            supported_platforms: List of supported platforms
            configuration_schema: JSON schema for configuration validation
            
        Raises:
            RunnerRegistrationError: If registration fails
        """
        if not issubclass(runner_class, BaseDistributedRunner):
            raise RunnerRegistrationError(
                f"Runner class {runner_class} must inherit from BaseDistributedRunner"
            )
        
        metadata = RunnerMetadata(
            runner_type=runner_type,
            runner_class=runner_class,
            description=description,
            version=version,
            dependencies=dependencies or [],
            supported_platforms=supported_platforms or ["linux", "darwin"],
            configuration_schema=configuration_schema
        )
        
        cls._runners[runner_type] = metadata
        cls._logger.info(f"Registered runner: {runner_type} v{version}")

    @classmethod
    def register_plugin(cls, plugin: RunnerPlugin):
        """Register a runner plugin.
        
        Args:
            plugin: Runner plugin instance
            
        Raises:
            RunnerRegistrationError: If plugin registration fails
        """
        try:
            metadata = plugin.get_metadata()
            cls._runners[metadata.runner_type] = metadata
            cls._logger.info(f"Registered plugin: {metadata.runner_type} v{metadata.version}")
        except Exception as e:
            raise RunnerRegistrationError(f"Failed to register plugin: {e}")

    @classmethod
    def discover_plugins(cls, package_name: str = "madengine.runners"):
        """Discover and register runner plugins automatically.
        
        Args:
            package_name: Package to search for plugins
        """
        try:
            package = importlib.import_module(package_name)
            package_path = package.__path__
            
            for _, module_name, _ in pkgutil.iter_modules(package_path):
                if module_name.endswith('_runner'):
                    try:
                        module = importlib.import_module(f"{package_name}.{module_name}")
                        
                        # Look for runner plugins
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                issubclass(attr, RunnerPlugin) and 
                                attr != RunnerPlugin):
                                try:
                                    plugin = attr()
                                    cls.register_plugin(plugin)
                                except Exception as e:
                                    cls._logger.warning(f"Failed to register plugin {attr_name}: {e}")
                                    
                    except ImportError as e:
                        cls._logger.debug(f"Could not import {module_name}: {e}")
                        
        except Exception as e:
            cls._logger.warning(f"Plugin discovery failed: {e}")

    @classmethod
    def create_runner(
        cls, 
        runner_type: str, 
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseDistributedRunner:
        """Create a runner instance with validation.

        Args:
            runner_type: Type of runner to create
            config: Configuration dictionary
            **kwargs: Additional arguments to pass to runner constructor

        Returns:
            Runner instance

        Raises:
            RunnerCreationError: If runner creation fails
        """
        if runner_type not in cls._runners:
            available_types = ", ".join(cls.get_available_runners())
            raise RunnerCreationError(
                f"Unknown runner type: {runner_type}. "
                f"Available types: {available_types}"
            )

        metadata = cls._runners[runner_type]
        
        # Check if runner is available
        if not metadata.is_available:
            missing_deps = []
            for dep in metadata.dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    missing_deps.append(dep)
            
            raise RunnerCreationError(
                f"Runner {runner_type} is not available. "
                f"Missing dependencies: {', '.join(missing_deps)}"
            )
        
        # Validate configuration if schema provided
        if config and metadata.configuration_schema:
            try:
                # In a real implementation, use jsonschema for validation
                cls._logger.debug(f"Validating configuration for {runner_type}")
            except Exception as e:
                raise RunnerCreationError(f"Configuration validation failed: {e}")
        
        try:
            runner_class = metadata.runner_class
            
            # Merge config into kwargs
            if config:
                kwargs.update(config)
                
            return runner_class(**kwargs)
            
        except Exception as e:
            raise RunnerCreationError(f"Failed to create runner {runner_type}: {e}")

    @classmethod
    def get_available_runners(cls) -> List[str]:
        """Get list of available runner types.

        Returns:
            List of registered runner types that are available
        """
        return [
            runner_type for runner_type, metadata in cls._runners.items()
            if metadata.is_available
        ]
    
    @classmethod
    def get_all_runners(cls) -> List[str]:
        """Get list of all registered runner types.

        Returns:
            List of all registered runner types
        """
        return list(cls._runners.keys())
    
    @classmethod
    def get_runner_info(cls, runner_type: str) -> Optional[RunnerMetadata]:
        """Get detailed information about a runner.
        
        Args:
            runner_type: Runner type to get info for
            
        Returns:
            Runner metadata or None if not found
        """
        return cls._runners.get(runner_type)
    
    @classmethod
    def get_runners_by_platform(cls, platform: str) -> List[str]:
        """Get runners that support a specific platform.
        
        Args:
            platform: Platform name (e.g., 'linux', 'darwin')
            
        Returns:
            List of runner types supporting the platform
        """
        return [
            runner_type for runner_type, metadata in cls._runners.items()
            if platform in metadata.supported_platforms and metadata.is_available
        ]
    
    @classmethod
    def validate_runner_config(cls, runner_type: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for a specific runner.
        
        Args:
            runner_type: Runner type
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            RunnerCreationError: If validation fails
        """
        if runner_type not in cls._runners:
            raise RunnerCreationError(f"Unknown runner type: {runner_type}")
        
        metadata = cls._runners[runner_type]
        
        if metadata.configuration_schema:
            # In a real implementation, use jsonschema
            cls._logger.debug(f"Validating configuration for {runner_type}")
            return True
        
        return True


def register_default_runners():
    """Register default runners with enhanced metadata."""
    try:
        from madengine.runners.ssh_runner import SSHDistributedRunner

        RunnerFactory.register_runner(
            runner_type="ssh",
            runner_class=SSHDistributedRunner,
            description="SSH-based distributed runner for remote execution",
            version="1.0.0",
            dependencies=["paramiko", "scp"],
            supported_platforms=["linux", "darwin", "win32"]
        )
    except ImportError as e:
        logging.warning(f"SSH runner not available: {e}")

    try:
        from madengine.runners.ansible_runner import AnsibleDistributedRunner

        RunnerFactory.register_runner(
            runner_type="ansible",
            runner_class=AnsibleDistributedRunner,
            description="Ansible-based distributed runner for infrastructure automation",
            version="1.0.0",
            dependencies=["ansible-runner", "ansible"],
            supported_platforms=["linux", "darwin"]
        )
    except ImportError as e:
        logging.warning(f"Ansible runner not available: {e}")

    try:
        from madengine.runners.k8s_runner import KubernetesDistributedRunner

        RunnerFactory.register_runner(
            runner_type="k8s",
            runner_class=KubernetesDistributedRunner,
            description="Kubernetes-based distributed runner for containerized workloads",
            version="1.0.0",
            dependencies=["kubernetes"],
            supported_platforms=["linux", "darwin", "win32"]
        )
        
        # Register alias
        RunnerFactory.register_runner(
            runner_type="kubernetes",
            runner_class=KubernetesDistributedRunner,
            description="Kubernetes-based distributed runner (alias for k8s)",
            version="1.0.0",
            dependencies=["kubernetes"],
            supported_platforms=["linux", "darwin", "win32"]
        )
    except ImportError as e:
        logging.warning(f"Kubernetes runner not available: {e}")


# Auto-register default runners and discover plugins
register_default_runners()
RunnerFactory.discover_plugins()
