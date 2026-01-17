"""SSH Multi-Node Runner for MAD Engine

This package provides SSH-based multi-node distributed training capabilities
for the MAD Engine framework.

Main Components:
- SSHMultiNodeRunner: Main orchestration class
- SSHClientManager: Robust SSH connection management
- MultiNodeConfig: Configuration management
- Configuration validation and setup instructions
- Utilities: Common helper functions

Example Usage:
    from runners.ssh import SSHMultiNodeRunner, MultiNodeConfig
    
    config = MultiNodeConfig.from_config_file('config.ini')
    runner = SSHMultiNodeRunner(config)
    success = runner.run()

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

from .config_manager import (
    MultiNodeConfig,
    SSHConfig,
    ClusterConfig,
    TrainingConfig,
    MadEngineConfig
)
from .ssh_client_manager import SSHClientManager
from .run import SSHMultiNodeRunner
from . import utils

__version__ = "1.0.0"
__author__ = "Advanced Micro Devices, Inc."

__all__ = [
    'SSHMultiNodeRunner',
    'SSHClientManager',
    'MultiNodeConfig',
    'SSHConfig',
    'ClusterConfig',
    'TrainingConfig',
    'MadEngineConfig',
    'utils'
]