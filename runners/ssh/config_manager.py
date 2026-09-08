#!/usr/bin/env python3
"""Configuration Management for SSH Multi-Node Runner

This module provides configuration validation and management for the SSH runner.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import configparser
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class SSHConfig:
    """SSH configuration settings."""
    user: str
    password: Optional[str] = None
    key_file: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate SSH configuration after initialization."""
        if not self.password and not self.key_file:
            raise ValueError("Either SSH password or key file must be specified")
        
        if self.key_file and not os.path.exists(self.key_file):
            raise FileNotFoundError(f"SSH key file not found: {self.key_file}")


@dataclass
class ClusterConfig:
    """Cluster configuration settings."""
    nodes: List[str]
    master_addr: Optional[str] = None
    master_port: int = 4000
    
    def __post_init__(self):
        """Validate cluster configuration after initialization."""
        if not self.nodes:
            raise ValueError("At least one node must be specified")
        
        # Clean up node list
        self.nodes = [node.strip() for node in self.nodes if node.strip()]
        
        if not self.nodes:
            raise ValueError("At least one valid node must be specified")
        
        # Set master address to first node if not specified
        if not self.master_addr:
            self.master_addr = self.nodes[0]


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    model: str
    shared_data_path: str = '/nfs/data'
    nccl_interface: str = 'ens14np0'
    gloo_interface: str = 'ens14np0'
    timeout: int = 3600
    additional_args: str = ''
    
    def __post_init__(self):
        """Validate training configuration after initialization."""
        if not self.model:
            raise ValueError("Model must be specified")


@dataclass
class MadEngineConfig:
    """MAD Engine specific configuration."""
    path: str = 'madengine'
    working_directory: str = 'MAD'


@dataclass
class MultiNodeConfig:
    """Complete multi-node runner configuration."""
    ssh: SSHConfig
    cluster: ClusterConfig
    training: TrainingConfig
    madengine: MadEngineConfig = field(default_factory=MadEngineConfig)
    
    @classmethod
    def from_args(cls, args) -> 'MultiNodeConfig':
        """Create configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Complete configuration object
        """
        # Parse nodes list
        nodes = [node.strip() for node in args.nodes.split(',') if node.strip()]
        
        ssh_config = SSHConfig(
            user=args.ssh_user,
            password=getattr(args, 'ssh_password', None),
            key_file=getattr(args, 'ssh_key', None),
            timeout=getattr(args, 'ssh_timeout', 30),
            max_retries=getattr(args, 'ssh_max_retries', 3)
        )
        
        cluster_config = ClusterConfig(
            nodes=nodes,
            master_addr=getattr(args, 'master_addr', None),
            master_port=getattr(args, 'master_port', 4000)
        )
        
        training_config = TrainingConfig(
            model=args.model,
            shared_data_path=getattr(args, 'shared_data_path', '/nfs/data'),
            nccl_interface=getattr(args, 'nccl_interface', 'ens14np0'),
            gloo_interface=getattr(args, 'gloo_interface', 'ens14np0'),
            timeout=getattr(args, 'timeout', 3600),
            additional_args=getattr(args, 'additional_args', '')
        )
        
        madengine_config = MadEngineConfig(
            path=getattr(args, 'madengine_path', 'madengine'),
            working_directory=getattr(args, 'working_directory', 'MAD')
        )
        
        return cls(
            ssh=ssh_config,
            cluster=cluster_config,
            training=training_config,
            madengine=madengine_config
        )
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'MultiNodeConfig':
        """Create configuration from INI file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Complete configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            configparser.Error: If config file is malformed
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Parse SSH configuration
        ssh_section = config['ssh'] if 'ssh' in config else {}
        ssh_config = SSHConfig(
            user=ssh_section.get('user'),
            password=ssh_section.get('password'),
            key_file=ssh_section.get('key_file'),
            timeout=int(ssh_section.get('timeout', 30)),
            max_retries=int(ssh_section.get('max_retries', 3))
        )
        
        # Parse cluster configuration
        cluster_section = config['cluster'] if 'cluster' in config else {}
        nodes_str = cluster_section.get('nodes', '')
        nodes = [node.strip() for node in nodes_str.split(',') if node.strip()]
        
        cluster_config = ClusterConfig(
            nodes=nodes,
            master_addr=cluster_section.get('master_addr'),
            master_port=int(cluster_section.get('master_port', 4000))
        )
        
        # Parse training configuration
        training_section = config['training'] if 'training' in config else {}
        training_config = TrainingConfig(
            model=training_section.get('model'),
            shared_data_path=training_section.get('shared_data_path', '/nfs/data'),
            nccl_interface=training_section.get('nccl_interface', 'ens14np0'),
            gloo_interface=training_section.get('gloo_interface', 'ens14np0'),
            timeout=int(training_section.get('timeout', 3600)),
            additional_args=training_section.get('additional_args', '')
        )
        
        # Parse madengine configuration
        madengine_section = config['madengine'] if 'madengine' in config else {}
        madengine_config = MadEngineConfig(
            path=madengine_section.get('path', 'madengine'),
            working_directory=madengine_section.get('working_directory', 'MAD')
        )
        
        return cls(
            ssh=ssh_config,
            cluster=cluster_config,
            training=training_config,
            madengine=madengine_config
        )
    
    def validate(self) -> None:
        """Validate the complete configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Configurations are validated in their __post_init__ methods
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'ssh': {
                'user': self.ssh.user,
                'has_password': bool(self.ssh.password),
                'has_key_file': bool(self.ssh.key_file),
                'timeout': self.ssh.timeout,
                'max_retries': self.ssh.max_retries
            },
            'cluster': {
                'nodes': self.cluster.nodes,
                'master_addr': self.cluster.master_addr,
                'master_port': self.cluster.master_port
            },
            'training': {
                'model': self.training.model,
                'shared_data_path': self.training.shared_data_path,
                'nccl_interface': self.training.nccl_interface,
                'gloo_interface': self.training.gloo_interface,
                'timeout': self.training.timeout,
                'additional_args': self.training.additional_args
            },
            'madengine': {
                'path': self.madengine.path,
                'working_directory': self.madengine.working_directory
            }
        }


def merge_config_file_with_args(config_path: str, args) -> 'MultiNodeConfig':
    """Merge configuration file with command line arguments.
    
    Command line arguments take precedence over config file values.
    
    Args:
        config_path: Path to configuration file
        args: Command line arguments
        
    Returns:
        Merged configuration
    """
    # Start with config file
    config = MultiNodeConfig.from_config_file(config_path)
    
    # Override with command line arguments if provided
    if hasattr(args, 'nodes') and args.nodes:
        nodes = [node.strip() for node in args.nodes.split(',') if node.strip()]
        config.cluster.nodes = nodes
    
    if hasattr(args, 'master_addr') and args.master_addr:
        config.cluster.master_addr = args.master_addr
    
    if hasattr(args, 'master_port') and args.master_port:
        config.cluster.master_port = args.master_port
    
    if hasattr(args, 'ssh_user') and args.ssh_user:
        config.ssh.user = args.ssh_user
    
    if hasattr(args, 'ssh_password') and args.ssh_password:
        config.ssh.password = args.ssh_password
    
    if hasattr(args, 'ssh_key') and args.ssh_key:
        config.ssh.key_file = args.ssh_key
    
    if hasattr(args, 'model') and args.model:
        config.training.model = args.model
    
    if hasattr(args, 'shared_data_path') and args.shared_data_path:
        config.training.shared_data_path = args.shared_data_path
    
    if hasattr(args, 'timeout') and args.timeout:
        config.training.timeout = args.timeout
    
    if hasattr(args, 'madengine_path') and args.madengine_path:
        config.madengine.path = args.madengine_path
    
    # Re-validate after merging
    config.validate()
    return config
