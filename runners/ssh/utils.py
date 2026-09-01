#!/usr/bin/env python3
"""Utility functions for SSH Multi-Node Runner

This module provides common utility functions used across the SSH runner.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import logging
import socket
import time
from typing import Dict, Any, Optional


def setup_logging(level: str = 'INFO', format_string: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)


def validate_network_connectivity(hostname: str, port: int, timeout: int = 5) -> bool:
    """Test network connectivity to a host and port.
    
    Args:
        hostname: Target hostname or IP address
        port: Target port number
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with socket.create_connection((hostname, port), timeout=timeout):
            return True
    except (socket.error, socket.timeout):
        return False


def wait_for_port_ready(hostname: str, port: int, max_wait_time: int = 60, check_interval: int = 2) -> bool:
    """Wait for a port to become available on a host.
    
    Args:
        hostname: Target hostname or IP address
        port: Target port number
        max_wait_time: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        
    Returns:
        True if port becomes available, False if timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        if validate_network_connectivity(hostname, port):
            return True
        time.sleep(check_interval)
    
    return False


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    
    if hours < 24:
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
    
    days = int(hours // 24)
    remaining_hours = int(hours % 24)
    
    return f"{days}d {remaining_hours}h {remaining_minutes}m {remaining_seconds}s"


def sanitize_hostname(hostname: str) -> str:
    """Sanitize hostname for safe use in file names and logs.
    
    Args:
        hostname: Raw hostname or IP address
        
    Returns:
        Sanitized hostname safe for file system use
    """
    # Replace common problematic characters
    safe_hostname = hostname.replace(':', '_').replace('/', '_').replace('\\', '_')
    # Remove any remaining problematic characters
    safe_hostname = ''.join(c for c in safe_hostname if c.isalnum() or c in '-_.')
    return safe_hostname


def create_node_summary(nodes: list, successful: list, failed: list) -> Dict[str, Any]:
    """Create a summary of node execution results.
    
    Args:
        nodes: List of all nodes
        successful: List of successful nodes
        failed: List of failed nodes
        
    Returns:
        Dictionary containing execution summary
    """
    return {
        'total_nodes': len(nodes),
        'successful_nodes': len(successful),
        'failed_nodes': len(failed),
        'success_rate': len(successful) / len(nodes) if nodes else 0.0,
        'successful_node_list': successful,
        'failed_node_list': failed,
        'all_successful': len(failed) == 0
    }


def validate_madengine_command(command: str) -> bool:
    """Validate that a madengine command looks reasonable.
    
    Args:
        command: Command string to validate
        
    Returns:
        True if command appears valid, False otherwise
    """
    required_parts = ['madengine', 'run', '--tags']
    return all(part in command for part in required_parts)


def escape_shell_argument(argument: str) -> str:
    """Escape shell argument for safe execution.
    
    Args:
        argument: Argument to escape
        
    Returns:
        Escaped argument safe for shell execution
    """
    # Simple escaping - wrap in single quotes and escape any single quotes
    return f"'{argument.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"


def parse_node_list(nodes_string: str) -> list:
    """Parse a comma-separated list of nodes.
    
    Args:
        nodes_string: Comma-separated string of node names/IPs
        
    Returns:
        List of cleaned node names
    """
    if not nodes_string:
        return []
    
    nodes = [node.strip() for node in nodes_string.split(',') if node.strip()]
    return nodes


def get_network_interfaces() -> Dict[str, str]:
    """Get available network interfaces on the local machine.
    
    Returns:
        Dictionary mapping interface names to IP addresses
    """
    interfaces = {}
    
    try:
        import socket
        # Get hostname
        hostname = socket.gethostname()
        # Get local IP
        local_ip = socket.gethostbyname(hostname)
        interfaces['local'] = local_ip
        
        # Try to get more detailed interface information
        try:
            import psutil
            for interface, addresses in psutil.net_if_addrs().items():
                for address in addresses:
                    if address.family == socket.AF_INET:
                        interfaces[interface] = address.address
                        break
        except ImportError:
            # psutil not available, use basic detection
            pass
            
    except Exception:
        # Fallback to basic detection
        pass
    
    return interfaces
