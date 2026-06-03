#!/usr/bin/env python3
"""SSH Client Manager for MAD Engine Multi-Node Runner

This module provides a robust SSH client management class with connection pooling,
error handling, and retry mechanisms.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import logging
import socket
import time
from contextlib import contextmanager
from typing import Optional, Tuple, Iterator

try:
    import paramiko
except ImportError:
    raise ImportError("paramiko is required but not installed. Please install it with: pip install paramiko")


class SSHClientManager:
    """Manages SSH connections with robust error handling and retry mechanisms."""
    
    def __init__(self, hostname: str, username: str, password: Optional[str] = None, 
                 key_filename: Optional[str] = None, timeout: int = 30, max_retries: int = 3):
        """Initialize SSH client manager.
        
        Args:
            hostname: Target hostname or IP address
            username: SSH username
            password: SSH password (if using password auth)
            key_filename: Path to SSH private key (if using key auth)
            timeout: Connection timeout in seconds
            max_retries: Maximum number of connection retries
        """
        self.hostname = hostname
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"{__name__}.{hostname}")
        
        if not password and not key_filename:
            raise ValueError("Either password or key_filename must be provided")
    
    def _create_client(self) -> paramiko.SSHClient:
        """Create and configure a new SSH client.
        
        Returns:
            Configured SSH client
        """
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.load_system_host_keys()
        return client
    
    def _connect_with_retry(self, client: paramiko.SSHClient) -> None:
        """Connect SSH client with retry mechanism.
        
        Args:
            client: SSH client to connect
            
        Raises:
            ConnectionError: If all connection attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                if self.key_filename:
                    client.connect(
                        hostname=self.hostname,
                        username=self.username,
                        key_filename=self.key_filename,
                        timeout=self.timeout
                    )
                else:
                    client.connect(
                        hostname=self.hostname,
                        username=self.username,
                        password=self.password,
                        timeout=self.timeout
                    )
                
                self.logger.debug(f"Successfully connected to {self.hostname} on attempt {attempt + 1}")
                return
                
            except (paramiko.ssh_exception.AuthenticationException,
                    paramiko.ssh_exception.SSHException,
                    socket.error) as e:
                last_exception = e
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        raise ConnectionError(f"Failed to connect to {self.hostname} after {self.max_retries} attempts. Last error: {last_exception}")
    
    @contextmanager
    def get_client(self) -> Iterator[paramiko.SSHClient]:
        """Get an SSH client with automatic cleanup.
        
        Yields:
            Connected SSH client
            
        Raises:
            ConnectionError: If connection fails
        """
        client = self._create_client()
        try:
            self._connect_with_retry(client)
            yield client
        finally:
            try:
                client.close()
            except Exception as e:
                self.logger.warning(f"Error closing SSH connection: {e}")
    
    def execute_command(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """Execute a command on the remote host.
        
        Args:
            command: Command to execute
            timeout: Command timeout (uses default if None)
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
            
        Raises:
            ConnectionError: If SSH connection fails
            TimeoutError: If command times out
        """
        with self.get_client() as client:
            try:
                stdin, stdout, stderr = client.exec_command(command, timeout=timeout or self.timeout)
                
                exit_code = stdout.channel.recv_exit_status()
                stdout_text = stdout.read().decode('utf-8', errors='replace').strip()
                stderr_text = stderr.read().decode('utf-8', errors='replace').strip()
                
                return exit_code, stdout_text, stderr_text
                
            except socket.timeout:
                raise TimeoutError(f"Command timed out after {timeout or self.timeout} seconds: {command}")
    
    def test_connectivity(self) -> bool:
        """Test connectivity to the remote host.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            exit_code, stdout, stderr = self.execute_command('echo "connectivity_test"')
            return exit_code == 0 and stdout.strip() == "connectivity_test"
        except Exception as e:
            self.logger.error(f"Connectivity test failed: {e}")
            return False
