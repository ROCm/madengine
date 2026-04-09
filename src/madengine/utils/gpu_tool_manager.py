#!/usr/bin/env python3
"""
Base GPU Tool Manager Architecture

Provides abstract base class and common infrastructure for GPU vendor-specific
tool managers (AMD ROCm, NVIDIA CUDA, etc.).

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import logging
import os
import subprocess
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseGPUToolManager(ABC):
    """Abstract base class for GPU vendor-specific tool managers.
    
    Provides common infrastructure for:
    - Tool availability checking
    - Command execution with timeout
    - Result caching
    - Consistent logging
    
    Subclasses implement vendor-specific logic for:
    - Version detection
    - Tool selection
    - Command execution with fallback
    """
    
    def __init__(self):
        """Initialize base GPU tool manager."""
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        
    @abstractmethod
    def get_version(self) -> Optional[str]:
        """Get GPU vendor tool version (e.g., ROCm version, CUDA version).
        
        Returns:
            Version string or None if unable to detect
        """
        pass
    
    @abstractmethod
    def execute_command(
        self,
        command: str,
        fallback_command: Optional[str] = None,
        timeout: int = 30
    ) -> str:
        """Execute command with optional fallback.
        
        Args:
            command: Primary command to execute
            fallback_command: Optional fallback command if primary fails
            timeout: Command timeout in seconds
            
        Returns:
            Command output as string
            
        Raises:
            RuntimeError: If both primary and fallback commands fail
        """
        pass
    
    def is_tool_available(self, tool_path: str) -> bool:
        """Check if a tool exists and is executable.
        
        Args:
            tool_path: Path to the tool (e.g., /opt/rocm/bin/amd-smi)
            
        Returns:
            True if tool exists and is executable, False otherwise
        """
        cache_key = f"tool_available:{tool_path}"
        
        # Check cache first
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Check if file exists and is executable
        result = os.path.isfile(tool_path) and os.access(tool_path, os.X_OK)
        
        # Cache the result
        with self._cache_lock:
            self._cache[cache_key] = result
        
        return result
    
    def _execute_shell_command(
        self,
        command: str,
        timeout: int = 30,
        check_returncode: bool = True
    ) -> Tuple[bool, str, str]:
        """Execute a shell command and return result.
        
        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            check_returncode: If True, only succeed on returncode 0
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = (result.returncode == 0) if check_returncode else True
            return success, result.stdout.strip(), result.stderr.strip()
            
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except FileNotFoundError:
            return False, "", f"Command not found: {command.split()[0]}"
        except Exception as e:
            return False, "", f"Command execution error: {str(e)}"
    
    def _cache_result(self, key: str, value: Any) -> None:
        """Cache a result for future use.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._cache_lock:
            self._cache[key] = value
    
    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get a cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._cache_lock:
            return self._cache.get(key)
    
    def _log_debug(self, message: str) -> None:
        """Log a debug message.
        
        Args:
            message: Debug message
        """
        logger.debug(f"[{self.__class__.__name__}] {message}")
    
    def _log_info(self, message: str) -> None:
        """Log an info message.
        
        Args:
            message: Info message
        """
        logger.info(f"[{self.__class__.__name__}] {message}")
    
    def _log_warning(self, message: str) -> None:
        """Log a warning message.
        
        Args:
            message: Warning message
        """
        logger.warning(f"[{self.__class__.__name__}] {message}")
    
    def _log_error(self, message: str) -> None:
        """Log an error message.
        
        Args:
            message: Error message
        """
        logger.error(f"[{self.__class__.__name__}] {message}")
    
    def clear_cache(self) -> None:
        """Clear all cached results.
        
        Useful for testing or when tools are installed/updated during runtime.
        """
        with self._cache_lock:
            self._cache.clear()
        self._log_debug("Cache cleared")

