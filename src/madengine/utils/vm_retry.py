#!/usr/bin/env python3
"""
Retry logic and error handling for VM operations.

Provides decorators and utilities for retrying failed VM operations
with exponential backoff and intelligent error handling.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import time
import functools
from typing import Callable, Type, Tuple, Optional
from enum import Enum


class VMError(Exception):
    """Base exception for VM-related errors."""
    pass


class VMCreationError(VMError):
    """VM creation failed."""
    pass


class VMStartError(VMError):
    """VM start/boot failed."""
    pass


class VMNetworkError(VMError):
    """VM network/SSH connection failed."""
    pass


class VMExecutionError(VMError):
    """Command execution inside VM failed."""
    pass


class VMCleanupError(VMError):
    """VM cleanup/destruction failed."""
    pass


class RetryStrategy(Enum):
    """Retry strategies for different failure types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


def retry_on_failure(
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        strategy: Retry strategy to use
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function called before each retry
        
    Example:
        @retry_on_failure(max_attempts=3, exceptions=(VMNetworkError,))
        def connect_to_vm(vm_ip):
            # ... connection logic ...
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        # Last attempt failed, re-raise
                        raise
                    
                    # Calculate delay based on strategy
                    if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    elif strategy == RetryStrategy.LINEAR_BACKOFF:
                        delay = min(base_delay * attempt, max_delay)
                    elif strategy == RetryStrategy.IMMEDIATE:
                        delay = 0
                    else:
                        # NO_RETRY - shouldn't reach here
                        raise
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt, max_attempts, delay, e)
                    
                    # Wait before retry
                    if delay > 0:
                        time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def retry_vm_operation(operation_name: str, max_attempts: int = 3):
    """
    Specialized retry decorator for VM operations with logging.
    
    Args:
        operation_name: Name of the operation for logging
        max_attempts: Maximum number of attempts
        
    Example:
        @retry_vm_operation("VM_START", max_attempts=3)
        def start_vm(vm):
            # ... start logic ...
            pass
    """
    def on_retry_callback(attempt, max_attempts, delay, exception):
        print(f"  ⚠ {operation_name} failed (attempt {attempt}/{max_attempts}): {exception}")
        if delay > 0:
            print(f"  Retrying in {delay:.1f}s...")
    
    return retry_on_failure(
        max_attempts=max_attempts,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=2.0,
        max_delay=30.0,
        exceptions=(VMError, OSError, TimeoutError),
        on_retry=on_retry_callback
    )


class VMOperationContext:
    """
    Context manager for VM operations with automatic cleanup on failure.
    
    Example:
        with VMOperationContext("Create VM") as ctx:
            vm = create_vm()
            ctx.set_resource(vm)
            ctx.set_cleanup(lambda: destroy_vm(vm))
            # If exception occurs, cleanup is automatically called
    """
    
    def __init__(self, operation_name: str):
        """
        Initialize context manager.
        
        Args:
            operation_name: Name of operation for logging
        """
        self.operation_name = operation_name
        self.resource = None
        self.cleanup_func = None
        self.success = False
    
    def set_resource(self, resource):
        """Set the resource being managed."""
        self.resource = resource
    
    def set_cleanup(self, cleanup_func: Callable):
        """Set the cleanup function to call on failure."""
        self.cleanup_func = cleanup_func
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with cleanup on failure."""
        if exc_type is not None and self.cleanup_func:
            # Exception occurred, run cleanup
            print(f"  ⚠ {self.operation_name} failed, running cleanup...")
            try:
                self.cleanup_func()
                print(f"  ✓ Cleanup completed")
            except Exception as cleanup_error:
                print(f"  ✗ Cleanup failed: {cleanup_error}")
        
        # Don't suppress the exception
        return False


def safe_cleanup(cleanup_func: Callable, resource_name: str = "resource"):
    """
    Safely execute cleanup function, catching and logging exceptions.
    
    Args:
        cleanup_func: Function to execute for cleanup
        resource_name: Name of resource for logging
        
    Returns:
        True if cleanup succeeded, False otherwise
    """
    try:
        cleanup_func()
        return True
    except Exception as e:
        print(f"  ⚠ Failed to cleanup {resource_name}: {e}")
        return False


class ErrorRecoveryManager:
    """
    Manages error recovery strategies for VM operations.
    
    Tracks failures and provides recovery recommendations.
    """
    
    def __init__(self):
        """Initialize error recovery manager."""
        self.failure_counts = {}
        self.error_history = []
    
    def record_failure(self, operation: str, error: Exception):
        """
        Record a failure for tracking.
        
        Args:
            operation: Name of operation that failed
            error: Exception that was raised
        """
        self.failure_counts[operation] = self.failure_counts.get(operation, 0) + 1
        self.error_history.append({
            "operation": operation,
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": time.time()
        })
    
    def should_abort(self, operation: str, threshold: int = 5) -> bool:
        """
        Check if operation should be aborted due to repeated failures.
        
        Args:
            operation: Name of operation
            threshold: Number of failures before aborting
            
        Returns:
            True if should abort
        """
        return self.failure_counts.get(operation, 0) >= threshold
    
    def get_recovery_suggestion(self, operation: str) -> str:
        """
        Get recovery suggestion based on failure history.
        
        Args:
            operation: Name of operation
            
        Returns:
            Recovery suggestion string
        """
        count = self.failure_counts.get(operation, 0)
        
        if count == 0:
            return "No failures recorded"
        elif count == 1:
            return "First failure - retry recommended"
        elif count <= 3:
            return f"Multiple failures ({count}) - check system resources"
        else:
            return f"Repeated failures ({count}) - manual intervention required"
    
    def reset(self):
        """Reset failure tracking."""
        self.failure_counts.clear()
        self.error_history.clear()


# Global error recovery manager instance
_global_recovery_manager = ErrorRecoveryManager()


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager instance."""
    return _global_recovery_manager
