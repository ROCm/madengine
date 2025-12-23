#!/usr/bin/env python3
"""
Unified Error Handling System for madengine

This module provides a centralized error handling system with structured
error types and consistent Rich console-based error reporting.
"""

import logging
import traceback
from dataclasses import dataclass
from typing import Optional, Any, Dict, List
from enum import Enum

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
except ImportError:
    raise ImportError("Rich is required for error handling. Install with: pip install rich")


class ErrorCategory(Enum):
    """Error category enumeration for classification."""
    
    VALIDATION = "validation"
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    RUNTIME = "runtime"
    BUILD = "build"
    DISCOVERY = "discovery"
    ORCHESTRATION = "orchestration"
    RUNNER = "runner"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"


@dataclass
class ErrorContext:
    """Context information for errors."""
    
    operation: str
    phase: Optional[str] = None
    component: Optional[str] = None
    model_name: Optional[str] = None
    node_id: Optional[str] = None
    file_path: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class MADEngineError(Exception):
    """Base exception for all madengine errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = False,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.context = context or ErrorContext(operation="unknown")
        self.cause = cause
        self.recoverable = recoverable
        self.suggestions = suggestions or []


class ValidationError(MADEngineError):
    """Validation and input errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.VALIDATION, 
            context, 
            recoverable=True,
            **kwargs
        )


class ConnectionError(MADEngineError):
    """Connection and network errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.CONNECTION, 
            context, 
            recoverable=True,
            **kwargs
        )


class AuthenticationError(MADEngineError):
    """Authentication and credential errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.AUTHENTICATION, 
            context, 
            recoverable=True,
            **kwargs
        )


class RuntimeError(MADEngineError):
    """Runtime execution errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.RUNTIME, 
            context, 
            recoverable=False,
            **kwargs
        )


class BuildError(MADEngineError):
    """Build and compilation errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.BUILD, 
            context, 
            recoverable=False,
            **kwargs
        )


class DiscoveryError(MADEngineError):
    """Model discovery errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.DISCOVERY, 
            context, 
            recoverable=True,
            **kwargs
        )


class OrchestrationError(MADEngineError):
    """Distributed orchestration errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.ORCHESTRATION, 
            context, 
            recoverable=False,
            **kwargs
        )


class RunnerError(MADEngineError):
    """Distributed runner errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.RUNNER, 
            context, 
            recoverable=True,
            **kwargs
        )


class ConfigurationError(MADEngineError):
    """Configuration and setup errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.CONFIGURATION, 
            context, 
            recoverable=True,
            **kwargs
        )


class TimeoutError(MADEngineError):
    """Timeout and duration errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.TIMEOUT, 
            context, 
            recoverable=True,
            **kwargs
        )


class ErrorHandler:
    """Unified error handler with Rich console integration."""
    
    def __init__(self, console: Optional[Console] = None, verbose: bool = False):
        self.console = console or Console()
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
    
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[ErrorContext] = None,
        show_traceback: Optional[bool] = None
    ) -> None:
        """Handle and display errors with rich formatting."""
        
        show_tb = show_traceback if show_traceback is not None else self.verbose
        
        if isinstance(error, MADEngineError):
            self._handle_madengine_error(error, show_tb)
        else:
            self._handle_generic_error(error, context, show_tb)
    
    def _handle_madengine_error(self, error: MADEngineError, show_traceback: bool) -> None:
        """Handle madengine structured errors."""
        
        # Determine error emoji and color
        category_info = {
            ErrorCategory.VALIDATION: ("⚠️", "yellow"),
            ErrorCategory.CONNECTION: ("🔌", "blue"),
            ErrorCategory.AUTHENTICATION: ("🔒", "red"),
            ErrorCategory.RUNTIME: ("💥", "red"),
            ErrorCategory.BUILD: ("🔨", "red"),
            ErrorCategory.DISCOVERY: ("🔍", "yellow"),
            ErrorCategory.ORCHESTRATION: ("⚡", "red"),
            ErrorCategory.RUNNER: ("🚀", "red"),
            ErrorCategory.CONFIGURATION: ("⚙️", "yellow"),
            ErrorCategory.TIMEOUT: ("⏱️", "yellow"),
        }
        
        emoji, color = category_info.get(error.category, ("❌", "red"))
        
        # Create error panel
        title = f"{emoji} {error.category.value.title()} Error"
        
        # Build error content
        content = Text()
        content.append(f"{error.message}\n", style=f"bold {color}")
        
        # Add context information
        if error.context:
            content.append("\n📋 Context:\n", style="bold cyan")
            if error.context.operation:
                content.append(f"  Operation: {error.context.operation}\n")
            if error.context.phase:
                content.append(f"  Phase: {error.context.phase}\n")
            if error.context.component:
                content.append(f"  Component: {error.context.component}\n")
            if error.context.model_name:
                content.append(f"  Model: {error.context.model_name}\n")
            if error.context.node_id:
                content.append(f"  Node: {error.context.node_id}\n")
            if error.context.file_path:
                content.append(f"  File: {error.context.file_path}\n")
        
        # Add cause information
        if error.cause:
            content.append(f"\n🔗 Caused by: {str(error.cause)}\n", style="dim")
        
        # Add suggestions
        if error.suggestions:
            content.append("\n💡 Suggestions:\n", style="bold green")
            for suggestion in error.suggestions:
                content.append(f"  • {suggestion}\n", style="green")
        
        # Add recovery information
        if error.recoverable:
            content.append("\n♻️  This error may be recoverable", style="bold blue")
        
        panel = Panel(
            content,
            title=title,
            border_style=color,
            expand=False
        )
        
        self.console.print(panel)
        
        # Show traceback if requested
        if show_traceback and error.cause:
            self.console.print("\n📚 [bold]Full Traceback:[/bold]")
            self.console.print_exception()
        
        # Log to file
        self.logger.error(
            f"{error.category.value}: {error.message}",
            extra={
                "context": error.context.__dict__ if error.context else {},
                "recoverable": error.recoverable,
                "suggestions": error.suggestions
            }
        )
    
    def _handle_generic_error(
        self, 
        error: Exception, 
        context: Optional[ErrorContext], 
        show_traceback: bool
    ) -> None:
        """Handle generic Python exceptions."""
        
        title = f"❌ {type(error).__name__}"
        
        content = Text()
        content.append(f"{str(error)}\n", style="bold red")
        
        if context:
            content.append("\n📋 Context:\n", style="bold cyan")
            content.append(f"  Operation: {context.operation}\n")
            if context.phase:
                content.append(f"  Phase: {context.phase}\n")
            if context.component:
                content.append(f"  Component: {context.component}\n")
        
        panel = Panel(
            content,
            title=title,
            border_style="red",
            expand=False
        )
        
        self.console.print(panel)
        
        if show_traceback:
            self.console.print("\n📚 [bold]Full Traceback:[/bold]")
            self.console.print_exception()
        
        # Log to file
        self.logger.error(f"{type(error).__name__}: {str(error)}")


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def set_error_handler(handler: ErrorHandler) -> None:
    """Set the global error handler."""
    global _global_error_handler
    _global_error_handler = handler


def get_error_handler() -> Optional[ErrorHandler]:
    """Get the global error handler."""
    return _global_error_handler


def handle_error(
    error: Exception, 
    context: Optional[ErrorContext] = None,
    show_traceback: Optional[bool] = None
) -> None:
    """Handle error using the global error handler."""
    if _global_error_handler:
        _global_error_handler.handle_error(error, context, show_traceback)
    else:
        # Fallback to basic logging
        logging.error(f"Error: {error}")
        if show_traceback:
            logging.exception("Exception details:")


def create_error_context(
    operation: str,
    phase: Optional[str] = None,
    component: Optional[str] = None,
    **kwargs
) -> ErrorContext:
    """Convenience function to create error context."""
    return ErrorContext(
        operation=operation,
        phase=phase,
        component=component,
        **kwargs
    )