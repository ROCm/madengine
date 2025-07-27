#!/usr/bin/env python3
"""
Monitoring and Observability for MADEngine

This module provides structured logging, metrics collection, and 
observability features for better monitoring and debugging.
"""

import json
import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
from collections import defaultdict, deque

from madengine.core.errors import MADEngineError


class MonitoringError(MADEngineError):
    """Monitoring system specific errors."""
    pass


@dataclass
class MetricSample:
    """A single metric sample."""
    
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str]
    metric_type: str = "gauge"  # gauge, counter, histogram, summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class LogEvent:
    """Structured log event."""
    
    timestamp: str
    level: str
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    labels: Dict[str, str]
    extra_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class MetricsCollector:
    """
    Thread-safe metrics collector for gathering system and application metrics.
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_samples: Maximum number of samples to keep in memory
        """
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self._lock = threading.RLock()
        self._start_time = time.time()
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric (instantaneous value)."""
        self._record_metric(name, value, labels, "gauge")
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric (cumulative value)."""
        self._record_metric(name, value, labels, "counter")
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric (distribution of values)."""
        self._record_metric(name, value, labels, "histogram")
    
    def _record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]], metric_type: str):
        """Internal method to record a metric."""
        with self._lock:
            sample = MetricSample(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=metric_type
            )
            self._metrics[name].append(sample)
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List[MetricSample]]:
        """
        Get collected metrics.
        
        Args:
            name: Specific metric name to retrieve (optional)
            
        Returns:
            Dictionary of metric names to sample lists
        """
        with self._lock:
            if name:
                return {name: list(self._metrics.get(name, []))}
            return {k: list(v) for k, v in self._metrics.items()}
    
    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Summary statistics or None if metric not found
        """
        with self._lock:
            samples = self._metrics.get(name)
            if not samples:
                return None
            
            values = [s.value for s in samples]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else None,
                "latest_timestamp": samples[-1].timestamp if samples else None
            }
    
    def clear_metrics(self, name: Optional[str] = None):
        """Clear metrics from memory."""
        with self._lock:
            if name:
                self._metrics.pop(name, None)
            else:
                self._metrics.clear()


class StructuredLogger:
    """
    Enhanced logger with structured logging capabilities.
    """
    
    def __init__(self, name: str, console=None, enable_json: bool = False):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            console: Rich console for formatted output
            enable_json: Enable JSON formatted logging
        """
        self.logger = logging.getLogger(name)
        self.console = console
        self.enable_json = enable_json
        self._context_data: Dict[str, Any] = {}
        
        # Set up structured formatter if JSON enabled
        if enable_json:
            self._setup_json_formatter()
    
    def _setup_json_formatter(self):
        """Set up JSON formatter for structured logging."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def set_context(self, **kwargs):
        """Set context data that will be included in all log messages."""
        self._context_data.update(kwargs)
    
    def clear_context(self):
        """Clear context data."""
        self._context_data.clear()
    
    def log_structured(
        self, 
        level: str, 
        message: str, 
        labels: Optional[Dict[str, str]] = None,
        **extra_data
    ):
        """
        Log a structured message.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            labels: Labels for categorization
            **extra_data: Additional structured data
        """
        # Merge context data
        merged_data = {**self._context_data, **extra_data}
        
        if self.enable_json:
            log_event = LogEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                level=level,
                message=message,
                logger_name=self.logger.name,
                module=__name__,
                function="log_structured",
                line_number=0,  # Would need inspection to get real line number
                thread_id=threading.get_ident(),
                process_id=0,  # Would need os.getpid()
                labels=labels or {},
                extra_data=merged_data
            )
            
            # Log as JSON
            json_message = json.dumps(log_event.to_dict())
            getattr(self.logger, level.lower())(json_message)
        else:
            # Standard logging with context
            context_str = ""
            if labels or merged_data:
                context_parts = []
                if labels:
                    context_parts.append(f"labels={labels}")
                if merged_data:
                    context_parts.append(f"data={merged_data}")
                context_str = f" [{', '.join(context_parts)}]"
            
            full_message = f"{message}{context_str}"
            getattr(self.logger, level.lower())(full_message)
        
        # Also log to console if available
        if self.console:
            style_map = {
                "DEBUG": "dim",
                "INFO": "blue", 
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold red"
            }
            style = style_map.get(level, "white")
            self.console.print(f"[{level}] {message}", style=style)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log_structured("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log_structured("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log_structured("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log_structured("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log_structured("CRITICAL", message, **kwargs)


class PerformanceTracker:
    """
    Performance tracking utility for measuring execution times and resource usage.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, logger: StructuredLogger):
        """
        Initialize performance tracker.
        
        Args:
            metrics_collector: Metrics collector instance
            logger: Structured logger instance
        """
        self.metrics = metrics_collector
        self.logger = logger
        self._active_timers: Dict[str, float] = {}
    
    @contextmanager
    def time_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            labels: Additional labels for the metric
        """
        start_time = time.time()
        
        try:
            self.logger.debug(f"Starting operation: {operation_name}", operation=operation_name)
            yield
        finally:
            duration = time.time() - start_time
            
            # Record timing metric
            timing_labels = {"operation": operation_name}
            if labels:
                timing_labels.update(labels)
            
            self.metrics.record_histogram(
                "operation_duration_seconds",
                duration,
                timing_labels
            )
            
            self.logger.info(
                f"Completed operation: {operation_name}",
                operation=operation_name,
                duration_seconds=duration
            )
    
    def start_timer(self, timer_name: str):
        """Start a named timer."""
        self._active_timers[timer_name] = time.time()
        self.logger.debug(f"Started timer: {timer_name}", timer=timer_name)
    
    def stop_timer(self, timer_name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Stop a named timer and record the duration.
        
        Args:
            timer_name: Name of the timer
            labels: Additional labels for the metric
            
        Returns:
            Duration in seconds
        """
        if timer_name not in self._active_timers:
            self.logger.warning(f"Timer {timer_name} not found")
            return 0.0
        
        duration = time.time() - self._active_timers.pop(timer_name)
        
        # Record timing metric
        timing_labels = {"timer": timer_name}
        if labels:
            timing_labels.update(labels)
        
        self.metrics.record_histogram(
            "timer_duration_seconds",
            duration,
            timing_labels
        )
        
        self.logger.debug(
            f"Stopped timer: {timer_name}",
            timer=timer_name,
            duration_seconds=duration
        )
        
        return duration


class MonitoringManager:
    """
    Central monitoring manager that coordinates logging, metrics, and performance tracking.
    """
    
    def __init__(self, console=None, enable_json_logging: bool = False):
        """
        Initialize monitoring manager.
        
        Args:
            console: Rich console for output
            enable_json_logging: Enable JSON structured logging
        """
        self.console = console
        self.metrics_collector = MetricsCollector()
        self.logger = StructuredLogger("madengine", console, enable_json_logging)
        self.performance_tracker = PerformanceTracker(self.metrics_collector, self.logger)
        
        # System-level metrics tracking
        self._system_metrics_enabled = False
        self._system_metrics_thread: Optional[threading.Thread] = None
        self._stop_system_metrics = threading.Event()
    
    def enable_system_metrics(self, interval: float = 30.0):
        """
        Enable automatic system metrics collection.
        
        Args:
            interval: Collection interval in seconds
        """
        if self._system_metrics_enabled:
            return
        
        self._system_metrics_enabled = True
        self._stop_system_metrics.clear()
        
        def collect_system_metrics():
            while not self._stop_system_metrics.wait(interval):
                try:
                    # Collect basic system metrics
                    import psutil
                    
                    # CPU usage
                    self.metrics_collector.record_gauge(
                        "system_cpu_percent",
                        psutil.cpu_percent()
                    )
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.metrics_collector.record_gauge(
                        "system_memory_percent",
                        memory.percent
                    )
                    self.metrics_collector.record_gauge(
                        "system_memory_used_bytes",
                        memory.used
                    )
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.metrics_collector.record_gauge(
                        "system_disk_percent",
                        (disk.used / disk.total) * 100
                    )
                    
                except ImportError:
                    # psutil not available
                    break
                except Exception as e:
                    self.logger.warning(f"System metrics collection failed: {e}")
        
        self._system_metrics_thread = threading.Thread(
            target=collect_system_metrics,
            daemon=True
        )
        self._system_metrics_thread.start()
        
        self.logger.info("System metrics collection enabled", interval=interval)
    
    def disable_system_metrics(self):
        """Disable automatic system metrics collection."""
        if not self._system_metrics_enabled:
            return
        
        self._stop_system_metrics.set()
        if self._system_metrics_thread:
            self._system_metrics_thread.join(timeout=5.0)
        
        self._system_metrics_enabled = False
        self.logger.info("System metrics collection disabled")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Health status dictionary
        """
        uptime = time.time() - self.metrics_collector._start_time
        
        health = {
            "status": "healthy",
            "uptime_seconds": uptime,
            "metrics_collected": len(self.metrics_collector._metrics),
            "system_metrics_enabled": self._system_metrics_enabled,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Check for any error conditions
        error_metrics = self.metrics_collector.get_metrics("error_count")
        if error_metrics and error_metrics.get("error_count"):
            recent_errors = sum(
                1 for sample in error_metrics["error_count"]
                if time.time() - sample.timestamp < 300  # Last 5 minutes
            )
            if recent_errors > 10:
                health["status"] = "degraded"
                health["recent_errors"] = recent_errors
        
        return health
    
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """
        Export metrics in specified format.
        
        Args:
            format_type: Export format (prometheus, json)
            
        Returns:
            Formatted metrics string
        """
        if format_type == "json":
            metrics_data = {}
            for name, samples in self.metrics_collector.get_metrics().items():
                metrics_data[name] = [s.to_dict() for s in samples]
            return json.dumps(metrics_data, indent=2)
        
        elif format_type == "prometheus":
            lines = []
            for name, samples in self.metrics_collector.get_metrics().items():
                if not samples:
                    continue
                
                # Add metric help and type
                lines.append(f"# HELP {name} MADEngine metric")
                lines.append(f"# TYPE {name} {samples[0].metric_type}")
                
                # Add samples
                for sample in samples[-10:]:  # Last 10 samples
                    labels_str = ""
                    if sample.labels:
                        label_pairs = [f'{k}="{v}"' for k, v in sample.labels.items()]
                        labels_str = "{" + ",".join(label_pairs) + "}"
                    
                    lines.append(f"{name}{labels_str} {sample.value} {int(sample.timestamp * 1000)}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def shutdown(self):
        """Shutdown monitoring manager and clean up resources."""
        self.disable_system_metrics()
        self.logger.info("Monitoring manager shutdown complete")


# Global monitoring instance
_monitoring_manager: Optional[MonitoringManager] = None


def get_monitoring_manager() -> MonitoringManager:
    """Get or create global monitoring manager instance."""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    return _monitoring_manager


def init_monitoring(console=None, enable_json_logging: bool = False, enable_system_metrics: bool = True):
    """
    Initialize global monitoring system.
    
    Args:
        console: Rich console for output
        enable_json_logging: Enable JSON structured logging
        enable_system_metrics: Enable automatic system metrics collection
    """
    global _monitoring_manager
    _monitoring_manager = MonitoringManager(console, enable_json_logging)
    
    if enable_system_metrics:
        _monitoring_manager.enable_system_metrics()
    
    return _monitoring_manager