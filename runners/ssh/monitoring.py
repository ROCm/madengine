#!/usr/bin/env python3
"""Monitoring and logging utilities for SSH Multi-Node Runner

This module provides enhanced monitoring and logging capabilities.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class NodeExecutionResult:
    """Result of execution on a single node."""
    hostname: str
    node_rank: int
    success: bool
    start_time: float
    end_time: float
    output: str
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def duration_formatted(self) -> str:
        """Get formatted duration string."""
        from .utils import format_duration
        return format_duration(self.duration)


@dataclass
class TrainingSession:
    """Complete training session information."""
    session_id: str
    model: str
    nodes: List[str]
    master_addr: str
    master_port: int
    start_time: float
    end_time: Optional[float] = None
    results: List[NodeExecutionResult] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []
    
    @property
    def duration(self) -> Optional[float]:
        """Get total session duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.success)
        return (successful / len(self.results)) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['duration'] = self.duration
        data['success_rate'] = self.success_rate
        return data


class SessionLogger:
    """Logger for training sessions with structured output."""
    
    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None):
        """Initialize session logger.
        
        Args:
            log_dir: Directory to store log files
            session_id: Unique session identifier (auto-generated if None)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.session_file = self.log_dir / f"{session_id}.json"
        self.log_file = self.log_dir / f"{session_id}.log"
        
        # Setup file logger
        self.logger = logging.getLogger(f"session.{session_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler
        file_handler = logging.FileHandler(str(self.log_file))
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        self.session: Optional[TrainingSession] = None
    
    def start_session(self, model: str, nodes: List[str], master_addr: str, master_port: int) -> None:
        """Start a new training session.
        
        Args:
            model: Model being trained
            nodes: List of node hostnames
            master_addr: Master node address
            master_port: Master node port
        """
        self.session = TrainingSession(
            session_id=self.session_id,
            model=model,
            nodes=nodes.copy(),
            master_addr=master_addr,
            master_port=master_port,
            start_time=time.time()
        )
        
        self.logger.info(f"Starting training session {self.session_id}")
        self.logger.info(f"Model: {model}")
        self.logger.info(f"Nodes: {', '.join(nodes)}")
        self.logger.info(f"Master: {master_addr}:{master_port}")
        
        self._save_session()
    
    def log_node_start(self, hostname: str, node_rank: int, command: str) -> None:
        """Log the start of execution on a node.
        
        Args:
            hostname: Node hostname
            node_rank: Node rank
            command: Command being executed
        """
        self.logger.info(f"Node {hostname} (rank {node_rank}) starting: {command}")
    
    def log_node_output(self, hostname: str, node_rank: int, line: str, is_error: bool = False) -> None:
        """Log output from a node.
        
        Args:
            hostname: Node hostname
            node_rank: Node rank
            line: Output line
            is_error: Whether this is error output
        """
        level = logging.ERROR if is_error else logging.INFO
        prefix = "ERROR" if is_error else "OUTPUT"
        self.logger.log(level, f"[{hostname}:{node_rank}] {prefix}: {line}")
    
    def log_node_result(self, result: NodeExecutionResult) -> None:
        """Log the result of execution on a node.
        
        Args:
            result: Node execution result
        """
        if self.session is None:
            raise RuntimeError("No active session")
        
        self.session.results.append(result)
        
        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(
            f"Node {result.hostname} (rank {result.node_rank}) {status} "
            f"in {result.duration_formatted}"
        )
        
        if not result.success and result.error_message:
            self.logger.error(f"Node {result.hostname} error: {result.error_message}")
        
        self._save_session()
    
    def end_session(self, success: bool) -> None:
        """End the training session.
        
        Args:
            success: Whether the overall session was successful
        """
        if self.session is None:
            raise RuntimeError("No active session")
        
        self.session.end_time = time.time()
        
        status = "SUCCESS" if success else "FAILED"
        duration = self.session.duration
        from .utils import format_duration
        duration_str = format_duration(duration) if duration else "unknown"
        
        self.logger.info(f"Training session {status} in {duration_str}")
        self.logger.info(f"Success rate: {self.session.success_rate:.1f}%")
        
        self._save_session()
    
    def _save_session(self) -> None:
        """Save session data to JSON file."""
        if self.session is None:
            return
        
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.session.to_dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save session data: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary.
        
        Returns:
            Dictionary containing session summary
        """
        if self.session is None:
            return {}
        
        return {
            'session_id': self.session.session_id,
            'model': self.session.model,
            'total_nodes': len(self.session.nodes),
            'completed_nodes': len(self.session.results),
            'successful_nodes': sum(1 for r in self.session.results if r.success),
            'failed_nodes': sum(1 for r in self.session.results if not r.success),
            'success_rate': self.session.success_rate,
            'duration': self.session.duration,
            'status': 'completed' if self.session.end_time else 'running'
        }


class ProgressMonitor:
    """Monitor training progress across nodes."""
    
    def __init__(self, total_nodes: int):
        """Initialize progress monitor.
        
        Args:
            total_nodes: Total number of nodes
        """
        self.total_nodes = total_nodes
        self.completed_nodes = 0
        self.successful_nodes = 0
        self.failed_nodes = 0
        self.start_time = time.time()
    
    def update(self, success: bool) -> None:
        """Update progress with a completed node.
        
        Args:
            success: Whether the node completed successfully
        """
        self.completed_nodes += 1
        if success:
            self.successful_nodes += 1
        else:
            self.failed_nodes += 1
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information.
        
        Returns:
            Dictionary containing progress information
        """
        elapsed_time = time.time() - self.start_time
        completion_rate = self.completed_nodes / self.total_nodes if self.total_nodes > 0 else 0
        
        # Estimate remaining time
        if completion_rate > 0:
            estimated_total_time = elapsed_time / completion_rate
            estimated_remaining_time = estimated_total_time - elapsed_time
        else:
            estimated_remaining_time = None
        
        return {
            'total_nodes': self.total_nodes,
            'completed_nodes': self.completed_nodes,
            'successful_nodes': self.successful_nodes,
            'failed_nodes': self.failed_nodes,
            'completion_rate': completion_rate,
            'elapsed_time': elapsed_time,
            'estimated_remaining_time': estimated_remaining_time,
            'success_rate': self.successful_nodes / self.completed_nodes if self.completed_nodes > 0 else 0
        }
    
    def print_progress(self) -> None:
        """Print current progress to console."""
        progress = self.get_progress()
        
        from .utils import format_duration
        elapsed = format_duration(progress['elapsed_time'])
        
        if progress['estimated_remaining_time']:
            remaining = format_duration(progress['estimated_remaining_time'])
            time_info = f"Elapsed: {elapsed}, Remaining: ~{remaining}"
        else:
            time_info = f"Elapsed: {elapsed}"
        
        print(f"Progress: {progress['completed_nodes']}/{progress['total_nodes']} "
              f"({progress['completion_rate']*100:.1f}%) - "
              f"Success: {progress['successful_nodes']}, Failed: {progress['failed_nodes']} - "
              f"{time_info}")


def load_session_history(log_dir: str = "logs") -> List[Dict[str, Any]]:
    """Load session history from log directory.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        List of session summaries
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return []
    
    sessions = []
    for json_file in log_path.glob("session_*.json"):
        try:
            with open(json_file, 'r') as f:
                session_data = json.load(f)
                sessions.append(session_data)
        except Exception:
            # Skip corrupted files
            continue
    
    # Sort by start time (most recent first)
    sessions.sort(key=lambda x: x.get('start_time', 0), reverse=True)
    return sessions
