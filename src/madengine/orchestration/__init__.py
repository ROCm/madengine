"""
Orchestration layer for madengine workflows.

Provides high-level workflow coordination for build and run phases.
This layer sits between the CLI (presentation) and execution/deployment layers.

Architecture:
- BuildOrchestrator: Manages Docker image building workflow
- RunOrchestrator: Manages model execution workflow (local or distributed)
"""

from .build_orchestrator import BuildOrchestrator
from .run_orchestrator import RunOrchestrator

__all__ = ["BuildOrchestrator", "RunOrchestrator"]

