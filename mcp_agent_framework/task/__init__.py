"""
Task module for MCP Agent Framework.

This module contains task orchestration, dependency management,
and execution components.
"""

from .orchestrator import orchestrate_task
from .dependency import create_dependency_tree
from .execution import execute_task

__all__ = ["orchestrate_task", "create_dependency_tree", "execute_task"]
