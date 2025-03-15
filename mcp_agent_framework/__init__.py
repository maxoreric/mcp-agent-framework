"""
MCP Agent Framework

A Python framework for creating, orchestrating, and managing networks 
of AI agents using the Model Context Protocol (MCP).
"""

__version__ = "0.1.0"

from .agent.agent import Agent
from .agent.hierarchy import AgentHierarchy
from .framework import AgentFramework

__all__ = ["Agent", "AgentHierarchy", "AgentFramework"]
