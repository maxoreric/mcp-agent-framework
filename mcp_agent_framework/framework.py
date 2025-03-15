"""
Main Framework Class for MCP Agent Framework.

This module provides the main entry point for the MCP Agent Framework,
combining all the components into a unified system.
"""

import asyncio
import os
import json
from typing import Dict, Any, Optional, List, Union, Callable

from mcp.server.fastmcp import Context

from .agent.hierarchy import AgentHierarchy
from .agent.factory import AgentFactory, AgentSpec
from .cli.interface import CommandLineInterface
from .task.orchestrator import orchestrate_task

class AgentFramework:
    """
    Main entry point for the MCP Agent Framework.
    
    This class provides a high-level interface for using the agent
    framework, including initialization, task submission, and CLI
    interaction.
    
    Attributes:
        config: Configuration dictionary
        hierarchy: AgentHierarchy instance
        factory: AgentFactory instance
    """
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize a new AgentFramework instance.
        
        Args:
            api_key: API key for language model access (optional)
            config_path: Path to a JSON configuration file (optional)
        """
        # Load configuration
        self.config = self._load_config(api_key, config_path)
        
        # Initialize agent hierarchy
        self.hierarchy = AgentHierarchy(self.config)
        
        # Initialize agent factory
        self.factory = AgentFactory(self.hierarchy, self.config)
    
    def _load_config(self, api_key: Optional[str], config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load and merge configuration from multiple sources.
        
        This method creates a configuration dictionary by combining
        defaults, config file, and explicit parameters.
        
        Args:
            api_key: API key for language model access
            config_path: Path to a JSON configuration file
        
        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        # Default configuration
        config = {
            "api_key": None,
            "model": "gpt-4",
            "max_retry_attempts": 3,
            "base_retry_delay": 1.0,
            "llm_provider": "deepseek",  # Default to deepseek
            "anthropic_api_key": None,
            "anthropic_model": "claude-3-opus-20240229",
            "deepseek_api_key": None,
            "deepseek_model": "deepseek-chat",
            "open_xml_tag": "<answer>",
            "close_xml_tag": "</answer>",
        }
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config from {config_path}: {str(e)}")
        
        # Override with explicit api_key if provided
        if api_key:
            config["api_key"] = api_key
        
        # Load from environment variables if not provided explicitly
        if not config["api_key"]:
            config["api_key"] = os.environ.get("OPENAI_API_KEY")
        
        if not config["anthropic_api_key"]:
            config["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY")
            
        if not config["deepseek_api_key"]:
            config["deepseek_api_key"] = os.environ.get("DEEPSEEK_API_KEY")
        
        return config
    
    async def initialize(self) -> None:
        """
        Initialize the agent framework.
        
        This method sets up the agent hierarchy and initializes
        the main agent.
        """
        await self.hierarchy.initialize()
    
    async def submit_task(self, description: str) -> str:
        """
        Submit a task to the agent framework.
        
        This method submits a task to the main agent and orchestrates
        its execution.
        
        Args:
            description: Description of the task to submit
        
        Returns:
            str: ID of the submitted task
        """
        # Ensure the framework is initialized
        if not self.hierarchy.main_agent:
            await self.initialize()
        
        # Create context
        ctx = Context()
        
        # Submit task to main agent
        task_id = await orchestrate_task(self.hierarchy, description, ctx)
        
        return task_id
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a submitted task.
        
        This method retrieves the current status and result of a task.
        
        Args:
            task_id: ID of the task to retrieve
        
        Returns:
            Optional[Dict[str, Any]]: Task information including status and result,
                or None if the task is not found
        """
        # Check if the main agent exists
        if not self.hierarchy.main_agent:
            return None
        
        # Check if the task exists in the main agent
        if task_id in self.hierarchy.main_agent.tasks:
            task = self.hierarchy.main_agent.tasks[task_id]
            return {
                "id": task_id,
                "description": task.get("description", "Unknown task"),
                "status": task.get("status", "UNKNOWN"),
                "result": task.get("result", None),
                "error": task.get("error", None),
            }
        
        # Check other agents for the task
        for agent_id, agent in self.hierarchy.agent_registry.items():
            if task_id in agent.tasks:
                task = agent.tasks[task_id]
                return {
                    "id": task_id,
                    "description": task.get("description", "Unknown task"),
                    "status": task.get("status", "UNKNOWN"),
                    "result": task.get("result", None),
                    "error": task.get("error", None),
                    "agent_id": agent_id,
                    "agent_name": agent.name,
                    "agent_role": agent.role,
                }
        
        return None
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Wait for a task to complete.
        
        This method blocks until the task is completed or fails,
        or until the timeout is reached.
        
        Args:
            task_id: ID of the task to wait for
            timeout: Maximum time to wait in seconds (None for no timeout)
        
        Returns:
            Optional[Dict[str, Any]]: Completed task information, or None if
                the task was not found or did not complete within the timeout
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check if the task exists and is completed or failed
            task_info = await self.get_task_result(task_id)
            
            if not task_info:
                # Task not found
                return None
            
            if task_info["status"] in ["COMPLETED", "FAILED"]:
                # Task is completed or failed
                return task_info
            
            # Check timeout
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    # Timeout reached
                    return None
            
            # Wait a bit before checking again
            await asyncio.sleep(0.5)
    
    def start_cli(self) -> None:
        """
        Start the command-line interface.
        
        This method launches the interactive CLI for the agent framework.
        """
        cli = CommandLineInterface(self.config)
        
        # Run the CLI in the event loop
        asyncio.run(cli.start())
    
    async def shutdown(self) -> None:
        """
        Shut down the agent framework.
        
        This method gracefully shuts down the agent hierarchy and
        cleans up resources.
        """
        await self.hierarchy.shutdown()
    
    async def create_agent(self, name: str, role: str, parent_id: Optional[str] = None) -> str:
        """
        Create a new agent in the hierarchy.
        
        This method creates a new agent with the specified name and role,
        optionally connecting it to a parent agent.
        
        Args:
            name: Name for the new agent
            role: Role for the new agent
            parent_id: Optional ID of the parent agent
        
        Returns:
            str: ID of the newly created agent
        """
        # Create agent specification
        spec = AgentSpec(name, role)
        
        # Create the agent
        agent = await self.factory.create_agent(spec, parent_id)
        
        return agent.id
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an agent.
        
        This method retrieves information about the specified agent.
        
        Args:
            agent_id: ID of the agent to retrieve
        
        Returns:
            Optional[Dict[str, Any]]: Agent information, or None if
                the agent is not found
        """
        agent = await self.hierarchy.get_agent(agent_id)
        
        if not agent:
            return None
        
        return {
            "id": agent.id,
            "name": agent.name,
            "role": agent.role,
            "tasks": list(agent.tasks.keys()),
            "child_agents": list(agent.child_agents.keys()),
        }
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """
        Destroy an agent in the hierarchy.
        
        This method destroys the specified agent, removing it from
        the hierarchy.
        
        Args:
            agent_id: ID of the agent to destroy
        
        Returns:
            bool: True if the agent was successfully destroyed,
                False otherwise
        """
        return await self.hierarchy.destroy_agent(agent_id)

# Convenience function for using the framework from the command line
def run_cli():
    """
    Run the MCP Agent Framework CLI.
    
    This function creates an AgentFramework instance and starts the CLI.
    """
    # Create framework instance
    framework = AgentFramework()
    
    # Start CLI
    framework.start_cli()

if __name__ == "__main__":
    run_cli()
