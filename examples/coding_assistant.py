"""
Coding Assistant Example for MCP Agent Framework.

This example demonstrates how to use the MCP Agent Framework to create
a specialized coding assistant that can help with programming tasks.
"""

import asyncio
import os
import sys
import argparse
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcp_agent_framework import AgentFramework
from mcp_agent_framework.agent.factory import AgentSpec
from mcp_agent_framework.agent.agent import Agent

async def setup_coding_assistant(framework: AgentFramework) -> Dict[str, str]:
    """
    Set up a coding assistant with specialized agents.
    
    This function creates a hierarchy of agents specialized in different
    aspects of software development.
    
    Args:
        framework: Initialized AgentFramework instance
    
    Returns:
        Dict[str, str]: Dictionary mapping agent roles to agent IDs
    """
    # Initialize the framework
    await framework.initialize()
    
    # Main agent is already created during initialization
    main_agent_id = framework.hierarchy.main_agent.id
    
    # Create specialized agents
    developer_id = await framework.create_agent(
        "Developer", 
        "developer",
        parent_id=main_agent_id
    )
    
    architect_id = await framework.create_agent(
        "Architect", 
        "software_architect",
        parent_id=main_agent_id
    )
    
    tester_id = await framework.create_agent(
        "Tester", 
        "qa_engineer",
        parent_id=main_agent_id
    )
    
    documentation_id = await framework.create_agent(
        "Documentation", 
        "technical_writer",
        parent_id=main_agent_id
    )
    
    # Return map of roles to IDs
    return {
        "main": main_agent_id,
        "developer": developer_id,
        "architect": architect_id,
        "tester": tester_id,
        "documentation": documentation_id
    }

async def run_coding_task(framework: AgentFramework, task_description: str) -> None:
    """
    Run a coding task using the coding assistant.
    
    This function submits a coding task to the framework and displays
    the results.
    
    Args:
        framework: Initialized AgentFramework instance
        task_description: Description of the coding task
    """
    print(f"Submitting coding task: {task_description}")
    
    # Submit task to framework
    task_id = await framework.submit_task(task_description)
    
    print(f"Task submitted with ID: {task_id}")
    print("Waiting for completion...")
    
    # Wait for task to complete
    task_info = await framework.wait_for_task(task_id)
    
    if task_info and task_info["status"] == "COMPLETED":
        print("\nTask completed successfully!")
        print("\nResult:")
        print("=" * 80)
        print(task_info["result"])
        print("=" * 80)
    elif task_info and task_info["status"] == "FAILED":
        print(f"\nTask failed: {task_info['error']}")
    else:
        print("\nTask did not complete within the timeout period.")

async def main():
    """
    Main entry point for the coding assistant example.
    """
    parser = argparse.ArgumentParser(description="MCP Agent Framework Coding Assistant Example")
    parser.add_argument("--task", help="Coding task to submit")
    parser.add_argument("--api-key", help="API key for language models")
    args = parser.parse_args()
    
    # Get API key from arguments or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: No API key provided.")
        print("Please provide an API key using --api-key or set the OPENAI_API_KEY environment variable.")
        return 1
    
    # Create framework
    framework = AgentFramework(api_key=api_key)
    
    # Set up coding assistant
    agent_ids = await setup_coding_assistant(framework)
    
    print("Coding Assistant initialized with the following agents:")
    for role, agent_id in agent_ids.items():
        agent_info = await framework.get_agent(agent_id)
        if agent_info:
            print(f"- {agent_info['name']} ({agent_info['role']}): {agent_id[:8]}...")
    
    if args.task:
        # Run specific task
        await run_coding_task(framework, args.task)
    else:
        # Interactive mode
        print("\nInteractive Coding Assistant")
        print("Type 'exit' to quit")
        
        while True:
            task = input("\nEnter coding task: ")
            
            if task.lower() == "exit":
                break
            
            if task:
                await run_coding_task(framework, task)
    
    # Shutdown framework
    await framework.shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
