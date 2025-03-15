"""
Research Assistant Example for MCP Agent Framework.

This example demonstrates how to use the MCP Agent Framework to create
a specialized research assistant that can help with research tasks.
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

async def setup_research_assistant(framework: AgentFramework) -> Dict[str, str]:
    """
    Set up a research assistant with specialized agents.
    
    This function creates a hierarchy of agents specialized in different
    aspects of research and information gathering.
    
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
    literature_id = await framework.create_agent(
        "Literature Reviewer", 
        "literature_reviewer",
        parent_id=main_agent_id
    )
    
    data_analyst_id = await framework.create_agent(
        "Data Analyst", 
        "data_analyst",
        parent_id=main_agent_id
    )
    
    fact_checker_id = await framework.create_agent(
        "Fact Checker", 
        "fact_checker",
        parent_id=main_agent_id
    )
    
    writer_id = await framework.create_agent(
        "Science Writer", 
        "science_writer",
        parent_id=main_agent_id
    )
    
    # Return map of roles to IDs
    return {
        "main": main_agent_id,
        "literature": literature_id,
        "data_analyst": data_analyst_id,
        "fact_checker": fact_checker_id,
        "writer": writer_id
    }

async def run_research_task(framework: AgentFramework, task_description: str) -> None:
    """
    Run a research task using the research assistant.
    
    This function submits a research task to the framework and displays
    the results.
    
    Args:
        framework: Initialized AgentFramework instance
        task_description: Description of the research task
    """
    print(f"Submitting research task: {task_description}")
    
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
    Main entry point for the research assistant example.
    """
    parser = argparse.ArgumentParser(description="MCP Agent Framework Research Assistant Example")
    parser.add_argument("--task", help="Research task to submit")
    parser.add_argument("--api-key", help="API key for language models")
    parser.add_argument("--model", help="Model to use", default="gpt-4")
    parser.add_argument("--use-anthropic", action="store_true", help="Use Anthropic Claude instead of OpenAI")
    args = parser.parse_args()
    
    # Get API keys from arguments or environment
    api_key = None
    anthropic_api_key = None
    
    if args.use_anthropic:
        anthropic_api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            print("Error: No Anthropic API key provided.")
            print("Please provide an API key using --api-key or set the ANTHROPIC_API_KEY environment variable.")
            return 1
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: No OpenAI API key provided.")
            print("Please provide an API key using --api-key or set the OPENAI_API_KEY environment variable.")
            return 1
    
    # Set up configuration
    config = {
        "api_key": api_key,
        "anthropic_api_key": anthropic_api_key,
        "model": args.model,
        "llm_provider": "anthropic" if args.use_anthropic else "openai"
    }
    
    # Create framework
    framework = AgentFramework(api_key=api_key, config_path=None)
    framework.config.update(config)
    
    # Set up research assistant
    agent_ids = await setup_research_assistant(framework)
    
    print("Research Assistant initialized with the following agents:")
    for role, agent_id in agent_ids.items():
        agent_info = await framework.get_agent(agent_id)
        if agent_info:
            print(f"- {agent_info['name']} ({agent_info['role']}): {agent_id[:8]}...")
    
    if args.task:
        # Run specific task
        await run_research_task(framework, args.task)
    else:
        # Interactive mode
        print("\nInteractive Research Assistant")
        print("Type 'exit' to quit")
        
        while True:
            task = input("\nEnter research task: ")
            
            if task.lower() == "exit":
                break
            
            if task:
                await run_research_task(framework, task)
    
    # Shutdown framework
    await framework.shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
