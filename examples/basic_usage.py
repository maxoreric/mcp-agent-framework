#!/usr/bin/env python3
"""
MCP Agent Framework - Basic Usage Example

This example demonstrates the basic usage of the MCP Agent Framework,
including creation of agents, task submission, and result retrieval.

基本用法示例，展示了MCP Agent Framework的基本用法，
包括创建Agent、提交任务和检索结果。
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Add parent directory to path to import the framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_agent_framework import AgentFramework, Agent, AgentHierarchy
from mcp_agent_framework.agent.factory import AgentSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mcp_agent_framework.log")
    ]
)

logger = logging.getLogger("mcp_example")

async def simple_example() -> None:
    """
    Simple example demonstrating the basic usage of the framework.
    
    简单示例，演示框架的基本用法。
    """
    # Create the framework instance with configuration
    config = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-4",
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "anthropic_model": "claude-3-opus-20240229",
        "deepseek_api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "deepseek_model": "deepseek-chat",
        "llm_provider": "deepseek",  # Options: openai, anthropic, deepseek
        "use_prefix_completion": True,
        "open_xml_tag": "<answer>",
        "close_xml_tag": "</answer>",
        "system_prompt": "You are a helpful AI assistant focused on providing precise, accurate information."
    }
    
    framework = AgentFramework(config=config)
    
    # Initialize the framework
    await framework.initialize()
    print("Framework initialized.")
    
    # Create specialized agents
    developer_agent_id = await framework.create_agent(
        name="Developer",
        role="developer",
        parent_id=framework.hierarchy.main_agent.id
    )
    
    researcher_agent_id = await framework.create_agent(
        name="Researcher",
        role="researcher",
        parent_id=framework.hierarchy.main_agent.id
    )
    
    print(f"Created agents: Developer ({developer_agent_id}), Researcher ({researcher_agent_id})")
    
    # Submit a simple task
    simple_task_id = await framework.submit_task(
        description="Calculate the sum of numbers from 1 to 10."
    )
    
    print(f"Submitted simple task: {simple_task_id}")
    
    # Wait for the simple task to complete
    simple_result = await framework.wait_for_task(simple_task_id, timeout=30)
    
    if simple_result:
        print("\n=== Simple Task Result ===")
        print(f"Status: {simple_result['status']}")
        print(f"Result: {simple_result['result']}")
    else:
        print("Simple task did not complete within the timeout period.")
    
    # Submit a complex task that requires multiple agents
    complex_task_id = await framework.submit_task(
        description="""
        Research the concept of transformers in machine learning and create a simple
        implementation example in Python. The research should include:
        1. Basic explanation of transformers
        2. Key components like self-attention
        3. Common applications
        
        Then implement a minimal working example of a transformer architecture.
        """
    )
    
    print(f"Submitted complex task: {complex_task_id}")
    
    # Wait for the complex task to complete
    complex_result = await framework.wait_for_task(complex_task_id, timeout=120)
    
    if complex_result:
        print("\n=== Complex Task Result ===")
        print(f"Status: {complex_result['status']}")
        print(f"Result: \n{complex_result['result']}")
    else:
        print("Complex task did not complete within the timeout period.")
    
    # Shutdown the framework
    await framework.shutdown()
    print("Framework shut down.")

async def create_team_example() -> None:
    """
    Example demonstrating the creation and use of a team of agents.
    
    示例演示创建和使用Agent团队。
    """
    # Create the framework instance
    framework = AgentFramework()
    
    # Initialize the framework
    await framework.initialize()
    print("Framework initialized for team example.")
    
    # Create a specialized team of agents using the factory
    factory = framework.factory
    
    team = await factory.create_agent_team(
        team_roles=["developer", "researcher", "writer", "analyst"],
        leader_role="ceo",
        name_prefix="Project",
        configs={
            "developer": {
                "programming_languages": ["Python", "JavaScript", "Rust"],
                "specialization": "Backend development"
            },
            "researcher": {
                "search_depth": 5,
                "specialization": "Technology trends"
            }
        }
    )
    
    print(f"Created agent team with {len(team)} members:")
    for role, agent in team.items():
        print(f"  - {agent.name} ({role}): {agent.id}")
    
    # Submit a task to the team leader (CEO)
    ceo = team["ceo"]
    
    task_id = await framework.submit_task(
        description="""
        Create a project plan for developing a new mobile application that uses
        machine learning to identify plants from photos. Include:
        1. Technical architecture overview
        2. Research on existing solutions
        3. Development timeline and milestones
        4. Market analysis
        """
    )
    
    print(f"Submitted task to CEO: {task_id}")
    
    # Wait for the task to complete
    result = await framework.wait_for_task(task_id, timeout=180)
    
    if result:
        print("\n=== Team Task Result ===")
        print(f"Status: {result['status']}")
        print(f"Result: \n{result['result']}")
    else:
        print("Team task did not complete within the timeout period.")
    
    # Shutdown the framework
    await framework.shutdown()
    print("Team example framework shut down.")

if __name__ == "__main__":
    # Run the examples
    print("=== Running Simple Example ===")
    asyncio.run(simple_example())
    
    print("\n\n=== Running Team Example ===")
    asyncio.run(create_team_example())
