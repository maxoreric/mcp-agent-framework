# MCP Agent Framework Architecture

## 1. Introduction

The MCP Agent Framework is a Python-based system designed to create, orchestrate, and manage networks of AI agents. Using the Model Context Protocol (MCP) as the standard communication protocol, this framework enables agents to dynamically create subordinate agents, forming tree-like organizational structures to solve complex problems.

### 1.1 System Overview

This architecture leverages the Model Context Protocol to implement a hierarchical agent system where:

- **Each agent functions as both a client and server** within the MCP ecosystem
- **Agents can create and manage child agents** with specialized roles
- **Task orchestration** follows a hierarchical decomposition pattern
- **Standardized XML-style communication** enables structured information exchange
- **Terminal-based user interface** provides task visualization and management

## 2. Core Architecture

The framework is built on a dual client-server architecture where each agent participates in the system as both:

### 2.1 Architectural Components

![Architecture Diagram](https://i.imgur.com/JG6jDqB.png)

1. **MCP Server**: Each agent exposes its capabilities as MCP tools, resources, and prompts
2. **MCP Client**: Each agent can connect to child agents and utilize their capabilities
3. **Agent Hierarchy**: A tree-structure of specialized agents with the main agent at the root
4. **Task Orchestration**: A system for breaking down tasks and distributing them to appropriate agents
5. **XML Protocol**: A structured communication format for agent interactions

### 2.2 Agent Model

The core agent model consists of:

- **Agent Identity**: Unique identification and role description
- **MCP Server**: Exposes the agent's capabilities to parent agents
- **MCP Client Connection**: Connects to child agents
- **Task Processing**: Handles task execution and management
- **LLM Integration**: Interfaces with language models for solving tasks

## 3. Agent Implementation

Each agent is implemented using MCP's FastMCP framework, combining both client and server capabilities.

### 3.1 Agent Base Implementation

```python
from mcp.server.fastmcp import FastMCP, Context
import uuid
from typing import Dict, List, Any, Optional

class Agent:
    """Agent implementation using MCP's FastMCP."""
    
    def __init__(self, name: str, role: str, config: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.config = config
        self.child_agents = {}  # Child agent connections
        self.tasks = {}  # Task tracking
        
        # Initialize FastMCP server
        self.mcp = FastMCP(f"agent-{self.name}")
        
        # Register MCP capabilities
        self._register_capabilities()
```

### 3.2 MCP Capabilities

Each agent exposes its capabilities as MCP tools, resources, and prompts:

```python
def _register_capabilities(self):
    """Register agent capabilities using FastMCP decorators."""
    
    @self.mcp.tool()
    async def create_child_agent(name: str, role: str, ctx: Context) -> str:
        """Create a child agent with the specified role."""
        ctx.info(f"Creating child agent: {name} with role: {role}")
        # Implementation for creating a child agent
        return "child_agent_id"  # Return the created agent's ID
    
    @self.mcp.tool()
    async def submit_task(description: str, dependencies: List[str] = None, ctx: Context) -> str:
        """Submit a task to this agent."""
        task_id = str(uuid.uuid4())
        ctx.info(f"Received task: {description}")
        
        # Task implementation
        # ...
        
        return task_id
    
    @self.mcp.resource("task://{task_id}")
    async def get_task(task_id: str) -> str:
        """Get task details as a resource."""
        if task_id not in self.tasks:
            return "<e>Task not found</e>"
        
        # Return task details in XML format
        task = self.tasks[task_id]
        return f"""<task id="{task_id}">
<description>{task['description']}</description>
<status>{task['status']}</status>
<r>{task.get('result', '')}</r>
</task>"""
    
    @self.mcp.prompt()
    def agent_prompt() -> str:
        """Return a prompt template for this agent."""
        return f"""<role>{self.role}</role>
You are a specialized agent with expertise in {self.role}.
Your task is to assist the main system by providing expert knowledge.

When given a task, analyze it carefully and provide the best solution based on your expertise.
"""
```

### 3.3 Agent Methods

Key functionality that each agent implements:

- **Task Submission**: Receive and process tasks
- **Task Delegation**: Break down complex tasks and delegate to child agents
- **Agent Creation**: Dynamically create specialized child agents
- **Result Aggregation**: Combine results from multiple subtasks
- **Status Reporting**: Report task and subtask status

## 4. Hierarchical Agent Structure

The framework implements a hierarchical structure with a main agent and dynamically created expert agents.

### 4.1 Hierarchy Management

```python
class AgentHierarchy:
    """Manager for hierarchical agent structure."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.main_agent = None
        self.agent_registry = {}  # Track all created agents
    
    async def initialize(self) -> None:
        """Initialize the agent hierarchy with a main agent."""
        # Create main agent with CEO role
        self.main_agent = await self._create_agent("Main", "CEO", parent_id=None)
    
    async def _create_agent(self, name: str, role: str, parent_id: Optional[str] = None) -> Agent:
        """Create an agent and register it."""
        # Create agent configuration
        agent_config = {
            "name": name,
            "role": role,
            "api_key": self.config.get("api_key"),
            "model": self.config.get("model", "gpt-4"),
        }
        
        # Create agent instance
        agent = Agent(name, role, agent_config)
        
        # Register agent
        self.agent_registry[agent.id] = agent
        
        # Connect to parent if needed
        if parent_id and parent_id in self.agent_registry:
            parent = self.agent_registry[parent_id]
            await self._connect_parent_child(parent, agent)
        
        return agent
    
    async def _connect_parent_child(self, parent: Agent, child: Agent) -> None:
        """Establish MCP connection between parent and child."""
        # Set up MCP client connection from parent to child
        # This would use MCP's client session to connect
        parent.child_agents[child.id] = {
            "name": child.name,
            "role": child.role,
            "connection": None  # Will hold MCP client session
        }
```

### 4.2 Agent Lifecycle

The lifecycle of agents in the hierarchy:

1. **Creation**: Agents are created by parent agents or the system
2. **Initialization**: MCP server and client connections are established
3. **Registration**: Agents are registered in the hierarchy
4. **Operation**: Agents process tasks and delegate subtasks
5. **Destruction**: Agents are destroyed when no longer needed

## 5. Task Orchestration

Tasks are orchestrated using MCP's tools and resources, following a hierarchical decomposition pattern.

### 5.1 Task Orchestration Process

```python
async def orchestrate_task(agent: Agent, task_description: str, ctx: Context) -> str:
    """Orchestrate a task using MCP tools and resources."""
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Analyze task complexity
    complexity = await analyze_task_complexity(agent, task_description, ctx)
    
    if complexity == "simple":
        # Execute directly
        await execute_task(agent, task_id, task_description, ctx)
    else:
        # Decompose into subtasks
        subtasks = await decompose_task(agent, task_description, ctx)
        
        # Create dependency tree
        dependency_tree = create_dependency_tree(subtasks)
        
        # Execute subtasks based on dependencies
        for subtask in dependency_tree:
            # Find appropriate agent for subtask
            target_agent = await find_agent_for_task(agent, subtask["description"], ctx)
            
            # Submit task to target agent
            await submit_task_to_agent(target_agent, subtask, ctx)
    
    return task_id
```

### 5.2 Task Execution

```python
async def execute_task(agent: Agent, task_id: str, description: str, ctx: Context) -> None:
    """Execute a task directly."""
    # Create XML-formatted task for LLM
    task_prompt = f"""<role>{agent.role}</role>
<task>{description}</task>
<context>
As an expert in {agent.role}, analyze and solve this task.
Provide a detailed and comprehensive solution.
</context>
"""
    
    # Process with LLM
    result = await process_with_llm(agent, task_prompt, ctx)
    
    # Update task with result
    agent.tasks[task_id] = {
        "id": task_id,
        "description": description,
        "status": "COMPLETED",
        "result": result
    }
```

### 5.3 Task Dependency Management

Tasks can have dependencies on other tasks, creating a directed acyclic graph (DAG) of task execution:

- **Task Dependency Tree**: Tracks which tasks depend on others
- **Execution Ordering**: Tasks are executed in an order that respects dependencies
- **Status Tracking**: Progress is monitored and reported for the entire dependency tree
- **Result Aggregation**: Results from subtasks are aggregated into final results

## 6. XML Communication Protocol

The framework uses an XML-style communication format for structured information exchange.

### 6.1 Message Format

```python
def format_task_message(task: Dict[str, Any]) -> str:
    """Format a task message using XML-style tags."""
    return f"""<task id="{task['id']}">
    <description>{task['description']}</description>
    <status>{task['status']}</status>
    {format_result(task)}
    {format_error(task)}
</task>"""

def format_result(task: Dict[str, Any]) -> str:
    """Format task result if available."""
    if task.get('result'):
        return f"<r>{task['result']}</r>"
    return ""

def format_error(task: Dict[str, Any]) -> str:
    """Format task error if available."""
    if task.get('error'):
        return f"<e>{task['error']}</e>"
    return ""
```

### 6.2 Message Types

The XML protocol includes message types for:

- **Task Descriptions**: Information about tasks to be performed
- **Role Descriptions**: Information about agent roles and capabilities
- **Context Information**: Contextual details for task execution
- **Result Data**: Results from completed tasks
- **Error Information**: Details about errors and exceptions

## 7. LLM Integration

Language models are integrated into the framework for task execution and decision-making.

### 7.1 LLM Request Processing

```python
async def process_with_llm(agent: Agent, prompt: str, ctx: Context) -> str:
    """Process a prompt with the LLM using MCP Context."""
    try:
        # Report progress
        ctx.info("Sending request to language model...")
        
        # Create LLM request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {agent.config['api_key']}"
                },
                json={
                    "model": agent.config.get('model', 'gpt-4'),
                    "messages": [
                        {"role": "system", "content": f"You are an expert agent specialized in {agent.role}."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                },
                timeout=30.0
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Process result
            content = result['choices'][0]['message']['content']
            ctx.info("Received response from language model")
            
            return content
            
    except Exception as e:
        ctx.error(f"Error processing with LLM: {str(e)}")
        raise
```

### 7.2 Prompt Management

The framework includes a system for managing prompts, including:

- **Role-specific Prompts**: Templates customized for different agent roles
- **Task-specific Prompts**: Templates for different types of tasks
- **Context Inclusion**: Methods for including relevant context in prompts
- **XML Formatting**: Structured XML format for improved parsing

## 8. Command-Line Interface

The framework includes a command-line interface for user interaction.

### 8.1 CLI Implementation

```python
class CommandLineInterface:
    """Command-line interface for the MCP Agent Framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hierarchy = AgentHierarchy(config)
    
    async def start(self) -> None:
        """Start the CLI interface."""
        print("Initializing agent system...")
        await self.hierarchy.initialize()
        print("Agent system initialized with main agent.")
        
        while True:
            try:
                command = input("\nEnter command (task/status/exit): ")
                
                if command.lower() == "exit":
                    await self.shutdown()
                    break
                
                if command.lower() == "task":
                    await self.handle_task()
                elif command.lower() == "status":
                    await self.handle_status()
                else:
                    print("Unknown command. Available commands: task, status, exit")
            
            except KeyboardInterrupt:
                await self.shutdown()
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    async def handle_task(self) -> None:
        """Handle task submission."""
        description = input("Enter task description: ")
        
        # Create context for task
        ctx = Context()
        
        # Submit task to main agent
        task_id = await orchestrate_task(self.hierarchy.main_agent, description, ctx)
        
        print(f"Task submitted with ID: {task_id}")
        
        # Display task progress
        await self.display_task_progress(task_id)
```

### 8.2 Task Visualization

The CLI includes visualization for task status and progress:

- **Task Trees**: Visualize the hierarchical structure of tasks and subtasks
- **Status Indicators**: Show the current status of each task
- **Progress Tracking**: Display progress of long-running tasks
- **Result Viewing**: Format and display task results

## 9. Implementation Timeline

The implementation of the MCP Agent Framework is divided into several phases:

### 9.1 Phase 1: Core Framework (Week 1)
- Implement Agent using FastMCP
- Implement hierarchical structure
- Set up basic task processing

### 9.2 Phase 2: Task Orchestration (Week 2)
- Implement task decomposition
- Implement dependency tracking
- Implement task execution

### 9.3 Phase 3: Agent Communication (Week 3)
- Implement parent-child MCP connections
- Implement XML-style messaging
- Implement task delegation

### 9.4 Phase 4: User Interface & Examples (Week 4)
- Implement CLI interface
- Create example applications
- Test and refine the system

## 10. Project Structure

```
📦 mcp_agent_framework
 ┣ 📂 mcp_agent_framework
 ┃ ┣ 📂 agent
 ┃ ┃ ┣ 📜 agent.py           # Agent implementation using FastMCP
 ┃ ┃ ┣ 📜 hierarchy.py       # Hierarchical agent management
 ┃ ┃ ┗ 📜 factory.py         # Agent creation and registration
 ┃ ┣ 📂 task
 ┃ ┃ ┣ 📜 orchestrator.py    # Task orchestration using MCP
 ┃ ┃ ┣ 📜 dependency.py      # Task dependency management
 ┃ ┃ ┗ 📜 execution.py       # Task execution handling
 ┃ ┣ 📂 llm
 ┃ ┃ ┣ 📜 integration.py     # LLM API integration
 ┃ ┃ ┗ 📜 prompts.py         # Prompt templates and management
 ┃ ┣ 📂 cli
 ┃ ┃ ┣ 📜 interface.py       # Command-line interface
 ┃ ┃ ┗ 📜 visualization.py   # Task visualization
 ┃ ┗ 📜 __main__.py          # Entry point for the framework
 ┣ 📂 examples
 ┃ ┣ 📜 coding_assistant.py  # Coding assistant example
 ┃ ┗ 📜 research_assistant.py # Research assistant example
 ┣ 📂 tests                  # Test suite
 ┣ 📜 setup.py               # Package setup
 ┗ 📜 requirements.txt       # Dependencies
```

## 11. Advantages of MCP Integration

This implementation leverages MCP's capabilities in several key ways:

1. **Agent Communication**: Using MCP's client-server architecture for parent-child communication
2. **Resource Sharing**: Using MCP resources for task status and results
3. **Tool-based Functionality**: Exposing agent capabilities as MCP tools
4. **Prompt Templates**: Using MCP prompts for agent-specific templates
5. **Context Protocol**: Using MCP's context protocol for task execution and progress reporting

The benefits of building on the MCP foundation include:

- **Standardized Communication**: Well-defined protocol for agent interactions
- **Extensibility**: Easy addition of new agent capabilities
- **Interoperability**: Potential integration with other MCP-compatible systems
- **Resource Management**: Built-in mechanisms for resource exposure and tracking
- **Task Orchestration**: Natural tools-based approach to task delegation

## 12. Conclusion

The MCP Agent Framework provides a powerful system for creating hierarchical agent networks that can collaborate to solve complex problems. By leveraging the Model Context Protocol, the framework ensures standardized communication, extensibility, and interoperability.

This architecture document outlines the core components, implementation details, and development plan for building a robust agent orchestration system using MCP.
