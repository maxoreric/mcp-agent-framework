"""
Command-line Interface for MCP Agent Framework.

This module provides a user-friendly command-line interface for
interacting with the MCP Agent Framework.
"""

import asyncio
import sys
import os
import json
import time
from typing import Dict, Any, List, Optional, Callable, Set

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.live import Live

from mcp.server.fastmcp import Context

from ..agent.hierarchy import AgentHierarchy
from ..task.orchestrator import orchestrate_task
from .visualization import TaskVisualizer

class CommandLineInterface:
    """
    Command-line interface for the MCP Agent Framework.
    
    This class provides a user-friendly interface for interacting with
    the agent framework, including task submission, status monitoring,
    and result visualization.
    
    Attributes:
        config: Configuration dictionary
        hierarchy: AgentHierarchy instance
        console: Rich console for output
        visualizer: TaskVisualizer for task visualization
        active_tasks: Set of currently active task IDs
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a new CommandLineInterface instance.
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        # Ensure we have the necessary configuration for LLM integration
        self.config = config.copy()  # Make a copy to avoid modifying the original
        
        # Add required XML tags if missing
        if "open_xml_tag" not in self.config:
            self.config["open_xml_tag"] = "<answer>"
        if "close_xml_tag" not in self.config:
            self.config["close_xml_tag"] = "</answer>"
        
        # Default to DeepSeek provider if not set
        if "llm_provider" not in self.config or not self.config["llm_provider"]:
            self.config["llm_provider"] = "deepseek"
        
        # Initialize components
        self.hierarchy = AgentHierarchy(self.config)
        self.console = Console()
        self.visualizer = TaskVisualizer(self.console)
        self.active_tasks: Set[str] = set()
    
    async def start(self) -> None:
        """
        Start the CLI interface.
        
        This method initializes the agent hierarchy and starts the
        interactive command loop.
        """
        # Display welcome message
        self.console.print(Panel.fit(
            "[bold blue]MCP Agent Framework CLI[/bold blue]",
            subtitle="Model Context Protocol Agent Framework"
        ))
        
        # Check for API keys
        if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            self.console.print("[yellow]Warning: No API keys found in environment variables.[/yellow]")
            self.console.print("Please set at least one of the following:")
            self.console.print("  - DEEPSEEK_API_KEY")
            self.console.print("  - OPENAI_API_KEY")
            self.console.print("  - ANTHROPIC_API_KEY")
            
            # Prompt user for DeepSeek API key
            api_key = Prompt.ask("\nPlease enter your DeepSeek API Key", password=True)
            if api_key:
                os.environ["DEEPSEEK_API_KEY"] = api_key
                self.config["deepseek_api_key"] = api_key
                self.console.print("[green]API key set.[/green]")
            else:
                self.console.print("[red]No API key provided. The framework may not function correctly.[/red]")
        
        # Initialize the agent hierarchy
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Initializing agent system...[/bold blue]"),
            transient=True,
        ) as progress:
            task = progress.add_task("Initializing", total=None)
            await self.hierarchy.initialize()
            progress.update(task, completed=True)
        
        self.console.print("[green]Agent system initialized with main agent.[/green]")
        
        if self.hierarchy.main_agent:
            self.console.print(f"Main Agent: [bold]{self.hierarchy.main_agent.name}[/bold] ({self.hierarchy.main_agent.role})")
        
        # Register event handlers
        if self.hierarchy.main_agent:
            self.hierarchy.main_agent.on("task_created", self._handle_task_created)
            self.hierarchy.main_agent.on("task_updated", self._handle_task_updated)
            self.hierarchy.main_agent.on("task_completed", self._handle_task_completed)
            self.hierarchy.main_agent.on("task_failed", self._handle_task_failed)
        
        # Start command loop
        await self._command_loop()
    
    async def _command_loop(self) -> None:
        """
        Run the interactive command loop.
        
        This internal method handles user commands and dispatches
        them to the appropriate handlers.
        """
        while True:
            try:
                self.console.print()
                command = Prompt.ask(
                    "[bold cyan]Enter command[/bold cyan]",
                    choices=["task", "status", "agents", "help", "exit"],
                    default="help"
                )
                
                if command.lower() == "exit":
                    if await self._shutdown():
                        break
                elif command.lower() == "task":
                    await self._handle_task_command()
                elif command.lower() == "status":
                    await self._handle_status_command()
                elif command.lower() == "agents":
                    await self._handle_agents_command()
                elif command.lower() == "help":
                    self._show_help()
                else:
                    self.console.print("[yellow]Unknown command. Type 'help' for available commands.[/yellow]")
            
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled.[/yellow]")
                if await self._shutdown_prompt():
                    break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    async def _handle_task_command(self) -> None:
        """
        Handle the 'task' command.
        
        This internal method processes task submission, allowing users
        to input task descriptions and submit them to the agent hierarchy.
        """
        self.console.print("[bold]Submit a new task to the agent system[/bold]")
        
        # Get task description
        description = Prompt.ask("[cyan]Enter task description[/cyan]")
        
        if not description:
            self.console.print("[yellow]Task submission cancelled.[/yellow]")
            return
        
        # Create progress context
        ctx = Context()
        
        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True
        ) as progress:
            # Add progress task
            progress_task = progress.add_task("Processing task...", total=None)
            
            # Register progress callback
            async def progress_callback(current: int, total: int) -> None:
                if total:
                    progress.update(progress_task, completed=current, total=total)
                    
            # Register log callback
            def log_callback(level: str, message: str) -> None:
                color = {
                    "info": "blue",
                    "warning": "yellow",
                    "error": "red"
                }.get(level, "white")
                
                progress.console.print(f"[{color}]{message}[/{color}]")
            
            # Create a custom Context adapter class to handle both awaited and non-awaited calls
            class ProgressContext:
                def __init__(self, ctx):
                    self.ctx = ctx
                
                async def info(self, msg):
                    log_callback("info", msg)
                    # This passthrough allows for more graceful handling of calls
                
                async def warning(self, msg):
                    log_callback("warning", msg) 
                    
                async def error(self, msg):
                    log_callback("error", msg)
                
                async def debug(self, msg):
                    # Silently handle debug messages
                    pass
                    
                # Non-async versions for compatibility with code that doesn't await
                def __getattr__(self, name):
                    # For any attribute not defined, return a function that logs but doesn't crash
                    return lambda msg: log_callback(name, msg)
            
            # Use the adapter instead of modifying ctx directly
            progress_ctx = ProgressContext(ctx)
            
            try:
                # Submit task to main agent
                task_id = await orchestrate_task(self.hierarchy, description, progress_ctx)
                
                # Store active task
                self.active_tasks.add(task_id)
                
                progress.update(progress_task, description="Task completed", completed=100, total=100)
                
                # Display task result if available
                if self.hierarchy.main_agent and task_id in self.hierarchy.main_agent.tasks:
                    task = self.hierarchy.main_agent.tasks[task_id]
                    if task.get("status") == "COMPLETED" and "result" in task:
                        self.console.print("\n[bold green]Task Result:[/bold green]")
                        self.console.print(Panel(task["result"], title=f"Task: {description}", expand=False))
                    elif task.get("status") == "FAILED" and "error" in task:
                        self.console.print("\n[bold red]Task Failed:[/bold red]")
                        self.console.print(f"Error: {task['error']}")
                
            except Exception as e:
                progress.update(progress_task, description="Task failed", completed=0, total=100)
                self.console.print(f"\n[bold red]Error submitting task:[/bold red] {str(e)}")
    
    async def _handle_status_command(self) -> None:
        """
        Handle the 'status' command.
        
        This internal method displays the status of all active tasks
        and the agent hierarchy.
        """
        self.console.print("[bold]System Status[/bold]")
        
        # Display agent status
        agent_count = len(self.hierarchy.agent_registry)
        self.console.print(f"\nActive Agents: [bold cyan]{agent_count}[/bold cyan]")
        
        # Collect all tasks from all agents
        all_tasks = {}
        for agent_id, agent in self.hierarchy.agent_registry.items():
            for task_id, task in agent.tasks.items():
                all_tasks[task_id] = {
                    "id": task_id,
                    "description": task.get("description", "Unknown task"),
                    "status": task.get("status", "UNKNOWN"),
                    "agent": agent.name,
                    "role": agent.role,
                    "created_at": task.get("created_at", 0),
                }
        
        # Display task status
        task_count = len(all_tasks)
        active_count = sum(1 for t in all_tasks.values() if t["status"] in ["PENDING", "IN_PROGRESS"])
        completed_count = sum(1 for t in all_tasks.values() if t["status"] == "COMPLETED")
        failed_count = sum(1 for t in all_tasks.values() if t["status"] == "FAILED")
        
        self.console.print(f"\nTasks:")
        self.console.print(f"  [bold]Total:[/bold] {task_count}")
        self.console.print(f"  [bold blue]Active:[/bold blue] {active_count}")
        self.console.print(f"  [bold green]Completed:[/bold green] {completed_count}")
        self.console.print(f"  [bold red]Failed:[/bold red] {failed_count}")
        
        # Display task table
        if task_count > 0:
            table = Table(title="Recent Tasks")
            table.add_column("ID", style="dim")
            table.add_column("Description")
            table.add_column("Status", justify="center")
            table.add_column("Agent")
            table.add_column("Role")
            
            # Sort tasks by creation time (most recent first)
            sorted_tasks = sorted(
                all_tasks.values(),
                key=lambda t: t.get("created_at", 0),
                reverse=True
            )
            
            # Display at most 10 recent tasks
            for task in sorted_tasks[:10]:
                status_style = {
                    "PENDING": "yellow",
                    "IN_PROGRESS": "blue",
                    "COMPLETED": "green",
                    "FAILED": "red",
                }.get(task["status"], "white")
                
                table.add_row(
                    task["id"][:8],
                    task["description"][:50] + ("..." if len(task["description"]) > 50 else ""),
                    f"[{status_style}]{task['status']}[/{status_style}]",
                    task["agent"],
                    task["role"]
                )
            
            self.console.print(table)
            
            # Display task tree visualization for active tasks
            if active_count > 0:
                self.console.print("\n[bold]Task Dependency Tree:[/bold]")
                await self.visualizer.draw_task_tree(self.hierarchy)
    
    async def _handle_agents_command(self) -> None:
        """
        Handle the 'agents' command.
        
        This internal method displays information about all active agents
        in the hierarchy.
        """
        self.console.print("[bold]Agent Hierarchy[/bold]")
        
        # Display agent table
        table = Table(title="Active Agents")
        table.add_column("ID", style="dim")
        table.add_column("Name")
        table.add_column("Role")
        table.add_column("Task Count")
        table.add_column("Child Agents")
        
        for agent_id, agent in self.hierarchy.agent_registry.items():
            task_count = len(agent.tasks)
            child_count = len(agent.child_agents)
            
            table.add_row(
                agent_id[:8],
                agent.name,
                agent.role,
                str(task_count),
                str(child_count)
            )
        
        self.console.print(table)
        
        # Display agent tree visualization
        self.console.print("\n[bold]Agent Hierarchy Tree:[/bold]")
        await self.visualizer.draw_agent_tree(self.hierarchy)
    
    def _show_help(self) -> None:
        """
        Display available commands and their usage.
        
        This internal method shows help information for the CLI.
        """
        help_text = """
# MCP Agent Framework CLI

## Available Commands

- **task**: Submit a new task to the agent system
- **status**: Display the status of all tasks and agents
- **agents**: Display information about all active agents
- **help**: Show this help message
- **exit**: Close the application

## Task Command

The **task** command allows you to submit a new task to the agent system.
You will be prompted to enter a task description, which can be any natural
language instruction for the agent hierarchy to process.

## Status Command

The **status** command displays the current status of all tasks and agents
in the system. This includes information about task progress, agent roles,
and hierarchical relationships.

## Agents Command

The **agents** command provides detailed information about all active agents
in the hierarchy, including their roles, tasks, and relationships.
        """
        
        self.console.print(Markdown(help_text))
    
    async def _shutdown_prompt(self) -> bool:
        """
        Prompt for confirmation before shutting down.
        
        Returns:
            bool: True if shutdown is confirmed, False otherwise
        """
        return Confirm.ask("[yellow]Do you want to exit the application?[/yellow]")
    
    async def _shutdown(self) -> bool:
        """
        Gracefully shut down the application.
        
        This method shuts down the agent hierarchy and cleans up resources.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        if not await self._shutdown_prompt():
            return False
        
        self.console.print("[yellow]Shutting down agent system...[/yellow]")
        
        try:
            # Shutdown the agent hierarchy
            await self.hierarchy.shutdown()
            self.console.print("[green]Agent system shut down successfully.[/green]")
            return True
        
        except Exception as e:
            self.console.print(f"[bold red]Error during shutdown:[/bold red] {str(e)}")
            return False
    
    async def _handle_task_created(self, data: Dict[str, Any]) -> None:
        """
        Handle task creation events.
        
        This callback is triggered when a new task is created in the system.
        
        Args:
            data: Event data containing task information
        """
        task_id = data.get("id")
        description = data.get("description")
        
        if task_id and description:
            self.console.print(f"[blue]Task created:[/blue] {description}")
    
    async def _handle_task_updated(self, data: Dict[str, Any]) -> None:
        """
        Handle task update events.
        
        This callback is triggered when a task's status is updated.
        
        Args:
            data: Event data containing task information
        """
        task_id = data.get("id")
        status = data.get("status")
        
        if task_id and status:
            status_style = {
                "PENDING": "yellow",
                "IN_PROGRESS": "blue",
                "COMPLETED": "green",
                "FAILED": "red",
            }.get(status, "white")
            
            if self.hierarchy.main_agent and task_id in self.hierarchy.main_agent.tasks:
                description = self.hierarchy.main_agent.tasks[task_id].get("description", "Unknown task")
                self.console.print(f"Task updated: {description} - Status: [{status_style}]{status}[/{status_style}]")
    
    async def _handle_task_completed(self, data: Dict[str, Any]) -> None:
        """
        Handle task completion events.
        
        This callback is triggered when a task is successfully completed.
        
        Args:
            data: Event data containing task information
        """
        task_id = data.get("id")
        
        if task_id:
            if self.hierarchy.main_agent and task_id in self.hierarchy.main_agent.tasks:
                description = self.hierarchy.main_agent.tasks[task_id].get("description", "Unknown task")
                self.console.print(f"[green]Task completed:[/green] {description}")
    
    async def _handle_task_failed(self, data: Dict[str, Any]) -> None:
        """
        Handle task failure events.
        
        This callback is triggered when a task fails.
        
        Args:
            data: Event data containing task information
        """
        task_id = data.get("id")
        error = data.get("error")
        
        if task_id:
            if self.hierarchy.main_agent and task_id in self.hierarchy.main_agent.tasks:
                description = self.hierarchy.main_agent.tasks[task_id].get("description", "Unknown task")
                self.console.print(f"[red]Task failed:[/red] {description}")
                if error:
                    self.console.print(f"[red]Error:[/red] {error}")
