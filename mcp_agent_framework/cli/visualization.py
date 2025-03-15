"""
Task and Agent Visualization for MCP Agent Framework.

MCP Agent Framework的任务和Agent可视化。

本模块提供了在终端中以用户友好的格式显示任务和Agent层次结构的可视化功能。

This module provides visualization functions for displaying task and
agent hierarchies in a user-friendly format in the terminal.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

from rich.console import Console
from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from ..agent.hierarchy import AgentHierarchy

# 配置模块级别日志器
# Configure module-level logger
logger = logging.getLogger(__name__)

class TaskVisualizer:
    """
    任务和Agent层次结构的可视化器。
    
    Visualizer for tasks and agent hierarchies.
    
    此类提供了用于创建任务树、Agent层次结构和其他系统组件的视觉表示的方法。
    
    This class provides methods for creating visual representations
    of task trees, agent hierarchies, and other system components.
    
    Attributes:
        console: 用于输出的Rich控制台
                Rich console for output
    """
    
    def __init__(self, console: Console):
        """
        初始化新的TaskVisualizer实例。
        
        Initialize a new TaskVisualizer instance.
        
        Args:
            console: 用于输出的Rich控制台
                    Rich console for output
        """
        self.console = console
    
    async def draw_task_tree(self, hierarchy: AgentHierarchy) -> None:
        """
        绘制任务依赖树的可视化表示。
        
        Draw a visual representation of the task dependency tree.
        
        此方法创建显示任务及其依赖关系的树状可视化。
        
        This method creates a tree visualization showing tasks and
        their dependencies.
        
        Args:
            hierarchy: 包含任务的Agent层次结构
                      Agent hierarchy containing the tasks
        """
        try:
            # 从所有Agent收集所有任务
            # Collect all tasks from all agents
            all_tasks = {}
            for agent_id, agent in hierarchy.agent_registry.items():
                for task_id, task in agent.tasks.items():
                    assigned_to = task.get("assigned_to")
                    if assigned_to in hierarchy.agent_registry:
                        agent_name = hierarchy.agent_registry[assigned_to].name
                        agent_role = hierarchy.agent_registry[assigned_to].role
                    else:
                        agent_name = "未知"
                        agent_role = "未知"
                        agent_name = "Unknown"
                        agent_role = "Unknown"
                    
                    all_tasks[task_id] = {
                        "id": task_id,
                        "description": task.get("description", "未知任务"),
                        "description_en": task.get("description", "Unknown task"),
                        "status": task.get("status", "UNKNOWN"),
                        "dependencies": task.get("dependencies", []),
                        "agent_name": agent_name,
                        "agent_role": agent_role,
                        "parent_task": task.get("parent_task", None),
                        "subtasks": task.get("subtasks", []),
                    }
            
            # 如果没有任务，显示消息并返回
            # If no tasks, show message and return
            if not all_tasks:
                self.console.print("[yellow]系统中没有任务。[/yellow]")
                self.console.print("[yellow]No tasks in the system.[/yellow]")
                return
            
            # 创建任务树
            # Create task tree
            tree = Tree("[bold]任务树[/bold]")
            tree = Tree("[bold]Task Tree[/bold]")
            
            # 查找根任务（没有父任务的任务）
            # Find root tasks (tasks with no parent)
            root_tasks = {
                task_id: task for task_id, task in all_tasks.items()
                if not task.get("parent_task") and not (isinstance(task.get("dependencies"), list) and task.get("dependencies"))
            }
            
            # 如果找不到根任务但有任务，将所有任务用作根
            # If no root tasks found but there are tasks, use all tasks as roots
            if not root_tasks and all_tasks:
                root_tasks = all_tasks
            
            # 按ID排序根任务（为了稳定的顺序）
            # Sort root tasks by ID (for stable order)
            sorted_root_tasks = sorted(
                root_tasks.items(),
                key=lambda x: x[0]
            )
            
            # 将根任务添加到树中
            # Add root tasks to tree
            for task_id, task in sorted_root_tasks:
                task_label = self._format_task_label(task)
                task_node = tree.add(task_label)
                
                # 递归添加子任务
                # Add subtasks recursively
                self._add_subtasks(task_node, task, all_tasks)
            
            # 显示树
            # Display the tree
            self.console.print(tree)
            
        except Exception as e:
            logger.error(f"绘制任务树时出错: {str(e)}")
            logger.error(f"Error drawing task tree: {str(e)}")
            self.console.print(f"[bold red]绘制任务树时出错:[/bold red] {str(e)}")
            self.console.print(f"[bold red]Error drawing task tree:[/bold red] {str(e)}")
    
    def _add_subtasks(self, parent_node: Tree, parent_task: Dict[str, Any], all_tasks: Dict[str, Dict[str, Any]]) -> None:
        """
        递归地将子任务添加到任务节点。
        
        Recursively add subtasks to a task node.
        
        此内部方法将子任务添加到可视化树中的父任务节点。
        
        This internal method adds child tasks to a parent task node
        in the visualization tree.
        
        Args:
            parent_node: 表示父任务的树节点
                        Tree node representing the parent task
            parent_task: 包含父任务信息的字典
                        Dictionary containing parent task information
            all_tasks: 系统中所有任务的字典
                      Dictionary of all tasks in the system
        """
        # 检查父任务中定义的子任务
        # Check for subtasks defined in the parent task
        subtasks = parent_task.get("subtasks", [])
        
        # 还检查将此任务作为父任务的任务
        # Also check for tasks that have this task as parent
        child_tasks = {
            task_id: task for task_id, task in all_tasks.items()
            if task.get("parent_task") == parent_task["id"]
        }
        
        # 还检查将此任务作为依赖项的任务
        # Also check for tasks that have this task as a dependency
        dependent_tasks = {
            task_id: task for task_id, task in all_tasks.items()
            if parent_task["id"] in task.get("dependencies", [])
        }
        
        # 合并所有子任务来源
        # Combine all child task sources
        child_task_ids = set(subtasks) | set(child_tasks.keys()) | set(dependent_tasks.keys())
        
        # 按ID排序子任务（为了稳定的顺序）
        # Sort child tasks by ID (for stable order)
        sorted_child_task_ids = sorted(child_task_ids)
        
        # 将子任务添加到树中
        # Add child tasks to tree
        for child_id in sorted_child_task_ids:
            if child_id in all_tasks:
                child_task = all_tasks[child_id]
                child_label = self._format_task_label(child_task)
                child_node = parent_node.add(child_label)
                
                # 递归添加子任务（避免循环）
                # Add subtasks recursively (avoid cycles)
                if child_id != parent_task["id"]:
                    self._add_subtasks(child_node, child_task, all_tasks)
    
    def _format_task_label(self, task: Dict[str, Any]) -> Text:
        """
        为树中显示格式化任务标签。
        
        Format a task label for display in the tree.
        
        此内部方法为任务创建格式丰富的文本标签，
        包括其状态和描述。
        
        This internal method creates a richly formatted text label
        for a task, including its status and description.
        
        Args:
            task: 包含任务信息的字典
                 Dictionary containing task information
        
        Returns:
            Text: 格式化的任务标签
                 Formatted task label
        """
        status = task.get("status", "UNKNOWN")
        description = task.get("description", "未知任务")
        description_en = task.get("description_en", "Unknown task")
        
        # 选择中文或英文描述，优先使用中文
        # Choose Chinese or English description, prioritize Chinese
        display_description = description if description != "未知任务" else description_en
        
        # 设置状态颜色
        # Set status color
        status_color = {
            "PENDING": "yellow",
            "IN_PROGRESS": "blue",
            "COMPLETED": "green",
            "FAILED": "red",
        }.get(status, "white")
        
        # 创建格式化标签
        # Create formatted label
        label = Text()
        label.append("[", style="dim")
        label.append(status, style=status_color)
        label.append("] ", style="dim")
        label.append(display_description[:50])
        if len(display_description) > 50:
            label.append("...", style="dim")
        label.append(f" ({task['agent_name']})", style="dim")
        
        return label
    
    async def draw_agent_tree(self, hierarchy: AgentHierarchy) -> None:
        """
        绘制Agent层次结构的可视化表示。
        
        Draw a visual representation of the agent hierarchy.
        
        此方法创建显示Agent及其父子关系的树状可视化。
        
        This method creates a tree visualization showing agents and
        their parent-child relationships.
        
        Args:
            hierarchy: 要可视化的Agent层次结构
                      Agent hierarchy to visualize
        """
        try:
            if not hierarchy.main_agent:
                self.console.print("[yellow]在层次结构中找不到主Agent。[/yellow]")
                self.console.print("[yellow]No main agent found in hierarchy.[/yellow]")
                return
            
            # 创建Agent树
            # Create agent tree
            tree = Tree("[bold]Agent层次结构[/bold]")
            tree = Tree("[bold]Agent Hierarchy[/bold]")
            
            # 将主Agent添加为根
            # Add main agent as root
            main_agent = hierarchy.main_agent
            main_label = self._format_agent_label(main_agent.name, main_agent.role, len(main_agent.tasks))
            main_node = tree.add(main_label)
            
            # 跟踪已访问的Agent以避免循环
            # Track visited agents to avoid cycles
            visited = {main_agent.id}
            
            # 递归添加子Agent
            # Add child agents recursively
            self._add_child_agents(main_node, main_agent.id, hierarchy, visited)
            
            # 检查是否还有其他未连接到主Agent的Agent
            # Check for other agents not connected to main agent
            unvisited_agents = set(hierarchy.agent_registry.keys()) - visited
            
            if unvisited_agents:
                # 创建一个单独的部分用于未连接的Agent
                # Create a separate section for unconnected agents
                unconnected_node = tree.add("[bold yellow]未连接的Agent[/bold yellow]")
                unconnected_node = tree.add("[bold yellow]Unconnected Agents[/bold yellow]")
                
                for agent_id in sorted(unvisited_agents):
                    agent = hierarchy.agent_registry[agent_id]
                    label = self._format_agent_label(agent.name, agent.role, len(agent.tasks), disconnected=True)
                    unconnected_node.add(label)
            
            # 显示树
            # Display the tree
            self.console.print(tree)
            
        except Exception as e:
            logger.error(f"绘制Agent树时出错: {str(e)}")
            logger.error(f"Error drawing agent tree: {str(e)}")
            self.console.print(f"[bold red]绘制Agent树时出错:[/bold red] {str(e)}")
            self.console.print(f"[bold red]Error drawing agent tree:[/bold red] {str(e)}")
    
    def _add_child_agents(self, parent_node: Tree, parent_id: str, hierarchy: AgentHierarchy, visited: Set[str]) -> None:
        """
        递归地将子Agent添加到父Agent节点。
        
        Recursively add child agents to a parent agent node.
        
        此内部方法将子Agent添加到可视化树中的父Agent节点。
        
        This internal method adds child agents to a parent agent node
        in the visualization tree.
        
        Args:
            parent_node: 表示父Agent的树节点
                        Tree node representing the parent agent
            parent_id: 父Agent的ID
                      ID of the parent agent
            hierarchy: 包含Agent的Agent层次结构
                      Agent hierarchy containing the agents
            visited: 已访问Agent ID的集合（避免循环）
                    Set of already visited agent IDs (to avoid cycles)
        """
        if parent_id not in hierarchy.agent_registry:
            return
        
        parent_agent = hierarchy.agent_registry[parent_id]
        
        # 添加每个子Agent
        # Add each child agent
        for child_id, child_info in parent_agent.child_agents.items():
            # 如果已访问则跳过（避免循环）
            # Skip if already visited (avoid cycles)
            if child_id in visited:
                continue
            
            visited.add(child_id)
            
            # 如果存在，获取子Agent
            # Get child agent if it exists
            if child_id in hierarchy.agent_registry:
                child_agent = hierarchy.agent_registry[child_id]
                child_label = self._format_agent_label(
                    child_agent.name, 
                    child_agent.role, 
                    len(child_agent.tasks)
                )
                child_node = parent_node.add(child_label)
                
                # 递归添加子Agent
                # Add child agents recursively
                self._add_child_agents(child_node, child_id, hierarchy, visited)
            else:
                # 子Agent存在于父的child_agents中，但不在注册表中
                # Child agent exists in parent's child_agents but not in registry
                name = child_info.get("name", "未知")
                name_en = child_info.get("name", "Unknown")
                role = child_info.get("role", "未知")
                role_en = child_info.get("role", "Unknown")
                
                # 选择中文或英文名称和角色，优先使用中文
                # Choose Chinese or English name and role, prioritize Chinese
                display_name = name if name != "未知" else name_en
                display_role = role if role != "未知" else role_en
                
                child_label = self._format_agent_label(display_name, display_role, 0, disconnected=True)
                parent_node.add(child_label)
    
    def _format_agent_label(self, name: str, role: str, task_count: int, disconnected: bool = False) -> Text:
        """
        为树中显示格式化Agent标签。
        
        Format an agent label for display in the tree.
        
        此内部方法为Agent创建格式丰富的文本标签，
        包括其名称、角色和任务计数。
        
        This internal method creates a richly formatted text label
        for an agent, including its name, role, and task count.
        
        Args:
            name: Agent的名称
                 Name of the agent
            role: Agent的角色
                 Role of the agent
            task_count: 分配给Agent的任务数量
                       Number of tasks assigned to the agent
            disconnected: Agent是否已断开连接
                         Whether the agent is disconnected
        
        Returns:
            Text: 格式化的Agent标签
                 Formatted agent label
        """
        # 创建格式化标签
        # Create formatted label
        label = Text()
        label.append(name, style="bold")
        label.append(f" ({role})")
        
        if task_count > 0:
            label.append(f" - {task_count} 个任务", style="blue")
            label.append(f" - {task_count} tasks", style="blue")
        
        if disconnected:
            label.append(" [已断开连接]", style="red")
            label.append(" [DISCONNECTED]", style="red")
        
        return label
    
    def draw_task_table(self, tasks: List[Dict[str, Any]]) -> None:
        """
        绘制任务表格。
        
        Draw a table of tasks.
        
        此方法创建任务的表格视图，包括它们的状态、描述和分配的Agent。
        
        This method creates a tabular view of tasks, including their
        status, description, and assigned agent.
        
        Args:
            tasks: 要可视化的任务字典列表
                  List of task dictionaries to visualize
        """
        # 创建任务表格
        # Create task table
        table = Table(title="任务")
        table = Table(title="Tasks")
        table.add_column("ID", style="dim")
        table.add_column("描述", header_style="bold")
        table.add_column("Description", header_style="bold")
        table.add_column("状态", justify="center")
        table.add_column("Status", justify="center")
        table.add_column("Agent")
        table.add_column("依赖项", style="dim")
        table.add_column("Dependencies", style="dim")
        
        for task in tasks:
            status = task.get("status", "UNKNOWN")
            
            # 设置状态颜色
            # Set status color
            status_style = {
                "PENDING": "yellow",
                "IN_PROGRESS": "blue",
                "COMPLETED": "green",
                "FAILED": "red",
            }.get(status, "white")
            
            # 格式化依赖项
            # Format dependencies
            dependencies = task.get("dependencies", [])
            if isinstance(dependencies, list):
                deps_str = ", ".join(d[:8] for d in dependencies)
            else:
                deps_str = str(dependencies)
            
            # 获取中文和英文描述
            # Get Chinese and English descriptions
            description = task.get("description", "未知任务")
            description_en = task.get("description_en", "Unknown task")
            
            # 选择要显示的描述
            # Choose description to display
            display_description = description if description != "未知任务" else description_en
            
            table.add_row(
                task["id"][:8],
                display_description[:50] + ("..." if len(display_description) > 50 else ""),
                f"[{status_style}]{status}[/{status_style}]",
                f"{task['agent_name']} ({task['agent_role']})",
                deps_str
            )
        
        # 显示表格
        # Display the table
        self.console.print(table)
    
    def draw_task_result(self, task: Dict[str, Any]) -> None:
        """
        绘制显示任务结果的面板。
        
        Draw a panel showing task results.
        
        此方法创建一个面板，显示已完成任务的详细结果。
        
        This method creates a panel displaying the detailed results
        of a completed task.
        
        Args:
            task: 包含任务信息的字典
                 Dictionary containing task information
        """
        # 获取中文和英文描述
        # Get Chinese and English descriptions
        description = task.get("description", "未知任务")
        description_en = task.get("description_en", "Unknown task")
        
        # 选择要显示的描述
        # Choose description to display
        display_description = description if description != "未知任务" else description_en
        
        result = task.get("result", "无结果可用")
        result_en = task.get("result", "No result available")
        
        # 选择要显示的结果
        # Choose result to display
        display_result = result if result != "无结果可用" else result_en
        
        # 创建面板，显示任务结果
        # Create panel, displaying task result
        panel = Panel(
            display_result,
            title=f"任务: {display_description}",
            title_align="left",
            expand=False
        )
        
        self.console.print(panel)
