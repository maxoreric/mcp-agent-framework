"""
Task Dependency Management for MCP Agent Framework.

This module handles the creation and management of task dependency trees,
ensuring that tasks are executed in the correct order based on their
dependencies.
"""

from typing import Dict, List, Any, Set, Optional
import copy

def create_dependency_tree(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a dependency tree from a list of subtasks.
    
    This function analyzes the dependencies between subtasks and returns
    an ordered list where tasks only appear after their dependencies.
    
    Args:
        subtasks: List of subtask specifications with dependencies
    
    Returns:
        List[Dict[str, Any]]: Ordered list of subtasks
    """
    # Create a copy of the subtasks to avoid modifying the original
    tasks = copy.deepcopy(subtasks)
    
    # Create a mapping of task ID to task
    task_map = {task["id"]: task for task in tasks}
    
    # Create a dependency graph
    dependency_graph: Dict[str, Set[str]] = {}
    for task in tasks:
        task_id = task["id"]
        dependencies = task.get("dependencies", [])
        
        # Convert to set for easier operations
        if isinstance(dependencies, list):
            dependency_graph[task_id] = set(dependencies)
        elif isinstance(dependencies, str) and dependencies:
            # Handle case where dependencies are comma-separated in a string
            dependency_graph[task_id] = set(dep.strip() for dep in dependencies.split(","))
        else:
            dependency_graph[task_id] = set()
    
    # Detect cycles in the dependency graph
    visited: Set[str] = set()
    temp: Set[str] = set()
    
    def has_cycle(node: str) -> bool:
        """
        Check if the dependency graph has a cycle starting from node.
        
        Args:
            node: Starting node for cycle detection
        
        Returns:
            bool: True if a cycle is detected, False otherwise
        """
        if node in temp:
            return True
        if node in visited:
            return False
        
        temp.add(node)
        
        for neighbor in dependency_graph.get(node, set()):
            if neighbor in task_map and has_cycle(neighbor):
                return True
        
        temp.remove(node)
        visited.add(node)
        return False
    
    # Check for cycles
    for task_id in task_map:
        if task_id not in visited:
            if has_cycle(task_id):
                raise ValueError(f"Dependency cycle detected in subtasks")
    
    # Topological sort
    result: List[Dict[str, Any]] = []
    visited = set()
    
    def dfs(node: str) -> None:
        """
        Depth-first search for topological sorting.
        
        Args:
            node: Current node in the DFS
        """
        if node in visited:
            return
        
        visited.add(node)
        
        for neighbor in dependency_graph.get(node, set()):
            if neighbor in task_map:
                dfs(neighbor)
        
        result.append(task_map[node])
    
    # Run topological sort
    for task_id in task_map:
        if task_id not in visited:
            dfs(task_id)
    
    # Reverse the result to get correct order
    result.reverse()
    
    return result

def find_ready_tasks(tasks: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Find tasks that are ready to be executed.
    
    A task is ready if it is in the PENDING state and all its
    dependencies have been completed.
    
    Args:
        tasks: Dictionary of tasks
    
    Returns:
        List[str]: List of task IDs that are ready for execution
    """
    ready_tasks = []
    
    for task_id, task in tasks.items():
        # Skip tasks that are not in PENDING state
        if task.get("status") != "PENDING":
            continue
        
        # Check dependencies
        dependencies = task.get("dependencies", [])
        
        # Convert to list if it's a string
        if isinstance(dependencies, str) and dependencies:
            dependencies = [dep.strip() for dep in dependencies.split(",")]
        
        # Check if all dependencies are satisfied
        all_dependencies_satisfied = True
        
        for dep_id in dependencies:
            if dep_id not in tasks or tasks[dep_id].get("status") != "COMPLETED":
                all_dependencies_satisfied = False
                break
        
        if all_dependencies_satisfied:
            ready_tasks.append(task_id)
    
    return ready_tasks

def get_dependent_tasks(task_id: str, tasks: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Find tasks that depend on the given task.
    
    This function identifies all tasks that have the given task
    as a dependency.
    
    Args:
        task_id: ID of the task to check
        tasks: Dictionary of tasks
    
    Returns:
        List[str]: List of task IDs that depend on the given task
    """
    dependent_tasks = []
    
    for other_id, other_task in tasks.items():
        # Skip the task itself
        if other_id == task_id:
            continue
        
        # Check dependencies
        dependencies = other_task.get("dependencies", [])
        
        # Convert to list if it's a string
        if isinstance(dependencies, str) and dependencies:
            dependencies = [dep.strip() for dep in dependencies.split(",")]
        
        # Check if this task depends on the given task
        if task_id in dependencies:
            dependent_tasks.append(other_id)
    
    return dependent_tasks

def calculate_critical_path(tasks: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Calculate the critical path through the task dependency graph.
    
    The critical path is the sequence of tasks that determines the
    minimum time needed to complete the entire project. Delays in
    critical path tasks will delay the entire project.
    
    Args:
        tasks: Dictionary of tasks
    
    Returns:
        List[str]: List of task IDs that form the critical path
    """
    # Create a copy of the tasks to avoid modifying the original
    tasks_copy = copy.deepcopy(tasks)
    
    # Calculate earliest start and finish times
    for task_id in topological_sort(tasks_copy):
        task = tasks_copy[task_id]
        
        # Set earliest start time
        earliest_start = 0
        
        # Check dependencies
        dependencies = task.get("dependencies", [])
        
        # Convert to list if it's a string
        if isinstance(dependencies, str) and dependencies:
            dependencies = [dep.strip() for dep in dependencies.split(",")]
        
        # Find the latest finish time among dependencies
        for dep_id in dependencies:
            if dep_id in tasks_copy:
                dep_task = tasks_copy[dep_id]
                dep_finish = dep_task.get("earliest_finish", 0)
                if dep_finish > earliest_start:
                    earliest_start = dep_finish
        
        # Set earliest finish time (assuming 1 time unit per task for simplicity)
        task["earliest_start"] = earliest_start
        task["earliest_finish"] = earliest_start + 1
    
    # Find the overall project finish time
    project_finish = max(task.get("earliest_finish", 0) for task in tasks_copy.values())
    
    # Calculate latest start and finish times
    for task_id in reversed(topological_sort(tasks_copy)):
        task = tasks_copy[task_id]
        
        # Set latest finish time
        latest_finish = project_finish
        
        # Find tasks that depend on this task
        dependent_tasks = get_dependent_tasks(task_id, tasks_copy)
        
        # Find the earliest start time among dependent tasks
        for dep_id in dependent_tasks:
            dep_task = tasks_copy[dep_id]
            dep_start = dep_task.get("latest_start", project_finish)
            if dep_start < latest_finish:
                latest_finish = dep_start
        
        # Set latest start time
        task["latest_finish"] = latest_finish
        task["latest_start"] = latest_finish - 1
    
    # Calculate slack for each task
    for task_id, task in tasks_copy.items():
        earliest_start = task.get("earliest_start", 0)
        latest_start = task.get("latest_start", 0)
        task["slack"] = latest_start - earliest_start
    
    # Critical path consists of tasks with zero slack
    critical_path = [
        task_id for task_id, task in tasks_copy.items()
        if task.get("slack", 0) == 0
    ]
    
    return critical_path

def topological_sort(tasks: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Perform a topological sort of tasks based on dependencies.
    
    This function returns a list of task IDs in an order where each
    task appears after all its dependencies.
    
    Args:
        tasks: Dictionary of tasks
    
    Returns:
        List[str]: Topologically sorted list of task IDs
    """
    # Create a dependency graph
    graph: Dict[str, Set[str]] = {}
    for task_id, task in tasks.items():
        dependencies = task.get("dependencies", [])
        
        # Convert to list if it's a string
        if isinstance(dependencies, str) and dependencies:
            dependencies = [dep.strip() for dep in dependencies.split(",")]
        
        # Add to graph
        graph[task_id] = set(dep for dep in dependencies if dep in tasks)
    
    # Perform topological sort
    result: List[str] = []
    visited: Set[str] = set()
    temp: Set[str] = set()
    
    def visit(node: str) -> None:
        """
        Visit a node in the dependency graph.
        
        Args:
            node: Current node being visited
        """
        if node in temp:
            raise ValueError(f"Dependency cycle detected involving task {node}")
        if node in visited:
            return
        
        temp.add(node)
        
        for neighbor in graph.get(node, set()):
            visit(neighbor)
        
        temp.remove(node)
        visited.add(node)
        result.append(node)
    
    # Visit all nodes
    for task_id in tasks:
        if task_id not in visited:
            visit(task_id)
    
    return list(reversed(result))
