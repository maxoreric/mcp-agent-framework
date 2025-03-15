"""
Task Execution for MCP Agent Framework.

This module handles the execution of tasks by agents, including direct
execution, LLM-based processing, and result formatting.
"""

import asyncio
from typing import Dict, Any, Optional

from mcp.server.fastmcp import Context

from ..agent.agent import Agent
from ..llm.integration import process_with_llm

async def execute_task(agent: Agent, task_id: str, ctx: Context) -> None:
    """
    Execute a task directly.
    
    This function handles the execution of a single task by an agent,
    using the agent's LLM to process the task and generate a result.
    
    Args:
        agent: Agent executing the task
        task_id: ID of the task to execute
        ctx: MCP Context for progress reporting
    """
    # Check if task exists
    if task_id not in agent.tasks:
        raise ValueError(f"Task {task_id} not found in agent {agent.name}")
    
    task = agent.tasks[task_id]
    
    # Check if task is in the correct state
    if task["status"] != "PENDING":
        ctx.info(f"Task {task_id} is not in PENDING state (current: {task['status']}), skipping execution")
        return
    
    # Update task status
    task["status"] = "IN_PROGRESS"
    await agent._trigger_event("task_updated", {"id": task_id, "status": "IN_PROGRESS"})
    
    try:
        # Create XML-formatted task for LLM
        task_prompt = format_task_prompt(agent, task)
        
        ctx.info(f"Executing task: {task['description']}")
        
        # Process with LLM
        result = await process_with_llm(agent, task_prompt, ctx)
        
        # Update task with result
        task["status"] = "COMPLETED"
        task["result"] = result
        
        # Trigger event handlers
        await agent._trigger_event("task_completed", {
            "id": task_id, 
            "result": task["result"]
        })
        
        ctx.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        # Update task with error
        task["status"] = "FAILED"
        task["error"] = str(e)
        
        # Trigger event handlers
        await agent._trigger_event("task_failed", {
            "id": task_id, 
            "error": task["error"]
        })
        
        ctx.error(f"Task {task_id} failed: {str(e)}")
        raise

def format_task_prompt(agent: Agent, task: Dict[str, Any]) -> str:
    """
    Format a task prompt for LLM processing.
    
    This function creates an XML-formatted prompt that provides the
    LLM with the task description and relevant context.
    
    Args:
        agent: Agent executing the task
        task: Task details
    
    Returns:
        str: Formatted prompt for the LLM
    """
    # Extract task description
    description = task["description"]
    
    # Create prompt
    return f"""<role>{agent.role}</role>
<task>{description}</task>
<context>
As an expert in {agent.role}, analyze and solve this task.
Provide a detailed and comprehensive solution.
</context>
"""

async def execute_subtasks(agent: Agent, task_id: str, subtask_ids: list[str], ctx: Context) -> None:
    """
    Execute a set of subtasks for a main task.
    
    This function handles the execution of multiple subtasks,
    potentially in parallel where dependencies allow.
    
    Args:
        agent: Agent orchestrating the subtasks
        task_id: ID of the main task
        subtask_ids: List of subtask IDs to execute
        ctx: MCP Context for progress reporting
    """
    # Check if main task exists
    if task_id not in agent.tasks:
        raise ValueError(f"Main task {task_id} not found in agent {agent.name}")
    
    main_task = agent.tasks[task_id]
    
    # Create a dictionary of subtasks
    subtasks = {}
    for subtask_id in subtask_ids:
        if subtask_id in agent.tasks:
            subtasks[subtask_id] = agent.tasks[subtask_id]
    
    # Update main task status
    main_task["status"] = "IN_PROGRESS"
    await agent._trigger_event("task_updated", {"id": task_id, "status": "IN_PROGRESS"})
    
    ctx.info(f"Executing {len(subtasks)} subtasks for main task {task_id}")
    
    # Process subtasks based on dependencies
    from .dependency import find_ready_tasks
    
    # Keep track of completed subtasks
    completed_subtasks = set()
    
    # Process subtasks until all are complete or failed
    while len(completed_subtasks) < len(subtasks):
        # Find tasks that are ready to execute
        ready_task_ids = find_ready_tasks(subtasks)
        
        if not ready_task_ids:
            # No tasks are ready, check if there are any incomplete tasks
            incomplete_tasks = [
                st_id for st_id in subtasks
                if subtasks[st_id]["status"] not in ["COMPLETED", "FAILED"]
            ]
            
            if incomplete_tasks:
                # There are incomplete tasks but none are ready
                # This indicates a dependency issue or a stuck task
                ctx.warning(f"No subtasks ready for execution, but {len(incomplete_tasks)} tasks are incomplete")
                # Wait a bit and try again
                await asyncio.sleep(1)
                continue
            else:
                # All tasks are either completed or failed
                break
        
        # Execute ready tasks (could be done in parallel)
        execution_tasks = []
        for st_id in ready_task_ids:
            # Skip tasks that have already been completed
            if st_id in completed_subtasks:
                continue
            
            # Execute the task
            execution_tasks.append(execute_task(agent, st_id, ctx))
        
        # Wait for all tasks to complete
        if execution_tasks:
            await asyncio.gather(*execution_tasks)
        
        # Update completed_subtasks
        for st_id in ready_task_ids:
            if st_id in agent.tasks and agent.tasks[st_id]["status"] == "COMPLETED":
                completed_subtasks.add(st_id)
    
    # Gather results from subtasks
    results = []
    for st_id in subtask_ids:
        if st_id in agent.tasks and "result" in agent.tasks[st_id]:
            st = agent.tasks[st_id]
            results.append(f"Subtask: {st['description']}\nResult: {st['result']}")
    
    # Update main task with aggregated results
    if all(agent.tasks[st_id]["status"] == "COMPLETED" for st_id in subtask_ids if st_id in agent.tasks):
        main_task["status"] = "COMPLETED"
        main_task["result"] = "\n\n".join(results)
        
        # Trigger event handlers
        await agent._trigger_event("task_completed", {
            "id": task_id, 
            "result": main_task["result"]
        })
        
        ctx.info(f"Main task {task_id} completed successfully")
    else:
        # Some subtasks failed
        failed_subtasks = [
            st_id for st_id in subtask_ids
            if st_id in agent.tasks and agent.tasks[st_id]["status"] == "FAILED"
        ]
        
        main_task["status"] = "FAILED"
        main_task["error"] = f"Some subtasks failed: {', '.join(failed_subtasks)}"
        
        # Trigger event handlers
        await agent._trigger_event("task_failed", {
            "id": task_id, 
            "error": main_task["error"]
        })
        
        ctx.error(f"Main task {task_id} failed due to failed subtasks")

async def retry_failed_task(agent: Agent, task_id: str, ctx: Context) -> bool:
    """
    Retry a failed task.
    
    This function attempts to re-execute a task that previously failed.
    
    Args:
        agent: Agent that owns the task
        task_id: ID of the task to retry
        ctx: MCP Context for progress reporting
    
    Returns:
        bool: True if retry was successful, False otherwise
    """
    # Check if task exists
    if task_id not in agent.tasks:
        raise ValueError(f"Task {task_id} not found in agent {agent.name}")
    
    task = agent.tasks[task_id]
    
    # Check if task is failed
    if task["status"] != "FAILED":
        ctx.warning(f"Cannot retry task {task_id}: not in FAILED state (current: {task['status']})")
        return False
    
    ctx.info(f"Retrying failed task: {task['description']}")
    
    # Reset task status
    task["status"] = "PENDING"
    await agent._trigger_event("task_updated", {"id": task_id, "status": "PENDING"})
    
    # Clear error
    if "error" in task:
        del task["error"]
    
    # Execute the task
    try:
        await execute_task(agent, task_id, ctx)
        return True
    except Exception:
        return False
