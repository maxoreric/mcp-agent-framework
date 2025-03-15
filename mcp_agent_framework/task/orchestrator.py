"""
Task Orchestration for MCP Agent Framework.

本模块定义了在分层结构的多个Agent之间协调任务执行的函数。它处理任务分解、委派和结果聚合。

This module defines functions for orchestrating task execution across
multiple agents in a hierarchical structure. It handles task decomposition,
delegation, and result aggregation.
"""

import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional, Set

from mcp.server.fastmcp import Context

from ..agent.agent import Agent
from ..agent.hierarchy import AgentHierarchy
from ..llm.integration import process_with_llm

# 配置模块级别日志器
# Configure module-level logger
logger = logging.getLogger(__name__)

async def analyze_task_complexity(agent: Agent, task_description: str, ctx: Context) -> str:
    """
    分析任务复杂度以确定处理策略。
    
    Analyze the complexity of a task to determine processing strategy.
    
    此函数使用Agent的LLM评估任务是否足够简单可以直接处理，或者是否复杂到需要分解为子任务。
    
    This function uses the agent's LLM to assess whether a task is simple
    enough to handle directly or complex enough to require decomposition
    into subtasks.
    
    Args:
        agent: 分析任务的Agent
               Agent analyzing the task
        task_description: 待分析任务的描述
                         Description of the task to analyze
        ctx: MCP上下文，用于进度报告
             MCP Context for progress reporting
    
    Returns:
        str: 复杂度评估结果 ("simple" 或 "complex")
             Complexity assessment ("simple" or "complex")
    """
    # 创建用于复杂度分析的prompt
    # Create prompt for complexity analysis
    prompt = f"""<role>{agent.role}</role>
<task>Analyze the complexity of the following task</task>
<task_description>{task_description}</task_description>

Classify this task as either "simple" or "complex".
- A "simple" task can be completed in a single step without breaking it down further.
- A "complex" task requires multiple steps or specialized knowledge from different domains.

Respond with just one word: either "simple" or "complex".
"""
    
    await ctx.info("分析任务复杂度...")
    await ctx.info("Analyzing task complexity...")
    
    # 使用LLM处理
    # Process with LLM
    result = await process_with_llm(agent, prompt, ctx)
    
    # 解析结果（simple或complex）
    # Parse result (simple or complex)
    if "complex" in result.lower():
        await ctx.info("任务被分类为复杂，将分解为子任务")
        await ctx.info("Task classified as complex, will decompose into subtasks")
        return "complex"
    else:
        await ctx.info("任务被分类为简单，将直接执行")
        await ctx.info("Task classified as simple, will execute directly")
        return "simple"

async def decompose_task(agent: Agent, task_description: str, ctx: Context) -> List[Dict[str, Any]]:
    """
    将复杂任务分解为更小的子任务。
    
    Decompose a complex task into smaller subtasks.
    
    此函数使用Agent的LLM将复杂任务分解为一系列更小、更易管理的子任务，并建立它们之间的依赖关系。
    
    This function uses the agent's LLM to break down a complex task
    into a series of smaller, more manageable subtasks with dependencies.
    
    Args:
        agent: 分解任务的Agent
               Agent decomposing the task
        task_description: 待分解任务的描述
                         Description of the task to decompose
        ctx: MCP上下文，用于进度报告
             MCP Context for progress reporting
    
    Returns:
        List[Dict[str, Any]]: 子任务规范列表
                             List of subtask specifications
    """
    # 创建用于任务分解的prompt
    # Create prompt for task decomposition
    prompt = f"""<role>{agent.role}</role>
<task>Decompose the following task into smaller subtasks</task>
<task_description>{task_description}</task_description>

Break down this task into 2-5 sequential subtasks. For each subtask, provide:
1. A brief description
2. The specialized role best suited to handle it (e.g., developer, researcher, writer)
3. Any dependencies on other subtasks (by subtask number)

Format your response as a structured list of subtasks like this:
<subtasks>
  <subtask id="1">
    <description>Research existing solutions</description>
    <role>researcher</role>
    <dependencies></dependencies>
  </subtask>
  <subtask id="2">
    <description>Develop prototype</description>
    <role>developer</role>
    <dependencies>1</dependencies>
  </subtask>
</subtasks>
"""
    
    await ctx.info("将任务分解为子任务...")
    await ctx.info("Decomposing task into subtasks...")
    
    # 使用LLM处理
    # Process with LLM
    result = await process_with_llm(agent, prompt, ctx)
    
    # 解析结果（XML格式的子任务）
    # Parse result (XML-formatted subtasks)
    # 注意：在实际实现中，我们应使用正确的XML解析器
    # Note: In a real implementation, we would use a proper XML parser
    
    # 解析XML中的子任务（简化的解析器）
    # Parse subtasks from the XML (simplified parser)
    subtasks = []
    
    # 简单的解析器，用于提取标签之间的内容
    # Simple parser that extracts content between tags
    # 这是一个非常基础的实现，在生产环境中应替换为正确的XML解析器
    # This is a very basic implementation and would need to be replaced with a proper XML parser
    def extract_between_tags(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return ""
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return ""
        return text[start:end].strip()
    
    # 提取每个子任务
    # Extract each subtask
    subtask_blocks = result.split("<subtask ")
    for block in subtask_blocks[1:]:  # 跳过第一个分割（在任何子任务之前）/ Skip the first split which is before any subtask
        id_part = block.split(">", 1)[0].strip()
        subtask_id = id_part.split("=", 1)[1].strip('"\'') if "=" in id_part else str(len(subtasks) + 1)
        
        subtask_text = block.split(">", 1)[1] if ">" in block else block
        
        description = extract_between_tags(subtask_text, "description")
        role = extract_between_tags(subtask_text, "role")
        dependencies_str = extract_between_tags(subtask_text, "dependencies")
        
        # 解析依赖关系
        # Parse dependencies
        dependencies = []
        if dependencies_str:
            dependencies = [dep.strip() for dep in dependencies_str.split(",")]
        
        # 创建子任务对象
        # Create subtask object
        subtask = {
            "id": subtask_id,
            "description": description,
            "role": role.lower(),
            "dependencies": dependencies
        }
        
        subtasks.append(subtask)
    
    # 日志记录分解结果
    # Log decomposition result
    await ctx.info(f"已分解为 {len(subtasks)} 个子任务")
    await ctx.info(f"Decomposed into {len(subtasks)} subtasks")
    
    # 验证子任务的合法性：确保不存在循环依赖
    # Validate subtask legality: ensure no cyclic dependencies
    if not await _validate_subtask_dependencies(subtasks, ctx):
        await ctx.warning("检测到子任务中存在循环依赖，将重置所有依赖")
        await ctx.warning("Detected cyclic dependencies in subtasks, resetting all dependencies")
        # 重置所有依赖关系
        # Reset all dependencies
        for i, subtask in enumerate(subtasks):
            if i == 0:
                subtask["dependencies"] = []
            else:
                # 只依赖前一个任务
                # Only depend on the previous task
                subtask["dependencies"] = [subtasks[i-1]["id"]]
    
    return subtasks

async def _validate_subtask_dependencies(subtasks: List[Dict[str, Any]], ctx: Context) -> bool:
    """
    验证子任务依赖关系，确保没有循环依赖。
    
    Validate subtask dependencies to ensure no cyclic dependencies exist.
    
    Args:
        subtasks: 要验证的子任务列表
                 List of subtasks to validate
        ctx: MCP上下文，用于日志记录
             MCP Context for logging
    
    Returns:
        bool: 如果依赖关系有效则为True，否则为False
              True if dependencies are valid, False otherwise
    """
    # 创建任务ID到索引的映射
    # Create mapping from task ID to index
    id_to_index = {subtask["id"]: i for i, subtask in enumerate(subtasks)}
    
    # 检查每个子任务的依赖
    # Check dependencies for each subtask
    for i, subtask in enumerate(subtasks):
        # 检查依赖的任务是否存在
        # Check if dependent tasks exist
        for dep_id in subtask["dependencies"]:
            if dep_id not in id_to_index:
                await ctx.warning(f"子任务 {subtask['id']} 依赖不存在的任务 {dep_id}")
                await ctx.warning(f"Subtask {subtask['id']} depends on non-existent task {dep_id}")
                return False
    
    # 检查循环依赖
    # Check for cyclic dependencies
    def has_cycle(task_index, visited, rec_stack):
        # 标记当前节点为已访问并加入递归栈
        # Mark current node as visited and add to recursion stack
        visited[task_index] = True
        rec_stack[task_index] = True
        
        # 递归检查所有依赖于此任务的任务
        # Recursively check all tasks that depend on this one
        task_id = subtasks[task_index]["id"]
        for i, subtask in enumerate(subtasks):
            if task_id in subtask["dependencies"]:
                if not visited[i]:
                    if has_cycle(i, visited, rec_stack):
                        return True
                elif rec_stack[i]:
                    return True
        
        # 从递归栈中移除当前节点
        # Remove current node from recursion stack
        rec_stack[task_index] = False
        return False
    
    # 对每个未访问的节点检查循环
    # Check cycles for each unvisited node
    visited = [False] * len(subtasks)
    rec_stack = [False] * len(subtasks)
    for i in range(len(subtasks)):
        if not visited[i]:
            if has_cycle(i, visited, rec_stack):
                await ctx.warning(f"检测到子任务之间存在循环依赖")
                await ctx.warning(f"Detected cyclic dependency between subtasks")
                return False
    
    return True

async def find_agent_for_task(hierarchy: AgentHierarchy, task_description: str, role: str, ctx: Context) -> Optional[Agent]:
    """
    根据角色为特定任务查找合适的Agent。
    
    Find an appropriate agent for a specific task based on role.
    
    此函数搜索具有所需角色的现有Agent，如果必要，创建新Agent。
    
    This function searches for an existing agent with the required role,
    or creates a new agent if necessary.
    
    Args:
        hierarchy: 要搜索的Agent层次结构
                  Agent hierarchy to search
        task_description: 任务描述
                         Description of the task
        role: Agent所需的角色
              Required role for the agent
        ctx: MCP上下文，用于进度报告
             MCP Context for progress reporting
    
    Returns:
        Agent: 选定的Agent，如果找不到合适的Agent则为None
              The selected agent, or None if no suitable agent could be found
    """
    # 首先，检查主Agent是否具有所需角色
    # First, check if the main agent has the required role
    if hierarchy.main_agent and hierarchy.main_agent.role.lower() == role.lower():
        await ctx.info(f"主Agent具有所需角色 '{role}'")
        await ctx.info(f"Main agent has the required role '{role}'")
        return hierarchy.main_agent
    
    # 接下来，查找具有所需角色的现有Agent
    # Next, look for existing agents with the required role
    for agent_id, agent in hierarchy.agent_registry.items():
        if agent.role.lower() == role.lower():
            await ctx.info(f"找到现有的'{role}'角色Agent: {agent.name}")
            await ctx.info(f"Found existing agent with role '{role}': {agent.name}")
            return agent
    
    # 如果不存在合适的Agent，在主Agent下创建一个新Agent
    # If no suitable agent exists, create a new one under the main agent
    await ctx.info(f"未找到'{role}'角色的现有Agent，创建新Agent")
    await ctx.info(f"No existing agent found for role '{role}', creating a new one")
    
    if not hierarchy.main_agent:
        await ctx.error("无法创建新Agent：不存在主Agent")
        await ctx.error("Cannot create new agent: No main agent exists")
        return None
    
    # 创建具有所需角色的新Agent
    # Create a new agent with the required role
    new_agent = await hierarchy.create_child_agent(
        hierarchy.main_agent.id,
        f"{role.capitalize()} Agent",
        role
    )
    
    if not new_agent:
        await ctx.error(f"创建'{role}'角色的新Agent失败")
        await ctx.error(f"Failed to create new agent with role '{role}'")
        return None
    
    await ctx.info(f"已创建新的'{role}'角色Agent: {new_agent.name}")
    await ctx.info(f"Created new agent with role '{role}': {new_agent.name}")
    return new_agent

async def process_subtasks(
    hierarchy: AgentHierarchy, 
    main_task_id: str, 
    subtasks: List[Dict[str, Any]], 
    ctx: Context
) -> Dict[str, str]:
    """
    处理一组子任务，考虑它们之间的依赖关系。
    
    Process a group of subtasks, considering dependencies between them.
    
    Args:
        hierarchy: Agent层次结构
                  Agent hierarchy
        main_task_id: 主任务ID
                     ID of the main task
        subtasks: 要处理的子任务列表
                 List of subtasks to process
        ctx: MCP上下文，用于进度报告
             MCP Context for progress reporting
             
    Returns:
        Dict[str, str]: 子任务ID到结果的映射
                       Mapping from subtask ID to results
    """
    # 子任务ID到Agent任务ID的映射
    # Mapping from subtask ID to agent task ID
    subtask_to_agent_task = {}
    
    # 子任务ID到Agent的映射
    # Mapping from subtask ID to agent
    subtask_agents = {}
    
    # 子任务结果
    # Subtask results
    results = {}
    
    # 待处理的子任务集合
    # Set of pending subtasks
    pending_subtasks = set(subtask["id"] for subtask in subtasks)
    
    # 准备工作：为每个子任务分配Agent并创建任务
    # Preparation: Assign agents and create tasks for each subtask
    for subtask in subtasks:
        # 查找或创建适合此子任务的Agent
        # Find or create an agent suitable for this subtask
        agent = await find_agent_for_task(
            hierarchy, 
            subtask["description"], 
            subtask["role"], 
            ctx
        )
        
        if not agent:
            await ctx.error(f"无法为角色'{subtask['role']}'找到或创建Agent")
            await ctx.error(f"Could not find or create agent for role '{subtask['role']}'")
            # 标记为失败，但继续其他子任务
            # Mark as failed but continue with other subtasks
            results[subtask["id"]] = "Error: Could not find suitable agent"
            pending_subtasks.remove(subtask["id"])
            continue
        
        # 创建子任务ID
        # Create subtask ID
        agent_task_id = str(uuid.uuid4())
        
        # 记录映射关系
        # Record mappings
        subtask_to_agent_task[subtask["id"]] = agent_task_id
        subtask_agents[subtask["id"]] = agent
        
        # 在目标Agent中创建任务
        # Create a task in the target agent
        agent.tasks[agent_task_id] = {
            "id": agent_task_id,
            "description": subtask["description"],
            "status": "PENDING",
            "dependencies": subtask["dependencies"],  # 直接存储子任务依赖关系标识符
            "assigned_to": agent.id,
            "created_at": asyncio.get_event_loop().time(),
            "parent_task": main_task_id,
            "subtask_id": subtask["id"]  # 添加对原始子任务ID的引用
        }
    
    # 处理子任务，直到所有子任务完成
    # Process subtasks until all are completed
    while pending_subtasks:
        # 跟踪此轮次处理的子任务
        # Track subtasks processed in this round
        processed_this_round = set()
        
        # Create a copy of pending_subtasks to iterate over to avoid set size changes during iteration
        pending_this_round = list(pending_subtasks)
        
        # 检查每个待处理的子任务
        # Check each pending subtask
        for subtask_id in pending_this_round:
            # 跳过已处理的子任务
            # Skip already processed subtasks
            if subtask_id in processed_this_round:
                continue
                
            # 获取对应的子任务信息
            # Get corresponding subtask info
            subtask = next((st for st in subtasks if st["id"] == subtask_id), None)
            if not subtask:
                await ctx.error(f"未找到子任务 {subtask_id}")
                await ctx.error(f"Subtask {subtask_id} not found")
                pending_subtasks.remove(subtask_id)
                continue
                
            # 获取Agent和Agent任务ID
            # Get agent and agent task ID
            agent = subtask_agents.get(subtask_id)
            agent_task_id = subtask_to_agent_task.get(subtask_id)
            
            if not agent or not agent_task_id:
                await ctx.error(f"子任务 {subtask_id} 缺少Agent或任务ID")
                await ctx.error(f"Subtask {subtask_id} missing agent or task ID")
                pending_subtasks.remove(subtask_id)
                continue
                
            # 检查所有依赖是否已完成
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in subtask["dependencies"]:
                # 检查依赖的子任务是否已处理完成
                # Check if dependent subtask has been processed
                if dep_id in pending_subtasks:
                    dependencies_met = False
                    break
                    
                # 检查依赖任务是否失败
                # Check if dependent task failed
                if dep_id in results and results[dep_id].startswith("Error:"):
                    await ctx.warning(f"子任务 {subtask_id} 的依赖任务 {dep_id} 失败")
                    await ctx.warning(f"Dependent task {dep_id} for subtask {subtask_id} failed")
                    agent.tasks[agent_task_id]["status"] = "FAILED"
                    agent.tasks[agent_task_id]["error"] = f"Dependent task failed: {results[dep_id]}"
                    results[subtask_id] = f"Error: Dependency {dep_id} failed"
                    pending_subtasks.remove(subtask_id)
                    processed_this_round.add(subtask_id)
                    dependencies_met = False
                    break
            
            # 如果依赖未满足，跳过此子任务
            # If dependencies not met, skip this subtask
            if not dependencies_met:
                continue
                
            # 处理子任务
            # Process the subtask
            await ctx.info(f"处理子任务: {subtask['description']}")
            await ctx.info(f"Processing subtask: {subtask['description']}")
            
            try:
                # 处理任务
                # Process the task
                await agent.process_task(agent_task_id)
                
                # 存储结果
                # Store result
                task = agent.tasks[agent_task_id]
                if task.get("status") == "COMPLETED":
                    results[subtask_id] = task.get("result", "Task completed with no result")
                else:
                    results[subtask_id] = f"Error: {task.get('error', 'Unknown error')}"
                    
                # 标记为已处理
                # Mark as processed
                pending_subtasks.remove(subtask_id)
                processed_this_round.add(subtask_id)
                
            except Exception as e:
                # 处理异常
                # Handle exceptions
                error_msg = str(e)
                await ctx.error(f"处理子任务 {subtask_id} 时出错: {error_msg}")
                await ctx.error(f"Error processing subtask {subtask_id}: {error_msg}")
                
                # 更新任务状态
                # Update task status
                agent.tasks[agent_task_id]["status"] = "FAILED"
                agent.tasks[agent_task_id]["error"] = error_msg
                
                # 存储错误结果
                # Store error result
                results[subtask_id] = f"Error: {error_msg}"
                
                # 标记为已处理
                # Mark as processed
                pending_subtasks.remove(subtask_id)
                processed_this_round.add(subtask_id)
        
        # 如果这一轮没有处理任何子任务，可能存在死锁
        # If no subtasks were processed in this round, potential deadlock
        if not processed_this_round:
            await ctx.warning("子任务处理停滞，可能存在死锁")
            await ctx.warning("Subtask processing stalled, possible deadlock")
            
            # 强制处理其中一个待处理的子任务
            # Force process one of the pending subtasks
            if pending_subtasks:
                stuck_id = next(iter(pending_subtasks))
                await ctx.warning(f"强制处理子任务 {stuck_id}，忽略依赖")
                await ctx.warning(f"Forcing processing of subtask {stuck_id}, ignoring dependencies")
                
                agent = subtask_agents.get(stuck_id)
                agent_task_id = subtask_to_agent_task.get(stuck_id)
                
                if agent and agent_task_id:
                    # 修改任务，清除依赖
                    # Modify task, clear dependencies
                    agent.tasks[agent_task_id]["dependencies"] = []
                    
                    try:
                        # 处理任务
                        # Process the task
                        await agent.process_task(agent_task_id)
                        
                        # 存储结果
                        # Store result
                        task = agent.tasks[agent_task_id]
                        if task.get("status") == "COMPLETED":
                            results[stuck_id] = task.get("result", "Task completed with no result")
                        else:
                            results[stuck_id] = f"Error: {task.get('error', 'Unknown error')}"
                            
                    except Exception as e:
                        # 处理异常
                        # Handle exceptions
                        error_msg = str(e)
                        results[stuck_id] = f"Error: {error_msg}"
                    
                    # 标记为已处理
                    # Mark as processed
                    pending_subtasks.remove(stuck_id)
                else:
                    # 如果找不到Agent或任务ID，直接移除
                    # If agent or task ID not found, simply remove
                    await ctx.error(f"无法强制处理子任务 {stuck_id}，缺少Agent或任务ID")
                    await ctx.error(f"Cannot force process subtask {stuck_id}, missing agent or task ID")
                    pending_subtasks.remove(stuck_id)
            else:
                # 不应该发生：没有待处理的子任务但循环仍在继续
                # Should not happen: no pending subtasks but loop continues
                break
        
        # 如果还有待处理的子任务，稍作等待
        # If there are still pending subtasks, wait a bit
        if pending_subtasks:
            await asyncio.sleep(0.1)
    
    return results

async def orchestrate_task(hierarchy: AgentHierarchy, task_description: str, ctx: Context) -> str:
    """
    使用MCP工具和资源编排任务。
    
    Orchestrate a task using MCP tools and resources.
    
    此函数处理任务的完整生命周期，从分析到执行，可能将子任务委派给专门的Agent。
    
    This function handles the complete lifecycle of a task, from analysis
    to execution, potentially delegating subtasks to specialized agents.
    
    Args:
        hierarchy: 用于任务执行的Agent层次结构
                  Agent hierarchy for task execution
        task_description: 待编排任务的描述
                         Description of the task to orchestrate
        ctx: MCP上下文，用于进度报告
             MCP Context for progress reporting
    
    Returns:
        str: 编排任务的ID
             ID of the orchestrated task
    """
    # 确保我们有一个主Agent
    # Ensure we have a main agent
    if not hierarchy.main_agent:
        raise ValueError("无法编排任务：不存在主Agent")
        raise ValueError("Cannot orchestrate task: No main agent exists")
    
    # 创建任务ID
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # 分析任务复杂度
    # Analyze task complexity
    complexity = await analyze_task_complexity(hierarchy.main_agent, task_description, ctx)
    
    main_agent = hierarchy.main_agent
    
    if complexity == "simple":
        # 直接在主Agent上执行
        # Execute directly on the main agent
        await ctx.info("直接在主Agent上执行简单任务")
        await ctx.info("Executing simple task directly on main agent")
        
        # 在主Agent中创建任务
        # Create a task in the main agent
        main_agent.tasks[task_id] = {
            "id": task_id,
            "description": task_description,
            "status": "PENDING",
            "dependencies": [],
            "assigned_to": main_agent.id,
            "created_at": asyncio.get_event_loop().time()
        }
        
        # 处理任务
        # Process the task
        await main_agent.process_task(task_id)
        
    else:
        # 分解为子任务
        # Decompose into subtasks
        subtasks = await decompose_task(main_agent, task_description, ctx)
        
        # 创建主任务以跟踪整体进度
        # Create the main task to track overall progress
        main_agent.tasks[task_id] = {
            "id": task_id,
            "description": task_description,
            "status": "IN_PROGRESS",
            "dependencies": [],
            "assigned_to": main_agent.id,
            "created_at": asyncio.get_event_loop().time(),
            "subtasks": [st["id"] for st in subtasks],
            "completed_subtasks": []
        }
        
        # 处理子任务
        # Process subtasks
        subtask_results = await process_subtasks(hierarchy, task_id, subtasks, ctx)
        
        # 检查是否所有子任务都成功完成
        # Check if all subtasks completed successfully
        all_succeeded = all(not result.startswith("Error:") for result in subtask_results.values())
        
        # 更新主任务，聚合结果
        # Update main task with aggregated results
        if all_succeeded:
            main_agent.tasks[task_id]["status"] = "COMPLETED"
            main_agent.tasks[task_id]["result"] = "\n\n".join([
                f"子任务: {next((st['description'] for st in subtasks if st['id'] == st_id), 'Unknown subtask')}"
                f"\n结果: {result}"
                for st_id, result in subtask_results.items()
            ])
        else:
            # 如果任何子任务失败，标记主任务为失败
            # If any subtask failed, mark the main task as failed
            failed_subtasks = [st_id for st_id, result in subtask_results.items() if result.startswith("Error:")]
            main_agent.tasks[task_id]["status"] = "FAILED"
            main_agent.tasks[task_id]["error"] = f"以下子任务失败: {', '.join(failed_subtasks)}"
            
            # 仍然提供所有结果作为参考
            # Still provide all results for reference
            main_agent.tasks[task_id]["result"] = "\n\n".join([
                f"子任务: {next((st['description'] for st in subtasks if st['id'] == st_id), 'Unknown subtask')}"
                f"\n状态: {'失败' if result.startswith('Error:') else '成功'}"
                f"\n结果: {result}"
                for st_id, result in subtask_results.items()
            ])
    
    return task_id
