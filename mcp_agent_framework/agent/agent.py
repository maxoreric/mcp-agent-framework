"""
Core Agent implementation for MCP Agent Framework.

MCP Agent Framework核心Agent实现。

本模块定义了Agent类，这是MCP Agent Framework的基本构建块。
每个Agent结合了MCP客户端和服务器功能，形成专业化Agent的层次结构。

This module defines the Agent class, which is the fundamental building
block of the MCP Agent Framework. Each agent combines MCP client and
server capabilities to form a hierarchical structure of specialized agents.
"""

import uuid
import asyncio
import inspect
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Set
from pydantic import BaseModel

from mcp.server.fastmcp import FastMCP, Context

# 配置模块级别日志器
# Configure module-level logger
logger = logging.getLogger(__name__)

class Task(BaseModel):
    """
    Agent可以处理的任务的数据模型。
    
    Data model for a task that can be processed by an agent.
    
    Attributes:
        id: 任务的唯一标识符
            Unique identifier for the task
        description: 任务的人类可读描述
                    Human-readable description of the task
        status: 任务的当前状态(PENDING, IN_PROGRESS, COMPLETED, FAILED)
                Current status of the task (PENDING, IN_PROGRESS, COMPLETED, FAILED)
        dependencies: 必须在此任务之前完成的任务ID列表
                      List of task IDs that must be completed before this task
        assigned_to: 分配给此任务的agent的ID
                     ID of the agent assigned to this task
        result: 任务执行的结果(如果已完成)
                Result of the task execution (if completed)
        error: 错误信息(如果失败)
               Error information (if failed)
    """
    id: str
    description: str
    status: str
    dependencies: List[str] = []
    assigned_to: str
    result: Optional[str] = None
    error: Optional[str] = None

class Agent:
    """
    使用MCP的FastMCP实现的Agent。
    
    Agent implementation using MCP's FastMCP.
    
    MCP Agent Framework中的Agent既是MCP服务器(向父Agent公开能力)，
    也是MCP客户端(连接到子Agent)。这种双重特性允许创建专业化Agent的层次结构。
    
    An Agent in the MCP Agent Framework is both an MCP server (exposing
    capabilities to parent agents) and an MCP client (connecting to child
    agents). This dual nature allows for the creation of hierarchical
    structures of specialized agents.
    
    Attributes:
        id: Agent的唯一标识符
            Unique identifier for the agent
        name: Agent的人类可读名称
              Human-readable name for the agent
        role: Agent的专业角色(例如, "CEO", "Developer", "Researcher")
              Specialized role of the agent (e.g., "CEO", "Developer", "Researcher")
        config: Agent的配置字典
                Configuration dictionary for the agent
        child_agents: 连接到此Agent的子Agent字典
                      Dictionary of child agents connected to this agent
        tasks: 分配给此Agent的任务字典
               Dictionary of tasks assigned to this agent
        mcp: 此Agent的FastMCP服务器实例
             FastMCP server instance for this agent
        event_handlers: 已注册事件处理程序的字典
                        Dictionary of registered event handlers
        processing_tasks: 当前正在处理的任务集合
                         Set of tasks currently being processed
    """
    
    def __init__(self, name: str, role: str, config: Dict[str, Any]):
        """
        初始化新的Agent实例。
        
        Initialize a new Agent instance.
        
        Args:
            name: Agent的人类可读名称
                 Human-readable name for the agent
            role: Agent的专业角色
                 Specialized role of the agent
            config: 包含API密钥和设置的配置字典
                   Configuration dictionary containing API keys and settings
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.config = config
        self.child_agents: Dict[str, Dict[str, Any]] = {}  # 子Agent连接 / Child agent connections
        self.tasks: Dict[str, Dict[str, Any]] = {}  # 任务跟踪 / Task tracking
        self.event_handlers: Dict[str, List[Callable]] = {
            "task_created": [],
            "task_updated": [],
            "task_completed": [],
            "task_failed": [],
            "agent_created": [],
            "agent_connected": [],
            "agent_disconnected": [],
        }
        self.processing_tasks: Set[str] = set()  # 跟踪正在处理的任务 / Track tasks being processed
        
        # 初始化FastMCP服务器
        # Initialize FastMCP server
        self.mcp = FastMCP(f"agent-{self.name}")
        
        # 注册MCP能力
        # Register MCP capabilities
        self._register_capabilities()
    
    def _register_capabilities(self) -> None:
        """
        使用FastMCP装饰器注册Agent能力。
        
        Register agent capabilities using FastMCP decorators.
        
        此方法设置定义Agent能力的MCP工具、资源和提示，并使它们对父Agent可用。
        
        This method sets up the MCP tools, resources, and prompts that
        define the agent's capabilities and make them available to
        parent agents.
        """
        
        @self.mcp.tool()
        async def create_child_agent(name: str, role: str, ctx: Context) -> str:
            """
            创建具有指定角色的子Agent。
            
            Create a child agent with the specified role.
            
            此工具允许Agent动态创建新的专业化Agent来处理特定类型的任务。
            
            This tool allows an agent to dynamically create a new
            specialized agent to handle specific types of tasks.
            
            Args:
                name: 新Agent的人类可读名称
                     Human-readable name for the new agent
                role: 新Agent的专业角色
                     Specialized role for the new agent
                ctx: 用于进度报告的MCP上下文
                     MCP Context for progress reporting
            
            Returns:
                str: 新创建Agent的ID
                     ID of the newly created agent
            """
            ctx.info(f"创建子Agent: {name}，角色: {role}")
            ctx.info(f"Creating child agent: {name} with role: {role}")
            
            # 在实际实现中，我们会创建一个新的Agent实例并与其建立MCP连接。
            # 现在，我们将用占位符模拟创建。
            # In a real implementation, we would create a new agent instance
            # and establish an MCP connection to it.
            # For now, we'll simulate the creation with a placeholder.
            child_id = str(uuid.uuid4())
            self.child_agents[child_id] = {
                "name": name,
                "role": role,
                "connection": None  # 在实际实现中会保存MCP客户端会话 / Would hold the MCP client session in a real implementation
            }
            
            # 触发事件处理程序
            # Trigger event handlers
            await self._trigger_event("agent_created", {"id": child_id, "name": name, "role": role})
            
            return child_id
        
        @self.mcp.tool()
        async def submit_task(description: str, ctx: Context, dependencies: List[str] = None) -> str:
            """
            向此Agent提交任务。
            
            Submit a task to this agent.
            
            此工具允许父Agent向此Agent提交任务进行处理。
            
            This tool allows parent agents to submit tasks to this agent
            for processing.
            
            Args:
                description: 任务的人类可读描述
                            Human-readable description of the task
                ctx: 用于进度报告的MCP上下文
                     MCP Context for progress reporting
                dependencies: 必须在此任务之前完成的任务ID的可选列表
                              Optional list of task IDs that must be completed before this task
            
            Returns:
                str: 创建的任务的ID
                     ID of the created task
            """
            ctx.info(f"接收到任务: {description}")
            ctx.info(f"Received task: {description}")
            
            # 创建新任务
            # Create a new task
            task_id = str(uuid.uuid4())
            deps = dependencies or []
            
            self.tasks[task_id] = {
                "id": task_id,
                "description": description,
                "status": "PENDING",
                "dependencies": deps,
                "assigned_to": self.id,
                "created_at": asyncio.get_event_loop().time()
            }
            
            # 触发事件处理程序
            # Trigger event handlers
            await self._trigger_event("task_created", {"id": task_id, "description": description})
            
            # 在实际实现中，我们会将任务排队等待处理
            # 现在，我们只返回任务ID
            # In a real implementation, we would queue the task for processing
            # For now, we'll just return the task ID
            return task_id
        
        @self.mcp.resource("task://{task_id}")
        async def get_task(task_id: str) -> str:
            """
            作为资源获取任务详情。
            
            Get task details as a resource.
            
            此资源提供对任务详情的访问，允许父Agent检查任务状态和结果。
            
            This resource provides access to task details, allowing
            parent agents to check task status and results.
            
            Args:
                task_id: 要检索的任务的ID
                        ID of the task to retrieve
            
            Returns:
                str: XML格式的任务详情
                     XML-formatted task details
            """
            if task_id not in self.tasks:
                return "<e>Task not found</e>"
            
            # 以XML格式返回任务详情
            # Return task details in XML format
            task = self.tasks[task_id]
            result = task.get('result', '')
            error = task.get('error', '')
            
            return f"""<task id="{task_id}">
    <description>{task['description']}</description>
    <status>{task['status']}</status>
    <dependencies>{','.join(task['dependencies'])}</dependencies>
    <assigned_to>{task['assigned_to']}</assigned_to>
    <r>{result}</r>
    <e>{error}</e>
</task>"""
        
        @self.mcp.prompt()
        def agent_prompt() -> str:
            """
            返回此Agent的提示模板。
            
            Return a prompt template for this agent.
            
            此提示为LLM提供了基于其专业角色与此Agent交互的指导。
            
            This prompt provides guidance to LLMs on how to interact
            with this agent based on its specialized role.
            
            Returns:
                str: XML格式的提示模板
                     XML-formatted prompt template
            """
            return f"""<role>{self.role}</role>
You are a specialized agent with expertise in {self.role}.
Your task is to assist the main system by providing expert knowledge.

When given a task, analyze it carefully and provide the best solution based on your expertise.
"""
    
    async def start(self) -> None:
        """
        启动Agent的服务。
        
        Start the agent's services.
        
        此方法初始化并启动Agent的MCP服务器，使其能力对父Agent可用。
        
        This method initializes and starts the agent's MCP server,
        making its capabilities available to parent agents.
        """
        # 在实际实现中，我们会在这里启动MCP服务器
        # 现在，我们只记录操作
        # In a real implementation, we would start the MCP server here
        # For now, we'll just log the action
        logger.info(f"启动Agent {self.name} ({self.id})，角色 {self.role}")
        logger.info(f"Starting agent {self.name} ({self.id}) with role {self.role}")
        print(f"启动Agent {self.name} ({self.id})，角色 {self.role}")
        print(f"Starting agent {self.name} ({self.id}) with role {self.role}")
    
    async def stop(self) -> None:
        """
        停止Agent的服务。
        
        Stop the agent's services.
        
        此方法优雅地关闭Agent的MCP服务器和与子Agent的客户端连接。
        
        This method gracefully shuts down the agent's MCP server and
        client connections to child agents.
        """
        # 在实际实现中，我们会停止MCP服务器并断开客户端连接
        # 现在，我们只记录操作
        # In a real implementation, we would stop the MCP server and disconnect clients
        # For now, we'll just log the action
        logger.info(f"停止Agent {self.name} ({self.id})")
        logger.info(f"Stopping agent {self.name} ({self.id})")
        print(f"停止Agent {self.name} ({self.id})")
        print(f"Stopping agent {self.name} ({self.id})")
    
    async def process_task(self, task_id: str) -> None:
        """
        处理分配给此Agent的任务。
        
        Process a task assigned to this agent.
        
        此方法处理任务的实际执行，如有必要，可能将子任务委派给子Agent。
        
        This method handles the actual execution of a task, potentially
        delegating subtasks to child agents if needed.
        
        Args:
            task_id: 要处理的任务的ID
                    ID of the task to process
                    
        Raises:
            ValueError: 如果找不到任务
                       If the task is not found
            RuntimeError: 如果任务已在处理中
                         If the task is already being processed
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务 {task_id} 未找到")
            raise ValueError(f"Task {task_id} not found")
        
        # 检查任务是否已在处理中
        # Check if task is already being processed
        if task_id in self.processing_tasks:
            logger.warning(f"任务 {task_id} 已在处理中")
            logger.warning(f"Task {task_id} is already being processed")
            raise RuntimeError(f"任务 {task_id} 已在处理中")
            raise RuntimeError(f"Task {task_id} is already being processed")
        
        # 将任务标记为正在处理
        # Mark task as being processed
        self.processing_tasks.add(task_id)
        
        task = self.tasks[task_id]
        
        # 检查所有依赖是否已完成
        # Check if all dependencies are completed
        for dep_id in task['dependencies']:
            dep_in_current_agent = False
            
            # 检查依赖是否在当前Agent中
            # Check if dependency is in current agent
            if dep_id in self.tasks:
                dep_in_current_agent = True
                if self.tasks[dep_id]['status'] != "COMPLETED":
                    # 无法立即处理此任务
                    # Cannot process this task yet
                    logger.info(f"任务 {task_id} 的依赖 {dep_id} 尚未完成")
                    logger.info(f"Dependency {dep_id} for task {task_id} not yet completed")
                    
                    # 将任务标记为不再处理中
                    # Mark task as no longer being processed
                    self.processing_tasks.remove(task_id)
                    return
            
            # 如果依赖不在当前Agent中，我们假设它是外部依赖并已满足
            # If dependency not in current agent, we assume it's an external dependency and is satisfied
            # 在实际实现中，可能需要更复杂的依赖检查
            # In a real implementation, more sophisticated dependency checking might be needed
            if not dep_in_current_agent:
                logger.warning(f"任务 {task_id} 的依赖 {dep_id} 不在当前Agent中，假设已满足")
                logger.warning(f"Dependency {dep_id} for task {task_id} not in current agent, assuming satisfied")
        
        # 更新任务状态
        # Update task status
        task['status'] = "IN_PROGRESS"
        await self._trigger_event("task_updated", {"id": task_id, "status": "IN_PROGRESS"})
        
        try:
            # 使用 LLM 实际生成内容
            # Actually generate content using LLM
            from ..llm.integration import process_with_llm
            from mcp.server.fastmcp import Context

            # 创建上下文请求
            # Create context for the request
            ctx = Context()
            
            # 根据角色准备提示
            # Prepare prompt based on role
            role_prompts = {
                "writer": "You are a skilled writer. Create a high-quality piece of writing based on the following task. Be creative and engaging.",
                "poet": "You are an accomplished poet. Write a beautiful poem based on the following task. Be artistic and evocative.",
                "lyricist": "You are a talented lyricist. Write meaningful and captivating lyrics based on the following task.",
                "researcher": "You are a thorough researcher. Provide comprehensive research results based on the following task.",
                "composer": "You are a creative composer. Describe a musical composition based on the following task.",
                "editor": "You are a precise editor. Edit and refine the content based on the following task.",
                "music producer": "You are an experienced music producer. Describe production details based on the following task.",
                "audio engineer": "You are a skilled audio engineer. Provide audio engineering details based on the following task."                
            }
            
            role_prompt = role_prompts.get(self.role.lower(), f"You are an expert {self.role}. Complete the following task with high quality results.")
            
            # 创建完整提示
            # Create complete prompt
            prompt = f"""{role_prompt}
            
Task: {task['description']}

Provide a detailed and creative response."""

            # 使用 LLM 处理提示
            # Process the prompt using LLM
            result = await process_with_llm(self, prompt, ctx)
            
            # 更新任务结果
            # Update task with result
            task['status'] = "COMPLETED"
            task['result'] = result
            
            # 触发事件处理程序
            # Trigger event handlers
            await self._trigger_event("task_completed", {
                "id": task_id, 
                "result": task['result']
            })
            
        except Exception as e:
            # 更新任务错误
            # Update task with error
            task['status'] = "FAILED"
            task['error'] = str(e)
            
            # 触发事件处理程序
            # Trigger event handlers
            await self._trigger_event("task_failed", {
                "id": task_id, 
                "error": task['error']
            })
            
            # 记录异常
            # Log the exception
            logger.error(f"处理任务 {task_id} 时出错: {str(e)}")
            logger.error(f"Error processing task {task_id}: {str(e)}")
            
            # 将任务标记为不再处理中
            # Mark task as no longer being processed
            self.processing_tasks.remove(task_id)
            
            # 重新抛出异常
            # Re-raise the exception
            raise
        
        finally:
            # 确保任务从处理集合中移除，即使出现异常
            # Ensure task is removed from processing set even if exception occurs
            if task_id in self.processing_tasks:
                self.processing_tasks.remove(task_id)
    
    def on(self, event: str, handler: Callable) -> None:
        """
        注册事件处理程序。
        
        Register an event handler.
        
        此方法允许外部组件为特定Agent事件注册回调。
        
        This method allows external components to register callbacks
        for specific agent events.
        
        Args:
            event: 要监听的事件的名称
                  Name of the event to listen for
            handler: 事件发生时执行的回调函数
                    Callback function to execute when the event occurs
                    
        Raises:
            ValueError: 如果事件未知
                       If the event is unknown
        """
        if event not in self.event_handlers:
            raise ValueError(f"未知事件: {event}")
            raise ValueError(f"Unknown event: {event}")
        
        self.event_handlers[event].append(handler)
    
    async def _trigger_event(self, event: str, data: Dict[str, Any]) -> None:
        """
        触发特定事件的事件处理程序。
        
        Trigger event handlers for a specific event.
        
        此内部方法使用提供的数据调用事件的所有已注册处理程序。
        
        This internal method calls all registered handlers for an event
        with the provided data.
        
        Args:
            event: 要触发的事件的名称
                  Name of the event to trigger
            data: 传递给事件处理程序的数据
                 Data to pass to event handlers
        """
        if event not in self.event_handlers:
            return
        
        for handler in self.event_handlers[event]:
            try:
                # 检查处理程序是否是协程函数或普通函数
                # Check if the handler is a coroutine function or a regular function
                if asyncio.iscoroutinefunction(handler) or inspect.isawaitable(handler):
                    # 对于协程函数，使用await调用
                    # For coroutine functions, use await
                    await handler(data)
                else:
                    # 对于常规函数，直接调用
                    # For regular functions, call directly
                    result = handler(data)
                    # 如果结果是协程，等待它
                    # If the result is a coroutine, await it
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as e:
                # 记录事件处理程序错误，但不中断其他处理程序
                # Log event handler errors but don't interrupt other handlers
                logger.error(f"{event} 事件处理程序错误: {str(e)}")
                logger.error(f"Error in event handler for {event}: {str(e)}")
                print(f"{event} 事件处理程序错误: {str(e)}")
                print(f"Error in event handler for {event}: {str(e)}")
