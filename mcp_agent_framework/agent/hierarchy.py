"""
Agent Hierarchy management for MCP Agent Framework.

MCP Agent Framework的Agent层次结构管理。

本模块定义了AgentHierarchy类，负责管理Agent的树状结构、它们的创建以及它们之间的连接。

This module defines the AgentHierarchy class, which is responsible for
managing the tree-like structure of agents, their creation, and the
connections between them.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .agent import Agent

# 配置模块级别日志器
# Configure module-level logger
logger = logging.getLogger(__name__)

class AgentHierarchy:
    """
    分层Agent结构的管理器。
    
    Manager for hierarchical agent structure.
    
    AgentHierarchy类管理树状结构中Agent的创建和互连。
    它维护所有活动Agent的注册表并处理它们的生命周期。
    
    The AgentHierarchy class manages the creation and interconnection
    of agents in a tree-like structure. It maintains a registry of all
    active agents and handles their lifecycle.
    
    Attributes:
        config: 层次结构的配置字典
               Configuration dictionary for the hierarchy
        main_agent: 根/主Agent(CEO)的引用
                   Reference to the root/main agent (CEO)
        agent_registry: 将Agent ID映射到Agent实例的字典
                       Dictionary mapping agent IDs to agent instances
        connections: 在Agent之间活动的MCP连接集合
                    Set of active MCP connections between agents
        _connection_by_agent_pair: 跟踪Agent间连接的字典
                                  Dictionary to track connections between agent pairs
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化新的AgentHierarchy实例。
        
        Initialize a new AgentHierarchy instance.
        
        Args:
            config: 包含API密钥和设置的配置字典
                   Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.main_agent: Optional[Agent] = None
        self.agent_registry: Dict[str, Agent] = {}
        self.connections: Set[Any] = set()  # 跟踪活动连接，用于清理 / Track active connections for cleanup
        self._connection_by_agent_pair: Dict[Tuple[str, str], Any] = {}  # 跟踪从父到子的连接 / Track connections from parent to child
    
    async def initialize(self) -> None:
        """
        用主Agent初始化Agent层次结构。
        
        Initialize the agent hierarchy with a main agent.
        
        此方法创建根Agent(CEO)并准备层次结构以进行操作。
        
        This method creates the root agent (CEO) and prepares the
        hierarchy for operation.
        """
        # 用CEO角色创建主Agent
        # Create main agent with CEO role
        self.main_agent = await self._create_agent("Main", "CEO", parent_id=None)
        
        # 启动主Agent
        # Start the main agent
        await self.main_agent.start()
        
        logger.info("已初始化Agent层次结构，并创建了主Agent")
        logger.info("Agent hierarchy initialized, and main agent created")
    
    async def shutdown(self) -> None:
        """
        优雅地关闭Agent层次结构。
        
        Gracefully shut down the agent hierarchy.
        
        此方法停止所有Agent并清理资源。
        
        This method stops all agents and cleans up resources.
        """
        # 按反向创建顺序停止所有Agent(先停子Agent)
        # Stop all agents in reverse creation order (children first)
        agent_ids = list(self.agent_registry.keys())
        for agent_id in reversed(agent_ids):
            agent = self.agent_registry[agent_id]
            await agent.stop()
        
        # 关闭所有连接
        # Close all connections
        for connection in self.connections:
            if hasattr(connection, 'close') and callable(connection.close):
                try:
                    await connection.close()
                except Exception as e:
                    logger.error(f"关闭连接时出错: {str(e)}")
                    logger.error(f"Error closing connection: {str(e)}")
        
        # 清理注册表
        # Clear registry
        self.agent_registry = {}
        self.main_agent = None
        self.connections = set()
        self._connection_by_agent_pair = {}
        
        logger.info("已关闭Agent层次结构")
        logger.info("Agent hierarchy shut down")
    
    async def _create_agent(self, name: str, role: str, parent_id: Optional[str] = None) -> Agent:
        """
        创建Agent并将其注册到层次结构中。
        
        Create an agent and register it in the hierarchy.
        
        此内部方法处理新Agent的创建，并在需要时建立父子关系。
        
        This internal method handles the creation of new agents and
        establishes the parent-child relationship if needed.
        
        Args:
            name: 新Agent的人类可读名称
                 Human-readable name for the new agent
            role: 新Agent的专业角色
                 Specialized role for the new agent
            parent_id: 可选的父Agent的ID
                      Optional ID of the parent agent
        
        Returns:
            Agent: 新创建的Agent实例
                  The newly created agent instance
        """
        # 创建Agent配置(从层次结构配置继承)
        # Create agent configuration (inherit from hierarchy config)
        agent_config = {
            "name": name,
            "role": role,
            "api_key": self.config.get("api_key"),
            "model": self.config.get("model", "gpt-4"),
            # 根据需要添加更多配置
            # Add more configuration as needed
        }
        
        # 从配置中复制所有键，以确保完整继承
        # Copy all keys from config to ensure complete inheritance
        for key, value in self.config.items():
            if key not in agent_config:
                agent_config[key] = value
        
        # 创建Agent实例
        # Create agent instance
        agent = Agent(name, role, agent_config)
        
        # 注册Agent
        # Register agent
        self.agent_registry[agent.id] = agent
        
        # 如果需要，连接到父Agent
        # Connect to parent if needed
        if parent_id and parent_id in self.agent_registry:
            parent = self.agent_registry[parent_id]
            await self._connect_parent_child(parent, agent)
        
        return agent
    
    async def _connect_parent_child(self, parent: Agent, child: Agent) -> None:
        """
        在父Agent和子Agent之间建立MCP连接。
        
        Establish MCP connection between parent and child agents.
        
        此方法从父Agent到子Agent创建MCP客户端连接，
        使父Agent能够访问子Agent的能力。
        
        This method creates an MCP client connection from the parent to
        the child, enabling the parent to access the child's capabilities.
        
        Args:
            parent: 父Agent
                   The parent agent
            child: 子Agent
                  The child agent
                  
        Raises:
            RuntimeError: 如果无法建立连接
                         If the connection cannot be established
        """
        # 实际实现MCP连接
        # Actually implement MCP connection
        try:
            # 在实际实现中，我们会：
            # In a real implementation, we would:
            # 1. 启动子Agent的MCP服务器
            #    Start the child's MCP server
            # 2. 创建父Agent的MCP客户端连接到子Agent
            #    Create an MCP client in the parent to connect to the child
            # 3. 注册连接以便清理
            #    Register the connection for cleanup
            
            # 目前，我们使用占位符模拟连接
            # For now, we're simulating the connection with a placeholder
            
            # 将子Agent添加到父Agent的子Agent列表
            # Add child agent to parent's child_agents list
            parent.child_agents[child.id] = {
                "name": child.name,
                "role": child.role,
                "connection": None  # 在实际实现中会保存MCP客户端会话 / Would hold the MCP client session in a real implementation
            }
            
            # 模拟连接对象，用于跟踪
            # Simulate a connection object for tracking
            connection = (parent.id, child.id)  # Use tuple instead of dict for hashability
            
            # 存储连接以便清理
            # Store connection for cleanup
            self.connections.add(connection)
            
            # 存储连接对应的Agent对
            # Store the connection for this agent pair
            self._connection_by_agent_pair[(parent.id, child.id)] = connection
            
            logger.info(f"已连接父Agent {parent.name} 到子Agent {child.name}")
            logger.info(f"Connected parent agent {parent.name} to child agent {child.name}")
            print(f"已连接父Agent {parent.name} 到子Agent {child.name}")
            print(f"Connected parent agent {parent.name} to child agent {child.name}")
            
        except Exception as e:
            error_msg = f"连接Agent {parent.name} 到 {child.name} 时出错: {str(e)}"
            error_msg_en = f"Error connecting agent {parent.name} to {child.name}: {str(e)}"
            logger.error(error_msg)
            logger.error(error_msg_en)
            print(error_msg)
            print(error_msg_en)
            raise RuntimeError(error_msg_en)
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        通过ID获取Agent。
        
        Get an agent by ID.
        
        Args:
            agent_id: 要检索的Agent的ID
                     ID of the agent to retrieve
        
        Returns:
            Agent: 请求的Agent实例，如果未找到则为None
                  The requested agent instance, or None if not found
        """
        return self.agent_registry.get(agent_id)
    
    async def create_child_agent(self, parent_id: str, name: str, role: str) -> Optional[Agent]:
        """
        在指定的父Agent下创建子Agent。
        
        Create a child agent under a specified parent.
        
        此方法创建具有给定名称和角色的新Agent，
        并与指定的父Agent建立父子关系。
        
        This method creates a new agent with the given name and role,
        and establishes a parent-child relationship with the specified parent.
        
        Args:
            parent_id: 父Agent的ID
                      ID of the parent agent
            name: 新Agent的人类可读名称
                 Human-readable name for the new agent
            role: 新Agent的专业角色
                 Specialized role for the new agent
        
        Returns:
            Agent: 新创建的Agent实例，如果未找到父Agent则为None
                  The newly created agent instance, or None if the parent was not found
        """
        parent = await self.get_agent(parent_id)
        if not parent:
            logger.error(f"创建子Agent失败: 未找到ID为 {parent_id} 的父Agent")
            logger.error(f"Failed to create child agent: Parent agent with ID {parent_id} not found")
            return None
        
        # 创建子Agent
        # Create the child agent
        child = await self._create_agent(name, role, parent_id=parent_id)
        
        # 启动子Agent
        # Start the child agent
        await child.start()
        
        return child
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """
        销毁Agent并将其从层次结构中移除。
        
        Destroy an agent and remove it from the hierarchy.
        
        此方法停止Agent，将其与父Agent断开连接，
        并将其从注册表中移除。
        
        This method stops the agent, disconnects it from its parent,
        and removes it from the registry.
        
        Args:
            agent_id: 要销毁的Agent的ID
                     ID of the agent to destroy
        
        Returns:
            bool: 如果Agent成功销毁则为True，否则为False
                 True if the agent was successfully destroyed, False otherwise
        """
        agent = await self.get_agent(agent_id)
        if not agent:
            logger.warning(f"尝试销毁不存在的Agent: {agent_id}")
            logger.warning(f"Attempted to destroy non-existent agent: {agent_id}")
            return False
        
        # 首先递归销毁所有子Agent
        # First, recursively destroy all child agents
        child_ids = list(agent.child_agents.keys())
        for child_id in child_ids:
            success = await self.destroy_agent(child_id)
            if not success:
                logger.warning(f"销毁子Agent {child_id} 失败")
                logger.warning(f"Failed to destroy child agent {child_id}")
        
        # 停止Agent
        # Stop the agent
        await agent.stop()
        
        # 从父Agent的child_agents中删除
        # Remove from parent's child_agents
        for other_id, other_agent in self.agent_registry.items():
            if agent_id in other_agent.child_agents:
                # 删除从父Agent到此Agent的连接
                # Remove connection from parent to this agent
                connection = self._connection_by_agent_pair.get((other_id, agent_id))
                if connection and connection in self.connections:
                    self.connections.remove(connection)
                    del self._connection_by_agent_pair[(other_id, agent_id)]
                
                # 从父Agent的子Agent列表中删除
                # Remove from parent's child agents list
                del other_agent.child_agents[agent_id]
        
        # 从注册表中删除
        # Remove from registry
        del self.agent_registry[agent_id]
        
        logger.info(f"已销毁Agent: {agent.name} ({agent_id})")
        logger.info(f"Destroyed agent: {agent.name} ({agent_id})")
        return True
