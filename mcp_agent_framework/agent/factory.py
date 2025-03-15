"""
Agent Factory for MCP Agent Framework.

MCP Agent Framework的Agent工厂。

本模块定义了AgentFactory类，负责基于模板和规范创建各种类型的Agent。

This module defines the AgentFactory class, which handles the creation
of various types of agents based on templates and specifications.
"""

import logging
from typing import Dict, Any, Optional, Type, List, Set

from .agent import Agent
from .hierarchy import AgentHierarchy

# 配置模块级别日志器
# Configure module-level logger
logger = logging.getLogger(__name__)

class AgentSpec:
    """
    Agent创建的规范。
    
    Specification for agent creation.
    
    此类封装了创建新Agent实例所需的参数。
    
    This class encapsulates the parameters needed to create
    a new agent instance.
    
    Attributes:
        name: Agent的人类可读名称
              Human-readable name for the agent
        role: Agent的专业角色
              Specialized role of the agent
        config: Agent的附加配置
                Additional configuration for the agent
    """
    
    def __init__(self, name: str, role: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化新的AgentSpec实例。
        
        Initialize a new AgentSpec instance.
        
        Args:
            name: Agent的人类可读名称
                 Human-readable name for the agent
            role: Agent的专业角色
                 Specialized role of the agent
            config: Agent的附加配置(可选)
                   Additional configuration for the agent (optional)
        """
        if not isinstance(name, str):
            raise TypeError("Agent名称必须是字符串")
            raise TypeError("Agent name must be a string")
        
        if not isinstance(role, str):
            raise TypeError("Agent角色必须是字符串")
            raise TypeError("Agent role must be a string")
            
        if config is not None and not isinstance(config, dict):
            raise TypeError("Agent配置必须是字典")
            raise TypeError("Agent configuration must be a dictionary")
            
        self.name = name
        self.role = role
        self.config = config or {}
        
    def __repr__(self) -> str:
        """返回AgentSpec的字符串表示"""
        return f"AgentSpec(name='{self.name}', role='{self.role}', config={self.config})"

class AgentFactory:
    """
    创建Agent实例的工厂。
    
    Factory for creating agent instances.
    
    AgentFactory类处理基于模板和规范创建各种类型的Agent。
    它与AgentHierarchy协作，注册和管理创建的Agent。
    
    The AgentFactory class handles the creation of various types of agents
    based on templates and specifications. It works with the AgentHierarchy
    to register and manage created agents.
    
    Attributes:
        hierarchy: Agent层次结构的引用
                  Reference to the agent hierarchy
        config: 配置字典
                Configuration dictionary
        templates: 已注册Agent模板的字典
                  Dictionary of registered agent templates
        supported_roles: 支持的角色集合
                        Set of supported roles
    """
    
    def __init__(self, hierarchy: AgentHierarchy, config: Dict[str, Any]):
        """
        初始化新的AgentFactory实例。
        
        Initialize a new AgentFactory instance.
        
        Args:
            hierarchy: Agent层次结构的引用
                      Reference to the agent hierarchy
            config: 配置字典
                   Configuration dictionary
        """
        self.hierarchy = hierarchy
        self.config = config
        self.templates: Dict[str, Type[Agent]] = {}
        self.supported_roles: Set[str] = {"ceo", "developer", "researcher", "writer", "analyst"}
        
        # 注册基本的Agent模板
        # Register basic agent templates
        self.register_template("default", Agent)
    
    def register_template(self, role: str, agent_class: Type[Agent]) -> None:
        """
        注册Agent模板。
        
        Register an agent template.
        
        此方法将专业Agent类与角色关联，使工厂能够在请求具有该角色的Agent时
        创建该类的实例。
        
        This method associates a specialized agent class with a role,
        allowing the factory to create instances of that class when
        an agent with that role is requested.
        
        Args:
            role: 此Agent类型的角色名称
                 Role name for this agent type
            agent_class: 用于此角色的Agent类
                        Agent class to use for this role
                        
        Raises:
            TypeError: 如果agent_class不是Agent的子类
                      If agent_class is not a subclass of Agent
        """
        if not issubclass(agent_class, Agent):
            raise TypeError(f"agent_class必须是Agent的子类，但收到了{agent_class.__name__}")
            raise TypeError(f"agent_class must be a subclass of Agent, but got {agent_class.__name__}")
            
        role_lower = role.lower()
        self.templates[role_lower] = agent_class
        self.supported_roles.add(role_lower)
        
        logger.info(f"为角色 '{role}' 注册了Agent模板: {agent_class.__name__}")
        logger.info(f"Registered agent template for role '{role}': {agent_class.__name__}")
    
    def _validate_role(self, role: str) -> str:
        """
        验证并规范化Agent角色。
        
        Validate and normalize agent role.
        
        Args:
            role: 要验证的Agent角色
                 Agent role to validate
                 
        Returns:
            str: 规范化的角色（小写）
                 Normalized role (lowercase)
                 
        Raises:
            ValueError: 如果角色未受支持且未启用自动创建
                       If role is not supported and auto-creation is not enabled
        """
        role_lower = role.lower()
        
        # 检查这是否是支持的角色
        # Check if this is a supported role
        if role_lower not in self.supported_roles:
            # 如果配置允许，自动添加新角色
            # Auto-add new roles if configuration allows
            if self.config.get("allow_new_roles", True):
                logger.warning(f"未知的Agent角色 '{role}'，但由于allow_new_roles=True，将允许创建")
                logger.warning(f"Unknown agent role '{role}', but will allow creation due to allow_new_roles=True")
                self.supported_roles.add(role_lower)
            else:
                logger.error(f"尝试创建不支持的Agent角色: {role}")
                logger.error(f"Attempted to create unsupported agent role: {role}")
                supported_list = ", ".join(sorted(self.supported_roles))
                raise ValueError(f"不支持的Agent角色: '{role}'。支持的角色包括: {supported_list}")
                raise ValueError(f"Unsupported agent role: '{role}'. Supported roles include: {supported_list}")
                
        return role_lower
        
    async def create_agent(self, spec: AgentSpec, parent_id: Optional[str] = None) -> Agent:
        """
        基于规范创建新Agent。
        
        Create a new agent based on specification.
        
        此方法基于提供的规范创建新的Agent实例，可选择将其连接到父Agent。
        
        This method creates a new agent instance based on the provided
        specification, optionally connecting it to a parent agent.
        
        Args:
            spec: 新Agent的规范
                 Specification for the new agent
            parent_id: 父Agent的可选ID
                      Optional ID of the parent agent
        
        Returns:
            Agent: 新创建的Agent实例
                  The newly created agent instance
                  
        Raises:
            ValueError: 如果父Agent未找到或角色无效
                       If parent agent is not found or role is invalid
            RuntimeError: 如果Agent创建失败
                         If agent creation fails
        """
        # 验证角色
        # Validate role
        normalized_role = self._validate_role(spec.role)
        
        # 合并配置
        # Merge configuration
        agent_config = self.config.copy()
        agent_config.update(spec.config)
        
        # 检查我们是否有此角色的模板
        # Check if we have a template for this role
        agent_class = self.templates.get(normalized_role, self.templates.get("default", Agent))
        
        try:
            # 创建Agent实例
            # Create agent instance
            logger.info(f"为角色 '{spec.role}' 创建Agent: {spec.name}")
            logger.info(f"Creating agent for role '{spec.role}': {spec.name}")
            
            agent = agent_class(spec.name, spec.role, agent_config)
            
            # A parent ID is provided, this is a child agent, register under the parent
            if parent_id:
                # 检查父Agent是否存在
                # Check if parent agent exists
                parent = await self.hierarchy.get_agent(parent_id)
                if not parent:
                    logger.error(f"无法创建子Agent: 未找到ID为 {parent_id} 的父Agent")
                    logger.error(f"Cannot create child agent: Parent agent with ID {parent_id} not found")
                    raise ValueError(f"未找到ID为 {parent_id} 的父Agent")
                    raise ValueError(f"Parent agent with ID {parent_id} not found")
                
                # 这将创建Agent并将其连接到父Agent
                # This will create the agent and connect it to the parent
                agent = await self.hierarchy.create_child_agent(parent_id, spec.name, spec.role)
                if not agent:
                    logger.error(f"创建子Agent时发生错误: {spec.name} ({spec.role})")
                    logger.error(f"Error occurred while creating child agent: {spec.name} ({spec.role})")
                    raise RuntimeError(f"创建子Agent时发生错误: {spec.name} ({spec.role})")
                    raise RuntimeError(f"Error occurred while creating child agent: {spec.name} ({spec.role})")
            else:
                # 只添加到注册表
                # Just add to the registry
                self.hierarchy.agent_registry[agent.id] = agent
                await agent.start()
            
            logger.info(f"已成功创建Agent: {spec.name} ({spec.role}), ID: {agent.id}")
            logger.info(f"Successfully created agent: {spec.name} ({spec.role}), ID: {agent.id}")
            return agent
            
        except Exception as e:
            logger.error(f"创建Agent时出错: {str(e)}")
            logger.error(f"Error creating agent: {str(e)}")
            raise RuntimeError(f"创建Agent失败: {str(e)}")
            raise RuntimeError(f"Failed to create agent: {str(e)}")
    
    async def create_specialized_agent(
        self, 
        role: str, 
        name: Optional[str] = None, 
        parent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """
        基于角色创建专业Agent。
        
        Create a specialized agent based on role.
        
        这是一个便捷方法，用于创建具有预定义角色的新Agent，
        使用该角色的默认设置。
        
        This is a convenience method that creates a new agent with
        a predefined role, using default settings for that role.
        
        Args:
            role: 新Agent的专业角色
                 Specialized role for the new agent
            name: Agent的可选名称(默认为角色名称)
                 Optional name for the agent (defaults to role name)
            parent_id: 父Agent的可选ID
                      Optional ID of the parent agent
            config: Agent的可选附加配置
                   Optional additional configuration for the agent
        
        Returns:
            Agent: 新创建的Agent实例
                  The newly created agent instance
        """
        # 如果未提供名称，则生成名称
        # Generate name if not provided
        if not name:
            name = f"{role.capitalize()} Agent"
        
        # 为此角色创建默认配置
        # Create default configuration for this role
        role_config = config or {}
        
        # 为不同角色定义特定配置
        # Define role-specific configurations
        role_lower = role.lower()
        
        if role_lower == "developer":
            role_defaults = {
                "programming_languages": ["Python", "JavaScript", "TypeScript"],
                "specialization": "Full-stack development",
                "tool_preference": "VSCode"
            }
        elif role_lower == "researcher":
            role_defaults = {
                "search_depth": 3,
                "specialization": "General research",
                "citation_style": "APA"
            }
        elif role_lower == "writer":
            role_defaults = {
                "writing_style": "professional",
                "specialization": "Technical writing",
                "tone": "Informative"
            }
        elif role_lower == "analyst":
            role_defaults = {
                "analysis_methods": ["statistical", "qualitative"],
                "data_formats": ["CSV", "JSON", "Excel"],
                "visualization_tools": ["Matplotlib", "Plotly"]
            }
        else:
            # 通用默认值
            # Generic defaults
            role_defaults = {
                "specialization": f"{role.capitalize()} tasks"
            }
        
        # 合并默认值和提供的配置
        # Merge defaults with provided config
        for key, value in role_defaults.items():
            if key not in role_config:
                role_config[key] = value
        
        # 创建Agent规范
        # Create agent specification
        spec = AgentSpec(name, role, role_config)
        
        # 创建并返回Agent
        # Create and return the agent
        return await self.create_agent(spec, parent_id)
    
    async def create_agent_team(
        self,
        team_roles: List[str],
        leader_role: str = "ceo",
        name_prefix: str = "Team",
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Agent]:
        """
        创建相互连接的Agent团队。
        
        Create a team of interconnected agents.
        
        此方法创建由领导Agent协调的多个专业Agent团队。
        
        This method creates a team of multiple specialized agents
        coordinated by a leader agent.
        
        Args:
            team_roles: 团队成员的角色列表
                       List of roles for team members
            leader_role: 领导Agent的角色(默认为"ceo")
                        Role for the leader agent (defaults to "ceo")
            name_prefix: Agent名称的前缀
                        Prefix for agent names
            configs: 按角色为Agent提供特定配置的字典
                    Dictionary providing specific configs for agents by role
        
        Returns:
            Dict[str, Agent]: 将角色映射到Agent实例的字典
                             Dictionary mapping roles to agent instances
        """
        agents = {}
        configs = configs or {}
        
        # 创建团队领导
        # Create team leader
        leader_name = f"{name_prefix} {leader_role.capitalize()}"
        leader_config = configs.get(leader_role, {})
        
        leader_spec = AgentSpec(leader_name, leader_role, leader_config)
        leader = await self.create_agent(leader_spec)
        agents[leader_role] = leader
        
        # 创建团队成员作为领导的子Agent
        # Create team members as children of the leader
        for role in team_roles:
            if role == leader_role:
                continue  # 跳过领导，已创建 / Skip leader, already created
                
            member_name = f"{name_prefix} {role.capitalize()}"
            member_config = configs.get(role, {})
            
            member = await self.create_specialized_agent(
                role=role,
                name=member_name,
                parent_id=leader.id,
                config=member_config
            )
            
            agents[role] = member
        
        logger.info(f"已创建包含 {len(agents)} 个Agent的团队")
        logger.info(f"Created team with {len(agents)} agents")
        return agents
