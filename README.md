# MCP Agent Framework

![MCP Agent Framework](https://via.placeholder.com/1200x300/4a90e2/ffffff?text=MCP+Agent+Framework)

一个强大的Python框架，用于创建、编排和管理使用Model Context Protocol (MCP)的AI代理网络。

A powerful Python framework for creating, orchestrating, and managing networks of AI agents using the Model Context Protocol (MCP).

## 🌟 特性 | Features

- **层次化代理结构** | **Hierarchical Agent Structure**
  - 创建CEO、开发者、研究员等专业角色的代理
  - Create agents with specialized roles like CEO, Developer, Researcher, etc.
  - 支持父子代理关系的树状层次结构
  - Tree-like hierarchy supporting parent-child agent relationships

- **智能任务编排** | **Intelligent Task Orchestration**
  - 自动分析任务复杂度并相应地分解任务
  - Automatically analyze task complexity and decompose tasks accordingly
  - 处理任务依赖关系和并行执行
  - Handle task dependencies and parallel execution

- **灵活的LLM集成** | **Flexible LLM Integration**
  - 支持多种LLM提供商（DeepSeek、OpenAI、Anthropic）
  - Support for multiple LLM providers (DeepSeek, OpenAI, Anthropic)
  - 结构化XML输出格式
  - Structured XML output format

- **用户友好的CLI** | **User-Friendly CLI**
  - 交互式命令行界面用于监控和控制
  - Interactive command-line interface for monitoring and control
  - 可视化层次结构和任务依赖关系
  - Visualization of hierarchies and task dependencies

- **完全异步** | **Fully Asynchronous**
  - 基于asyncio构建，实现高效并发
  - Built on asyncio for efficient concurrency
  - 异步事件处理系统
  - Asynchronous event handling system

## 🚀 入门 | Getting Started

### 安装 | Installation

```bash
# 克隆仓库
# Clone the repository
git clone https://github.com/maxoreric/mcp-agent-framework.git
cd mcp_agent_framework

# 安装依赖
# Install dependencies
pip install -r requirements.txt
```

### 基本使用 | Basic Usage

```python
import asyncio
from mcp_agent_framework import AgentFramework

async def main():
    # 创建框架实例
    # Create framework instance
    framework = AgentFramework()
    
    # 初始化框架
    # Initialize the framework
    await framework.initialize()
    
    # 创建专业代理
    # Create specialized agents
    developer_id = await framework.create_agent(
        name="Developer",
        role="developer"
    )
    
    # 提交任务
    # Submit a task
    task_id = await framework.submit_task(
        description="Create a Python function to calculate Fibonacci numbers."
    )
    
    # 等待任务完成
    # Wait for task to complete
    result = await framework.wait_for_task(task_id)
    
    print(f"Task result: {result['result']}")
    
    # 关闭框架
    # Shutdown the framework
    await framework.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 启动CLI | Start the CLI

```bash
python -m mcp_agent_framework
```

## 🔧 配置 | Configuration

使用配置字典或JSON文件创建`AgentFramework`实例：

Create an `AgentFramework` instance with a configuration dictionary or JSON file:

```python
config = {
    "api_key": "your-openai-api-key",
    "model": "gpt-4",
    "anthropic_api_key": "your-anthropic-api-key",
    "anthropic_model": "claude-3-opus-20240229",
    "deepseek_api_key": "your-deepseek-api-key",
    "deepseek_model": "deepseek-chat",
    "llm_provider": "deepseek",  # 选项: openai, anthropic, deepseek
    "use_prefix_completion": True,
    "open_xml_tag": "<answer>",
    "close_xml_tag": "</answer>"
}

framework = AgentFramework(config=config)
```

或者，您可以使用环境变量：

Alternatively, you can use environment variables:

```bash
export OPENAI_API_KEY=your-openai-api-key
export ANTHROPIC_API_KEY=your-anthropic-api-key
export DEEPSEEK_API_KEY=your-deepseek-api-key
```

## 📖 示例 | Examples

### 创建代理团队 | Creating an Agent Team

```python
from mcp_agent_framework import AgentFramework

async def create_team():
    framework = AgentFramework()
    await framework.initialize()
    
    team = await framework.factory.create_agent_team(
        team_roles=["developer", "researcher", "writer"],
        leader_role="ceo",
        name_prefix="Project"
    )
    
    # 使用团队领导（CEO）提交任务
    # Submit a task using the team leader (CEO)
    ceo = team["ceo"]
    task_id = await framework.submit_task(
        description="Create a market analysis report for AI in healthcare."
    )
    
    result = await framework.wait_for_task(task_id)
    print(result['result'])
```

### 使用CLI | Using the CLI

命令行界面支持以下命令：

The command-line interface supports the following commands:

- `task`: 提交新任务 | Submit a new task
- `status`: 显示所有任务和代理的状态 | Display the status of all tasks and agents
- `agents`: 显示有关所有活动代理的信息 | Display information about all active agents
- `help`: 显示帮助信息 | Show help message
- `exit`: 关闭应用 | Close the application

## 🧩 架构 | Architecture

MCP Agent Framework是围绕以下核心组件构建的：

The MCP Agent Framework is built around the following core components:

- **Agent**: 代表具有专业角色的AI代理
  Represents an AI agent with a specialized role
  
- **AgentHierarchy**: 管理代理之间的树状关系
  Manages the tree-like relationships between agents
  
- **AgentFactory**: 创建不同类型的代理基于模板和规范
  Creates different types of agents based on templates and specifications
  
- **TaskOrchestrator**: 处理任务分解、委派和结果聚合
  Handles task decomposition, delegation, and result aggregation
  
- **LLM集成**: 与各种LLM提供商集成以处理提示
  Integration with various LLM providers for prompt processing
  
- **CLI**: 提供用户友好的界面用于与框架交互
  Provides a user-friendly interface for interacting with the framework

## 📝 许可证 | License

本项目根据MIT许可证授权 - 有关详细信息，请参阅[LICENSE](LICENSE)文件。

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 贡献 | Contributing

欢迎贡献！请随时提交问题和拉取请求。

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork仓库
   Fork the repository
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
   Create your feature branch (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
   Commit your changes (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
   Push to the branch (`git push origin feature/amazing-feature`)
5. 打开拉取请求
   Open a Pull Request

## 📞 联系 | Contact

项目维护者 - Zhang Jianfeng - jianfeng.zhang@example.com

Project Maintainer - Zhang Jianfeng - jianfeng.zhang@example.com

项目链接: [https://github.com/maxoreric/mcp-agent-framework](https://github.com/maxoreric/mcp-agent-framework)

Project Link: [https://github.com/maxoreric/mcp-agent-framework](https://github.com/maxoreric/mcp-agent-framework)
