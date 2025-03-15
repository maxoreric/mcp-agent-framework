"""
LLM Integration for MCP Agent Framework using DeepSeek API.

本模块实现使用 DeepSeek API 进行 LLM 请求处理，满足以下要求：
1. 父 agent 传递 XML 样式标签，其中 open_xml_tag 用作前缀，close_xml_tag 用作停止符（stop 参数），
   以确保子 agent 的返回内容被包装在该标签中，防止输出过多。
2. 如启用 prefix completion 模式（use_prefix_completion=True），则通过 Beta 接口将 open_xml_tag 作为 assistant 消息前缀发送。

请确保在 agent 配置中传入以下必要参数：
  - deepseek_api_key: Deepseek API密钥
  - open_xml_tag: XML开始标签（如 <answer>）
  - close_xml_tag: XML结束标签（如 </answer>）
  - use_prefix_completion（可选，默认为 False）: 是否使用前缀补全API
  - 其他可选参数（deepseek_model, system_prompt, request_timeout, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, deepseek_base_url, max_retries, retry_delay）

可通过最新的Deepseek API支持prefix completion（前缀补全）功能，适合XML标签模式处理。

This module implements LLM request processing using the DeepSeek API, meeting the following requirements:
1. Parent agent passes XML-style tags, where open_xml_tag serves as a prefix and close_xml_tag as a stop token (stop parameter),
   ensuring the child agent's return content is wrapped in this tag, preventing excessive output.
2. If prefix completion mode is enabled (use_prefix_completion=True), the Beta API is used to send open_xml_tag as an assistant message prefix.

Please ensure the following required parameters are passed in the agent configuration:
  - deepseek_api_key: Deepseek API key
  - open_xml_tag: XML opening tag (e.g., <answer>)
  - close_xml_tag: XML closing tag (e.g., </answer>)
  - use_prefix_completion (optional, default is False): Whether to use prefix completion API
  - Other optional parameters (deepseek_model, system_prompt, request_timeout, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, deepseek_base_url, max_retries, retry_delay)

The latest Deepseek API supports prefix completion functionality, suitable for XML tag pattern processing.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union, Tuple

import httpx
from mcp.server.fastmcp import Context

from ..agent.agent import Agent


class SafeContext:
    """A wrapper for Context that handles both sync and async calls safely."""
    
    def __init__(self, ctx):
        self.ctx = ctx
    
    def info(self, msg):
        # If the original context has an async info method, call it in a way that won't cause warnings
        if hasattr(self.ctx, 'info') and callable(self.ctx.info):
            # Create a new task but don't await it (it will run independently)
            asyncio.create_task(self._safe_call(self.ctx.info, msg))
    
    def warning(self, msg):
        if hasattr(self.ctx, 'warning') and callable(self.ctx.warning):
            asyncio.create_task(self._safe_call(self.ctx.warning, msg))
    
    def error(self, msg):
        if hasattr(self.ctx, 'error') and callable(self.ctx.error):
            asyncio.create_task(self._safe_call(self.ctx.error, msg))
    
    def debug(self, msg):
        if hasattr(self.ctx, 'debug') and callable(self.ctx.debug):
            asyncio.create_task(self._safe_call(self.ctx.debug, msg))
    
    async def _safe_call(self, func, msg):
        """Safely call an async function, catching any exceptions."""
        try:
            if asyncio.iscoroutinefunction(func):
                await func(msg)
            else:
                func(msg)
        except Exception:
            # Silently ignore any errors in context methods
            pass


async def process_with_llm(agent: Agent, prompt: str, ctx: Context) -> str:
    """
    使用 LLM 处理 prompt，目前支持 DeepSeek 模型。
    
    Process a prompt using LLM, currently supporting DeepSeek models.
    
    Args:
        agent: 发起请求的 Agent
              Agent initiating the request
        prompt: 待处理的 prompt
               Prompt to be processed
        ctx: MCP 上下文，用于日志和进度报告
             MCP Context for logging and progress reporting
    
    Returns:
        str: LLM 返回的文本响应
             Text response from the LLM
    """
    # Create a safe wrapper for the context
    safe_ctx = SafeContext(ctx)
    
    # 根据配置选择 LLM 实现
    # Choose LLM implementation based on configuration
    llm_provider = agent.config.get("llm_provider", "deepseek").lower()
    
    # 默认使用 DeepSeek LLM
    # Use DeepSeek LLM by default
    if llm_provider == "deepseek":
        return await process_with_deepseek(agent, prompt, safe_ctx)
    else:
        # 未来可以在这里添加对其他LLM提供商的支持
        # Future support for other LLM providers can be added here
        safe_ctx.warning(f"不支持的LLM提供商: {llm_provider}，回退到DeepSeek")
        safe_ctx.warning(f"Unsupported LLM provider: {llm_provider}, falling back to DeepSeek")
        return await process_with_deepseek(agent, prompt, safe_ctx)


def _validate_xml_tag(tag: str, expected_format: str, ctx: Context) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    辅助函数，处理XML标签，确保标签格式正确且匹配。
    
    Helper function to process XML tags, ensuring they are properly formed and matched.
    
    Args:
        tag: 要检查的XML标签
             XML tag to check
        expected_format: 预期的格式描述（用于日志记录）
                         The expected format description (for logging)
        ctx: MCP上下文，用于日志记录
             MCP context for logging
        
    Returns:
        tuple: (is_valid, tag_name, error_message)
               (是否有效, 标签名, 错误信息)
    """
    # 检查基本格式
    # Check basic format
    if not tag or not isinstance(tag, str):
        return False, None, f"XML标签不存在或类型错误: {type(tag).__name__}"
    
    # 检查开始标签
    # Check opening tag
    if expected_format == "opening":
        if not tag.startswith('<') or not tag.endswith('>'):
            return False, None, f"XML标签格式错误，应以'<'开头并以'>'结尾: {tag}"
        
        # 提取标签名
        # Extract tag name
        tag_content = tag[1:-1].strip()
        if not tag_content or ' ' in tag_content.split(' ')[0]:
            return False, None, f"XML标签名无效: {tag}"
        
        tag_name = tag_content.split(' ')[0]
        return True, tag_name, None
    
    # 检查关闭标签
    # Check closing tag
    elif expected_format == "closing":
        if not tag.startswith('</') or not tag.endswith('>'):
            return False, None, f"XML关闭标签格式错误，应以'</'开头并以'>'结尾: {tag}"
        
        # 提取标签名
        # Extract tag name
        tag_name = tag[2:-1].strip()
        if not tag_name:
            return False, None, f"XML关闭标签名无效: {tag}"
        
        return True, tag_name, None
    
    return False, None, f"未知的标签格式期望: {expected_format}"


async def _post_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_delay: float,
    ctx: Context,
    provider: str,
) -> Dict[str, Any]:
    """
    内部辅助函数：使用重试机制向指定 API 发送 POST 请求。
    
    Internal helper function: Send POST requests to specified API with retry mechanism.

    Args:
        url: API 请求地址
             API request URL
        headers: 请求头
                Request headers
        payload: 请求体（JSON 格式）
                Request body (JSON format)
        timeout: 请求超时时间（秒）
                Request timeout in seconds
        max_retries: 最大重试次数
                    Maximum number of retry attempts
        retry_delay: 初始重试延迟（后续使用指数退避）
                    Initial retry delay (exponential backoff is used for subsequent attempts)
        ctx: MCP 上下文，用于日志记录
             MCP context for logging
        provider: 提供商名称（用于日志输出）
                 Provider name (for log output)

    Returns:
        响应 JSON 对象的字典形式
        Response JSON object as a dictionary

    Raises:
        ValueError: 若所有重试均失败，则抛出异常
                   If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                ctx.info(f"发送 {provider} API 请求 (尝试 {attempt + 1}/{max_retries})")
                ctx.info(f"Sending {provider} API request (attempt {attempt + 1}/{max_retries})")
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            ctx.error(f"{provider} API 第 {attempt + 1} 次调用失败：{str(e)}")
            ctx.error(f"{provider} API call {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                ctx.warning(f"等待 {wait_time:.1f} 秒后重试...")
                ctx.warning(f"Retrying after {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                raise ValueError(f"{provider} API 重试 {max_retries} 次后仍失败：{str(e)}")
    raise ValueError("未知错误")


async def process_with_deepseek(agent: Agent, prompt: str, ctx: Context) -> str:
    """
    使用 DeepSeek API 处理 prompt，仅支持 DeepSeek 模型。
    
    Process prompt using DeepSeek API, supporting only DeepSeek models.

    父 agent 必须传递 XML 样式标签，其中：
      - open_xml_tag 用作前缀（prefix）：如果启用了 prefix completion 模式，
        将通过 assistant 消息发送该标签作为输出前缀；
      - close_xml_tag 用作停止符（stop 参数），防止生成内容过多。
    
    Parent agent must pass XML-style tags, where:
      - open_xml_tag serves as a prefix: if prefix completion mode is enabled,
        this tag is sent as an output prefix via the assistant message;
      - close_xml_tag serves as a stop token (stop parameter), preventing excessive content generation.

    Args:
        agent: 发起请求的 Agent
               Agent initiating the request
        prompt: 待处理的 prompt
               Prompt to be processed
        ctx: MCP 上下文，用于日志和进度报告
             MCP Context for logging and progress reporting

    Returns:
        DeepSeek 返回的文本响应，确保以 XML 结束标签结尾
        Text response from DeepSeek, ensuring it ends with the XML closing tag

    Raises:
        ValueError: 当API密钥未提供或XML标签未提供时
                   When API key or XML tags are not provided
    """
    # 获取 API key
    # Get API key
    api_key = agent.config.get("deepseek_api_key", os.getenv("DEEPSEEK_API_KEY"))
    if not api_key:
        raise ValueError("未在 agent 配置或环境变量中找到 DeepSeek API Key")
        
    # 获取模型、系统提示和请求超时设置
    # Get model, system prompt, and request timeout settings
    model = agent.config.get("deepseek_model", "deepseek-chat")
    if model not in ["deepseek-chat", "deepseek-reasoner"]:
        ctx.warning(f"不支持的 DeepSeek 模型：{model}，回退至 deepseek-chat")
        ctx.warning(f"Unsupported DeepSeek model: {model}, falling back to deepseek-chat")
        model = "deepseek-chat"
    system = agent.config.get("system_prompt", f"You are {agent.name}, an AI assistant.")
    timeout = agent.config.get("request_timeout", 120)

    # 获取父 agent 传递的 XML 标签：open_xml_tag 用作前缀，close_xml_tag 用作停止符
    # Get XML tags passed by parent agent: open_xml_tag as prefix, close_xml_tag as stop token
    xml_open_tag = agent.config.get("open_xml_tag")
    xml_close_tag = agent.config.get("close_xml_tag")
    if not xml_open_tag or not xml_close_tag:
        raise ValueError("必须提供 open_xml_tag 和 close_xml_tag 用于包装输出")

    # 判断是否启用 prefix completion 模式
    # Check if prefix completion mode is enabled
    prefix_completion = agent.config.get("use_prefix_completion", False)
    
    # 检查并验证 XML 标签
    # Check and validate XML tags
    open_valid, open_tag_name, open_error = _validate_xml_tag(xml_open_tag, "opening", ctx)
    close_valid, close_tag_name, close_error = _validate_xml_tag(xml_close_tag, "closing", ctx)
    
    if not open_valid:
        ctx.warning(f"XML开始标签验证失败: {open_error}")
        ctx.warning(f"XML opening tag validation failed: {open_error}")
    
    if not close_valid:
        ctx.warning(f"XML关闭标签验证失败: {close_error}")
        ctx.warning(f"XML closing tag validation failed: {close_error}")
    
    # 检查标签名匹配
    # Check tag name matching
    if open_valid and close_valid and open_tag_name != close_tag_name:
        ctx.warning(f"XML标签名不匹配: <{open_tag_name}> 与 </{close_tag_name}>")
        ctx.warning(f"XML tag names don't match: <{open_tag_name}> vs </{close_tag_name}>")

    # 尝试加载 DeepSeek LLM 类
    # Try to load the DeepSeekLLM class
    _deepseek_available = False
    try:
        from .deepseek import DeepSeekLLM, ChatMessage
        _deepseek_available = True
    except ImportError:
        ctx.warning("DeepSeekLLM 类无法加载，使用原生 API 调用")
        ctx.warning("DeepSeekLLM class could not be loaded, using native API calls")
    
    # 检查是否可能使用 DeepSeekLLM 类
    # Check if we can use the DeepSeekLLM class
    if _deepseek_available and agent.config.get("use_deepseek_llm_class", True):
        ctx.info("使用 DeepSeekLLM 类处理请求")
        ctx.info("Using DeepSeekLLM class to process request")
        try:
            # 创建 DeepSeekLLM 实例
            # Create DeepSeekLLM instance
            llm = DeepSeekLLM(
                api_key=api_key,
                model=model,
                base_url=agent.config.get("deepseek_base_url", "https://api.deepseek.com"),
                max_tokens=agent.config.get("max_tokens", 2048),
                temperature=agent.config.get("temperature", 0.7),
                timeout=timeout
            )
            
            # 创建消息列表
            # Create message list
            messages = [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=prompt)
            ]
            
            # 使用 generate_with_xml_tags 生成文本
            # Generate text with XML tags
            result = await llm.generate_with_xml_tags(
                messages=messages,
                open_tag=xml_open_tag,
                close_tag=xml_close_tag,
                use_prefix_completion=prefix_completion,
                temperature=agent.config.get("temperature", 0.7),
                max_tokens=agent.config.get("max_tokens", 2048),
                # 传递其他可选参数
                # Pass other optional parameters
                **{param: agent.config[param] for param in [
                    "top_p", "frequency_penalty", "presence_penalty"
                ] if param in agent.config}
            )
            
            ctx.debug(f"使用 DeepSeekLLM 类生成文本，长度：{len(result)} 字符")
            ctx.debug(f"Generated text using DeepSeekLLM class, length: {len(result)} characters")
            return result
        
        except Exception as e:
            ctx.error(f"使用 DeepSeekLLM 类失败，回退到原生 API 调用: {str(e)}")
            ctx.error(f"Failed to use DeepSeekLLM class, falling back to native API calls: {str(e)}")
            # 作为 fallback，继续执行下面的原生 API 调用
            # As a fallback, continue with the native API calls below
    else:
        ctx.info("使用原生 API 调用处理请求")
        ctx.info("Using native API calls to process request")

    # 构造消息列表：系统消息、用户消息
    # Construct message list: system message, user message
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    
    # 正确处理prefix completion和XML标签
    # Correctly handle prefix completion and XML tags
    if prefix_completion:
        # 如果启用 prefix completion，则添加一条 assistant 消息，
        # 将 open_xml_tag 作为前缀发送，并设置 "prefix": True 以启用 DeepSeek 的前缀补全功能
        # If prefix completion is enabled, add an assistant message,
        # sending open_xml_tag as a prefix and setting "prefix": True to enable DeepSeek's prefix completion
        messages.append({"role": "assistant", "content": xml_open_tag, "prefix": True})
    else:
        # 非 prefix 模式：将 open_xml_tag 直接拼接到用户消息的内容后面
        # Non-prefix mode: Append open_xml_tag directly to the user message content
        messages[-1]["content"] = f"{prompt}\n\n{xml_open_tag}"

    # 构造请求 payload，包含必需参数和停止 token（使用 XML 的结束标签）
    # Construct request payload with required parameters and stop token (using XML closing tag)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": agent.config.get("max_tokens", 2048),
        "temperature": agent.config.get("temperature", 0.7),
        "stop": [xml_close_tag],
    }
    
    # 添加可选参数
    # Add optional parameters
    for param in ["top_p", "frequency_penalty", "presence_penalty"]:
        if param in agent.config:
            payload[param] = agent.config[param]

    # 根据是否启用 prefix completion 模式选择适当的 API endpoint
    # Select the appropriate API endpoint based on whether prefix completion mode is enabled
    base_url = agent.config.get("deepseek_base_url", "https://api.deepseek.com").rstrip("/")
    if prefix_completion:
        # Beta 接口支持 prefix completion 模式
        # Beta interface supports prefix completion mode
        api_url = f"{base_url}/beta/chat/completions"
        ctx.info(f"使用 Deepseek Beta API 进行前缀补全: {api_url}")
        ctx.info(f"Using Deepseek Beta API for prefix completion: {api_url}")
    else:
        api_url = f"{base_url}/chat/completions"
        ctx.info(f"使用 Deepseek 标准 API: {api_url}")
        ctx.info(f"Using Deepseek standard API: {api_url}")

    ctx.debug(f"向 DeepSeek ({model}) 发送请求，prompt 长度：{len(prompt)} 字符")
    ctx.debug(f"Sending request to DeepSeek ({model}), prompt length: {len(prompt)} characters")
    ctx.debug(f"API URL: {api_url}")
    ctx.debug(f"请求负载：{json.dumps(payload, indent=2)}")
    ctx.debug(f"Request payload: {json.dumps(payload, indent=2)}")

    max_retries = agent.config.get("max_retries", 3)
    retry_delay = agent.config.get("retry_delay", 1.0)

    # 发起请求并采用重试机制
    # Make request with retry mechanism
    try:
        result = await _post_with_retries(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            payload=payload,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            ctx=ctx,
            provider="DeepSeek"
        )
    except ValueError as e:
        # 错误处理：记录错误并返回简单的错误输出
        # Error handling: Log the error and return a simple error output
        ctx.error(f"DeepSeek API 请求失败: {str(e)}")
        ctx.error(f"DeepSeek API request failed: {str(e)}")
        # 返回一个包含错误信息的XML格式结果
        # Return an XML-formatted result containing the error message
        return f"{xml_open_tag}Error: Failed to get response from DeepSeek API. {str(e)}{xml_close_tag}"

    ctx.debug(f"DeepSeek 原始响应：{json.dumps(result, indent=2)}")
    ctx.debug(f"DeepSeek raw response: {json.dumps(result, indent=2)}")
    if "choices" in result and result["choices"]:
        content = result["choices"][0]["message"]["content"]
        ctx.debug(f"接收到 DeepSeek 响应，字符数：{len(content)}")
        ctx.debug(f"Received DeepSeek response, character count: {len(content)}")
        
        # 处理内容的前缀（若未正确处理）
        # Handle content prefix (if not correctly processed)
        if not content.startswith(xml_open_tag) and not prefix_completion:
            ctx.debug(f"补充XML开始标签: {xml_open_tag}")
            ctx.debug(f"Adding missing XML opening tag: {xml_open_tag}")
            content = f"{xml_open_tag}{content}"
            
        # 确保返回的内容以 XML 的结束标签结尾
        # Ensure the returned content ends with the XML closing tag
        if not content.endswith(xml_close_tag):
            ctx.debug(f"补充XML结束标签: {xml_close_tag}")
            ctx.debug(f"Adding missing XML closing tag: {xml_close_tag}")
            content += xml_close_tag
            
        # 检查内容格式是否正确（调试用）
        # Check if the content format is correct (for debugging)
        tag_name = open_tag_name if open_valid else "unknown"
        expected_format = f"{xml_open_tag}内容{xml_close_tag}"
        ctx.debug(f"检查XML格式是否符合: {expected_format}")
        ctx.debug(f"Checking if XML format matches: {expected_format}")
        
        # 扫描发现内部XML标签插入问题
        # Scan for internal XML tag injection issues
        content_without_tags = content.replace(xml_open_tag, "").replace(xml_close_tag, "")
        if xml_open_tag in content_without_tags or xml_close_tag in content_without_tags:
            ctx.warning(f"XML标签可能嵌套不正确：内容中发现额外的XML标签")
            ctx.warning(f"XML tags may be improperly nested: additional XML tags found in content")
            # 尝试修复一些常见问题 
            # Try to fix some common issues
            duplicate_open_index = content.find(xml_open_tag, len(xml_open_tag))
            if duplicate_open_index > 0:
                ctx.debug(f"删除重复的开始标签")
                ctx.debug(f"Removing duplicate opening tag")
                content = content[:duplicate_open_index] + content[duplicate_open_index + len(xml_open_tag):]
            
            duplicate_close_index = content.rfind(xml_close_tag, 0, content.rfind(xml_close_tag))
            if duplicate_close_index > 0:
                ctx.debug(f"删除重复的结束标签")
                ctx.debug(f"Removing duplicate closing tag")
                content = content[:duplicate_close_index] + content[duplicate_close_index + len(xml_close_tag):]
        
        # 检查是否有有效响应内容（去除XML标签后）
        # Check if there is any valid response content (after removing XML tags)
        if not content_without_tags.strip():
            ctx.warning("DeepSeek 响应内容为空（仅有XML标签）")
            ctx.warning("DeepSeek response content is empty (XML tags only)")
        
        return content
    else:
        ctx.warning("DeepSeek 返回空响应")
        ctx.warning("DeepSeek returned an empty response")
        return f"{xml_open_tag}{xml_close_tag}"  # 返回空的XML标签对 / Return empty XML tag pair
