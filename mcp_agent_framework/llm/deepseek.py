"""
DeepSeek LLM integration for MCP Agent Framework.

此模块实现了DeepSeek大语言模型的集成，使MCP Agent Framework能够使用DeepSeek Chat和Reasoner模型。
提供了XML标签处理和前缀补全(prefix completion)功能，适用于创建结构化的代理间通信。

This module implements the integration with DeepSeek large language models,
enabling the MCP Agent Framework to use DeepSeek Chat and Reasoner models.
It provides XML tag handling and prefix completion functionality suitable
for creating structured communication between agents.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union, Tuple

import httpx

# 导入ChatMessage类型作为NamedTuple
# Import ChatMessage as a NamedTuple
from typing import NamedTuple


class ChatMessage(NamedTuple):
    """
    Chat message structure for LLMs.
    
    聊天消息的数据结构，用于与大语言模型的交互。
    
    Attributes:
        role: 消息角色，可选值为 'system'、'user' 或 'assistant'
              Message role, either 'system', 'user', or 'assistant'
        content: 消息内容文本
                 Message content text
    """
    role: str  # 'system', 'user', or 'assistant'
    content: str  # message content


class BaseLLM:
    """
    Abstract base class for LLM implementations.
    
    大语言模型实现的抽象基类，定义了通用接口。
    
    所有LLM实现都应该继承此类并实现generate方法。
    All LLM implementations should inherit from this class
    and implement the generate method.
    """
    async def generate(self, messages: List[ChatMessage], **kwargs) -> str:
        """
        Generate text from the LLM based on messages.
        
        基于消息列表从大语言模型生成文本。
        
        Args:
            messages: 聊天消息列表
                     List of chat messages
            **kwargs: 额外的参数，传递给具体的LLM实现
                     Additional parameters passed to the specific LLM implementation
                     
        Returns:
            str: 生成的文本
                 Generated text
        
        Raises:
            NotImplementedError: 子类必须实现此方法
                               Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement generate method")


# 配置模块级别日志器
# Configure module-level logger
logger = logging.getLogger(__name__)


class DeepSeekLLM(BaseLLM):
    """
    DeepSeek LLM implementation for MCP Agent Framework.
    
    DeepSeek大语言模型的实现，用于MCP Agent Framework。
    支持XML标签处理和前缀补全，适用于结构化代理间通信。
    
    Implements DeepSeek LLM integration for MCP Agent Framework.
    Supports XML tag handling and prefix completion for structured
    agent-to-agent communication.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 60,
        **kwargs,
    ):
        """
        Initialize DeepSeek LLM.
        
        初始化DeepSeek大语言模型客户端。

        Args:
            api_key: DeepSeek API密钥。如果未提供，将尝试从DEEPSEEK_API_KEY环境变量获取。
                    DeepSeek API key. If not provided, will try to get from DEEPSEEK_API_KEY env var.
            model: DeepSeek模型名称，可选["deepseek-chat", "deepseek-reasoner"]
                  DeepSeek model to use, one of ["deepseek-chat", "deepseek-reasoner"]
            base_url: DeepSeek API基础URL
                     Base URL for DeepSeek API
            max_tokens: 生成的最大token数量
                       Maximum tokens to generate
            temperature: 采样温度(0.0到2.0)
                        Temperature for sampling (0.0 to 2.0)
            timeout: API调用超时时间（秒）
                    Timeout for API calls in seconds
            **kwargs: 传递给API的其他参数
                     Additional parameters to pass to the API
        
        Raises:
            ValueError: 当未提供API密钥且环境变量不存在时
                       When API key is not provided and environment variable doesn't exist
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not provided and DEEPSEEK_API_KEY environment variable not set"
            )
        
        self.model = model
        if model not in ["deepseek-chat", "deepseek-reasoner"]:
            logger.warning(f"不支持的 DeepSeek 模型：{model}，回退至 deepseek-chat")
            logger.warning(f"Unsupported DeepSeek model: {model}, falling back to deepseek-chat")
            self.model = "deepseek-chat"
            
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.kwargs = kwargs

    def _build_api_payload(
        self,
        messages: List[ChatMessage],
        stop: Optional[Union[str, List[str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build payload for DeepSeek API request.
        
        构建DeepSeek API请求的负载。

        Args:
            messages: 聊天消息列表
                     List of chat messages
            stop: 遇到时停止生成的字符串或字符串列表
                 String or list of strings that will stop generation when encountered
            max_tokens: 生成的最大token数量（如果提供，将覆盖实例默认值）
                       Maximum tokens to generate (overrides instance default if provided)
            temperature: 采样温度（如果提供，将覆盖实例默认值）
                        Temperature for sampling (overrides instance default if provided)
            **kwargs: 传递给API的其他参数
                     Additional parameters to pass to the API

        Returns:
            Dict[str, Any]: 包含API请求负载的字典
                           Dictionary containing the payload for the API request
        """
        # 将ChatMessage转换为DeepSeek API格式
        # Convert ChatMessage to DeepSeek API format
        formatted_messages = []
        for msg in messages:
            formatted_msg = {
                "role": msg.role,
                "content": msg.content
            }
            
            # 处理前缀标志（如果存在）
            # Handle prefix flag if present
            if kwargs.get('prefix_for_assistant') and msg.role == 'assistant':
                formatted_msg["prefix"] = True
                
            formatted_messages.append(formatted_msg)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }

        # 如果提供了停止序列，添加到负载中
        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                payload["stop"] = [stop]
            else:
                payload["stop"] = stop

        # 添加任何其他参数
        # Add any additional arguments
        for k, v in {**self.kwargs, **kwargs}.items():
            if k not in payload and k != 'prefix_for_assistant' and v is not None:
                payload[k] = v

        return payload

    async def generate(
        self,
        messages: List[ChatMessage],
        stop: Optional[Union[str, List[str]]] = None,
        prefix: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a completion using DeepSeek API.
        
        使用DeepSeek API生成文本补全。

        Args:
            messages: 聊天消息列表
                     List of chat messages
            stop: 遇到时停止生成的字符串或字符串列表
                 String or list of strings that will stop generation when encountered
            prefix: 要添加到生成文本前的字符串（用于XML标签处理）
                   String to prepend to the generated text (for XML tag handling)
            **kwargs: 传递给API的其他参数，包括：
                     Additional parameters to pass to the API, including:
                     - use_prefix_completion (bool): 是否使用前缀补全模式
                                                   Whether to use prefix completion mode

        Returns:
            str: 生成的文本
                 Generated text from the model
                 
        Raises:
            ValueError: 当API请求失败时
                       When the API request fails
        """
        # 检查是否使用prefix completion模式，专为XML标签设计
        # Check if using prefix completion mode, specifically designed for XML tags
        use_prefix_completion = kwargs.get('use_prefix_completion', False)
        
        # 处理XML标签前缀
        # Handle XML tag prefix
        if prefix and isinstance(prefix, str):
            # 检测是否是XML标签
            # Detect if it's an XML tag
            is_xml_tag = prefix.startswith('<') and '>' in prefix
            xml_close_tag = None
            
            if is_xml_tag:
                # 尝试提取XML标签名称，用于后续推断关闭标签
                # Try to extract XML tag name for inferring closing tag later
                tag_text = prefix[1:prefix.find('>')].split()[0].strip()
                if tag_text and not tag_text.startswith('/'):
                    # 推断对应的关闭标签
                    # Infer corresponding closing tag
                    xml_close_tag = f"</{tag_text}>"
                    
                    # 记录日志，便于调试
                    # Log for debugging
                    logger.debug(f"检测到XML标签: {prefix}, 关闭标签: {xml_close_tag}")
                    logger.debug(f"Detected XML tag: {prefix}, closing tag: {xml_close_tag}")
                    
                    # 将关闭标签添加到stop参数中
                    # Add closing tag to stop parameter
                    if stop is None:
                        stop = [xml_close_tag]
                    elif isinstance(stop, str):
                        stop = [stop, xml_close_tag]
                    elif isinstance(stop, list) and xml_close_tag not in stop:
                        stop = stop + [xml_close_tag]
            
            # 根据prefix_completion处理模式不同
            # Different handling based on prefix_completion mode
            if use_prefix_completion:
                # 使用前缀补全模式
                # Use prefix completion mode
                # 1. 创建带有prefix文本的新assistant消息
                #    Create new assistant message with prefix text
                # 2. 标记为前缀模式
                #    Mark as prefix mode
                messages = list(messages) + [ChatMessage(role="assistant", content=prefix)]
                kwargs['prefix_for_assistant'] = True
            else:
                # 非前缀补全模式：将前缀添加到最后一条用户消息
                # Non-prefix completion mode: Add prefix to the last user message
                if messages:
                    last_message = messages[-1]
                    # 确保在用户消息和前缀之间有足够的空白
                    # Ensure proper spacing between user message and prefix
                    separator = "\n\n" if not last_message.content.endswith("\n") else "\n"
                    messages = list(messages[:-1]) + [
                        ChatMessage(
                            role=last_message.role,
                            content=f"{last_message.content}{separator}{prefix}"
                        )
                    ]

        # 确定使用的API端点
        # Determine API endpoint to use
        if kwargs.get('prefix_for_assistant', False):
            # 使用Beta API端点支持前缀补全
            # Use Beta API endpoint for prefix completion
            if "beta" not in self.base_url:
                api_url = f"{self.base_url}/beta/chat/completions"
            else:
                api_url = f"{self.base_url}/chat/completions"
            logger.info(f"使用DeepSeek前缀补全(Beta)端点: {api_url}")
            logger.info(f"Using DeepSeek prefix completion (Beta) endpoint: {api_url}")
        else:
            # 使用标准API端点
            # Use standard API endpoint
            api_url = f"{self.base_url}/chat/completions"
            logger.info(f"使用DeepSeek标准API端点: {api_url}")
            logger.info(f"Using DeepSeek standard API endpoint: {api_url}")
        
        # 构建API请求头
        # Build API request headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # 从kwargs中移除内部参数标记
        # Remove internal flag parameters from kwargs
        internal_kwargs = {}
        for key in ['prefix_for_assistant', 'use_prefix_completion']:
            if key in kwargs:
                internal_kwargs[key] = kwargs.pop(key)
        
        # 构建API负载
        # Build API payload
        payload = self._build_api_payload(messages, stop, **kwargs)
        
        logger.debug(f"DeepSeek API 请求: {json.dumps(payload, indent=2)}")
        logger.debug(f"DeepSeek API request: {json.dumps(payload, indent=2)}")
        
        # 发送API请求
        # Send API request
        xml_data = {
            "prefix": prefix,
            "stop": stop,
            "is_xml": prefix and prefix.startswith('<') and '>' in prefix,
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    api_url, headers=headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                response_data = response.json()
                logger.debug(f"DeepSeek API 响应: {json.dumps(response_data, indent=2)}")
                logger.debug(f"DeepSeek API response: {json.dumps(response_data, indent=2)}")
                
                # 提取生成的内容
                # Extract generated content
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    generated_text = response_data["choices"][0]["message"]["content"]
                    
                    # 处理可能的XML标签问题
                    # Handle possible XML tag issues
                    if xml_data["is_xml"]:
                        generated_text = self._process_xml_response(
                            generated_text, 
                            xml_data["prefix"],
                            xml_data["stop"],
                            internal_kwargs.get('use_prefix_completion', False)
                        )
                    
                    return generated_text
                else:
                    raise ValueError("DeepSeek API响应格式异常，未找到生成的内容")
                
            except httpx.HTTPStatusError as e:
                error_message = f"DeepSeek API HTTP错误: {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_message += f" - {error_data['error']}"
                except:
                    pass
                logger.error(error_message)
                raise ValueError(error_message)
            
            except httpx.RequestError as e:
                error_message = f"DeepSeek API请求错误: {str(e)}"
                logger.error(error_message)
                raise ValueError(error_message)
            
            except Exception as e:
                error_message = f"调用DeepSeek API时发生意外错误: {str(e)}"
                logger.error(error_message)
                raise ValueError(error_message)
    
    def _process_xml_response(
        self, 
        text: str, 
        open_tag: str,
        stop_tags: Optional[Union[str, List[str]]],
        was_prefix_completion: bool
    ) -> str:
        """
        处理XML响应，确保正确的标签包装。
        
        Process XML response to ensure proper tag wrapping.
        
        Args:
            text: 从API接收的原始文本
                 Raw text received from API
            open_tag: 开放XML标签
                     Opening XML tag
            stop_tags: 停止标签（可能包含关闭标签）
                      Stop tags (may contain closing tag)
            was_prefix_completion: 是否使用了前缀补全模式
                                  Whether prefix completion was used
                                  
        Returns:
            str: 处理后的XML文本
                 Processed XML text
        """
        # 提取关闭标签（如果存在）
        # Extract closing tag if present
        close_tag = None
        
        if isinstance(stop_tags, str) and stop_tags.startswith('</'):
            close_tag = stop_tags
        elif isinstance(stop_tags, list):
            for tag in stop_tags:
                if isinstance(tag, str) and tag.startswith('</'):
                    close_tag = tag
                    break
        
        # 如果无法确定关闭标签，尝试从开放标签推断
        # If closing tag can't be determined, try to infer from opening tag
        if not close_tag and open_tag.startswith('<') and '>' in open_tag:
            tag_name = open_tag[1:open_tag.find('>')].split()[0].strip()
            if tag_name:
                close_tag = f"</{tag_name}>"
        
        # 开始处理文本
        # Start processing text
        logger.debug(f"处理XML响应，开放标签: {open_tag}, 关闭标签: {close_tag}")
        logger.debug(f"Processing XML response, open tag: {open_tag}, close tag: {close_tag}")
        
        # 确保文本以开放标签开始（如果在前缀补全模式中可能已经包含）
        # Ensure text starts with opening tag (may already be included in prefix completion mode)
        if not text.startswith(open_tag) and not was_prefix_completion:
            text = f"{open_tag}{text}"
        
        # 确保文本以关闭标签结束
        # Ensure text ends with closing tag
        if close_tag and not text.endswith(close_tag):
            text = f"{text}{close_tag}"
        
        # 检查是否有嵌套的标签或其他格式问题
        # Check for nested tags or other formatting issues
        if open_tag in text[len(open_tag):] or (close_tag and text.count(close_tag) > 1):
            logger.warning("检测到XML响应中可能存在标签嵌套问题")
            logger.warning("Detected possible tag nesting issues in XML response")
            
            # 修复：只保留第一个开放标签和最后一个关闭标签
            # Fix: Keep only the first opening tag and the last closing tag
            content = text
            if content.startswith(open_tag):
                content = content[len(open_tag):]
            
            if close_tag and content.endswith(close_tag):
                content = content[:-len(close_tag)]
            
            # 移除所有额外的开放和关闭标签
            # Remove all additional opening and closing tags
            content = content.replace(open_tag, "").replace(close_tag, "")
            
            # 重新构建正确的XML结构
            # Rebuild proper XML structure
            text = f"{open_tag}{content}{close_tag if close_tag else ''}"
        
        return text

    async def generate_with_xml_tags(
        self,
        messages: List[ChatMessage],
        open_tag: str,
        close_tag: str,
        **kwargs,
    ) -> str:
        """
        Generate a completion wrapped in XML tags.
        
        生成被XML标签包装的文本补全。

        Args:
            messages: 聊天消息列表
                     List of chat messages
            open_tag: 开放XML标签（如 <answer>）
                     Opening XML tag (e.g., <answer>)
            close_tag: 关闭XML标签，也用作停止序列（如 </answer>）
                      Closing XML tag (also used as stop sequence)
            **kwargs: 传递给API的其他参数
                     Additional parameters to pass to the API

        Returns:
            str: 被XML标签包装的生成文本
                 Generated text wrapped in XML tags
        """
        # 验证标签格式
        # Validate tag format
        if not open_tag.startswith('<') or not '>' in open_tag:
            logger.warning(f"开放标签格式无效: {open_tag}，应为 '<tag>'")
            logger.warning(f"Invalid opening tag format: {open_tag}, should be '<tag>'")
        
        if not close_tag.startswith('</') or not close_tag.endswith('>'):
            logger.warning(f"关闭标签格式无效: {close_tag}，应为 '</tag>'")
            logger.warning(f"Invalid closing tag format: {close_tag}, should be '</tag>'")
        
        # 确保标签名匹配
        # Ensure tag names match
        open_tag_name = open_tag[1:open_tag.find('>')].split()[0].strip() if '>' in open_tag else ""
        close_tag_name = close_tag[2:-1].strip() if close_tag.startswith('</') and close_tag.endswith('>') else ""
        
        if open_tag_name and close_tag_name and open_tag_name != close_tag_name:
            logger.warning(f"标签名不匹配: '{open_tag_name}' vs '{close_tag_name}'")
            logger.warning(f"Tag names don't match: '{open_tag_name}' vs '{close_tag_name}'")
            
        # 确定是否使用prefix completion模式
        # Determine whether to use prefix completion mode
        use_prefix_completion = kwargs.get('use_prefix_completion', False)
        
        # 使用open_tag作为前缀，close_tag作为停止序列
        # Use open_tag as prefix and close_tag as stop sequence
        # Create a copy of kwargs to avoid modifying the original
        local_kwargs = kwargs.copy()
        
        # Set use_prefix_completion in our local kwargs
        local_kwargs['use_prefix_completion'] = use_prefix_completion
        
        result = await self.generate(
            messages=messages,
            prefix=open_tag,
            stop=close_tag,
            **local_kwargs
        )
        
        # 最终验证：确保结果正确地包装在XML标签中
        # Final validation: Ensure the result is properly wrapped in XML tags
        if not result.startswith(open_tag):
            result = f"{open_tag}{result}"
        
        if not result.endswith(close_tag):
            result = f"{result}{close_tag}"
        
        return result
