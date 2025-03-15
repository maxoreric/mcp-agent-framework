"""
LLM integration module for MCP Agent Framework.

This module contains components for interacting with
language models and managing prompts.
"""

from .integration import process_with_deepseek
from .prompts import format_prompt
from .deepseek import DeepSeekLLM, ChatMessage, BaseLLM

# 提供process_with_llm别名用于向后兼容
process_with_llm = process_with_deepseek

__all__ = ["process_with_llm", "process_with_deepseek", "format_prompt", 
          "DeepSeekLLM", "ChatMessage", "BaseLLM"]
