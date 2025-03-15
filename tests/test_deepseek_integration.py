"""
Unit tests for DeepSeek integration in MCP Agent Framework.

这个模块包含测试MCP Agent Framework中DeepSeek API集成的单元测试。
该测试确保XML标签处理和前缀补全功能按预期工作。

This module contains unit tests for the DeepSeek API integration in MCP Agent Framework.
These tests ensure XML tag handling and prefix completion functionality work as expected.
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# 添加项目根目录到系统路径，以便导入MCP Agent Framework模块
# Add project root directory to system path to import MCP Agent Framework modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入被测试的模块
# Import modules being tested
from mcp_agent_framework.llm.deepseek import DeepSeekLLM, ChatMessage, BaseLLM
from mcp_agent_framework.llm.integration import process_with_deepseek
from mcp.server.fastmcp import Context


class MockAgent:
    """
    Mock Agent class for testing.
    
    模拟测试用的Agent类，提供必要的config属性。
    """
    def __init__(self, config=None):
        self.name = "TestAgent"
        self.config = config or {}


class MockContext:
    """
    Mock Context class for testing.
    
    模拟测试用的Context类，实现必要的日志方法。
    """
    def __init__(self):
        self.logs = []
    
    def debug(self, message):
        self.logs.append(("DEBUG", message))
    
    def info(self, message):
        self.logs.append(("INFO", message))
    
    def warning(self, message):
        self.logs.append(("WARNING", message))
    
    def error(self, message):
        self.logs.append(("ERROR", message))


class TestDeepSeekLLM(unittest.TestCase):
    """
    Test DeepSeekLLM class functionality.
    
    测试DeepSeekLLM类的基本功能。
    """
    
    def setUp(self):
        """
        Set up test environment.
        
        设置测试环境，包括模拟的API key和请求响应。
        """
        # 设置环境变量，模拟API key / Set environment variable to mock API key
        os.environ["DEEPSEEK_API_KEY"] = "mock-api-key"
        
        # 创建DeepSeekLLM实例 / Create DeepSeekLLM instance
        self.llm = DeepSeekLLM()
    
    @patch('httpx.AsyncClient.post')
    async def test_generate_with_xml_tags(self, mock_post):
        """
        Test generate_with_xml_tags method.
        
        测试generate_with_xml_tags方法，确保XML标签处理正确。
        """
        # 设置模拟响应 / Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response"
                    }
                }
            ]
        }
        mock_response.raise_for_status = AsyncMock()
        mock_post.return_value = mock_response
        
        # 测试消息 / Test messages
        messages = [
            ChatMessage(role="system", content="You are a test assistant"),
            ChatMessage(role="user", content="This is a test message")
        ]
        
        # 测试XML标签 / Test XML tags
        open_tag = "<answer>"
        close_tag = "</answer>"
        
        # 调用方法 / Call method
        result = await self.llm.generate_with_xml_tags(
            messages=messages,
            open_tag=open_tag,
            close_tag=close_tag
        )
        
        # 验证结果 / Verify result
        self.assertTrue(result.startswith(open_tag))
        self.assertTrue(result.endswith(close_tag))
        self.assertIn("This is a test response", result)
        
        # 验证API调用 / Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        self.assertEqual(call_args["headers"]["Authorization"], "Bearer mock-api-key")
        
        # 检查请求负载 / Check request payload
        request_json = call_args["json"]
        self.assertEqual(request_json["model"], "deepseek-chat")
        self.assertEqual(len(request_json["messages"]), 2)
    
    def tearDown(self):
        """Clean up after tests."""
        # 清除环境变量 / Clear environment variable
        if "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]


class TestProcessWithDeepseek(unittest.TestCase):
    """
    Test process_with_deepseek function.
    
    测试process_with_deepseek函数的功能。
    """
    
    @patch('mcp_agent_framework.llm.integration._post_with_retries')
    async def test_process_with_deepseek(self, mock_post_with_retries):
        """
        Test the main process_with_deepseek function.
        
        测试主要的process_with_deepseek函数，确保API调用和响应处理正确。
        """
        # 设置模拟返回值 / Setup mock return value
        mock_post_with_retries.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "<answer>This is a test response</answer>"
                    }
                }
            ]
        }
        
        # 创建模拟agent和context / Create mock agent and context
        mock_agent = MockAgent({
            "deepseek_api_key": "mock-api-key",
            "deepseek_model": "deepseek-chat",
            "open_xml_tag": "<answer>",
            "close_xml_tag": "</answer>",
            "use_prefix_completion": False,
            "max_tokens": 1024,
            "temperature": 0.7
        })
        mock_ctx = MockContext()
        
        # 调用函数 / Call function
        result = await process_with_deepseek(
            agent=mock_agent,
            prompt="Test prompt",
            ctx=mock_ctx
        )
        
        # 验证结果 / Verify result
        self.assertEqual(result, "<answer>This is a test response</answer>")
        
        # 验证API调用 / Verify API call
        mock_post_with_retries.assert_called_once()
        call_args = mock_post_with_retries.call_args[0]
        self.assertIn("/chat/completions", call_args[0])  # URL
        self.assertEqual(call_args[1]["Authorization"], "Bearer mock-api-key")  # Headers
        
        # 检查请求负载 / Check request payload
        request_payload = call_args[2]
        self.assertEqual(request_payload["model"], "deepseek-chat")
        self.assertEqual(len(request_payload["messages"]), 2)
        self.assertEqual(request_payload["max_tokens"], 1024)
        self.assertEqual(request_payload["temperature"], 0.7)
        self.assertEqual(request_payload["stop"], ["</answer>"])
    
    @patch('mcp_agent_framework.llm.integration._post_with_retries')
    async def test_process_with_deepseek_prefix_completion(self, mock_post_with_retries):
        """
        Test process_with_deepseek with prefix completion enabled.
        
        测试启用前缀补全模式的process_with_deepseek函数。
        """
        # 设置模拟返回值 / Setup mock return value
        mock_post_with_retries.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response"
                    }
                }
            ]
        }
        
        # 创建模拟agent和context / Create mock agent and context
        mock_agent = MockAgent({
            "deepseek_api_key": "mock-api-key",
            "deepseek_model": "deepseek-chat",
            "open_xml_tag": "<answer>",
            "close_xml_tag": "</answer>",
            "use_prefix_completion": True,
            "max_tokens": 1024,
            "temperature": 0.7
        })
        mock_ctx = MockContext()
        
        # 调用函数 / Call function
        result = await process_with_deepseek(
            agent=mock_agent,
            prompt="Test prompt",
            ctx=mock_ctx
        )
        
        # 验证结果 / Verify result
        self.assertEqual(result, "<answer>This is a test response</answer>")
        
        # 验证API调用 / Verify API call
        mock_post_with_retries.assert_called_once()
        call_args = mock_post_with_retries.call_args[0]
        self.assertIn("/beta/chat/completions", call_args[0])  # Beta URL
        
        # 检查请求负载 / Check request payload
        request_payload = call_args[2]
        self.assertEqual(request_payload["model"], "deepseek-chat")
        self.assertEqual(len(request_payload["messages"]), 3)  # 3 messages with assistant prefix
        self.assertEqual(request_payload["messages"][2]["role"], "assistant")
        self.assertEqual(request_payload["messages"][2]["content"], "<answer>")
        self.assertEqual(request_payload["messages"][2]["prefix"], True)


def async_test(coro):
    """
    Decorator to run async tests.
    
    用于运行异步测试的装饰器。
    """
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper


if __name__ == "__main__":
    # 通过装饰器运行异步测试 / Run async tests through decorator
    TestDeepSeekLLM.test_generate_with_xml_tags = async_test(TestDeepSeekLLM.test_generate_with_xml_tags)
    TestProcessWithDeepseek.test_process_with_deepseek = async_test(TestProcessWithDeepseek.test_process_with_deepseek)
    TestProcessWithDeepseek.test_process_with_deepseek_prefix_completion = async_test(TestProcessWithDeepseek.test_process_with_deepseek_prefix_completion)
    
    unittest.main()
