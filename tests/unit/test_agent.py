"""
Unit tests for the Agent class.

This module contains tests for the core Agent implementation in the
MCP Agent Framework.
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp_agent_framework.agent.agent import Agent
from mcp.server.fastmcp import Context, FastMCP

class TestAgent(unittest.TestCase):
    """
    Test cases for the Agent class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.config = {
            "api_key": "test-api-key",
            "model": "gpt-4"
        }
        self.agent = Agent("Test", "tester", self.config)
    
    def test_initialization(self):
        """
        Test agent initialization.
        """
        self.assertEqual(self.agent.name, "Test")
        self.assertEqual(self.agent.role, "tester")
        self.assertEqual(self.agent.config, self.config)
        self.assertIsNotNone(self.agent.id)
        self.assertEqual(len(self.agent.tasks), 0)
        self.assertEqual(len(self.agent.child_agents), 0)
        self.assertIsInstance(self.agent.mcp, FastMCP)
    
    def test_event_handlers(self):
        """
        Test event handler registration.
        """
        # Create a mock handler
        handler = MagicMock()
        
        # Register handler
        self.agent.on("task_created", handler)
        
        # Verify registration
        self.assertIn(handler, self.agent.event_handlers["task_created"])
        
        # Test invalid event
        with self.assertRaises(ValueError):
            self.agent.on("invalid_event", handler)
    
    @patch.object(Agent, '_trigger_event')
    async def test_process_task(self, mock_trigger_event):
        """
        Test task processing.
        """
        # Create a task
        task_id = "test-task-id"
        self.agent.tasks[task_id] = {
            "id": task_id,
            "description": "Test task",
            "status": "PENDING",
            "dependencies": [],
            "assigned_to": self.agent.id,
            "created_at": asyncio.get_event_loop().time()
        }
        
        # Process task
        await self.agent.process_task(task_id)
        
        # Check task status
        self.assertEqual(self.agent.tasks[task_id]["status"], "COMPLETED")
        self.assertIn("result", self.agent.tasks[task_id])
        
        # Check event triggering
        self.assertEqual(mock_trigger_event.call_count, 2)  # task_updated and task_completed
    
    async def test_trigger_event(self):
        """
        Test event triggering.
        """
        # Create mock handlers
        handler1 = MagicMock()
        handler2 = MagicMock()
        
        # Register handlers
        self.agent.on("task_created", handler1)
        self.agent.on("task_created", handler2)
        
        # Trigger event
        data = {"id": "test-id", "description": "Test task"}
        await self.agent._trigger_event("task_created", data)
        
        # Check handler calls
        handler1.assert_called_once_with(data)
        handler2.assert_called_once_with(data)
    
    def test_non_existent_task_processing(self):
        """
        Test processing a non-existent task.
        """
        with self.assertRaises(ValueError):
            asyncio.run(self.agent.process_task("non-existent-task"))

if __name__ == '__main__':
    unittest.main()
