#!/usr/bin/env python3
"""
Tests for Agent Coordination System

Tests:
- ShadowAgent creation and execution
- ShadowCrew task assignment and kickoff
- AgentSwarm coordination
- Message passing
- Tool execution
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowd.agent_coordination import (
    ShadowAgent,
    ShadowCrew,
    AgentSwarm,
    AgentRole,
    TaskStatus,
    AgentMessage,
    ShadowTask,
    AgentTool
)
from shadowd.block_memory import BlockBasedMemory


class TestAgentMessage:
    """Tests for AgentMessage."""
    
    def test_create_message(self):
        """Test creating a message."""
        msg = AgentMessage(
            sender="agent1",
            receiver="agent2",
            content="Hello!"
        )
        assert msg.sender == "agent1"
        assert msg.receiver == "agent2"
        assert msg.content == "Hello!"
        assert isinstance(msg.timestamp, datetime)
    
    def test_message_serialization(self):
        """Test message to_dict and from_dict."""
        msg = AgentMessage(
            sender="agent1",
            receiver="agent2",
            content="Test message",
            metadata={"priority": "high"}
        )
        
        msg_dict = msg.to_dict()
        assert msg_dict["sender"] == "agent1"
        assert msg_dict["content"] == "Test message"
        
        msg2 = AgentMessage.from_dict(msg_dict)
        assert msg2.sender == msg.sender
        assert msg2.content == msg.content
        assert msg2.metadata == msg.metadata


class TestShadowTask:
    """Tests for ShadowTask."""
    
    def test_create_task(self):
        """Test creating a task."""
        task = ShadowTask(
            id="task_1",
            description="Test task",
            goal="Execute test"
        )
        assert task.id == "task_1"
        assert task.status == TaskStatus.PENDING
        assert task.assigned_agent is None
    
    def test_task_status_transition(self):
        """Test task status changes."""
        task = ShadowTask(
            id="task_1",
            description="Test task",
            goal="Execute test"
        )
        
        task.status = TaskStatus.ASSIGNED
        assert task.status == TaskStatus.ASSIGNED
        
        task.status = TaskStatus.IN_PROGRESS
        assert task.status == TaskStatus.IN_PROGRESS
        
        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED
    
    def test_task_dependencies(self):
        """Test task dependencies."""
        task1 = ShadowTask(id="task_1", description="First", goal="Test")
        task2 = ShadowTask(
            id="task_2",
            description="Second",
            goal="Test",
            dependencies=["task_1"]
        )
        
        assert "task_1" in task2.dependencies
        assert len(task2.dependencies) == 1
    
    def test_task_serialization(self):
        """Test task to_dict and from_dict."""
        task = ShadowTask(
            id="task_1",
            description="Test task",
            goal="Execute test",
            context={"key": "value"},
            dependencies=["task_0"]
        )
        
        task_dict = task.to_dict()
        assert task_dict["id"] == "task_1"
        assert task_dict["goal"] == "Execute test"
        
        task2 = ShadowTask.from_dict(task_dict)
        assert task2.id == task.id
        assert task2.description == task.description
        assert task2.dependencies == task.dependencies


class TestAgentTool:
    """Tests for AgentTool."""
    
    def test_sync_tool(self):
        """Test synchronous tool execution."""
        def sync_handler(query: str):
            return f"Result for: {query}"
        
        tool = AgentTool(
            name="test_tool",
            handler=sync_handler,
            description="Test tool"
        )
        
        # asyncio.get_event_loop() is deprecated and may raise RuntimeError
        # in 3.10+ when called outside a coroutine without an existing loop.
        # Use asyncio.run, which creates and tears down its own loop.
        result = asyncio.run(tool.execute(query="test"))
        assert result == "Result for: test"

    def test_async_tool(self):
        """Test async tool execution."""
        async def async_handler(query: str):
            await asyncio.sleep(0.01)
            return f"Async result for: {query}"

        tool = AgentTool(
            name="async_tool",
            handler=async_handler,
            description="Async test tool"
        )

        async def run_test():
            result = await tool.execute(query="test")
            assert result == "Async result for: test"

        asyncio.run(run_test())


class TestShadowAgent:
    """Tests for ShadowAgent."""
    
    def test_create_agent(self):
        """Test creating an agent."""
        agent = ShadowAgent(
            agent_id="test_agent",
            role=AgentRole.RESEARCHER,
            goal="Find information",
            backstory="Expert researcher"
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.role == AgentRole.RESEARCHER
        assert agent.goal == "Find information"
        assert len(agent.inbox) == 0
        assert len(agent.outbox) == 0
    
    def test_agent_memory_initialization(self):
        """Test agent memory initialization."""
        agent = ShadowAgent(
            agent_id="test_agent",
            role=AgentRole.EXECUTOR,
            goal="Execute tasks"
        )
        
        assert agent.memory is not None
        assert "role" in agent.memory.core_memory.blocks
    
    def test_system_prompt(self):
        """Test agent system prompt generation."""
        agent = ShadowAgent(
            agent_id="test_agent",
            role=AgentRole.COORDINATOR,
            goal="Coordinate tasks",
            backstory="Expert coordinator"
        )
        
        prompt = agent.system_prompt()
        assert "test_agent" in prompt
        assert "coordinator" in prompt.lower()
        assert "Coordinate tasks" in prompt
    
    @pytest.mark.asyncio
    async def test_agent_with_tools(self):
        """Test agent with tools."""
        async def search_web(query: str) -> str:
            return f"results for {query}"
        
        tool = AgentTool(name="search", handler=search_web, description="Search")
        
        agent = ShadowAgent(
            agent_id="tool_agent",
            role=AgentRole.EXECUTOR,
            goal="Execute tasks",
            tools=[tool]
        )
        
        assert "search" in agent.tools
        
        result = await agent.use_tool("search", query="test query")
        assert "results" in result
    
    @pytest.mark.asyncio
    async def test_agent_receive_message(self):
        """Test agent receiving a message."""
        agent = ShadowAgent(
            agent_id="agent1",
            role=AgentRole.COORDINATOR,
            goal="Coordinate"
        )
        
        msg = AgentMessage(
            sender="agent2",
            receiver="agent1",
            content="Hello agent1"
        )
        
        await agent.receive_message(msg)
        
        assert len(agent.inbox) == 1
        assert agent.inbox[0].content == "Hello agent1"
    
    @pytest.mark.asyncio
    async def test_agent_send_message(self):
        """Test agent sending a message."""
        agent = ShadowAgent(
            agent_id="agent1",
            role=AgentRole.COORDINATOR,
            goal="Coordinate"
        )
        
        await agent.send_message("agent2", "Hello from agent1")
        
        assert len(agent.outbox) == 1
        assert agent.outbox[0].content == "Hello from agent1"
    
    def test_agent_get_status(self):
        """Test agent status retrieval."""
        agent = ShadowAgent(
            agent_id="test_agent",
            role=AgentRole.EXECUTOR,
            goal="Execute"
        )
        
        status = agent.get_status()
        assert status["agent_id"] == "test_agent"
        assert status["role"] == "executor"
        assert status["inbox_count"] == 0
        assert status["task_history_count"] == 0


class TestShadowCrew:
    """Tests for ShadowCrew."""
    
    def test_create_crew(self):
        """Test creating a crew."""
        crew = ShadowCrew(
            crew_id="test_crew",
            name="Test Crew",
            process="sequential"
        )
        
        assert crew.crew_id == "test_crew"
        assert crew.name == "Test Crew"
        assert crew.process == "sequential"
        assert len(crew.agents) > 0  # Default agents
    
    def test_add_custom_agent(self):
        """Test adding custom agent to crew."""
        crew = ShadowCrew(crew_id="test_crew")
        
        custom_agent = ShadowAgent(
            agent_id="custom",
            role=AgentRole.ANALYZER,
            goal="Analyze data"
        )
        
        crew.add_agent(custom_agent)
        
        assert "custom" in crew.agents
        assert crew.agents["custom"].role == AgentRole.ANALYZER
    
    def test_create_and_assign_task(self):
        """Test creating and assigning tasks."""
        crew = ShadowCrew(crew_id="test_crew")
        
        task = crew.create_task(
            description="Research topic",
            goal="research"
        )
        
        assert task.id in crew.tasks
        assert task.status == TaskStatus.PENDING
        
        assigned = crew.assign_task(task.id, "researcher")
        assert assigned is True
        assert task.assigned_agent == "researcher"
        assert task.status == TaskStatus.ASSIGNED
    
    def test_task_dependencies(self):
        """Test task dependency handling."""
        crew = ShadowCrew(crew_id="test_crew")
        
        task1 = crew.create_task(
            description="First task",
            goal="research"
        )
        
        task2 = crew.create_task(
            description="Second task",
            goal="analysis",
            dependencies=[task1.id]
        )
        
        assert task1.id in task2.dependencies
    
    def test_dependencies_met_check(self):
        """Test dependencies met checking."""
        crew = ShadowCrew(crew_id="test_crew")
        
        task1 = crew.create_task(
            description="First task",
            goal="research"
        )
        
        task2 = crew.create_task(
            description="Second task",
            goal="analysis",
            dependencies=[task1.id]
        )
        
        assert not crew._dependencies_met(task2)
        
        task1.status = TaskStatus.COMPLETED
        
        assert crew._dependencies_met(task2)
    
    def test_agent_selection(self):
        """Test agent selection for tasks."""
        crew = ShadowCrew(crew_id="test_crew")
        
        task = crew.create_task(
            description="Find information",
            goal="research"
        )
        
        agent_id = crew._select_agent_for_task(task)
        assert agent_id is not None
        assert "researcher" in agent_id
    
    @pytest.mark.asyncio
    async def test_crew_kickoff(self):
        """Test crew kickoff."""
        crew = ShadowCrew(crew_id="test_crew", name="Test Crew")
        
        agent = ShadowAgent(
            agent_id="worker1",
            role=AgentRole.EXECUTOR,
            goal="Complete tasks"
        )
        crew.add_agent(agent)
        
        task = ShadowTask(
            id="task1",
            description="Test task",
            goal="Test goal"
        )
        crew.add_task(task)
        
        result = await crew.kickoff(goal="Complete the test task")
        
        assert "tasks_completed" in result
        assert len(result["tasks_completed"]) > 0
    
    def test_swarm_status(self):
        """Test swarm status retrieval."""
        swarm = AgentSwarm()
        crew = ShadowCrew(crew_id="test_crew")
        swarm.add_crew(crew)
        
        status = swarm.get_status()
        assert status["crews_count"] == 1
        assert "test_crew" in status["crews"]


class TestIntegration:
    """Integration tests for agent coordination."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full workflow with agents and crew."""
        swarm = AgentSwarm()
        
        crew = ShadowCrew(crew_id="workflow_crew", name="Workflow Crew")
        
        agent = ShadowAgent(
            agent_id="agent1",
            role=AgentRole.EXECUTOR,
            goal="Execute tasks"
        )
        crew.add_agent(agent)
        
        task = ShadowTask(
            id="task1",
            description="Process user input",
            goal="Process input"
        )
        crew.add_task(task)
        
        swarm.add_crew(crew)
        
        result = await crew.kickoff("Process user input")
        
        assert "tasks_completed" in result
        assert len(result["tasks_completed"]) > 0
        
        agent_status = crew.get_agent_status()
        assert len(agent_status) > 0
    
    def test_agent_memory_persistence(self):
        """Test agent memory persistence across tasks."""
        memory = BlockBasedMemory(Path(tempfile.mkdtemp()))
        
        agent = ShadowAgent(
            agent_id="mem_agent",
            role=AgentRole.EXECUTOR,
            goal="Execute",
            memory=memory
        )
        
        memory.core_append("human", "User likes Python")
        
        assert "Python" in memory.core_get("human")
        
        assert agent.memory.core_get("human") == memory.core_get("human")
    
    @pytest.mark.asyncio
    async def test_tool_execution_in_task(self):
        """Test tool execution during task."""
        def search_tool(query: str):
            return {"results": ["result1", "result2"]}
        
        tool = AgentTool(
            name="search",
            handler=search_tool,
            description="Search tool"
        )
        
        agent = ShadowAgent(
            agent_id="tool_agent",
            role=AgentRole.RESEARCHER,
            goal="Research",
            tools=[tool]
        )
        
        task = ShadowTask(
            id="task_1",
            description="Search for information",
            goal="research"
        )
        
        result = await agent.execute_task(task)
        assert result["task_id"] == "task_1"
        assert len(agent.task_history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])