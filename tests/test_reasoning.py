#!/usr/bin/env python3
"""
Tests for State-Based Reasoning Engine and Context Protocol
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

# State Graph imports
from shadowd.reasoning.state_graph import (
    State,
    StateTransition,
    StateNode,
    Transition,
    Checkpoint,
    ExecutionResult,
    StateGraph,
    StateGraphBuilder,
)

# Context Protocol imports
from shadowd.reasoning.context_protocol import (
    Permission,
    ResourceType,
    Capability,
    SecurityContext,
    ContextMessage,
    Resource,
    ContextProtocol,
)


class TestStateNode:
    """Test StateNode."""
    
    def test_create_node(self):
        """Test creating a state node."""
        node = StateNode(
            id="test_node",
            name="Test Node",
            state=State.EXECUTING,
        )
        
        assert node.id == "test_node"
        assert node.state == State.EXECUTING
        assert node.timeout == 60.0
    
    def test_node_to_dict(self):
        """Test node serialization."""
        node = StateNode(
            id="test",
            name="Test",
            state=State.PLANNING,
            timeout=30.0,
            max_retries=5,
        )
        
        data = node.to_dict()
        
        assert data["id"] == "test"
        assert data["state"] == "planning"
        assert data["timeout"] == 30.0


class TestTransition:
    """Test Transition."""
    
    def test_simple_transition(self):
        """Test simple state transition."""
        transition = Transition(
            from_state=State.INIT,
            to_state=State.PLANNING,
            event=StateTransition.START,
        )
        
        assert transition.matches(State.INIT, StateTransition.START, {})
        assert not transition.matches(State.EXECUTING, StateTransition.START, {})
    
    def test_conditional_transition(self):
        """Test transition with condition."""
        transition = Transition(
            from_state=State.EXECUTING,
            to_state=State.COMPLETED,
            event=StateTransition.COMPLETE,
            condition=lambda ctx: ctx.get("success", False),
        )
        
        assert transition.matches(State.EXECUTING, StateTransition.COMPLETE, {"success": True})
        assert not transition.matches(State.EXECUTING, StateTransition.COMPLETE, {"success": False})


class TestCheckpoint:
    """Test Checkpoint."""
    
    def test_create_checkpoint(self):
        """Test creating checkpoint."""
        checkpoint = Checkpoint(
            id="test_checkpoint",
            state=State.EXECUTING,
            context={"data": "test"},
            created_at=datetime.now(timezone.utc),
        )
        
        assert checkpoint.state == State.EXECUTING
        assert "data" in checkpoint.context
    
    def test_checkpoint_serialization(self):
        """Test checkpoint serialization."""
        checkpoint = Checkpoint(
            id="test",
            state=State.PLANNING,
            context={"key": "value"},
            created_at=datetime.now(timezone.utc),
            node_id="node_1",
        )
        
        data = checkpoint.to_dict()
        restored = Checkpoint.from_dict(data)
        
        assert restored.id == checkpoint.id
        assert restored.state == checkpoint.state
        assert restored.context == checkpoint.context


class TestStateGraph:
    """Test StateGraph."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def graph(self, temp_dir):
        return StateGraph(name="test_graph", checkpoint_dir=temp_dir)
    
    def test_create_graph(self, graph):
        """Test creating state graph."""
        assert graph.name == "test_graph"
        assert len(graph.nodes) == 0
    
    def test_add_node(self, graph):
        """Test adding nodes."""
        node = StateNode(id="start", name="Start", state=State.INIT)
        graph.add_node(node)
        
        assert "start" in graph.nodes
        assert graph.start_node == node
    
    def test_add_transition(self, graph):
        """Test adding transitions."""
        graph.add_transition(Transition(
            from_state=State.INIT,
            to_state=State.PLANNING,
            event=StateTransition.START,
        ))
        
        assert len(graph.transitions) == 1
    
    @pytest.mark.asyncio
    async def test_execute_simple_graph(self, graph):
        """Test executing a simple graph."""
        # Create nodes
        async def start_action(ctx):
            return {"started": True}
        
        async def plan_action(ctx):
            return {"plan": "test_plan"}
        
        node1 = StateNode(id="start", name="Start", state=State.INIT, action=start_action)
        node2 = StateNode(id="plan", name="Plan", state=State.PLANNING, action=plan_action)
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.set_start("start")
        
        # Add transitions
        graph.add_transition(Transition(State.INIT, State.PLANNING, StateTransition.START))
        graph.add_transition(Transition(State.PLANNING, State.COMPLETED, StateTransition.COMPLETE))
        
        # Execute
        result = await graph.execute(initial_context={})
        
        assert result.success
    
    def test_checkpoint_persistence(self, graph, temp_dir):
        """Test checkpoint persistence."""
        checkpoint = graph._create_checkpoint(State.EXECUTING)
        
        assert checkpoint.id is not None
        
        # Load from file
        loaded = graph.load_checkpoint(checkpoint.id)
        
        assert loaded is not None
        assert loaded.state == State.EXECUTING


class TestStateGraphBuilder:
    """Test StateGraphBuilder."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_build_simple_graph(self, temp_dir):
        """Test building simple graph."""
        async def plan(ctx):
            return {"plan": "test"}
        
        async def execute(ctx):
            return {"executed": True}
        
        graph = StateGraphBuilder("test").graph
        graph.checkpoint_dir = temp_dir
        
        # Add nodes manually
        start = StateNode(id="start", name="Start", state=State.INIT)
        plan_node = StateNode(id="planning", name="Planning", state=State.PLANNING, action=plan)
        exec_node = StateNode(id="executing", name="Executing", state=State.EXECUTING, action=execute)
        
        graph.add_node(start)
        graph.add_node(plan_node)
        graph.add_node(exec_node)
        graph.set_start("start")
        
        assert len(graph.nodes) == 3
        assert graph.start_node is not None
    
    def test_build_with_input(self, temp_dir):
        """Test building graph with input."""
        builder = StateGraphBuilder("input_test")
        builder.graph.checkpoint_dir = temp_dir
        
        start = StateNode(id="start", name="Start", state=State.INIT)
        builder.graph.add_node(start)
        builder.graph.set_start("start")
        
        wait_node = StateNode(
            id="waiting_input",
            name="Waiting",
            state=State.WAITING_INPUT,
        )
        builder.graph.add_node(wait_node)
        
        assert "waiting_input" in builder.graph.nodes


class TestCapability:
    """Test Capability."""
    
    def test_create_capability(self):
        """Test creating capability."""
        cap = Capability(
            name="file_access",
            permissions=[Permission.READ, Permission.WRITE],
            resources=[ResourceType.FILE],
        )
        
        assert cap.name == "file_access"
        assert Permission.READ in cap.permissions
    
    def test_capability_serialization(self):
        """Test capability serialization."""
        cap = Capability(
            name="test_cap",
            version="2.0",
            permissions=[Permission.ADMIN],
            resources=[ResourceType.MEMORY, ResourceType.TOOL],
        )
        
        data = cap.to_dict()
        restored = Capability.from_dict(data)
        
        assert restored.name == cap.name
        assert restored.version == cap.version


class TestSecurityContext:
    """Test SecurityContext."""
    
    def test_create_context(self):
        """Test creating security context."""
        context = SecurityContext(
            entity_id="agent_1",
            entity_type="agent",
            capabilities=[Capability(name="test")],
        )
        
        assert context.entity_id == "agent_1"
        assert context.session_id is not None
    
    def test_access_check(self):
        """Test access checking."""
        cap = Capability(
            name="file_ops",
            permissions=[Permission.READ, Permission.WRITE],
            resources=[ResourceType.FILE],
        )
        
        context = SecurityContext(
            entity_id="user_1",
            entity_type="user",
            capabilities=[cap],
        )
        
        # Should have access
        assert context.can_access(ResourceType.FILE, Permission.READ)
        assert context.can_access(ResourceType.FILE, Permission.WRITE)
        
        # Should not have access
        assert not context.can_access(ResourceType.NETWORK, Permission.EXECUTE)
    
    def test_expired_context(self):
        """Test expired context."""
        context = SecurityContext(
            entity_id="expired",
            entity_type="agent",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        
        # Should be expired
        assert not context.can_access(ResourceType.FILE, Permission.READ)


class TestResource:
    """Test Resource."""
    
    def test_create_resource(self):
        """Test creating resource."""
        resource = Resource(
            resource_id="file_1",
            resource_type=ResourceType.FILE,
            path="/data/file.txt",
            owner_id="user_1",
            required_permissions=[Permission.READ],
        )
        
        assert resource.resource_id == "file_1"
        assert resource.resource_type == ResourceType.FILE
    
    def test_restricted_resource(self):
        """Test restricted resource."""
        resource = Resource(
            resource_id="secret",
            resource_type=ResourceType.MEMORY,
            path="/secret/data",
            owner_id="admin",
            required_permissions=[Permission.ADMIN],
            allowed_entities=["admin", "user_2"],
        )
        
        assert resource.allowed_entities is not None
        assert "admin" in resource.allowed_entities


class TestContextProtocol:
    """Test ContextProtocol."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def protocol(self, temp_dir):
        return ContextProtocol(storage_path=temp_dir)
    
    def test_register_entity(self, protocol):
        """Test registering entity."""
        context = protocol.register_entity(
            entity_id="agent_1",
            entity_type="agent",
            capabilities=[Capability(name="test")],
        )
        
        assert context.entity_id == "agent_1"
        assert protocol.get_context("agent_1") is not None
    
    def test_unregister_entity(self, protocol):
        """Test unregistering entity."""
        protocol.register_entity("agent_1", "agent", [])
        protocol.unregister_entity("agent_1")
        
        assert protocol.get_context("agent_1") is None
    
    def test_register_resource(self, protocol):
        """Test registering resource."""
        resource = protocol.register_resource(
            resource_id="file_1",
            resource_type=ResourceType.FILE,
            path="/test/file.txt",
            owner_id="user_1",
            required_permissions=[Permission.READ],
        )
        
        assert resource.resource_id == "file_1"
        assert protocol.get_resource("file_1") is not None
    
    def test_request_access(self, protocol):
        """Test access request."""
        # Register entity with permissions
        protocol.register_entity(
            entity_id="user_1",
            entity_type="user",
            capabilities=[Capability(
                name="file_access",
                permissions=[Permission.READ],
                resources=[ResourceType.FILE],
            )],
        )
        
        # Register resource
        protocol.register_resource(
            resource_id="file_1",
            resource_type=ResourceType.FILE,
            path="/test/file",
            owner_id="admin",
            required_permissions=[Permission.READ],
        )
        
        # Request access
        can_access = protocol.request_access("user_1", "file_1", Permission.READ)
        
        assert can_access
    
    def test_negotiate_capabilities(self, protocol):
        """Test capability negotiation."""
        # Register two entities
        cap1 = Capability(
            name="shared",
            permissions=[Permission.READ, Permission.WRITE],
            resources=[ResourceType.FILE],
        )
        
        cap2 = Capability(
            name="shared",
            permissions=[Permission.READ],
            resources=[ResourceType.FILE, ResourceType.MEMORY],
        )
        
        protocol.register_entity("agent_1", "agent", [cap1])
        protocol.register_entity("agent_2", "agent", [cap2])
        
        # Negotiate
        result = protocol.negotiate_capabilities(
            "agent_1",
            "agent_2",
            [cap1],
        )
        
        assert result["success"]
        assert len(result["capabilities"]) > 0
    
    def test_create_message(self, protocol):
        """Test creating message."""
        protocol.register_entity("sender", "agent", [])
        protocol.register_entity("receiver", "agent", [])
        
        message = protocol.create_message(
            sender_id="sender",
            receiver_id="receiver",
            message_type="test",
            payload={"data": "test"},
        )
        
        assert message.id is not None
        assert message.signature is not None
    
    def test_verify_message(self, protocol):
        """Test message verification."""
        protocol.register_entity("sender", "agent", [])
        protocol.register_entity("receiver", "agent", [])
        
        message = protocol.create_message(
            sender_id="sender",
            receiver_id="receiver",
            message_type="test",
            payload={"data": "test"},
            sign=True,
        )
        
        assert protocol.verify_message(message)
    
    def test_list_accessible_resources(self, protocol):
        """Test listing accessible resources."""
        # Register entity
        protocol.register_entity(
            entity_id="user_1",
            entity_type="user",
            capabilities=[Capability(
                name="file_ops",
                permissions=[Permission.READ],
                resources=[ResourceType.FILE],
            )],
        )
        
        # Register resources
        protocol.register_resource(
            "file_1", ResourceType.FILE, "/data/file1.txt", "owner",
            [Permission.READ],
        )
        protocol.register_resource(
            "file_2", ResourceType.FILE, "/data/file2.txt", "owner",
            [Permission.WRITE],  # Requires write permission
        )
        protocol.register_resource(
            "mem_1", ResourceType.MEMORY, "/mem/data", "owner",
            [Permission.READ],
        )
        
        resources = protocol.list_accessible_resources("user_1", resource_type=ResourceType.FILE)
        
        # Should only see file_1 (has read permission)
        assert len(resources) == 1
        assert resources[0].resource_id == "file_1"
    
    def test_message_handler(self, protocol):
        """Test message handler registration."""
        received = []
        
        def handler(message: ContextMessage):
            received.append(message)
        
        protocol.register_entity("receiver", "agent", [])
        protocol.register_entity("sender", "agent", [])
        protocol.register_handler("receiver", "test_type", handler)
        
        message = protocol.create_message(
            sender_id="sender",
            receiver_id="receiver",
            message_type="test_type",
            payload={"test": "data"},
        )
        
        protocol.send_message(message)
        
        assert len(received) == 1
        assert received[0].payload["test"] == "data"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])