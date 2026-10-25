#!/usr/bin/env python3
"""
Tests for StateGraph (LangGraph-style workflow engine) - Fixed for actual implementation.

Tests cover:
- Node creation and execution
- Edge routing (conditional and unconditional)
- State management
- Checkpointing
- Error handling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Dict, Any, TypedDict
import pytest
import asyncio

from shadowd.state_graph import StateGraph, StateCheckpoint, Node, Edge, NodeStatus


class GraphState(TypedDict):
    """Test state type."""
    input: str
    output: str
    count: int
    history: list


class TestNodeCreation:
    """Tests for creating and managing nodes."""
    
    def test_create_simple_node(self):
        """Test creating a basic node."""
        def start_node(state: Dict) -> Dict:
            return {"output": "started"}
        
        graph = StateGraph()
        graph.add_node("start", start_node)
        
        assert "start" in graph.nodes
        assert graph.nodes["start"].name == "start"
        assert graph.nodes["start"].action == start_node
    
    def test_create_node_with_multiple_outputs(self):
        """Test node that returns multiple state updates."""
        def multi_output(state: Dict) -> Dict:
            return {
                "count": state.get("count", 0) + 1,
                "output": "incremented",
                "history": state.get("history", []) + ["multi"]
            }
        
        graph = StateGraph()
        graph.add_node("increment", multi_output)
        
        assert "increment" in graph.nodes
    
    def test_node_naming_conventions(self):
        """Test various valid node names."""
        graph = StateGraph()
        
        graph.add_node("start", lambda s: s)
        graph.add_node("end", lambda s: s)
        graph.add_node("process_data", lambda s: s)
        graph.add_node("node123", lambda s: s)
        
        assert len(graph.nodes) == 4
    
    def test_duplicate_node_names(self):
        """Test that duplicate node names are handled."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        
        # Should update the node
        graph.add_node("start", lambda s: {"output": "new"})
        
        assert "start" in graph.nodes
    
    def test_node_with_description(self):
        """Test node with description."""
        def process(state: Dict) -> Dict:
            return {"output": "processed"}
        
        graph = StateGraph()
        graph.add_node("process", process, description="Process the input")
        
        assert graph.nodes["process"].description == "Process the input"
    
    def test_node_with_retry_count(self):
        """Test node with custom retry count."""
        graph = StateGraph()
        graph.add_node("retry_node", lambda s: s, retry_count=5)
        
        assert graph.nodes["retry_node"].retry_count == 5


class TestEdgeRouting:
    """Tests for edge creation and routing."""
    
    def test_create_simple_edge(self):
        """Test creating an unconditional edge."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.add_node("end", lambda s: s)
        graph.add_edge("start", "end")
        
        assert len(graph.edges) == 1
        assert graph.edges[0].source == "start"
        assert graph.edges[0].target == "end"
    
    def test_create_conditional_edge(self):
        """Test creating a conditional edge."""
        def condition(state: Dict) -> bool:
            return state.get("count", 0) > 5
        
        graph = StateGraph()
        graph.add_node("check", lambda s: s)
        graph.add_node("yes", lambda s: s)
        graph.add_node("no", lambda s: s)
        graph.add_edge("check", "yes", condition=condition)
        
        assert graph.edges[0].condition == condition
    
    def test_edge_with_label(self):
        """Test edge with label."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.add_node("end", lambda s: s)
        graph.add_edge("start", "end", label="success")
        
        assert graph.edges[0].label == "success"
    
    def test_multiple_edges_from_node(self):
        """Test multiple edges from single node."""
        graph = StateGraph()
        graph.add_node("branch", lambda s: s)
        graph.add_node("path1", lambda s: s)
        graph.add_node("path2", lambda s: s)
        graph.add_edge("branch", "path1")
        graph.add_edge("branch", "path2")
        
        assert len(graph.edges) == 2


class TestStateManagement:
    """Tests for state management."""
    
    def test_initial_state(self):
        """Test initial state setup."""
        graph = StateGraph()
        
        assert graph.current_state == {}
        assert graph.entry_point is None
        assert graph.finish_nodes == []
    
    def test_set_entry_point(self):
        """Test setting entry point."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.set_entry_point("start")
        
        assert graph.entry_point == "start"
    
    def test_set_finish_point(self):
        """Test setting finish point."""
        graph = StateGraph()
        graph.add_node("end", lambda s: s)
        graph.set_finish_point("end")
        
        assert "end" in graph.finish_nodes
    
    def test_fluent_api(self):
        """Test fluent API for building graphs."""
        graph = StateGraph()
        
        graph.add_node("start", lambda s: s) \
             .add_node("end", lambda s: s) \
             .add_edge("start", "end") \
             .set_entry_point("start") \
             .set_finish_point("end")
        
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.entry_point == "start"
        assert "end" in graph.finish_nodes


class TestExecution:
    """Tests for graph execution."""
    
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test executing a simple graph."""
        async def start_node(state: Dict) -> Dict:
            return {"output": "started"}
        
        graph = StateGraph()
        graph.add_node("start", start_node)
        graph.set_entry_point("start")
        graph.set_finish_point("start")
        
        result = await graph.execute({"input": "test"})
        
        assert result["output"] == "started"
        assert result["input"] == "test"
    
    @pytest.mark.asyncio
    async def test_multi_node_execution(self):
        """Test executing a multi-node graph."""
        async def node1(state: Dict) -> Dict:
            return {"count": 1}
        
        async def node2(state: Dict) -> Dict:
            return {"count": state.get("count", 0) + 1}
        
        graph = StateGraph()
        graph.add_node("start", node1)
        graph.add_node("increment", node2)
        graph.add_edge("start", "increment")
        graph.set_entry_point("start")
        graph.set_finish_point("increment")
        
        result = await graph.execute({})
        
        assert result["count"] == 2
    
    @pytest.mark.asyncio
    async def test_conditional_execution(self):
        """Test executing with conditional edges."""
        async def check(state: Dict) -> Dict:
            return {"checked": True}
        
        async def yes_path(state: Dict) -> Dict:
            return {"result": "yes"}
        
        async def no_path(state: Dict) -> Dict:
            return {"result": "no"}
        
        def go_yes(state: Dict) -> bool:
            return state.get("value", 0) > 5
        
        graph = StateGraph()
        graph.add_node("check", check)
        graph.add_node("yes", yes_path)
        graph.add_node("no", no_path)
        graph.add_edge("check", "yes", condition=go_yes)
        graph.add_edge("check", "no")
        graph.set_entry_point("check")
        graph.set_finish_point("yes")
        graph.set_finish_point("no")
        
        result = await graph.execute({"value": 10})
        assert result["result"] == "yes"


class TestCheckpointing:
    """Tests for checkpoint management."""
    
    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        checkpoint = StateCheckpoint(
            node_id="test_node",
            timestamp=datetime.now(),
            state={"count": 5},
            status=NodeStatus.SUCCESS
        )
        
        assert checkpoint.node_id == "test_node"
        assert checkpoint.status == NodeStatus.SUCCESS
    
    def test_checkpoint_serialization(self):
        """Test checkpoint to_dict and from_dict."""
        checkpoint = StateCheckpoint(
            node_id="node1",
            timestamp=datetime.now(),
            state={"key": "value"},
            status=NodeStatus.SUCCESS,
            error=None
        )
        
        d = checkpoint.to_dict()
        assert d["node_id"] == "node1"
        
        restored = StateCheckpoint.from_dict(d)
        assert restored.node_id == checkpoint.node_id
        assert restored.status == NodeStatus.SUCCESS
    
    def test_checkpoint_with_error(self):
        """Test checkpoint with error."""
        checkpoint = StateCheckpoint(
            node_id="failed_node",
            timestamp=datetime.now(),
            state={},
            status=NodeStatus.FAILED,
            error="Something went wrong"
        )
        
        assert checkpoint.error == "Something went wrong"
    
    def test_graph_checkpoints_list(self):
        """Test graph maintains checkpoint list."""
        graph = StateGraph()
        
        assert graph.checkpoints == []


class TestNodeStatus:
    """Tests for NodeStatus enum."""
    
    def test_status_values(self):
        """Test NodeStatus values."""
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.RUNNING.value == "running"
        assert NodeStatus.SUCCESS.value == "success"
        assert NodeStatus.FAILED.value == "failed"
        assert NodeStatus.SKIPPED.value == "skipped"


class TestNodeDataclass:
    """Tests for Node dataclass."""
    
    def test_node_creation(self):
        """Test creating a Node."""
        def action(state: Dict) -> Dict:
            return state
        
        node = Node(
            id="test",
            name="test",
            action=action,
            description="Test node"
        )
        
        assert node.id == "test"
        assert node.name == "test"
        assert node.action == action
        assert node.description == "Test node"
        assert node.retry_count == 3  # default
        assert node.timeout == 300  # default


class TestEdgeDataclass:
    """Tests for Edge dataclass."""
    
    def test_edge_creation(self):
        """Test creating an Edge."""
        edge = Edge(source="start", target="end")
        
        assert edge.source == "start"
        assert edge.target == "end"
        assert edge.condition is None
        assert edge.label == ""
    
    def test_edge_with_condition(self):
        """Test edge with condition."""
        def cond(state: Dict) -> bool:
            return True
        
        edge = Edge(source="start", target="end", condition=cond)
        
        assert edge.condition == cond
    
    def test_edge_with_label(self):
        """Test edge with label."""
        edge = Edge(source="start", target="end", label="success")
        
        assert edge.label == "success"


class TestGraphMethods:
    """Tests for additional graph methods."""
    
    def test_get_execution_path(self):
        """Test getting execution path."""
        graph = StateGraph()
        
        path = graph.get_execution_path()
        
        assert path == []
    
    def test_get_state_at_node(self):
        """Test getting state at specific node."""
        graph = StateGraph()
        
        state = graph.get_state_at_node("nonexistent")
        
        assert state is None
    
    def test_visualize(self):
        """Test graph visualization."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.add_node("end", lambda s: s)
        graph.add_edge("start", "end")
        
        viz = graph.visualize()
        
        assert "start" in viz
        assert "end" in viz


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])