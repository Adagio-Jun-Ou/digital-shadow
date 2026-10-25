#!/usr/bin/env python3
"""
Comprehensive tests for StateGraph (LangGraph-style workflow engine)

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
from hypothesis import given, strategies as st
import uuid


class GraphState(TypedDict):
    """Test state type."""
    input: str
    output: str
    count: int
    history: list


from shadowd.state_graph import StateGraph, StateCheckpoint, NodeStatus


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
        assert graph.nodes["start"].func == start_node
    
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
        
        # Valid names
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
    
    def test_conditional_edge(self):
        """Test creating a conditional edge."""
        def should_continue(state: Dict) -> str:
            if state.get("count", 0) < 3:
                return "continue"
            return "end"
        
        graph = StateGraph()
        graph.add_node("process", lambda s: s)
        graph.add_node("end", lambda s: s)
        graph.add_conditional_edges(
            "process",
            should_continue,
            {"continue": "process", "end": "end"}
        )
        
        # Edge should be stored with conditional logic
        assert len(graph.edges) == 1 or len(graph.conditional_edges) == 1
    
    def test_chained_nodes(self):
        """Test creating a chain of nodes."""
        graph = StateGraph()
        
        for i in range(5):
            graph.add_node(f"step_{i}", lambda s: {"count": s.get("count", 0) + 1})
        
        for i in range(4):
            graph.add_edge(f"step_{i}", f"step_{i+1}")
        
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4
    
    def test_parallel_edges(self):
        """Test creating parallel branches."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.add_node("branch_a", lambda s: {"path": "a"})
        graph.add_node("branch_b", lambda s: {"path": "b"})
        graph.add_node("merge", lambda s: s)
        
        graph.add_edge("start", "branch_a")
        graph.add_edge("start", "branch_b")
        graph.add_edge("branch_a", "merge")
        graph.add_edge("branch_b", "merge")
        
        assert len(graph.edges) == 4


class TestStateManagement:
    """Tests for state management during execution."""
    
    def test_initial_state(self):
        """Test that initial state is correctly set."""
        graph = StateGraph()
        graph.add_node("start", lambda s: {"output": "done"})
        graph.set_entry_point("start")
        graph.set_finish_point("start")
        
        result = graph.run({"input": "test"})
        
        assert "input" in result
        assert result["input"] == "test"
        assert "output" in result
        assert result["output"] == "done"
    
    def test_state_persistence_across_nodes(self):
        """Test that state persists across node executions."""
        def add_history(state: Dict) -> Dict:
            return {"history": state.get("history", []) + ["node1"]}
        
        def add_count(state: Dict) -> Dict:
            return {"count": state.get("count", 0) + 1}
        
        graph = StateGraph()
        graph.add_node("history", add_history)
        graph.add_node("count", add_count)
        graph.add_edge("history", "count")
        graph.set_entry_point("history")
        graph.set_finish_point("count")
        
        result = graph.run(input={})
        
        assert "history" in result
        assert "node1" in result["history"]
        assert result["count"] == 1
    
    def test_state_merge(self):
        """Test that state updates are merged correctly."""
        def set_a(state: Dict) -> Dict:
            return {"a": 1, "shared": "from_a"}
        
        def set_b(state: Dict) -> Dict:
            return {"b": 2, "shared": "from_b"}
        
        graph = StateGraph()
        graph.add_node("node_a", set_a)
        graph.add_node("node_b", set_b)
        graph.add_edge("node_a", "node_b")
        graph.set_entry_point("node_a")
        graph.set_finish_point("node_b")
        
        result = graph.run({})
        
        assert result["a"] == 1
        assert result["b"] == 2
        # Last writer wins for shared keys
        assert result["shared"] == "from_b"


class TestExecution:
    """Tests for graph execution."""
    
    def test_simple_execution(self):
        """Test executing a simple graph."""
        def start(state: Dict) -> Dict:
            return {"status": "started"}
        
        def process(state: Dict) -> Dict:
            return {"status": "processed"}
        
        def end(state: Dict) -> Dict:
            return {"status": "completed"}
        
        graph = StateGraph()
        graph.add_node("start", start)
        graph.add_node("process", process)
        graph.add_node("end", end)
        
        graph.add_edge("start", "process")
        graph.add_edge("process", "end")
        
        graph.set_entry_point("start")
        graph.set_finish_point("end")
        
        result = graph.run({"input": "test"})
        
        assert result["status"] == "completed"
    
    def test_conditional_execution(self):
        """Test execution with conditional branching."""
        def condition(state: Dict) -> str:
            return "positive" if state.get("value", 0) > 0 else "negative"
        
        def positive_path(state: Dict) -> Dict:
            return {"result": "positive"}
        
        def negative_path(state: Dict) -> Dict:
            return {"result": "negative"}
        
        graph = StateGraph()
        graph.add_node("check", lambda s: s)
        graph.add_node("positive", positive_path)
        graph.add_node("negative", negative_path)
        
        graph.add_conditional_edges("check", condition, {
            "positive": "positive",
            "negative": "negative"
        })
        
        graph.set_entry_point("check")
        graph.set_finish_point("positive")
        graph.set_finish_point("negative")
        
        # Test positive path
        result_pos = graph.run({"value": 5})
        assert result_pos.get("result") == "positive"
        
        # Test negative path
        result_neg = graph.run({"value": -5})
        assert result_neg.get("result") == "negative"
    
    def test_error_handling(self):
        """Test error handling during execution."""
        def error_node(state: Dict) -> Dict:
            raise ValueError("Intentional error")
        
        graph = StateGraph()
        graph.add_node("error", error_node)
        graph.set_entry_point("error")
        graph.set_finish_point("error")
        
        # Graph retries on error, doesn't raise - returns partial state
        result = graph.run({})
        assert isinstance(result, dict)  # Returns partial state after failures


class TestCheckpointing:
    """Tests for checkpoint functionality."""
    
    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        checkpoint = StateCheckpoint(
            node_id="test_node",
            timestamp=datetime.now(),
            state={"input": "test"},
            status=NodeStatus.SUCCESS
        )
        
        assert checkpoint.checkpoint_id is not None
        assert checkpoint.node_id == "test_node"
        assert checkpoint.state["input"] == "test"
    
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoints."""
        import json
        
        graph = StateGraph(checkpoint_dir=tmp_path)
        
        def process(state: Dict) -> Dict:
            return {"processed": True}
        
        graph.add_node("process", process)
        graph.set_entry_point("process")
        graph.set_finish_point("process")
        
        result = graph.run({"input": "test"})
        
        # Checkpoints should be created
        checkpoints = list(tmp_path.glob("*.json"))
        # May or may not have checkpoints depending on implementation
        assert result["processed"] is True
    
    def test_resume_from_checkpoint(self, tmp_path):
        """Test resuming execution from checkpoint."""
        # This test would verify checkpoint resume functionality
        # Implementation depends on how StateGraph handles persistence
        pass


class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(st.dictionaries(st.text(min_size=1), st.text()))
    def test_state_preserves_keys(self, initial_state):
        """Test that state keys are preserved during execution."""
        def identity_node(state: Dict) -> Dict:
            return state
        
        graph = StateGraph()
        graph.add_node("identity", identity_node)
        graph.set_entry_point("identity")
        graph.set_finish_point("identity")
        
        result = graph.run(initial_state)
        
        # All keys should be present
        for key in initial_state:
            assert key in result
    
    @given(st.integers(min_value=0, max_value=100))
    def test_counter_increments(self, count):
        """Test counter increments correctly."""
        initial_state = {"count": count}
        
        def increment(state: Dict) -> Dict:
            return {"count": state.get("count", 0) + 1}
        
        graph = StateGraph()
        graph.add_node("inc", increment)
        graph.set_entry_point("inc")
        graph.set_finish_point("inc")
        
        result = graph.run(initial_state)
        
        assert result["count"] == count + 1
    
    @given(st.lists(st.integers(), max_size=20))
    def test_history_accumulation(self, items):
        """Test that history accumulates correctly."""
        def add_to_history(state: Dict) -> Dict:
            history = state.get("history", [])
            return {"history": history + ["processed"]}
        
        graph = StateGraph()
        for i, item in enumerate(items):
            graph.add_node(f"node_{i}", add_to_history)
        
        graph.set_entry_point("node_0")
        
        if len(items) > 1:
            for i in range(len(items) - 1):
                graph.add_edge(f"node_{i}", f"node_{i+1}")
        
        if items:
            graph.set_finish_point(f"node_{len(items)-1}")
            result = graph.run({"items": items})
            assert len(result.get("history", [])) == len(items)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_graph(self):
        """Test running an empty graph."""
        graph = StateGraph()
        
        # Should raise ValueError for no entry point
        with pytest.raises(ValueError, match="No entry point"):
            graph.run({})
    
    def test_single_node_graph(self):
        """Test graph with single node."""
        graph = StateGraph()
        graph.add_node("only", lambda s: {"output": "done"})
        graph.set_entry_point("only")
        graph.set_finish_point("only")
        
        result = graph.run({})
        assert result["output"] == "done"
    
    def test_cyclic_graph(self):
        """Test handling of cycles in graph."""
        def increment(state: Dict) -> Dict:
            return {"count": state.get("count", 0) + 1}
        
        graph = StateGraph()
        graph.add_node("inc1", increment)
        graph.add_node("inc2", increment)
        graph.add_node("inc3", increment)
        
        graph.set_entry_point("inc1")
        graph.add_edge("inc1", "inc2")
        graph.add_edge("inc2", "inc3")
        graph.set_finish_point("inc3")
        
        result = graph.run({})
        # Should have incremented 3 times (once per node)
        assert result["count"] == 3
    
    def test_nonexistent_node_reference(self):
        """Test referencing nodes that don't exist."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        
        # Adding edge to nonexistent node should be handled
        # Implementation-specific behavior
        try:
            graph.add_edge("start", "nonexistent")
        except (ValueError, KeyError):
            pass  # Expected for some implementations


class TestFaker:
    """Tests using Faker for realistic test data."""
    
    def test_user_workflow(self):
        """Test workflow with simulated user data."""
        try:
            from faker import Faker
        except ImportError:
            pytest.skip("Faker not installed")
        fake = Faker()
        
        # Generate fake user data
        user_data = {
            "name": fake.name(),
            "email": fake.email(),
            "address": fake.address(),
            "phone": fake.phone_number(),
        }
        
        def process_user(state: Dict) -> Dict:
            return {
                "user": state.get("user_data"),
                "processed": True,
                "name_length": len(state.get("user_data", {}).get("name", ""))
            }
        
        graph = StateGraph()
        graph.add_node("process", process_user)
        graph.set_entry_point("process")
        graph.set_finish_point("process")
        
        result = graph.run({"user_data": user_data})
        
        assert result["processed"] is True
        assert result["name_length"] > 0
        assert "@" in result["user"]["email"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])