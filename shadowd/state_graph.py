#!/usr/bin/env python3
"""
State Graph Implementation - Based on LangGraph Research

Implements:
- StateGraph with nodes and edges
- Conditional routing
- Durable execution state
- Checkpointing
"""

from typing import Dict, List, Optional, Any, Callable, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import os
import re as _re
import json
import uuid
import copy
import threading
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_SANITIZE_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StateCheckpoint:
    """Checkpoint for durable execution."""
    node_id: str
    timestamp: datetime
    state: Dict[str, Any]
    status: NodeStatus
    error: Optional[str] = None
    checkpoint_id: str = ""  # Optional checkpoint ID
    
    def __post_init__(self):
        if not self.checkpoint_id:
            self.checkpoint_id = f"cp_{uuid.uuid4().hex[:12]}"
        self.node_id = _SANITIZE_STRICT_RE.sub('', str(self.node_id))[:100]
        if self.error:
            self.error = _SANITIZE_RE.sub('', str(self.error))[:200]
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "state": copy.deepcopy(self.state) if self.state else {},
            "status": self.status.value,
            "error": self.error[:200] if self.error else None,
            "checkpoint_id": self.checkpoint_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StateCheckpoint':
        _ts = data.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(_ts) if _ts else datetime.now(timezone.utc)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)
        return cls(
            node_id=data["node_id"],
            timestamp=timestamp,
            state=data["state"],
            status=NodeStatus(data["status"]),
            error=data.get("error"),
            checkpoint_id=data.get("checkpoint_id", "")
        )


@dataclass
class Node:
    """A node in the state graph."""
    id: str
    name: str
    action: Callable
    description: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: int = 300
    
    @property
    def func(self) -> Callable:
        """Alias for action (for compatibility)."""
        return self.action


@dataclass
class Edge:
    """An edge between nodes."""
    source: str
    target: str
    condition: Optional[Callable] = None
    label: str = ""


class StateGraph:
    """
    State graph for agent workflows.
    
    Based on LangGraph pattern:
    - Define nodes with actions
    - Define edges with optional conditions
    - Execute with state management
    - Support checkpointing for durability
    """
    
    def __init__(self, state_schema: TypedDict = None, name: str = "workflow", checkpoint_dir: Path = None):
        self.name = _SANITIZE_STRICT_RE.sub('', str(name))[:200]
        self.state_schema = state_schema
        self.checkpoint_dir = checkpoint_dir
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.conditional_edges: List[Dict] = []  # Track conditional edge definitions
        self.entry_point: Optional[str] = None
        self.finish_nodes: List[str] = []
        self.checkpoints: List[StateCheckpoint] = []
        self._MAX_CHECKPOINTS = 100
        self.current_state: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
    def add_node(self, node_id: str, action: Callable, description: str = "", 
                  retry_count: int = 3, timeout: int = 300) -> 'StateGraph':
        """Add a node to the graph."""
        node = Node(
            id=_SANITIZE_STRICT_RE.sub('', str(node_id))[:100],
            name=_SANITIZE_STRICT_RE.sub('', str(node_id))[:100],
            action=action,
            description=_SANITIZE_RE.sub('', str(description))[:500],
            retry_count=max(0, min(retry_count, 10)),
            timeout=max(1, min(timeout, 3600))
        )
        with self._lock:
            if len(self.nodes) >= 200:
                oldest_key = next(iter(self.nodes))
                logger.warning("Evicting node %s", oldest_key)
                del self.nodes[oldest_key]
            self.nodes[node.id] = node
        return self
    
    def add_edge(self, source: str, target: str, condition: Callable = None, 
                  label: str = "") -> 'StateGraph':
        """Add an edge between nodes."""
        edge = Edge(
            source=_SANITIZE_STRICT_RE.sub('', str(source))[:100],
            target=_SANITIZE_STRICT_RE.sub('', str(target))[:100],
            condition=condition,
            label=_SANITIZE_RE.sub('', str(label))[:200]
        )
        with self._lock:
            if len(self.edges) >= 1000:
                self.edges = self.edges[-999:]
            self.edges.append(edge)
        return self
    
    def add_conditional_edges(self, source: str, condition: Callable, 
                               targets: Dict[str, str]) -> 'StateGraph':
        """Add conditional edges from a node."""
        with self._lock:
            if len(self.conditional_edges) >= 500:
                self.conditional_edges = self.conditional_edges[-499:]
            self.conditional_edges.append({
                "source": _SANITIZE_STRICT_RE.sub('', str(source))[:100],
                "condition": condition,
                "targets": targets
            })
            for result_value, target in targets.items():
                def condition_wrapper(state, val=result_value):
                    cond_result = condition(state)
                    return cond_result == val
                edge = Edge(
                    source=_SANITIZE_STRICT_RE.sub('', str(source))[:100],
                    target=_SANITIZE_STRICT_RE.sub('', str(target))[:100],
                    condition=condition_wrapper,
                    label=_SANITIZE_RE.sub('', str(result_value))[:200]
                )
                self.edges.append(edge)
        return self
    
    def run(self, initial_state: Dict[str, Any] = None, input: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous run method (wraps async execute)."""
        import asyncio
        # Support both positional and keyword arguments
        state = input or initial_state or {}
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            # Already in async context, return a coroutine
            return self.execute(state, self.checkpoint_dir)
        else:
            return asyncio.run(self.execute(state, self.checkpoint_dir))
    
    def set_entry_point(self, node_id: str) -> 'StateGraph':
        """Set the entry point for the graph."""
        with self._lock:
            self.entry_point = _SANITIZE_STRICT_RE.sub('', str(node_id))[:100]
        return self
    
    def set_finish_point(self, node_id: str) -> 'StateGraph':
        """Mark a node as a finish point."""
        with self._lock:
            self.finish_nodes.append(_SANITIZE_STRICT_RE.sub('', str(node_id))[:100])
        return self
    
    async def execute(self, initial_state: Dict[str, Any], 
                       checkpoint_dir: Path = None) -> Dict[str, Any]:
        """
        Execute the state graph.
        
        Args:
            initial_state: Starting state
            checkpoint_dir: Directory to save checkpoints (for durability)
        
        Returns:
            Final state
        """
        if not self.entry_point:
            raise ValueError("No entry point set")
        
        self.current_state = copy.deepcopy(initial_state) if initial_state else {}
        current_node_id = self.entry_point
        visited = set()
        
        while current_node_id and current_node_id not in visited:
            visited.add(current_node_id)
            
            # Get node
            node = self.nodes.get(current_node_id)
            if not node:
                logger.error("Node %s not found", current_node_id)
                break
            
            # Create checkpoint before execution
            checkpoint = StateCheckpoint(
                node_id=current_node_id,
                timestamp=datetime.now(timezone.utc),
                state=copy.deepcopy(self.current_state) if self.current_state else {},
                status=NodeStatus.RUNNING
            )
            self.checkpoints.append(checkpoint)
            if len(self.checkpoints) > self._MAX_CHECKPOINTS:
                self.checkpoints = self.checkpoints[-self._MAX_CHECKPOINTS:]
            
            # Execute node action
            success = await self._execute_node(node)
            
            if not success:
                # Handle failure
                checkpoint.status = NodeStatus.FAILED
                if checkpoint_dir:
                    self._save_checkpoint(checkpoint, checkpoint_dir)
                break
            
            checkpoint.status = NodeStatus.SUCCESS
            
            # Check if finish node
            if current_node_id in self.finish_nodes:
                break
            
            # Find next node
            next_node_id = self._find_next_node(current_node_id)
            current_node_id = next_node_id
        
        return copy.deepcopy(self.current_state)
    
    async def _execute_node(self, node: Node) -> bool:
        """Execute a single node."""
        import asyncio
        import inspect
        
        for attempt in range(node.retry_count):
            try:
                # Handle both sync and async functions
                action_result = node.action(self.current_state)
                if inspect.iscoroutine(action_result):
                    result = await asyncio.wait_for(action_result, timeout=node.timeout)
                else:
                    result = action_result
                
                # Update state with result
                if result:
                    self.current_state.update(result)
                
                logger.info("Node %s executed successfully", node.id)
                return True
            
            except asyncio.TimeoutError:
                logger.warning("Node %s timed out (attempt %d)", node.id, attempt + 1)
                if attempt == node.retry_count - 1:
                    return False
            
            except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError) as e:
                logger.error("Node %s failed (attempt %d): %s", node.id, attempt + 1, type(e).__name__)
                if attempt == node.retry_count - 1:
                    return False
        
        return False
    
    def _find_next_node(self, current_node_id: str) -> Optional[str]:
        """Find the next node to execute."""
        # Get all edges from current node
        outgoing_edges = [e for e in self.edges if e.source == current_node_id]
        
        if not outgoing_edges:
            return None
        
        # Evaluate conditional edges first, collect unconditional fallback
        unconditional = None
        for edge in outgoing_edges:
            if edge.condition:
                try:
                    if edge.condition(self.current_state):
                        return edge.target
                except (RuntimeError, ValueError, TypeError, KeyError) as e:
                    logger.error("Condition failed for edge %s->%s: %s", edge.source, edge.target, type(e).__name__)
                    continue
            else:
                unconditional = edge
        
        # Fall through to unconditional edge only if no condition matched
        if unconditional:
            return unconditional.target
        return None
    
    def _save_checkpoint(self, checkpoint: StateCheckpoint, checkpoint_dir: Path):
        """Save checkpoint to disk atomically."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint.checkpoint_id}.json"
        tmp = checkpoint_file.with_suffix(f'.json.tmp.{uuid.uuid4().hex[:8]}')
        data = json.dumps(checkpoint.to_dict(), indent=2, default=str)
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, data.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(str(tmp), str(checkpoint_file))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save checkpoint: %s", type(e).__name__)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    
    def load_checkpoint(self, checkpoint_file: Path) -> StateCheckpoint:
        """Load checkpoint from disk."""
        try:
            data = json.loads(checkpoint_file.read_text())
            return StateCheckpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error("Failed to load checkpoint: %s", type(e).__name__)
            raise
    
    def get_execution_path(self) -> List[str]:
        """Get the path of executed nodes."""
        with self._lock:
            return [c.node_id for c in self.checkpoints]
    
    def get_state_at_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get state at a specific node."""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint.node_id == node_id:
                return copy.deepcopy(checkpoint.state)
        return None
    
    def visualize(self) -> str:
        """Generate Mermaid diagram of the graph."""
        with self._lock:
            lines = ["graph TD"]
            
            # Add nodes
            for node_id, node in self.nodes.items():
                label = _SANITIZE_RE.sub('', node.name.replace('"', "'"))[:100]
                lines.append('    %s["%s"]' % (node_id, label))
            
            # Add edges
            for edge in self.edges:
                label = '|%s|' % _SANITIZE_RE.sub('', str(edge.label))[:50] if edge.label else ''
                lines.append('    %s -->%s %s' % (edge.source, label, edge.target))
            
            # Mark entry and finish
            if self.entry_point:
                lines.insert(1, '    START([%s])' % self.entry_point)
            
            for finish_node in self.finish_nodes:
                lines.append('    %s:::finish' % finish_node)
            
            lines.append("    classDef finish fill:#90EE90,stroke:#333")
            
            return "\n".join(lines)


# Example usage
async def example_workflow():
    """Example state graph workflow."""
    
    # Define state schema
    class WorkflowState(TypedDict):
        query: str
        documents: List[str]
        answer: str
        score: float
    
    # Create graph
    graph = StateGraph(name="example")
    
    # Add nodes
    async def retrieve_step(state: Dict) -> Dict:
        # Simulate retrieval
        return {"documents": ["doc1", "doc2", "doc3"]}
    
    async def generate_step(state: Dict) -> Dict:
        # Simulate generation
        docs = state.get("documents", [])
        return {"answer": f"Answer based on {len(docs)} documents"}
    
    async def score_step(state: Dict) -> Dict:
        # Simulate scoring
        return {"score": 0.95}
    
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("generate", generate_step)
    graph.add_node("score", score_step)
    
    # Add edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "score")
    
    # Set entry and finish
    graph.set_entry_point("retrieve")
    graph.set_finish_point("score")
    
    # Execute
    result = await graph.execute({"query": "What is X?"})
    print(f"Final state: {result}")
    
    # Visualize
    print(f"\nGraph visualization:\n{graph.visualize()}")
    
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_workflow())