"""
State-Based Reasoning Engine - LangGraph-inspired State Machine

Implements:
- State graphs with conditional transitions
- Checkpoint/resume (durable execution)
- Human-in-the-loop interrupts
- Automatic retry with exponential backoff
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re as _re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional
import hashlib
import uuid

logger = logging.getLogger(__name__)

_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize_content(text: str, max_length: int = 50000) -> str:
    if text is None:
        return ""
    return _SANITIZE_CONTENT_RE.sub('', str(text)).strip()[:max_length]


def _sanitize_strict(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()[:max_length]


class State(str, Enum):
    INIT = "init"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_INPUT = "waiting_input"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class StateTransition(str, Enum):
    START = "start"
    PLAN_COMPLETE = "plan_complete"
    EXECUTE_NEXT = "execute_next"
    NEEDS_INPUT = "needs_input"
    INPUT_RECEIVED = "input_received"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETRY = "retry"
    COMPLETE = "complete"
    FAIL = "fail"
    PAUSE = "pause"
    RESUME = "resume"


@dataclass
class StateNode:
    """Node in state graph."""
    id: str
    name: str
    state: State
    action: Callable[[dict], Coroutine[Any, Any, dict]] | None = None
    timeout: float = 60.0
    max_retries: int = 3
    on_enter: Callable[[dict], None] | None = None
    on_exit: Callable[[dict], None] | None = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "timeout": round(self.timeout, 3),
            "max_retries": self.max_retries,
        }

    def __post_init__(self):
        self.id = _sanitize_strict(self.id, 100)
        self.name = _sanitize_content(self.name, 200)


@dataclass
class Transition:
    """State transition with condition."""
    from_state: State
    to_state: State
    event: StateTransition
    condition: Callable[[dict], bool] | None = None
    
    def matches(self, current_state: State, event: StateTransition, context: dict) -> bool:
        if self.from_state != current_state or self.event != event:
            return False
        if self.condition and not self.condition(context):
            return False
        return True


@dataclass
class Checkpoint:
    """Execution checkpoint for resume."""
    id: str
    state: State
    context: dict
    created_at: datetime
    node_id: str | None = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "state": self.state.value,
            "context": copy.deepcopy(self.context),
            "created_at": self.created_at.isoformat(),
            "node_id": self.node_id,
            "metadata": copy.deepcopy(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        _ts = data.get("created_at")
        try:
            created_at = datetime.fromisoformat(_ts) if _ts else datetime.now(timezone.utc)
        except (ValueError, TypeError):
            created_at = datetime.now(timezone.utc)
        return cls(
            id=_sanitize_strict(str(data.get("id", "")), 100),
            state=State(data["state"]),
            context=data["context"],
            created_at=created_at,
            node_id=_sanitize_strict(data.get("node_id"), 100) if data.get("node_id") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionResult:
    """Result of state execution."""
    success: bool
    state: State
    output: dict
    error: str | None = None
    needs_input: bool = False
    input_prompt: str | None = None
    needs_review: bool = False
    review_data: dict | None = None
    elapsed_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "state": self.state.value,
            "output": copy.deepcopy(self.output),
            "error": self.error,
            "needs_input": self.needs_input,
            "input_prompt": self.input_prompt,
            "needs_review": self.needs_review,
            "review_data": copy.deepcopy(self.review_data) if self.review_data else None,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class StateGraph:
    """
    State-based reasoning engine.
    
    Features:
    - Conditional state transitions
    - Async node execution
    - Human-in-the-loop interrupts
    - Checkpoint persistence
    - Automatic retry with backoff
    """
    
    def __init__(self, name: str, checkpoint_dir: Path | None = None):
        self.name = name
        self.nodes: dict[str, StateNode] = {}
        self.transitions: list[Transition] = []
        self.start_node: StateNode | None = None
        self.end_states: set[State] = {State.COMPLETED, State.FAILED}
        self._max_checkpoints = 100
        self._max_interrupt_handlers = 50
        
        self.checkpoint_dir = checkpoint_dir or Path.home() / "digital-shadow" / "data" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_context: dict = {}
        self._current_node: StateNode | None = None
        self._checkpoints: dict[str, Checkpoint] = {}
        self._input_queue: Optional[asyncio.Queue] = None
        self._waiting_for_input: bool = False
        self._interrupt_handlers: dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def add_node(self, node: StateNode) -> str:
        """Add node to graph."""
        with self._lock:
            self.nodes[node.id] = node
            if node.state == State.INIT and self.start_node is None:
                self.start_node = node
        return node.id
    
    def add_transition(self, transition: Transition):
        """Add state transition."""
        with self._lock:
            self.transitions.append(transition)
    
    def set_start(self, node_id: str):
        """Set starting node."""
        with self._lock:
            if node_id in self.nodes:
                self.start_node = self.nodes[node_id]
    
    def on_interrupt(self, interrupt_id: str, handler: Callable):
        """Register handler for human input."""
        with self._lock:
            if len(self._interrupt_handlers) >= self._max_interrupt_handlers:
                oldest_key = next(iter(self._interrupt_handlers))
                logger.warning("Evicting interrupt handler %s", oldest_key)
                del self._interrupt_handlers[oldest_key]
            self._interrupt_handlers[interrupt_id] = handler
    
    async def execute(
        self,
        initial_context: dict | None = None,
        checkpoint_id: str | None = None,
    ) -> ExecutionResult:
        """Execute state graph."""
        start_time = time.perf_counter()
        
        # Resume from checkpoint if provided
        if checkpoint_id and checkpoint_id in self._checkpoints:
            checkpoint = self._checkpoints[checkpoint_id]
            self._current_context = checkpoint.context
            current_state = checkpoint.state
        else:
            self._current_context = initial_context or {}
            current_state = State.INIT
        
        if self.start_node is None:
            return ExecutionResult(
                success=False,
                state=current_state,
                output={},
                error="No start node defined",
                elapsed_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        self._current_node = self.start_node
        retries = 0
        
        while current_state not in self.end_states:
            # Find next transition
            if self._current_node is None:
                break
            
            node = self._current_node
            
            # Execute node action
            try:
                if node.on_enter:
                    node.on_enter(self._current_context)
                
                result = await self._execute_node(node, retries)
                
                if node.on_exit:
                    node.on_exit(self._current_context)
                
                # Handle different result types
                if result.needs_input:
                    # Human-in-the-loop: input needed
                    self._waiting_for_input = True
                    input_response = await self._wait_for_input()
                    self._current_context.update(input_response)
                    self._waiting_for_input = False
                    current_state = State.EXECUTING
                
                elif result.needs_review:
                    # Human-in-the-loop: review needed
                    review_response = await self._wait_for_review(result.review_data)
                    if review_response.get("approved"):
                        current_state = State.EXECUTING
                    else:
                        current_state = State.FAILED
                
                elif result.success:
                    # Update context with result
                    self._current_context.update(result.output)
                    current_state = result.state
                    retries = 0  # Reset retries on success
                
                else:
                    # Handle failure
                    retries += 1
                    if retries >= node.max_retries:
                        current_state = State.FAILED
                    else:
                        # Exponential backoff
                        backoff = min(2 ** retries, 30)
                        await asyncio.sleep(backoff)
                
            except asyncio.TimeoutError:
                retries += 1
                if retries >= node.max_retries:
                    current_state = State.FAILED
                continue
            
            except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError) as e:
                logger.error("Error executing node %s: %s", node.id, type(e).__name__)
                retries += 1
                if retries >= node.max_retries:
                    current_state = State.FAILED
                    continue
            
            # Find next node
            next_node = self._find_next_node(current_state, self._current_context)
            if next_node:
                self._current_node = next_node
            else:
                # No valid transition
                if current_state not in self.end_states:
                    current_state = State.COMPLETED
        
        # Save final checkpoint
        self._create_checkpoint(current_state)
        
        return ExecutionResult(
            success=current_state == State.COMPLETED,
            state=current_state,
            output=copy.deepcopy(self._current_context),
            elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
        )
    
    async def _execute_node(self, node: StateNode, retry_count: int) -> ExecutionResult:
        """Execute a single node."""
        if node.action is None:
            return ExecutionResult(
                success=True,
                state=node.state,
                output={"node": node.id},
            )
        
        try:
            result = await asyncio.wait_for(
                node.action(self._current_context),
                timeout=node.timeout
            )
            
            # Check for interrupt requests
            if isinstance(result, dict):
                if result.get("needs_input"):
                    return ExecutionResult(
                        success=False,
                        state=node.state,
                        output=result.get("output", {}),
                        needs_input=True,
                        input_prompt=result.get("input_prompt"),
                    )
                
                if result.get("needs_review"):
                    return ExecutionResult(
                        success=False,
                        state=node.state,
                        output=result.get("output", {}),
                        needs_review=True,
                        review_data=result.get("review_data"),
                    )
            
            return ExecutionResult(
                success=True,
                state=node.state,
                output=result if isinstance(result, dict) else {"result": result},
            )
        
        except asyncio.TimeoutError:
            raise
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError) as e:
            return ExecutionResult(
                success=False,
                state=node.state,
                output={},
                error=f"Execution failed: {type(e).__name__}",
            )
    
    def _find_next_node(self, current_state: State, context: dict) -> StateNode | None:
        """Find next node based on state transitions."""
        for transition in self.transitions:
            if transition.from_state == current_state:
                # Find node with matching state
                for node in self.nodes.values():
                    if node.state == transition.to_state:
                        if transition.condition is None or transition.condition(context):
                            return node
        
        # Default: find node by state progression
        state_order = [State.INIT, State.PLANNING, State.EXECUTING, State.COMPLETED]
        try:
            current_idx = state_order.index(current_state)
            if current_idx + 1 < len(state_order):
                for node in self.nodes.values():
                    if node.state == state_order[current_idx + 1]:
                        return node
        except ValueError:
            pass
        
        return None
    
    async def _wait_for_input(self) -> dict:
        """Wait for human input."""
        if self._input_queue is None:
            self._input_queue = asyncio.Queue()
        return await self._input_queue.get()
    
    async def _wait_for_review(self, review_data: dict | None) -> dict:
        """Wait for human review."""
        # Default auto-approve
        return {"approved": True, "feedback": "auto-approved"}
    
    def provide_input(self, input_data: dict):
        """Provide input for waiting state."""
        with self._lock:
            if self._waiting_for_input:
                if self._input_queue is None:
                    self._input_queue = asyncio.Queue()
                sanitized = {str(k): v for k, v in input_data.items()}
                self._input_queue.put_nowait(sanitized)
    
    def _create_checkpoint(self, state: State) -> Checkpoint:
        """Create execution checkpoint."""
        checkpoint = Checkpoint(
            id=f"cp_{uuid.uuid4().hex[:12]}",
            state=state,
            context=copy.deepcopy(self._current_context),
            created_at=datetime.now(timezone.utc),
            node_id=self._current_node.id if self._current_node else None,
        )
        
        with self._lock:
            self._checkpoints[checkpoint.id] = checkpoint
            
            if len(self._checkpoints) > self._max_checkpoints:
                oldest = list(self._checkpoints.keys())[:len(self._checkpoints) - self._max_checkpoints]
                for k in oldest:
                    logger.warning("Evicting checkpoint %s", k)
                    del self._checkpoints[k]
        
        # Persist to disk (atomic)
        checkpoint_file = self.checkpoint_dir / f"{checkpoint.id}.json"
        tmp = checkpoint_file.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, json.dumps(checkpoint.to_dict(), indent=2, default=str).encode())
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(str(tmp), str(checkpoint_file))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save checkpoint: %s", type(e).__name__)
        
        return checkpoint
    
    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Load checkpoint from disk."""
        checkpoint_id = _sanitize_strict(checkpoint_id, 100)
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        if not checkpoint_file.exists():
            return None
        
        try:
            data = json.loads(checkpoint_file.read_text())
            checkpoint = Checkpoint.from_dict(data)
            with self._lock:
                self._checkpoints[checkpoint_id] = checkpoint
            return checkpoint
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error("Failed to load checkpoint: %s", type(e).__name__)
            return None
    
    def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints."""
        return [copy.deepcopy(cp) for cp in self._checkpoints.values()]
    
    def pause(self):
        """Pause execution."""
        if self._current_node:
            self._create_checkpoint(State.PAUSED)
    
    def to_dict(self) -> dict:
        """Serialize graph."""
        with self._lock:
            return {
                "name": self.name,
                "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
                "transitions": [
                    {"from": t.from_state.value, "to": t.to_state.value, "event": t.event.value}
                    for t in self.transitions
                ],
                "start_node": self.start_node.id if self.start_node else None,
            }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StateGraph":
        """Deserialize graph."""
        graph = cls(name=_sanitize_strict(str(data.get("name", "workflow")), 200))
        
        for nid, ndata in data.get("nodes", {}).items():
            node = StateNode(
                id=_sanitize_strict(nid, 100),
                name=_sanitize_content(str(ndata.get("name", "")), 200),
                state=State(ndata["state"]),
                timeout=ndata.get("timeout", 60.0),
                max_retries=ndata.get("max_retries", 3),
            )
            graph.add_node(node)
        
        for tdata in data.get("transitions", []):
            graph.add_transition(Transition(
                from_state=State(tdata["from"]),
                to_state=State(tdata["to"]),
                event=StateTransition(tdata["event"]),
            ))
        
        if data.get("start_node"):
            graph.set_start(data["start_node"])
        
        return graph


class StateGraphBuilder:
    """Builder for creating state graphs."""
    
    def __init__(self, name: str):
        self.graph = StateGraph(name)
    
    def start(self, action: Callable | None = None) -> "StateGraphBuilder":
        """Set start node."""
        node = StateNode(
            id="start",
            name="Start",
            state=State.INIT,
            action=action,
        )
        self.graph.add_node(node)
        self.graph.set_start("start")
        return self
    
    def plan(self, action: Callable) -> "StateGraphBuilder":
        """Add planning node."""
        node = StateNode(
            id="planning",
            name="Planning",
            state=State.PLANNING,
            action=action,
        )
        self.graph.add_node(node)
        
        self.graph.add_transition(Transition(
            from_state=State.INIT,
            to_state=State.PLANNING,
            event=StateTransition.START,
        ))
        
        return self
    
    def execute(self, action: Callable) -> "StateGraphBuilder":
        """Add execution node."""
        node = StateNode(
            id="executing",
            name="Executing",
            state=State.EXECUTING,
            action=action,
        )
        self.graph.add_node(node)
        
        self.graph.add_transition(Transition(
            from_state=State.PLANNING,
            to_state=State.EXECUTING,
            event=StateTransition.PLAN_COMPLETE,
        ))
        
        return self
    
    def wait_input(
        self,
        prompt: str,
        action: Callable | None = None
    ) -> "StateGraphBuilder":
        """Add input waiting node."""
        async def wait_action(context: dict) -> dict:
            return {
                "needs_input": True,
                "input_prompt": prompt,
            }
        
        node = StateNode(
            id="waiting_input",
            name="Waiting for Input",
            state=State.WAITING_INPUT,
            action=action or wait_action,
        )
        self.graph.add_node(node)
        
        self.graph.add_transition(Transition(
            from_state=State.EXECUTING,
            to_state=State.WAITING_INPUT,
            event=StateTransition.NEEDS_INPUT,
        ))
        
        self.graph.add_transition(Transition(
            from_state=State.WAITING_INPUT,
            to_state=State.EXECUTING,
            event=StateTransition.INPUT_RECEIVED,
        ))
        
        return self
    
    def review(
        self,
        review_data: dict | None = None,
        action: Callable | None = None
    ) -> "StateGraphBuilder":
        """Add review node."""
        async def review_action(context: dict) -> dict:
            return {
                "needs_review": True,
                "review_data": review_data or context,
            }
        
        node = StateNode(
            id="reviewing",
            name="Reviewing",
            state=State.REVIEWING,
            action=action or review_action,
        )
        self.graph.add_node(node)
        
        self.graph.add_transition(Transition(
            from_state=State.EXECUTING,
            to_state=State.REVIEWING,
            event=StateTransition.NEEDS_REVIEW,
        ))
        
        self.graph.add_transition(Transition(
            from_state=State.REVIEWING,
            to_state=State.COMPLETED,
            event=StateTransition.APPROVED,
        ))
        
        self.graph.add_transition(Transition(
            from_state=State.REVIEWING,
            to_state=State.FAILED,
            event=StateTransition.REJECTED,
            condition=lambda ctx: ctx.get("retries", 0) >= 3,
        ))
        
        self.graph.add_transition(Transition(
            from_state=State.REVIEWING,
            to_state=State.EXECUTING,
            event=StateTransition.RETRY,
        ))
        
        return self
    
    def complete(self) -> "StateGraphBuilder":
        """Mark completion."""
        self.graph.add_transition(Transition(
            from_state=State.EXECUTING,
            to_state=State.COMPLETED,
            event=StateTransition.COMPLETE,
        ))
        return self
    
    def on_failure(self, handler: Callable) -> "StateGraphBuilder":
        """Handle failures."""
        node = StateNode(
            id="failure_handler",
            name="Failure Handler",
            state=State.FAILED,
            action=handler,
        )
        self.graph.add_node(node)
        return self
    
    def build(self) -> StateGraph:
        """Build final graph."""
        return self.graph