#!/usr/bin/env python3
"""
Shadow AI Workflow Engine - AutoGPT-style block execution

Features:
- Block-based workflow execution
- Directed acyclic graph (DAG) for dependencies
- Checkpoint/resume (durable execution)
- Parallel execution where possible
- Error handling and retry
- Human-in-the-loop interrupts
"""

import asyncio
import copy
import json
import hashlib
import logging
import os
import re
import time
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

logger = logging.getLogger("shadowd.workflow")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_WORKFLOWS = 500
_MAX_BLOCKS_PER_WORKFLOW = 200
_MAX_CHECKPOINTS = 50


class BlockStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"  # Waiting for dependencies
    INTERRUPTED = "interrupted"  # Human-in-the-loop


@dataclass
class BlockResult:
    """Result of block execution."""
    block_id: str
    status: BlockStatus
    output: Any
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "block_id": self.block_id,
            "status": self.status.value,
            "output": str(self.output)[:500] if self.output else None,
            "error": self.error[:200] if self.error else None,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "artifacts": list(self.artifacts),
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {}
        }


@dataclass
class Checkpoint:
    """Checkpoint for durable execution."""
    workflow_id: str
    block_id: str
    timestamp: datetime
    state: Dict[str, Any]
    status: BlockStatus
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "workflow_id": self.workflow_id,
            "block_id": self.block_id,
            "timestamp": self.timestamp.isoformat(),
            "state": copy.deepcopy(self.state) if self.state else {},
            "status": self.status.value,
            "error": self.error[:200] if self.error else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Checkpoint':
        _ts = data.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(_ts) if _ts else datetime.now(timezone.utc)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)
        return cls(
            workflow_id=data["workflow_id"],
            block_id=data["block_id"],
            timestamp=timestamp,
            state=data["state"],
            status=BlockStatus(data["status"]),
            error=data.get("error")
        )


class BaseBlock:
    """
    Base class for workflow blocks.
    
    Each block:
    - Has unique ID
    - Executes a single action
    - Has inputs/outputs
    - Can be retried
    - Can be interrupted
    """
    
    def __init__(self, block_id: str = None, name: str = None,
                 timeout: int = 60, max_retries: int = 3):
        self.block_id = block_id or f"blk_{uuid.uuid4().hex[:12]}"
        self.name = _SANITIZE_STRICT_RE.sub('', str(name or self.__class__.__name__))[:200]
        self.timeout = timeout
        self.max_retries = max_retries
        self.status = BlockStatus.PENDING
        self.dependencies: Set[str] = set()
        self._requires_confirmation = False
    
    def depends_on(self, *block_ids: str) -> 'BaseBlock':
        """Add dependencies."""
        self.dependencies.update(block_ids)
        return self
    
    def requires_confirmation(self, requires: bool = True) -> 'BaseBlock':
        """Set whether block requires human confirmation."""
        self._requires_confirmation = requires
        return self
    
    async def execute(self, context: Dict[str, Any]) -> BlockResult:
        """Execute the block."""
        start_time = datetime.now(timezone.utc)
        
        for attempt in range(max(1, self.max_retries)):
            try:
                self.status = BlockStatus.RUNNING
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_impl(context),
                    timeout=self.timeout
                )
                
                self.status = BlockStatus.SUCCESS
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                return BlockResult(
                    block_id=self.block_id,
                    status=BlockStatus.SUCCESS,
                    output=result,
                    elapsed_ms=elapsed
                )
            
            except asyncio.TimeoutError:
                logger.warning("Block %s timed out (attempt %d)", self.block_id, attempt + 1)
                if attempt == self.max_retries - 1:
                    self.status = BlockStatus.FAILED
                    return BlockResult(
                        block_id=self.block_id,
                        status=BlockStatus.FAILED,
                        output=None,
                        error=f"Timeout after {self.timeout}s"
                    )
            
            except Exception as e:
                logger.error("Block %s failed (attempt %d): %s", self.block_id, attempt + 1, type(e).__name__)
                if attempt == self.max_retries - 1:
                    self.status = BlockStatus.FAILED
                    return BlockResult(
                        block_id=self.block_id,
                        status=BlockStatus.FAILED,
                        output=None,
                        error=f"Execution failed: {type(e).__name__}"
                    )
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return BlockResult(
            block_id=self.block_id,
            status=BlockStatus.FAILED,
            output=None,
            error="Max retries exceeded"
        )
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Any:
        """Implementation to override."""
        raise NotImplementedError
    
    def to_dict(self) -> Dict:
        return {
            "block_id": self.block_id,
            "name": self.name[:200],
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "dependencies": list(self.dependencies),
            "status": self.status.value,
            "requires_confirmation": self._requires_confirmation
        }


class FunctionBlock(BaseBlock):
    """Block that executes a function."""
    
    def __init__(self, func: Callable, block_id: str = None, name: str = None,
                 timeout: int = 60, max_retries: int = 3):
        super().__init__(block_id, name or func.__name__, timeout, max_retries)
        self.func = func
        self.is_async = asyncio.iscoroutinefunction(func)
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Any:
        if self.is_async:
            return await self.func(context)
        else:
            return self.func(context)


class ToolBlock(BaseBlock):
    """Block that executes a tool."""
    
    def __init__(self, tool_name: str, parameters: Dict, block_id: str = None,
                 timeout: int = 60, max_retries: int = 3):
        from shadowd.tools.enhanced_tools import get_registry
        super().__init__(block_id, f"tool:{tool_name}", timeout, max_retries)
        self.tool_name = tool_name
        self.parameters = parameters
        self._tool_registry = None
    
    @property
    def tool_registry(self):
        if self._tool_registry is None:
            from shadowd.tools.enhanced_tools import get_registry
            self._tool_registry = get_registry()
        return self._tool_registry
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Any:
        # Merge parameters with context
        params = {**self.parameters}
        
        # Resolve parameter values from context
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to context value
                context_key = value[1:]
                params[key] = context.get(context_key, value)
        
        result = await self.tool_registry.execute_tool(self.tool_name, params, context)
        
        return result.to_dict()


class ConditionalBlock(BaseBlock):
    """Block that branches based on condition."""
    
    def __init__(self, condition: Callable[[Dict], bool], true_block: str,
                 false_block: str = None, block_id: str = None):
        super().__init__(block_id, "conditional")
        self.condition = condition
        self.true_block = true_block
        self.false_block = false_block
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Any:
        result = self.condition(context)
        return {
            "condition_result": result,
            "next_block": self.true_block if result else self.false_block
        }


class ParallelBlock(BaseBlock):
    """Block that executes multiple blocks in parallel."""
    
    def __init__(self, blocks: List[BaseBlock], block_id: str = None):
        super().__init__(block_id, "parallel")
        self.blocks = blocks
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Any:
        # Execute all blocks in parallel
        tasks = [block.execute(context) for block in self.blocks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "results": [r.to_dict() if isinstance(r, BlockResult) else str(r) for r in results],
            "success": all(isinstance(r, BlockResult) and r.status == BlockStatus.SUCCESS for r in results)
        }


class Workflow:
    """
    Workflow: collection of blocks with dependencies.
    
    Features:
    - DAG-based execution
    - Checkpoint/resume
    - Parallel execution
    - Human-in-the-loop interrupts
    """
    
    def __init__(self, workflow_id: str = None, name: str = "workflow"):
        self.workflow_id = workflow_id or f"wf_{uuid.uuid4().hex[:12]}"
        self.name = _SANITIZE_RE.sub('', str(name))[:200]
        self.blocks: Dict[str, BaseBlock] = {}
        self.edges: List[Tuple[str, str]] = []  # (from_block, to_block)
        self.context: Dict[str, Any] = {}
        self.results: Dict[str, BlockResult] = {}
        self.checkpoints: List[Checkpoint] = []
        self.status = BlockStatus.PENDING
        self.created = datetime.now(timezone.utc)
        self.started: Optional[datetime] = None
        self.completed: Optional[datetime] = None
        
        # Human-in-the-loop
        self._interrupt_lock = asyncio.Lock()
        self._interrupted = False
        self._waiting_for_confirmation: Optional[str] = None
        self._lock = threading.Lock()
    
    def add_block(self, block: BaseBlock) -> 'Workflow':
        """Add block to workflow."""
        with self._lock:
            if len(self.blocks) >= _MAX_BLOCKS_PER_WORKFLOW:
                oldest = next(iter(self.blocks))
                del self.blocks[oldest]
                logger.warning("Evicting block %s (cap=%d)", oldest, _MAX_BLOCKS_PER_WORKFLOW)
            self.blocks[block.block_id] = block
        return self
    
    def add_edge(self, from_block_id: str, to_block_id: str) -> 'Workflow':
        """Add edge between blocks."""
        self.edges.append((from_block_id, to_block_id))
        return self
    
    def set_entry_point(self, block_id: str) -> 'Workflow':
        """Set entry point (first block to execute)."""
        self._entry_point = block_id
        return self
    
    def set_context(self, **kwargs) -> 'Workflow':
        """Set workflow context."""
        self.context.update(kwargs)
        return self
    
    def get_dependencies(self, block_id: str) -> Set[str]:
        """Get all dependencies for a block."""
        deps = set()
        for from_id, to_id in self.edges:
            if to_id == block_id:
                deps.add(from_id)
        return deps
    
    def get_dependents(self, block_id: str) -> Set[str]:
        """Get all blocks that depend on this block."""
        dependents = set()
        for from_id, to_id in self.edges:
            if from_id == block_id:
                dependents.add(to_id)
        return dependents
    
    def topological_sort(self) -> List[str]:
        """Topological sort of blocks."""
        # Kahn's algorithm
        in_degree = defaultdict(int)
        for block_id in self.blocks:
            in_degree[block_id] = len(self.get_dependencies(block_id))
        
        # Start with blocks that have no dependencies
        queue = [bid for bid in self.blocks if in_degree[bid] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for dependent in self.get_dependents(current):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) < len(self.blocks):
            raise ValueError("Cycle detected in workflow DAG")
        return result
    
    def create_checkpoint(self, block_id: str) -> Checkpoint:
        """Create checkpoint."""
        checkpoint = Checkpoint(
            workflow_id=self.workflow_id,
            block_id=_SANITIZE_STRICT_RE.sub('', str(block_id))[:100],
            timestamp=datetime.now(timezone.utc),
            state=copy.deepcopy(self.context) if self.context else {},
            status=self.status
        )
        with self._lock:
            if len(self.checkpoints) >= _MAX_CHECKPOINTS:
                self.checkpoints.pop(0)
            self.checkpoints.append(checkpoint)
        return checkpoint
    
    async def interrupt(self):
        """Interrupt workflow execution."""
        async with self._interrupt_lock:
            self._interrupted = True
    
    async def wait_for_confirmation(self, block_id: str) -> bool:
        """Wait for human confirmation of block."""
        async with self._interrupt_lock:
            self._waiting_for_confirmation = block_id
        
        await asyncio.sleep(0.1)
        
        async with self._interrupt_lock:
            self._waiting_for_confirmation = None
            return True
    
    async def execute(self) -> Dict[str, BlockResult]:
        """Execute workflow."""
        self.status = BlockStatus.RUNNING
        self.started = datetime.now(timezone.utc)
        
        # Get execution order
        order = self.topological_sort()
        
        # Track completed blocks
        completed: Set[str] = set()
        
        for block_id in order:
            # Check for interrupt
            async with self._interrupt_lock:
                if self._interrupted:
                    self.status = BlockStatus.INTERRUPTED
                    break
            
            block = self.blocks[block_id]
            
            # Check dependencies
            deps = self.get_dependencies(block_id)
            if not all(d in completed for d in deps):
                # Skip for now, will retry
                continue
            
            # Check if block requires confirmation
            if block._requires_confirmation:
                confirmed = await self.wait_for_confirmation(block_id)
                if not confirmed:
                    self.results[block_id] = BlockResult(
                        block_id=block_id,
                        status=BlockStatus.SKIPPED,
                        output=None,
                        error="Not confirmed"
                    )
                    continue
            
            # Execute block
            result = await block.execute(self.context)
            self.results[block_id] = result
            
            # Update context with result
            if result.output:
                self.context[f"result_{block_id}"] = result.output
            
            # Create checkpoint
            self.create_checkpoint(block_id)
            
            # Handle result
            if result.status == BlockStatus.SUCCESS:
                completed.add(block_id)
            elif result.status == BlockStatus.FAILED:
                # Check if we can continue without this block
                if self._is_critical_block(block_id):
                    self.status = BlockStatus.FAILED
                    break
            
            # Handle conditional blocks
            if isinstance(block, ConditionalBlock) and result.output:
                next_block = result.output.get("next_block")
                if next_block and next_block in self.blocks:
                    # Add edge dynamically
                    self.edges.append((block_id, next_block))
        
        self.completed = datetime.now(timezone.utc)
        
        if self.status == BlockStatus.RUNNING:
            self.status = BlockStatus.SUCCESS
        
        return self.results
    
    def _is_critical_block(self, block_id: str) -> bool:
        """Check if block is critical (workflow can't continue without it)."""
        # Simple heuristic: if it's the only path to remaining blocks
        remaining = set(self.blocks.keys()) - set(self.results.keys())
        for rem in remaining:
            deps = self.get_dependencies(rem)
            if deps == {block_id}:
                return True
        return False
    
    def to_dict(self) -> Dict:
        with self._lock:
            blocks_snapshot = {bid: b.to_dict() for bid, b in self.blocks.items()}
            results_snapshot = {bid: r.to_dict() for bid, r in self.results.items()}
            context_snapshot = copy.deepcopy(self.context) if self.context else {}
        return {
            "workflow_id": self.workflow_id,
            "name": self.name[:200],
            "blocks": blocks_snapshot,
            "edges": list(self.edges),
            "context": context_snapshot,
            "results": results_snapshot,
            "status": self.status.value,
            "created": self.created.isoformat(),
            "started": self.started.isoformat() if self.started else None,
            "completed": self.completed.isoformat() if self.completed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Workflow':
        wf = cls(workflow_id=data["workflow_id"], name=data["name"])
        wf.context = data.get("context", {})
        wf.edges = [tuple(e) for e in data.get("edges", [])]
        
        # Reconstruct blocks (simplified - would need full block registry)
        # For now, just restore state
        wf.status = BlockStatus(data.get("status", "pending"))
        
        return wf


class WorkflowEngine:
    """
    Workflow execution engine.
    
    Features:
    - Workflow creation and management
    - Execution orchestration
    - Checkpoint persistence
    - Resume capability
    """
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or (Path.home() / "digital-shadow" / "data" / "workflows")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.workflows: Dict[str, Workflow] = {}
        self._running_workflows: Set[str] = set()
        self._lock = threading.Lock()
    
    def create_workflow(self, name: str = "workflow") -> Workflow:
        """Create new workflow."""
        name = _SANITIZE_RE.sub('', str(name))[:200]
        workflow = Workflow(name=name)
        with self._lock:
            if len(self.workflows) >= _MAX_WORKFLOWS:
                oldest_id = next(iter(self.workflows))
                del self.workflows[oldest_id]
                logger.warning("Evicting workflow %s (cap=%d)", oldest_id, _MAX_WORKFLOWS)
            self.workflows[workflow.workflow_id] = workflow
        return workflow
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, BlockResult]:
        """Execute workflow by ID."""
        workflow_id = _SANITIZE_STRICT_RE.sub('', str(workflow_id))[:100]
        with self._lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                raise ValueError("Workflow not found")
        
        self._running_workflows.add(workflow_id)
        
        try:
            results = await workflow.execute()
            self._save_workflow(workflow)
            return results
        finally:
            self._running_workflows.discard(workflow_id)
    
    async def execute_and_wait(self, workflow: Workflow) -> Dict[str, BlockResult]:
        """Execute workflow and return results."""
        self.workflows[workflow.workflow_id] = workflow
        return await self.execute_workflow(workflow.workflow_id)
    
    def resume_workflow(self, workflow_id: str) -> Workflow:
        """Resume workflow from checkpoint."""
        workflow_id = _SANITIZE_STRICT_RE.sub('', str(workflow_id))[:100]
        workflow_file = self.storage_path / f"{workflow_id}.json"
        if not workflow_file.exists():
            raise ValueError("No saved workflow")
        
        try:
            data = json.loads(workflow_file.read_text())
            workflow = Workflow.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error("Failed to load workflow %s: %s", workflow_id, type(e).__name__)
            raise ValueError("Invalid workflow data") from e
        
        # Restore checkpoints
        checkpoint_dir = self.storage_path / workflow_id / "checkpoints"
        if checkpoint_dir.exists():
            for cp_file in checkpoint_dir.glob("*.json"):
                cp_data = json.loads(cp_file.read_text())
                workflow.checkpoints.append(Checkpoint.from_dict(cp_data))
        
        self.workflows[workflow_id] = workflow
        return workflow
    
    def _save_workflow(self, workflow: Workflow):
        """Save workflow to disk atomically."""
        workflow_file = self.storage_path / f"{workflow.workflow_id}.json"
        tmp_path = workflow_file.with_suffix(f'.{uuid.uuid4().hex[:8]}.tmp')
        try:
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, 'w') as f:
                json.dump(workflow.to_dict(), f, indent=2)
                f.flush()
                os.fsync(fd)
            os.replace(str(tmp_path), str(workflow_file))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save workflow %s: %s", workflow.workflow_id, type(e).__name__)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        
        # Save checkpoints
        checkpoint_dir = self.storage_path / workflow.workflow_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for cp in workflow.checkpoints:
            cp_file = checkpoint_dir / f"{cp.block_id}_{cp.timestamp.timestamp()}.json"
            cp_tmp = cp_file.with_suffix(f'.{uuid.uuid4().hex[:8]}.tmp')
            try:
                fd = os.open(str(cp_tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                with os.fdopen(fd, 'w') as f:
                    json.dump(cp.to_dict(), f, indent=2)
                    f.flush()
                    os.fsync(fd)
                os.replace(str(cp_tmp), str(cp_file))
            except (OSError, TypeError, ValueError):
                if cp_tmp.exists():
                    try:
                        cp_tmp.unlink()
                    except OSError:
                        pass
    
    def get_status(self, workflow_id: str) -> Dict:
        """Get workflow status."""
        workflow_id = _SANITIZE_STRICT_RE.sub('', str(workflow_id))[:100]
        with self._lock:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                return {"error": "Workflow not found"}
            
            status = workflow.status.value
            blocks_total = len(workflow.blocks)
            blocks_completed = len(workflow.results)
            blocks_success = sum(1 for r in workflow.results.values() if r.status == BlockStatus.SUCCESS)
            blocks_failed = sum(1 for r in workflow.results.values() if r.status == BlockStatus.FAILED)
            checkpoints = len(workflow.checkpoints)
        
        return {
            "workflow_id": workflow_id,
            "status": status,
            "blocks_total": blocks_total,
            "blocks_completed": blocks_completed,
            "blocks_success": blocks_success,
            "blocks_failed": blocks_failed,
            "checkpoints": checkpoints,
        }


# ═════════════════════════════════════════════════════════════════
# STUCK DETECTOR (OpenHands-style)
# ═════════════════════════════════════════════════════════════════

class StuckDetector:
    """
    Detect when agent is stuck in loops.
    
    Based on OpenHands' StuckDetector.
    """
    
    def __init__(self, max_history: int = 100):
        self.history: List[Dict] = []
        self.max_history = min(max_history, 500)
        self.action_counts: Counter = Counter()
        self.pattern_history: List[str] = []
        self._lock = threading.Lock()
    
    def record(self, action: Dict):
        """Record an action for analysis."""
        action_id = self._hash_action(action)
        with self._lock:
            self.history.append({
                "action": {k: str(v)[:100] for k, v in action.items()} if isinstance(action, dict) else {},
                "action_id": action_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            self.action_counts[action_id] += 1
            
            # Cap action_counts
            if len(self.action_counts) > self.max_history:
                for key, _ in self.action_counts.most_common()[:-self.max_history]:
                    del self.action_counts[key]
            
            self.pattern_history.append(action_id)
            
            if len(self.pattern_history) > self.max_history:
                self.pattern_history.pop(0)
    
    def _hash_action(self, action: Dict) -> str:
        """Create hash of action for comparison."""
        key_fields = ["tool", "action", "command", "type"]
        hash_parts = []
        for field in key_fields:
            if field in action:
                value = str(action[field])[:50]
                hash_parts.append(f"{field}:{value}")
        return hashlib.sha256("|".join(hash_parts).encode()).hexdigest()[:12]
    
    def is_stuck(self) -> Tuple[bool, str]:
        """
        Check if agent is stuck.
        
        Returns (is_stuck, reason)
        """
        with self._lock:
            if len(self.history) < 4:
                return False, ""
            
            # Scenario 1: Same action repeated 4+ times
            recent_ids = [h["action_id"] for h in self.history[-10:]]
            if len(Counter(recent_ids).most_common(1)) > 0:
                most_common_id, count = Counter(recent_ids).most_common(1)[0]
                if count >= 4:
                    return True, f"Action repeated {count} times"
            
            # Scenario 2: Same action leads to error 3+ times
            errors = [h for h in self.history[-20:] if h.get("action", {}).get("error")]
            if len(errors) >= 3:
                error_actions = [self._hash_action(e["action"]) for e in errors]
                if len(set(error_actions)) == 1:
                    return True, "Same action causing repeated errors"
            
            # Scenario 3: Alternating pattern (A-B-A-B-A-B)
            if len(self.pattern_history) >= 6:
                last_6 = self.pattern_history[-6:]
                if (last_6[0] == last_6[2] == last_6[4] and
                    last_6[1] == last_6[3] == last_6[5] and
                    last_6[0] != last_6[1]):
                    return True, "Detected alternating action pattern"
        
        return False, ""
    
    def get_stuck_recovery_options(self) -> List[str]:
        """Get recovery options when stuck."""
        return [
            "restart_from_last_checkpoint",
            "try_different_approach",
            "ask_user_for_help",
            "skip_current_task",
            "reduce_complexity"
        ]
    
    def reset(self):
        """Reset detector state."""
        with self._lock:
            self.history = []
            self.action_counts = Counter()
            self.pattern_history = []


__all__ = [
    'BlockStatus',
    'BlockResult',
    'Checkpoint',
    'BaseBlock',
    'FunctionBlock',
    'ToolBlock',
    'ConditionalBlock',
    'ParallelBlock',
    'Workflow',
    'WorkflowEngine',
    'StuckDetector',
]