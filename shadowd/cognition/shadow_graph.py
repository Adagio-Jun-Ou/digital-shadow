#!/usr/bin/env python3
"""
ShadowGraph - LangGraph-Inspired State Machine for Shadow Cognition

Implements a graph-based cognitive architecture where the shadow's mind operates
as a directed state machine with durable execution, checkpointing, and conditional
branching. Every cognitive transition is explicit, auditable, and resumable.

Key Concepts (from LangGraph + 12-Factor Agents):
- Nodes: Cognitive modes (listening, reasoning, responding, delegating, learning)
- Edges: Transitions triggered by events with conditional routing
- State: Typed dictionary flowing through the graph
- Checkpoints: Persistent state snapshots for crash recovery
- Human-in-the-Loop: Interrupt at any node for approval

Architecture:
  LISTEN → ANALYZE → [condition] → RESPOND | DELEGATE | ESCALATE | LEARN
                                              ↕              ↕
                                          ShadowCrew      Human Approval
                                              ↕
                                          ACT → VERIFY → LEARN → LISTEN
"""

import asyncio
import copy
import json
import hashlib
import os
import threading
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict

from ..atomic_io import fsync_dir

logger = logging.getLogger("shadowd.cognition.shadow_graph")

import re as _re

_SANITIZE_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 10000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text).strip()
    return text[:max_length]


class CognitiveMode(str, Enum):
    LISTEN = "listen"
    ANALYZE = "analyze"
    REASON = "reason"
    RESPOND = "respond"
    DELEGATE = "delegate"
    ESCALATE = "escalate"
    ACT = "act"
    VERIFY = "verify"
    LEARN = "learn"
    REFLECT = "reflect"
    IDLE = "idle"
    DORMANT = "dormant"


class TransitionTrigger(str, Enum):
    EVENT_RECEIVED = "event_received"
    INSIGHT_GENERATED = "insight_generated"
    ACTION_REQUIRED = "action_required"
    HUMAN_REQUEST = "human_request"
    PATTERN_DETECTED = "pattern_detected"
    ABILITY_RESULT = "ability_result"
    VOICE_INPUT = "voice_input"
    SCHEDULED = "scheduled"
    ERROR = "error"
    TIMEOUT = "timeout"


class ApprovalLevel(str, Enum):
    AUTO = "auto"
    NOTIFY = "notify"
    ASK = "ask"
    FORBID = "forbid"


@dataclass
class CognitiveState:
    shadow_id: str = "default"
    mode: CognitiveMode = CognitiveMode.IDLE
    previous_mode: Optional[CognitiveMode] = None
    current_query: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    observations: List[Dict] = field(default_factory=list)
    pending_escalations: List[Dict] = field(default_factory=list)
    active_abilities: List[str] = field(default_factory=list)
    decisions: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    urgency: float = 0.0
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "shadow_id": self.shadow_id,
            "mode": self.mode.value,
            "previous_mode": self.previous_mode.value if self.previous_mode else None,
            "current_query": self.current_query,
            "context": copy.deepcopy(self.context),
            "observations_count": len(self.observations),
            "pending_escalations": len(self.pending_escalations),
            "active_abilities": list(self.active_abilities),
            "decisions_count": len(self.decisions),
            "confidence": round(self.confidence, 3),
            "urgency": round(self.urgency, 3),
            "domain": self.domain,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def checkpoint_hash(self) -> str:
        data = f"{self.shadow_id}:{self.mode.value}:{self.confidence}:{len(self.observations)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def transition_to(self, new_mode: CognitiveMode) -> 'CognitiveState':
        self.previous_mode = self.mode
        self.mode = new_mode
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return self


@dataclass
class ShadowNode:
    name: str
    mode: CognitiveMode
    handler: Optional[Callable] = None
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    timeout: float = 300.0
    max_retries: int = 3
    approval_level: ApprovalLevel = ApprovalLevel.AUTO
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return int(hashlib.sha256(self.name.encode()).hexdigest()[:8], 16)

    def __eq__(self, other):
        return isinstance(other, ShadowNode) and self.name == other.name


@dataclass
class ShadowEdge:
    source: str
    target: str
    trigger: Optional[TransitionTrigger] = None
    condition: Optional[Callable[[CognitiveState], bool]] = None
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_traverse(self, state: CognitiveState, trigger: Optional[TransitionTrigger] = None) -> bool:
        if self.trigger and trigger and self.trigger != trigger:
            return False
        if self.condition:
            try:
                return self.condition(state)
            except (TypeError, ValueError, AttributeError, KeyError):
                return False
        return True


@dataclass
class Checkpoint:
    checkpoint_id: str
    state: Dict[str, Any]
    node_name: str
    timestamp: str
    execution_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "node_name": self.node_name,
            "timestamp": self.timestamp,
            "execution_id": self.execution_id,
            "state_keys": list(self.state.keys()) if isinstance(self.state, dict) else [],
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ExecutionTrace:
    execution_id: str
    start_time: str
    nodes_visited: List[str] = field(default_factory=list)
    edges_traversed: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    status: str = "running"
    error: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "execution_id": self.execution_id,
            "start_time": self.start_time,
            "nodes_visited": list(self.nodes_visited),
            "edges_traversed": list(self.edges_traversed),
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
            "checkpoints": list(self.checkpoints),
        }


class ShadowGraph:
    """
    Graph-based cognitive state machine for Shadow AI.

    Inspired by LangGraph's graph-based agent architecture but adapted
    for a persistent digital shadow that:
    - Operates continuously (not just request-response)
    - Has cognitive modes like a mind
    - Can interrupt for human approval
    - Checkpoints for crash recovery
    - Traces every transition for audit
    """

    _MAX_STATE_ESCALATIONS = 100
    _MAX_STATE_ABILITIES = 50
    _MAX_STATE_DECISIONS = 500
    _MAX_METADATA_KEYS = 50

    def __init__(self, graph_id: str = "main", data_dir: Optional[str] = None):
        self.graph_id = graph_id
        self.nodes: Dict[str, ShadowNode] = {}
        self.edges: List[ShadowEdge] = []
        self.adjacency: Dict[str, List[ShadowEdge]] = defaultdict(list)
        self.state = CognitiveState(shadow_id=graph_id)
        self.checkpoints: List[Checkpoint] = []
        self.traces: List[ExecutionTrace] = []
        self.execution_counter = 0
        self._current_execution: Optional[ExecutionTrace] = None
        self._interrupt_callback: Optional[Callable] = None
        self._handlers: Dict[str, Callable] = {}
        self._transition_hooks: List[Callable] = []
        self._error_hooks: List[Callable] = []
        self._MAX_HOOKS = 20
        self._MAX_OBSERVATIONS_PER_EVENT = 50
        self._state_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._max_traces = 500

        self._data_dir = Path(data_dir) if data_dir else Path.home() / "digital-shadow" / "data" / "graphs"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._stats = {
            "transitions": 0,
            "checkpoints_created": 0,
            "errors": 0,
            "interrupts": 0,
            "human_approvals": 0,
            "by_mode": defaultdict(int),
        }

        self._build_default_graph()

    def _build_default_graph(self):
        self.add_node(ShadowNode(
            name="listen", mode=CognitiveMode.LISTEN,
            approval_level=ApprovalLevel.AUTO,
            timeout=60.0,
            metadata={"description": "Passive observation, ingesting events"}
        ))
        self.add_node(ShadowNode(
            name="analyze", mode=CognitiveMode.ANALYZE,
            approval_level=ApprovalLevel.AUTO,
            timeout=120.0,
            metadata={"description": "Deep analysis of observations and patterns"}
        ))
        self.add_node(ShadowNode(
            name="reason", mode=CognitiveMode.REASON,
            approval_level=ApprovalLevel.AUTO,
            timeout=180.0,
            metadata={"description": "LLM-powered reasoning and synthesis"}
        ))
        self.add_node(ShadowNode(
            name="respond", mode=CognitiveMode.RESPOND,
            approval_level=ApprovalLevel.AUTO,
            timeout=60.0,
            metadata={"description": "Generate response to user"}
        ))
        self.add_node(ShadowNode(
            name="delegate", mode=CognitiveMode.DELEGATE,
            approval_level=ApprovalLevel.NOTIFY,
            timeout=300.0,
            metadata={"description": "Delegate to specialized sub-agent"}
        ))
        self.add_node(ShadowNode(
            name="escalate", mode=CognitiveMode.ESCALATE,
            approval_level=ApprovalLevel.ASK,
            timeout=float('inf'),
            metadata={"description": "Escalate to human for approval"}
        ))
        self.add_node(ShadowNode(
            name="act", mode=CognitiveMode.ACT,
            approval_level=ApprovalLevel.NOTIFY,
            timeout=120.0,
            metadata={"description": "Execute an action autonomously"}
        ))
        self.add_node(ShadowNode(
            name="verify", mode=CognitiveMode.VERIFY,
            approval_level=ApprovalLevel.AUTO,
            timeout=60.0,
            metadata={"description": "Verify action outcome"}
        ))
        self.add_node(ShadowNode(
            name="learn", mode=CognitiveMode.LEARN,
            approval_level=ApprovalLevel.AUTO,
            timeout=120.0,
            metadata={"description": "Learn from outcomes and update models"}
        ))
        self.add_node(ShadowNode(
            name="reflect", mode=CognitiveMode.REFLECT,
            approval_level=ApprovalLevel.AUTO,
            timeout=300.0,
            metadata={"description": "Periodic self-reflection and model update"}
        ))

        self.add_edge("listen", "analyze", trigger=TransitionTrigger.EVENT_RECEIVED)
        self.add_edge("listen", "analyze", trigger=TransitionTrigger.VOICE_INPUT)
        self.add_edge("listen", "reason", trigger=TransitionTrigger.HUMAN_REQUEST)
        self.add_edge("analyze", "reason", trigger=TransitionTrigger.INSIGHT_GENERATED)
        self.add_edge("analyze", "reason", trigger=TransitionTrigger.PATTERN_DETECTED)
        self.add_edge("reason", "respond", condition=lambda s: s.urgency < 0.7 and s.confidence > 0.5)
        self.add_edge("reason", "delegate", condition=lambda s: s.urgency >= 0.7 and s.confidence > 0.3)
        self.add_edge("reason", "escalate", condition=lambda s: s.confidence < 0.3)
        self.add_edge("respond", "learn")
        self.add_edge("delegate", "act", trigger=TransitionTrigger.ABILITY_RESULT)
        self.add_edge("delegate", "escalate", condition=lambda s: len(s.pending_escalations) > 0)
        self.add_edge("act", "verify")
        self.add_edge("verify", "learn", condition=lambda s: s.confidence > 0.5)
        self.add_edge("verify", "escalate", condition=lambda s: s.confidence <= 0.5)
        self.add_edge("learn", "listen")
        self.add_edge("escalate", "act", trigger=TransitionTrigger.HUMAN_REQUEST)
        self.add_edge("escalate", "learn")
        self.add_edge("listen", "reflect", trigger=TransitionTrigger.SCHEDULED)

        self.state.mode = CognitiveMode.LISTEN

    _MAX_NODES = 50

    def add_node(self, node: ShadowNode) -> 'ShadowGraph':
        with self._sync_lock:
            node.name = _sanitize(node.name, 100)
            # Evict oldest node if over cap (skip default graph nodes)
            if len(self.nodes) >= self._MAX_NODES and node.name not in self.nodes:
                # Find a non-default node to evict (skip first 11 default nodes)
                default_nodes = {"listen", "analyze", "reason", "respond", "delegate",
                                  "escalate", "act", "verify", "learn", "reflect", "idle"}
                evictable = [n for n in self.nodes if n not in default_nodes]
                if evictable:
                    evicted_name = evictable[0]
                    del self.nodes[evicted_name]
                    logger.warning("Evicting node %s (at cap %d)", evicted_name, self._MAX_NODES)
            self.nodes[node.name] = node
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        trigger: Optional[TransitionTrigger] = None,
        condition: Optional[Callable[[CognitiveState], bool]] = None,
        priority: int = 5,
    ) -> 'ShadowGraph':
        with self._sync_lock:
            source = _sanitize(source, 100)
            target = _sanitize(target, 100)
            if len(self.edges) >= 100:
                evicted = self.edges.pop(0)
                logger.warning("Evicting edge %s->%s (trigger=%s), edges at cap 100", evicted.source, evicted.target, evicted.trigger.value if evicted.trigger else "none")
                if evicted.source in self.adjacency:
                    self.adjacency[evicted.source] = [
                        e for e in self.adjacency[evicted.source]
                        if e is not evicted
                    ]
            edge = ShadowEdge(
                source=source, target=target,
                trigger=trigger, condition=condition, priority=priority
            )
            self.edges.append(edge)
            self.adjacency[source].append(edge)
        return self

    def register_handler(self, node_name: str, handler: Callable[[CognitiveState], CognitiveState]):
        with self._sync_lock:
            self._handlers[node_name] = handler
            if node_name in self.nodes:
                self.nodes[node_name].handler = handler
        return self

    def on_transition(self, hook: Callable[[CognitiveState, CognitiveState, str], None]):
        if len(self._transition_hooks) >= self._MAX_HOOKS:
            self._transition_hooks.pop(0)
        self._transition_hooks.append(hook)
        return self

    def on_error(self, hook: Callable[[CognitiveState, Exception], None]):
        if len(self._error_hooks) >= self._MAX_HOOKS:
            self._error_hooks.pop(0)
        self._error_hooks.append(hook)
        return self

    def set_interrupt_callback(self, callback: Callable[[CognitiveState, str], Any]):
        self._interrupt_callback = callback

    async def process_event(
        self,
        event: Dict[str, Any],
        trigger: TransitionTrigger = TransitionTrigger.EVENT_RECEIVED,
    ) -> CognitiveState:
        async with self._state_lock:
            self.execution_counter += 1
            start_time = time.monotonic()
            execution_id = f"exec_{self.execution_counter}_{int(start_time)}"
            trace = ExecutionTrace(
                execution_id=execution_id,
                start_time=datetime.now(timezone.utc).isoformat(),
            )
            self.traces.append(trace)
            if len(self.traces) > self._max_traces:
                self.traces = self.traces[-self._max_traces:]
            self._current_execution = trace

            if event.get("query"):
                self.state.current_query = _sanitize_content(event["query"], 10000)
            if event.get("domain"):
                self.state.domain = _sanitize(event["domain"], 100)
            if event.get("observations"):
                new_obs = [_sanitize_content(o, 2000) for o in event["observations"][:self._MAX_OBSERVATIONS_PER_EVENT]]
                self.state.observations.extend(new_obs)
                if len(self.state.observations) > 1000:
                    self.state.observations = self.state.observations[-1000:]
            if event.get("urgency") is not None:
                self.state.urgency = max(0.0, min(1.0, float(event["urgency"])))
            if event.get("confidence") is not None:
                self.state.confidence = max(0.0, min(1.0, float(event["confidence"])))

            if event.get("metadata"):
                clean_meta = {}
                for k, v in list(event["metadata"].items())[:self._MAX_METADATA_KEYS]:
                    if isinstance(v, str):
                        clean_meta[_sanitize(k, 100)] = _sanitize_content(v, 500)
                    else:
                        clean_meta[_sanitize(k, 100)] = v
                self.state.metadata.update(clean_meta)

            if event.get("active_abilities"):
                self.state.active_abilities.extend(event["active_abilities"])
                self.state.active_abilities = self.state.active_abilities[-self._MAX_STATE_ABILITIES:]

            if event.get("decisions"):
                self.state.decisions.extend(event["decisions"])
                self.state.decisions = self.state.decisions[-self._MAX_STATE_DECISIONS:]

            current_trigger = trigger
            try:
                result = await self._execute_graph(current_trigger, trace)
                trace.status = "completed"
                trace.duration_ms = (time.monotonic() - start_time) * 1000
                return result
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                trace.status = "error"
                trace.error = type(e).__name__
                with self._stats_lock:
                    self._stats["errors"] += 1
                for hook in self._error_hooks:
                    try:
                        hook(self.state, e)
                    except (ValueError, TypeError, AttributeError, RuntimeError):
                        pass
                raise

    async def _execute_graph(self, initial_trigger: TransitionTrigger, trace: ExecutionTrace) -> CognitiveState:
        current_node_name = self._mode_to_node_name(self.state.mode)
        current_trigger = initial_trigger
        max_steps = 20
        steps = 0

        while steps < max_steps:
            if current_node_name not in self.nodes:
                logger.warning("Node %s not found, returning to listen", current_node_name[:50].replace(chr(10), ' '))
                self.state.transition_to(CognitiveMode.LISTEN)
                break

            node = self.nodes[current_node_name]
            trace.nodes_visited.append(current_node_name)
            with self._stats_lock:
                self._stats["by_mode"][node.mode.value] += 1

            if node.on_enter:
                try:
                    node.on_enter(self.state)
                except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                    logger.error("on_enter error for %s: %s", node.name[:50].replace(chr(10), ' '), type(e).__name__)

            if node.approval_level in (ApprovalLevel.ASK, ApprovalLevel.FORBID):
                with self._stats_lock:
                    self._stats["interrupts"] += 1
                if node.approval_level == ApprovalLevel.FORBID:
                    esc_id = f"esc_{uuid.uuid4().hex[:12]}"
                    self.state.pending_escalations.append({
                        "id": esc_id,
                        "node": node.name,
                        "reason": "forbidden_action",
                        "state": self.state.to_dict(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    self.state.pending_escalations = self.state.pending_escalations[-self._MAX_STATE_ESCALATIONS:]
                    self.state.transition_to(CognitiveMode.ESCALATE)
                    self._create_checkpoint(node.name, trace.execution_id)
                    break
                elif node.approval_level == ApprovalLevel.ASK:
                    if self._interrupt_callback:
                        try:
                            approved = await self._interrupt_callback(self.state, node.name)
                            if not approved:
                                esc_id = f"esc_{uuid.uuid4().hex[:12]}"
                                self.state.pending_escalations.append({
                                    "id": esc_id,
                                    "node": node.name,
                                    "reason": "human_denied",
                                    "state": self.state.to_dict(),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })
                                self.state.pending_escalations = self.state.pending_escalations[-self._MAX_STATE_ESCALATIONS:]
                                self.state.transition_to(CognitiveMode.ESCALATE)
                                self._create_checkpoint(node.name, trace.execution_id)
                                break
                        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError):
                            self.state.transition_to(CognitiveMode.ESCALATE)
                            break
                    self._stats["human_approvals"] += 1

            handler = self._handlers.get(current_node_name) or node.handler
            if handler:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result_state = await handler(self.state)
                    else:
                        result_state = handler(self.state)
                    if isinstance(result_state, CognitiveState):
                        self.state = result_state
                    else:
                        logger.error("Handler for %s returned %s, expected CognitiveState; keeping previous state", node.name[:50].replace(chr(10), ' '), type(result_state).__name__)
                        with self._stats_lock:
                            self._stats["errors"] += 1
                except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
                    logger.error("Handler error for %s: %s", node.name[:50].replace(chr(10), ' '), type(e).__name__)
                    with self._stats_lock:
                        self._stats["errors"] += 1
                    self.state.transition_to(CognitiveMode.ESCALATE)
                    self._create_checkpoint(node.name, trace.execution_id)
                    for hook in self._error_hooks:
                        try:
                            hook(self.state, e)
                        except (ValueError, TypeError, AttributeError, RuntimeError):
                            pass
                    break

            self._create_checkpoint(node.name, trace.execution_id)

            if node.on_exit:
                try:
                    node.on_exit(self.state)
                except (ValueError, TypeError, AttributeError, RuntimeError):
                    pass

            next_edges = self.adjacency.get(current_node_name, [])
            if not next_edges:
                break

            trigger_edges = [e for e in next_edges if e.trigger is not None and e.trigger == current_trigger]
            condition_edges = [e for e in next_edges if e.trigger is None and e.condition is not None]
            default_edges = [e for e in next_edges if e.trigger is None and e.condition is None]

            suitable_edges = trigger_edges or condition_edges or default_edges
            if not suitable_edges:
                break

            suitable_edges.sort(key=lambda e: e.priority, reverse=True)
            chosen_edge = suitable_edges[0]
            trace.edges_traversed.append(f"{chosen_edge.source}->{chosen_edge.target}")

            prev_state = copy.deepcopy(self.state)
            target_node = self.nodes.get(chosen_edge.target)
            if target_node:
                self.state.transition_to(target_node.mode)
                with self._stats_lock:
                    self._stats["transitions"] += 1

                for hook in self._transition_hooks:
                    try:
                        hook(prev_state, self.state, chosen_edge.source)
                    except (ValueError, TypeError, AttributeError, RuntimeError):
                        pass

                current_node_name = chosen_edge.target
            else:
                break

            steps += 1

            if self.state.mode == CognitiveMode.LISTEN and steps > 2:
                break

        return self.state

    def _mode_to_node_name(self, mode: CognitiveMode) -> str:
        if mode.value in self.nodes:
            return mode.value
        for name, node in self.nodes.items():
            if node.mode == mode:
                return name
        return "listen"

    def _create_checkpoint(self, node_name: str, execution_id: str) -> Checkpoint:
        checkpoint = Checkpoint(
            checkpoint_id=f"ckpt_{uuid.uuid4().hex[:12]}",
            state=self.state.to_dict(),
            node_name=node_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_id=execution_id,
        )
        self.checkpoints.append(checkpoint)
        with self._stats_lock:
            self._stats["checkpoints_created"] += 1

        if len(self.checkpoints) > 100:
            self.checkpoints = self.checkpoints[-100:]

        self._persist_checkpoint(checkpoint)
        return checkpoint

    def _persist_checkpoint(self, checkpoint: Checkpoint):
        try:
            checkpoint_file = self._data_dir / f"{self.graph_id}_latest.json"
            tmp_file = self._data_dir / f"{self.graph_id}_latest.json.tmp"
            data = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "state": checkpoint.state,
                "node_name": checkpoint.node_name,
                "timestamp": checkpoint.timestamp,
                "execution_id": checkpoint.execution_id,
            }
            tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            fd = os.open(str(tmp_file), os.O_WRONLY)
            os.fsync(fd)
            os.close(fd)
            fsync_dir(self._data_dir)
            tmp_file.rename(checkpoint_file)
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to persist checkpoint: %s", type(e).__name__)

    async def resume_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> CognitiveState:
        async with self._state_lock:
            if checkpoint_id:
                checkpoints = [c for c in self.checkpoints if c.checkpoint_id == checkpoint_id]
                if not checkpoints:
                    raise ValueError(f"Checkpoint {checkpoint_id} not found")
                checkpoint = checkpoints[-1]
                state_data = checkpoint.state
                mode_str = state_data.get("mode", "listen")
                try:
                    mode = CognitiveMode(mode_str)
                except ValueError:
                    logger.warning("Invalid mode '%s' in checkpoint, defaulting to LISTEN", mode_str)
                    mode = CognitiveMode.LISTEN
                self.state = CognitiveState(
                    shadow_id=state_data.get("shadow_id", self.graph_id),
                    mode=mode,
                    context=state_data.get("context", {}),
                    confidence=state_data.get("confidence", 0.0),
                    urgency=state_data.get("urgency", 0.0),
                    domain=state_data.get("domain"),
                )
                logger.info("Resumed from checkpoint: %s", checkpoint_id)
            else:
                checkpoint_file = self._data_dir / f"{self.graph_id}_latest.json"
                if checkpoint_file.exists():
                    try:
                        data = json.loads(checkpoint_file.read_text())
                        mode_str = data["state"].get("mode", "listen")
                        try:
                            mode = CognitiveMode(mode_str)
                        except ValueError:
                            logger.warning("Invalid mode '%s' in checkpoint, defaulting to LISTEN", mode_str)
                            mode = CognitiveMode.LISTEN
                        self.state = CognitiveState(
                            shadow_id=data["state"].get("shadow_id", self.graph_id),
                            mode=mode,
                            context=data["state"].get("context", {}),
                            confidence=data["state"].get("confidence", 0.0),
                            urgency=data["state"].get("urgency", 0.0),
                            domain=data["state"].get("domain"),
                        )
                        with self._stats_lock:
                            self._stats.update(data.get("stats", {}))
                        logger.info("Resumed from checkpoint: %s", data.get('checkpoint_id'))
                    except (json.JSONDecodeError, KeyError, OSError) as e:
                        logger.warning("Failed to load checkpoint: %s, starting fresh", type(e).__name__)
                        self.state = CognitiveState(shadow_id=self.graph_id)
                else:
                    self.state = CognitiveState(shadow_id=self.graph_id)
            return self.state

    def get_status(self) -> Dict[str, Any]:
        with self._sync_lock:
            with self._stats_lock:
                stats_copy = {k: v for k, v in self._stats.items() if k != "by_mode"}
                stats_copy["by_mode"] = dict(self._stats["by_mode"])
            return {
                "graph_id": self.graph_id,
                "current_mode": self.state.mode.value,
                "previous_mode": self.state.previous_mode.value if self.state.previous_mode else None,
                "current_query": self.state.current_query,
                "confidence": round(self.state.confidence, 3),
                "urgency": self.state.urgency,
                "domain": self.state.domain,
                "nodes": list(self.nodes.keys()),
                "edges_count": len(self.edges),
                "checkpoints_count": len(self.checkpoints),
                "traces_count": len(self.traces),
                "stats": stats_copy,
            }

    def get_graph_visualization(self) -> str:
        with self._sync_lock:
            lines = ["digraph ShadowGraph {"]
            lines.append('  rankdir=LR;')
            lines.append('  node [shape=box, style=filled, fillcolor="#1a1a2e", fontcolor="white"];')
            lines.append('  edge [color="#e94560", fontcolor="#e94560"];')

            current = self._mode_to_node_name(self.state.mode)
            for name, node in self.nodes.items():
                color = "#0f3460" if name == current else "#1a1a2e"
                lines.append(f'  "{name}" [fillcolor="{color}", label="{name}\\n({node.mode.value})"];')

            for edge in self.edges:
                label = edge.trigger.value if edge.trigger else ""
                label += " [cond]" if edge.condition else ""
                lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{label}"];')

            lines.append("}")
            return "\n".join(lines)

    def get_trace_summary(self, limit: int = 10) -> List[Dict]:
        with self._sync_lock:
            return [t.to_dict() for t in self.traces[-limit:]]

    def get_checkpoint_summary(self, limit: int = 10) -> List[Dict]:
        with self._sync_lock:
            return [c.to_dict() for c in self.checkpoints[-limit:]]

    async def approve_escalation(self, escalation_id: str, approved: bool) -> bool:
        async with self._state_lock:
            for esc in list(self.state.pending_escalations):
                if esc.get("id") == escalation_id:
                    self.state.pending_escalations.remove(esc)
                    if approved:
                        self.state.transition_to(CognitiveMode.ACT)
                    else:
                        self.state.transition_to(CognitiveMode.LEARN)
                    with self._stats_lock:
                        self._stats["human_approvals"] += 1
                    return True
            return False

    def reset(self):
        with self._sync_lock:
            self.state = CognitiveState(shadow_id=self.graph_id)
            self.checkpoints.clear()
            self.traces.clear()
            with self._stats_lock:
                self._stats = {
                    "transitions": 0,
                    "checkpoints_created": 0,
                    "errors": 0,
                    "interrupts": 0,
                    "human_approvals": 0,
                    "by_mode": defaultdict(int),
                }

    def export_graph_state(self) -> Dict[str, Any]:
        with self._sync_lock:
            with self._stats_lock:
                node_visit_counts = {name: self._stats["by_mode"].get(node.mode.value, 0) for name, node in self.nodes.items()}
                stats_snapshot = {**dict(self._stats), "by_mode": dict(self._stats["by_mode"])}
            last_n_traces = [t.to_dict() for t in self.traces[-20:]]
            checkpoint_data = [c.to_dict() for c in self.checkpoints]
            edge_data = [{"source": e.source, "target": e.target, "trigger": e.trigger.value if e.trigger else None, "has_condition": e.condition is not None} for e in self.edges]
            return {
                "graph_id": self.graph_id,
                "state": self.state.to_dict(),
                "checkpoints": checkpoint_data,
                "last_traces": last_n_traces,
                "node_visit_counts": node_visit_counts,
                "statistics": stats_snapshot,
                "nodes": list(self.nodes.keys()),
                "edges": edge_data,
                "execution_counter": self.execution_counter,
            }

    def import_graph_state(self, data: Dict[str, Any]):
        with self._sync_lock:
            if "graph_id" in data:
                self.graph_id = data["graph_id"]
            if "state" in data:
                s = data["state"]
                mode_str = s.get("mode", "idle")
                try:
                    mode = CognitiveMode(mode_str)
                except ValueError:
                    logger.warning("Invalid mode '%s' in import, defaulting to IDLE", mode_str)
                    mode = CognitiveMode.IDLE
                self.state = CognitiveState(
                    shadow_id=s.get("shadow_id", self.graph_id),
                    mode=mode,
                    context=s.get("context", {}),
                    confidence=max(0.0, min(1.0, s.get("confidence", 0.0))),
                    urgency=max(0.0, min(1.0, s.get("urgency", 0.0))),
                    domain=s.get("domain"),
                )
                if s.get("previous_mode"):
                    try:
                        self.state.previous_mode = CognitiveMode(s["previous_mode"])
                    except ValueError:
                        logger.warning("Invalid previous_mode '%s' in import", s["previous_mode"])
                if s.get("current_query"):
                    self.state.current_query = s["current_query"]
            if "execution_counter" in data:
                self.execution_counter = data["execution_counter"]
            if "statistics" in data:
                stats = data["statistics"]
                with self._stats_lock:
                    for key in ("transitions", "checkpoints_created", "errors", "interrupts", "human_approvals"):
                        if key in stats:
                            self._stats[key] = stats[key]
                    if "by_mode" in stats and isinstance(stats["by_mode"], dict):
                        self._stats["by_mode"] = defaultdict(int, stats["by_mode"])

    def get_node_statistics(self) -> Dict[str, Dict[str, Any]]:
        with self._sync_lock:
            result = {}
            transition_counts = defaultdict(int)
            for trace in self.traces:
                for i in range(len(trace.nodes_visited) - 1):
                    src = trace.nodes_visited[i]
                    dst = trace.nodes_visited[i + 1]
                    transition_counts[(src, dst)] += 1

            with self._stats_lock:
                by_mode_snapshot = dict(self._stats["by_mode"])

            for name, node in self.nodes.items():
                visit_count = by_mode_snapshot.get(node.mode.value, 0)
                outgoing = {f"{src}->{dst}": cnt for (src, dst), cnt in transition_counts.items() if src == name}
                incoming = {f"{src}->{dst}": cnt for (src, dst), cnt in transition_counts.items() if dst == name}
                result[name] = {
                    "mode": node.mode.value,
                    "visit_count": visit_count,
                    "transitions_from": outgoing,
                    "transitions_to": incoming,
                }
            return result