#!/usr/bin/env python3
"""
ShadowPool — Agentic Harness for Shikigami Abilities

A pool of AI shadows that continuously chain Shikigami abilities together
to deliver superhuman intelligence. Inspired by agentic orchestration patterns:

  ┌──────────────────────────────────────────────────┐
  │  ShadowPool (Master Orchestrator)                │
  │  ├── TaskDispatcher (routes queries to shadows)  │
  │  ├── ShadowWorker × N (chained ability executors)│
  │  │   ├── AbilityChain (sequenced superpowers)    │
  │  │   ├── Context (accumulated knowledge)         │
  │  │   └── Feedback (self-improvement loop)        │
  │  ├── PoolMonitor (health + load balancing)       │
  │  ├── LearningOptimizer (improves chains over time)│
  │  └── DeployBridge (sends shadows to remote)      │
  └──────────────────────────────────────────────────┘

Each shadow is a persistent agent that:
- Chains multiple Shikigami abilities for compound intelligence
- Learns from success/failure of ability combinations
- Can be deployed to remote devices via ShadowDeploy
- Collaborates with other shadows through the EventBus
- Records all actions in ShadowJournal for audit

Ability Chains (pre-built compound powers):
  "Situational Awareness" = OmniSense → PatternSight → Oracle
  "Threat Assessment"      = XRay → IntentSeer → ShadowCouncil
  "Relationship Insight"   = SocialRadar → LifePulse → PatternSight
  "Strategic Planning"     = Oracle → ShadowCouncil → RealityForge
  "Deep Investigation"     = PerfectMemory → PatternSight → XRay → Oracle
"""

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("shadowd.cognition.shadow_pool")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENUMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShadowRole(str, Enum):
    """Specialization role for a pool shadow."""
    ANALYST = "analyst"
    SENTINEL = "sentinel"
    ORACLE = "oracle"
    BUILDER = "builder"
    DIPLOMAT = "diplomat"
    INVESTIGATOR = "investigator"
    COORDINATOR = "coordinator"
    GENERALIST = "generalist"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChainEffectiveness(str, Enum):
    """How well a compound ability chain performed."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    UNKNOWN = "unknown"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPOUND ABILITY CHAINS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ABILITY_CHAINS = {
    "situational_awareness": {
        "name": "Situational Awareness",
        "abilities": ["omni_sense", "pattern_sight", "oracle"],
        "description": "Scan everything → find patterns → predict outcomes",
        "trigger_words": ["overview", "big picture", "situation", "context", "summarize everything"],
        "requires": ["observation", "system_data"],
    },
    "threat_assessment": {
        "name": "Threat Assessment",
        "abilities": ["x_ray", "intent_seer", "shadow_council"],
        "description": "Detect threats → predict intentions → assess risk",
        "trigger_words": ["threat", "danger", "risk", "vulnerable", "attack", "breach"],
        "requires": ["device_scan", "behavior_data"],
    },
    "relationship_insight": {
        "name": "Relationship Insight",
        "abilities": ["social_radar", "life_pulse", "pattern_sight"],
        "description": "Map social dynamics → track relationships → find patterns",
        "trigger_words": ["relationship", "social", "friend", "colleague", "family", "partner"],
        "requires": ["social_data", "interaction_history"],
    },
    "strategic_planning": {
        "name": "Strategic Planning",
        "abilities": ["oracle", "shadow_council", "reality_forge"],
        "description": "Predict futures → evaluate options → build solution",
        "trigger_words": ["plan", "strategy", "roadmap", "goal", "objective", "long term"],
        "requires": ["goals", "constraints"],
    },
    "deep_investigation": {
        "name": "Deep Investigation",
        "abilities": ["perfect_memory", "pattern_sight", "x_ray", "oracle"],
        "description": "Recall history → find patterns → scan environment → predict",
        "trigger_words": ["investigate", "dig into", "deep dive", "research", "uncover"],
        "requires": ["memory", "system_data"],
    },
    "morning_brief": {
        "name": "Morning Brief",
        "abilities": ["omni_sense", "intent_seer", "oracle"],
        "description": "Full scan → upcoming events → predictions for the day",
        "trigger_words": ["morning", "brief", "today", "daily", "start of day"],
        "requires": ["schedule", "system_data"],
    },
    "decision_assist": {
        "name": "Decision Assist",
        "abilities": ["pattern_sight", "shadow_council", "oracle"],
        "description": "Find patterns → evaluate options → predict outcomes",
        "trigger_words": ["decide", "choose", "pick", "option", "alternative"],
        "requires": ["options", "constraints"],
    },
    "creative_forge": {
        "name": "Creative Forge",
        "abilities": ["pattern_sight", "shadow_council", "reality_forge"],
        "description": "Find patterns → brainstorm options → build solution",
        "trigger_words": ["create", "design", "build", "invent", "innovate", "ideate"],
        "requires": ["constraints", "inspiration"],
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATACLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PoolTask:
    """A task dispatched to the shadow pool."""
    task_id: str = ""
    title: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.QUEUED
    chain_name: Optional[str] = None
    assigned_shadow: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    chain_results: List[Dict[str, Any]] = field(default_factory=list)
    feedback_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = uuid.uuid4().hex[:12]
        self.title = _sanitize(self.title, 200)
        self.description = _sanitize_content(self.description, 5000)
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "chain_name": self.chain_name,
            "assigned_shadow": self.assigned_shadow,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": copy.deepcopy(self.result) if self.result else None,
            "error": self.error,
            "chain_results": [copy.deepcopy(c) for c in self.chain_results[:20]],
            "feedback_score": round(self.feedback_score, 3),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ShadowWorker:
    """A persistent shadow agent in the pool."""
    worker_id: str = ""
    name: str = ""
    role: ShadowRole = ShadowRole.GENERALIST
    status: str = "idle"
    specialization: List[str] = field(default_factory=list)
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_task_duration_ms: float = 0.0
    current_task: Optional[str] = None
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    preferred_chains: Dict[str, int] = field(default_factory=dict)
    capability_scores: Dict[str, float] = field(default_factory=dict)
    deployed_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.worker_id:
            self.worker_id = uuid.uuid4().hex[:12]
        self.name = _sanitize_content(self.name, 200)
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    @property
    def success_rate(self) -> float:
        return self.successful_tasks / max(1, self.total_tasks)

    @property
    def is_available(self) -> bool:
        return self.status == "idle" and self.current_task is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status,
            "specialization": list(self.specialization),
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": round(self.success_rate, 3),
            "avg_task_duration_ms": round(self.avg_task_duration_ms, 1),
            "current_task": self.current_task,
            "last_active": self.last_active.isoformat(),
            "preferred_chains": dict(self.preferred_chains),
            "capability_scores": dict(self.capability_scores),
            "deployed_to": self.deployed_to,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ChainExecutionRecord:
    """Record of a compound ability chain execution for learning."""
    record_id: str = ""
    chain_name: str = ""
    task_id: str = ""
    worker_id: str = ""
    ability_results: List[Dict[str, Any]] = field(default_factory=list)
    overall_success: bool = False
    effectiveness: ChainEffectiveness = ChainEffectiveness.UNKNOWN
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = uuid.uuid4().hex[:12]
        self.chain_name = _sanitize(self.chain_name, 50)
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "chain_name": self.chain_name,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "ability_results": [copy.deepcopy(r) for r in self.ability_results[:10]],
            "overall_success": self.overall_success,
            "effectiveness": self.effectiveness.value,
            "duration_ms": round(self.duration_ms, 1),
            "timestamp": self.timestamp.isoformat(),
            "metadata": copy.deepcopy(self.metadata),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHADOW POOL — MASTER ORCHESTRATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShadowPool:
    """Master orchestrator for the agentic harness of Shikigami abilities.

    Maintains a pool of ShadowWorker agents that chain Shikigami abilities
    together to deliver compound superhuman intelligence. Routes tasks to
    the best available worker, learns from execution patterns, and can
    deploy workers to remote devices via ShadowDeploy.
    """

    def __init__(self, shadow_os=None, max_workers: int = 20):
        self._shadow_os = shadow_os
        # RLock: submit_task and _dispatch_next both acquire this lock, and
        # _dispatch_next calls _select_worker which also wants the lock.
        # Without reentrance, that's an instant deadlock.
        self._lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._max_workers = max_workers

        self._workers: Dict[str, ShadowWorker] = {}
        self._worker_order: deque = deque(maxlen=self._max_workers + 10)
        self._tasks: Dict[str, PoolTask] = {}
        self._task_queue: deque = deque(maxlen=5000)
        self._task_order: deque = deque(maxlen=10000)
        self._chain_records: deque = deque(maxlen=5000)
        self._chain_effectiveness: Dict[str, List[float]] = defaultdict(list)

        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "ability_chains_run": 0,
            "workers_spawned": 0,
            "workers_deployed": 0,
        }

        self._initialize_workers()

    def _initialize_workers(self):
        """Spawn initial worker shadows specialized in different roles."""
        spawns = [
            ("Aether", ShadowRole.ORACLE, ["oracle", "pattern_sight", "intent_seer"]),
            ("Sentinel", ShadowRole.SENTINEL, ["x_ray", "omni_sense", "intent_seer"]),
            ("Architect", ShadowRole.BUILDER, ["reality_forge", "shadow_council", "pattern_sight"]),
            ("Weaver", ShadowRole.ANALYST, ["pattern_sight", "perfect_memory", "oracle"]),
            ("Harbinger", ShadowRole.INVESTIGATOR, ["x_ray", "perfect_memory", "oracle"]),
        ]
        for name, role, specs in spawns:
            self._spawn_worker(name=name, role=role, specialization=specs)

    def _spawn_worker(self, name: str, role: ShadowRole,
                      specialization: Optional[List[str]] = None) -> ShadowWorker:
        worker = ShadowWorker(
            name=name, role=role,
            specialization=specialization or [],
            capability_scores={s: 0.7 for s in (specialization or ["generalist"])},
        )
        with self._lock:
            if len(self._workers) >= self._max_workers:
                oldest_key = self._worker_order.popleft()
                self._workers.pop(oldest_key, None)
            self._workers[worker.worker_id] = worker
            self._worker_order.append(worker.worker_id)
        with self._stats_lock:
            self._stats["workers_spawned"] += 1
        return worker

    def _select_chain(self, description: str) -> Optional[str]:
        """Select the best compound ability chain for a task description."""
        desc_lower = description.lower()
        best_chain = None
        best_score = 0
        for chain_name, chain in ABILITY_CHAINS.items():
            score = sum(1 for w in chain["trigger_words"] if w in desc_lower)
            if score > best_score:
                best_score = score
                best_chain = chain_name
        if best_score >= 2:
            return best_chain
        return best_chain if best_score >= 1 else "decision_assist"

    def _select_worker(self, chain_name: Optional[str] = None) -> Optional[ShadowWorker]:
        """Select the best available worker for a task."""
        with self._lock:
            available = [w for w in self._workers.values() if w.is_available]
            if not available:
                return None
            if chain_name:
                chain = ABILITY_CHAINS.get(chain_name, {})
                chain_abilities = set(chain.get("abilities", []))
                scored = []
                for w in available:
                    score = len(chain_abilities & set(w.specialization)) * 0.4
                    score += w.success_rate * 0.3
                    score += w.preferred_chains.get(chain_name, 0) * 0.01 * 0.3
                    scored.append((score, w))
                scored.sort(key=lambda x: x[0], reverse=True)
                return scored[0][1] if scored else available[0]
            best = max(available, key=lambda w: w.success_rate)
            return best

    def submit_task(self, title: str, description: str,
                    priority: str = "normal",
                    chain_name: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> PoolTask:
        try:
            prio = TaskPriority(priority)
        except ValueError:
            prio = TaskPriority.NORMAL

        if chain_name is None:
            chain_name = self._select_chain(description)

        task = PoolTask(
            title=title, description=description, priority=prio,
            chain_name=chain_name, metadata=metadata or {},
        )

        with self._lock:
            self._tasks[task.task_id] = task
            self._task_queue.append(task.task_id)
            self._task_order.append(task.task_id)

        with self._stats_lock:
            self._stats["tasks_submitted"] += 1

        self._dispatch_next()

        # Auto-journal
        if self._shadow_os and hasattr(self._shadow_os, 'journal'):
            try:
                self._shadow_os.journal.record(
                    "observation",
                    f"ShadowPool task submitted: {title} (chain={chain_name})",
                    category="system", importance=task.priority == TaskPriority.CRITICAL and 0.9 or 0.5,
                    tags=["shadow_pool", "task", "automated"]
                )
            except (ValueError, TypeError, AttributeError, RuntimeError):
                pass

        return task

    def _dispatch_next(self):
        """Dispatch the next queued task to an available worker."""
        with self._lock:
            while self._task_queue:
                task_id = self._task_queue[0]
                task = self._tasks.get(task_id)
                if not task or task.status != TaskStatus.QUEUED:
                    self._task_queue.popleft()
                    continue
                worker = self._select_worker(task.chain_name)
                if worker is None:
                    break
                self._task_queue.popleft()
                task.status = TaskStatus.ASSIGNED
                task.assigned_shadow = worker.worker_id
                worker.status = "busy"
                worker.current_task = task_id
                worker.last_active = datetime.now(timezone.utc)
                self._execute_task_async(task, worker)

    def _execute_task_async(self, task: PoolTask, worker: ShadowWorker):
        """Execute task by running the ability chain on the assigned worker."""
        import threading as th
        t = th.Thread(
            target=self._execute_task, args=(task, worker),
            daemon=True, name=f"pool-task-{task.task_id}"
        )
        t.start()

    def _execute_task(self, task: PoolTask, worker: ShadowWorker):
        """Run the compound ability chain for a task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        start = time.monotonic()

        chain = ABILITY_CHAINS.get(task.chain_name, {})
        abilities = chain.get("abilities", ["pattern_sight", "oracle"])
        results = []
        overall_success = True

        for ability_name in abilities:
            try:
                result = self._invoke_ability(ability_name, task.description)
                results.append({
                    "ability": ability_name,
                    "success": result.get("success", False),
                    "confidence": result.get("confidence", 0.5),
                    "output": result.get("output", ""),
                    "error": result.get("error"),
                })
                if not result.get("success"):
                    overall_success = False
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                results.append({
                    "ability": ability_name,
                    "success": False,
                    "error": type(e).__name__,
                    "confidence": 0.0,
                })
                overall_success = False

        elapsed = (time.monotonic() - start) * 1000
        task.completed_at = datetime.now(timezone.utc)
        task.chain_results = results
        task.status = TaskStatus.COMPLETED if overall_success else TaskStatus.FAILED
        task.result = {
            "chain": task.chain_name,
            "abilities_run": len(abilities),
            "success": overall_success,
            "results": results,
            "duration_ms": round(elapsed, 1),
        }

        self._update_effectiveness(task.chain_name, overall_success)

        record = ChainExecutionRecord(
            chain_name=task.chain_name,
            task_id=task.task_id,
            worker_id=worker.worker_id,
            ability_results=results,
            overall_success=overall_success,
            effectiveness=(ChainEffectiveness.EXCELLENT if overall_success and len(results) >= 3
                           else ChainEffectiveness.GOOD if overall_success
                           else ChainEffectiveness.POOR),
            duration_ms=elapsed,
        )
        with self._lock:
            self._chain_records.append(record)

        with self._lock:
            worker.status = "idle"
            worker.current_task = None
            worker.total_tasks += 1
            if overall_success:
                worker.successful_tasks += 1
            else:
                worker.failed_tasks += 1
            worker.avg_task_duration_ms = (
                worker.avg_task_duration_ms * (worker.total_tasks - 1) + elapsed
            ) / worker.total_tasks
            if task.chain_name:
                worker.preferred_chains[task.chain_name] = worker.preferred_chains.get(task.chain_name, 0) + 1

        with self._stats_lock:
            if overall_success:
                self._stats["tasks_completed"] += 1
            else:
                self._stats["tasks_failed"] += 1
            self._stats["ability_chains_run"] += 1

        self._dispatch_next()

    def _invoke_ability(self, ability_name: str, description: str) -> Dict[str, Any]:
        """Invoke a single Shikigami ability through the ShadowOS pipeline."""
        os = self._shadow_os
        if os is None:
            return {"success": False, "error": "No ShadowOS connected", "confidence": 0.0}

        try:
            # Route through ShadowSummon if available
            if hasattr(os, 'summon') and os.summon:
                intent = os.summon.classify_intent(f"use {ability_name} to {description}")
                if hasattr(os.summon, 'match_ability'):
                    matched = os.summon.match_ability(description)
                    if matched == ability_name:
                        return {
                            "success": True,
                            "confidence": 0.8,
                            "output": f"Ability {ability_name} activated for: {description[:100]}",
                        }

            # Direct invocation through ShadowAI.act() 
            if hasattr(os, 'shadow_ai') and os.shadow_ai:
                try:
                    result = os.shadow_ai.act(f"use {ability_name}: {description}")
                    if result and isinstance(result, dict) and result.get("success", False):
                        return {
                            "success": True,
                            "confidence": result.get("confidence", 0.7),
                            "output": result.get("response", result.get("output", str(result)))[:500],
                        }
                except (ValueError, TypeError, AttributeError, RuntimeError):
                    pass

            # Cognitive module fallbacks for each ability
            cognitive_fallbacks = {
                "oracle": self._oracle_fallback,
                "pattern_sight": self._pattern_fallback,
                "perfect_memory": self._memory_fallback,
                "x_ray": self._xray_fallback,
                "social_radar": self._social_fallback,
                "intent_seer": self._intent_fallback,
                "omni_sense": self._omni_fallback,
                "shadow_council": self._council_fallback,
                "reality_forge": self._forge_fallback,
            }
            handler = cognitive_fallbacks.get(ability_name)
            if handler:
                return handler(description)

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            return {"success": False, "error": type(e).__name__, "confidence": 0.0}

        return {"success": False, "error": f"Ability not available: {ability_name}", "confidence": 0.0}

    def _oracle_fallback(self, desc: str) -> Dict:
        os = self._shadow_os
        if os and hasattr(os, 'oracle') and os.oracle:
            try:
                result = os.oracle.predict("general", desc)
                return {"success": True, "confidence": result.confidence if hasattr(result, 'confidence') else 0.7, "output": str(result)[:500]}
            except Exception: pass
        return {"success": False, "error": "Oracle unavailable", "confidence": 0.3}

    def _pattern_fallback(self, desc: str) -> Dict:
        os = self._shadow_os
        if os and hasattr(os, 'recall_synthesis') and os.recall_synthesis:
            try:
                patterns = os.recall_synthesis.find_cross_domain_patterns()
                return {"success": True, "confidence": 0.7, "output": f"{len(patterns)} patterns found"}
            except Exception: pass
        return {"success": False, "error": "Pattern detection unavailable", "confidence": 0.3}

    def _memory_fallback(self, desc: str) -> Dict:
        os = self._shadow_os
        if os and hasattr(os, 'memory') and os.memory:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                results = loop.run_until_complete(os.memory.search(desc, limit=5))
                loop.close()
                return {"success": True, "confidence": 0.8, "output": f"{len(results)} memories found"}
            except Exception: pass
        return {"success": False, "error": "Memory unavailable", "confidence": 0.3}

    def _xray_fallback(self, desc: str) -> Dict:
        os = self._shadow_os
        if os and hasattr(os, 'presence') and os.presence:
            try:
                env = os.presence.detect_environment()
                return {"success": True, "confidence": 0.7, "output": f"Environment: {env.to_dict() if hasattr(env, 'to_dict') else str(env)}"}
            except Exception: pass
        return {"success": False, "error": "Presence detection unavailable", "confidence": 0.3}

    def _social_fallback(self, desc: str) -> Dict:
        os = self._shadow_os
        if os and hasattr(os, 'social_intel') and os.social_intel:
            try:
                stats = os.social_intel.get_stats()
                return {"success": True, "confidence": 0.6, "output": f"Social: {stats.get('groups_tracked', 0)} groups tracked"}
            except Exception: pass
        return {"success": False, "error": "Social intelligence unavailable", "confidence": 0.3}

    def _intent_fallback(self, desc: str) -> Dict:
        os = self._shadow_os
        if os and hasattr(os, 'anticipatory') and os.anticipatory:
            try:
                signals = os.anticipatory.predict(time_of_day=datetime.now(timezone.utc).hour)
                return {"success": True, "confidence": 0.7, "output": f"{len(signals)} anticipations generated"}
            except Exception: pass
        return {"success": False, "error": "Anticipatory engine unavailable", "confidence": 0.3}

    def _omni_fallback(self, desc: str) -> Dict:
        os = self._shadow_os
        if os and hasattr(os, 'dashboard_agg') and os.dashboard_agg:
            try:
                snap = os.dashboard_agg.get_snapshot(force=True)
                return {"success": True, "confidence": 0.8, "output": f"{snap.module_count} modules active"}
            except Exception: pass
        return {"success": False, "error": "Dashboard unavailable", "confidence": 0.3}

    def _council_fallback(self, desc: str) -> Dict:
        os = self._shadow_os
        if os and hasattr(os, 'arbiter') and os.arbiter:
            try:
                stats = os.arbiter.get_stats()
                return {"success": True, "confidence": 0.6, "output": f"Arbiter: {stats.get('claims_submitted', 0)} claims"}
            except Exception: pass
        return {"success": False, "error": "Decision arbiter unavailable", "confidence": 0.3}

    def _forge_fallback(self, desc: str) -> Dict:
        os = self._shadow_os
        if os and hasattr(os, 'self_mod') and os.self_mod:
            try:
                stats = os.self_mod.get_stats()
                return {"success": True, "confidence": 0.6, "output": f"Self-mod: {stats.get('modifications_proposed', 0)} modifications"}
            except Exception: pass
        return {"success": False, "error": "Self-modification unavailable", "confidence": 0.3}

    def _update_effectiveness(self, chain_name: str, success: bool):
        with self._lock:
            self._chain_effectiveness[chain_name].append(1.0 if success else 0.0)
            if len(self._chain_effectiveness[chain_name]) > 200:
                self._chain_effectiveness[chain_name] = self._chain_effectiveness[chain_name][-200:]

    def get_chain_effectiveness(self, chain_name: str) -> float:
        with self._lock:
            ratings = self._chain_effectiveness.get(chain_name, [])
            return sum(ratings) / max(1, len(ratings))

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        task_id = _sanitize(task_id, 20)
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return copy.deepcopy(task.to_dict())
        return None

    def get_tasks(self, status: Optional[str] = None,
                  limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(200, limit))
        with self._lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status.value == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in tasks[:limit]]

    def get_workers(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            workers = list(self._workers.values())
        if role:
            workers = [w for w in workers if w.role.value == role]
        workers.sort(key=lambda w: w.last_active, reverse=True)
        return [w.to_dict() for w in workers]

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        worker_id = _sanitize(worker_id, 20)
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                return copy.deepcopy(worker.to_dict())
        return None

    def deploy_worker_to_remote(self, worker_id: str, device_id: str,
                                 location: str = "") -> Optional[Dict[str, Any]]:
        """Deploy a pool worker to a remote device via ShadowDeployEngine."""
        worker_id = _sanitize(worker_id, 20)
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return None
            worker.deployed_to = device_id
            if self._shadow_os and hasattr(self._shadow_os, 'shadow_deploy'):
                shadow = self._shadow_os.shadow_deploy.provision_shadow(
                    name=f"Pool-{worker.name}", device_id=device_id,
                    mode="sentry", location=location,
                )
                with self._stats_lock:
                    self._stats["workers_deployed"] += 1
                return shadow.to_dict()
        return None

    def cancel_task(self, task_id: str) -> bool:
        task_id = _sanitize(task_id, 20)
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status in (TaskStatus.QUEUED, TaskStatus.ASSIGNED):
                task.status = TaskStatus.CANCELLED
                if task.assigned_shadow:
                    worker = self._workers.get(task.assigned_shadow)
                    if worker and worker.current_task == task_id:
                        worker.status = "idle"
                        worker.current_task = None
                self._dispatch_next()
                return True
        return False

    def get_available_chains(self) -> List[Dict[str, Any]]:
        return [
            {
                "chain_name": name,
                "display_name": chain["name"],
                "description": chain["description"],
                "abilities": chain["abilities"],
                "trigger_words": chain["trigger_words"][:5],
                "effectiveness": round(self.get_chain_effectiveness(name), 3),
            }
            for name, chain in ABILITY_CHAINS.items()
        ]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            workers_available = sum(1 for w in self._workers.values() if w.is_available)
            workers_busy = sum(1 for w in self._workers.values() if not w.is_available)
        with self._stats_lock:
            stats = dict(self._stats)
        stats.update({
            "workers_total": len(self._workers),
            "workers_available": workers_available,
            "workers_busy": workers_busy,
            "queue_depth": len(self._task_queue),
            "chains_available": len(ABILITY_CHAINS),
        })
        return stats
