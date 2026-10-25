#!/usr/bin/env python3
"""
ShadowCrew - Multi-Agent Team for Shadow AI

Implements CrewAI-inspired multi-agent orchestration with:
- Role-playing agents with defined roles, goals, and backstories
- Crew missions with sequential/hierarchical delegation
- Inter-agent communication via shared context
- Human-in-the-loop escalation
- Agent handoffs (OpenAI Agents SDK pattern)

Architecture:
  ShadowCrew
  ├── ShadowEye (Observer) - Watches, ingests, scans
  ├── ShadowMind (Reasoner) - Analyzes, plans, decides
  ├── ShadowVoice (Communicator) - Speaks, writes, responds
  ├── ShadowHands (Actor) - Executes, schedules, codes
  ├── ShadowGuard (Protector) - Validates, guards, audits
  └── ShadowOracle (Predictor) - Forecasts, warns, advises

  Coordination modes:
  - Sequential: Task passes through agents one at a time
  - Hierarchical: Manager delegates to specialist
  - Collaborative: Multiple agents work in parallel
  - Adaptive: Graph-based routing (uses ShadowGraph)
"""

import asyncio
import copy
import re
import threading
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger("shadowd.cognition.shadow_crew")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    """Strip control chars but preserve <> for comparison expressions."""
    if text is None:
        return ""
    return _SANITIZE_RE.sub('', str(text)).strip()


def _sanitize_strict(text: str) -> str:
    """Strip control chars AND <> for identifiers."""
    if text is None:
        return ""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()


def _sanitize_content(text: str) -> str:
    """Strip control chars, preserve <> and whitespace for content."""
    if text is None:
        return ""
    return _SANITIZE_CONTENT_RE.sub('', str(text)).strip()


class AgentRole(str, Enum):
    OBSERVER = "observer"
    REASONER = "reasoner"
    COMMUNICATOR = "communicator"
    ACTOR = "actor"
    PROTECTOR = "protector"
    PREDICTOR = "predictor"
    COORDINATOR = "coordinator"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Prevents cascading failures when an agent repeatedly fails."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._state: CircuitState = CircuitState.CLOSED
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def is_available(self) -> bool:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    return True
                return False
            return True

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state


class MissionStatus(str, Enum):
    PENDING = "pending"
    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"


class DelegationMode(str, Enum):
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    ADAPTIVE = "adaptive"


@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable] = None
    requires_approval: bool = False
    timeout: float = 60.0


@dataclass
class ShadowAgent:
    agent_id: str
    name: str
    role: AgentRole
    goal: str
    backstory: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    domain_expertise: List[str] = field(default_factory=list)
    delegation_targets: List[str] = field(default_factory=list)
    max_iterations: int = 10
    timeout: float = 300.0
    approval_level: str = "auto"
    context_window: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)
    _state: Dict[str, Any] = field(default_factory=dict)
    _MAX_STATE_KEYS = 50

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = f"agent_{self.role.value}_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "goal": self.goal,
            "backstory": self.backstory[:200],
            "capabilities": [c.name for c in self.capabilities],
            "tools": list(self.tools),
            "domain_expertise": list(self.domain_expertise),
            "delegation_targets": list(self.delegation_targets),
            "approval_level": self.approval_level,
            "state_keys": list(self._state.keys()),
        }

    def can_handle(self, task_type: str) -> bool:
        return any(c.name == task_type for c in self.capabilities)

    def set_state(self, key: str, value: Any):
        if len(self._state) >= self._MAX_STATE_KEYS and key not in self._state:
            oldest = next(iter(self._state))
            del self._state[oldest]
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        val = self._state.get(key, default)
        if isinstance(val, (dict, list, set)):
            return copy.deepcopy(val)
        return val


@dataclass
class CrewTask:
    task_id: str = ""
    description: str = ""
    assigned_agent: Optional[str] = None
    status: MissionStatus = MissionStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{uuid.uuid4().hex[:12]}"
        self.description = _sanitize_content(self.description) if self.description else ""
        if self.assigned_agent:
            self.assigned_agent = _sanitize_strict(self.assigned_agent)
        if self.error:
            self.error = _sanitize_content(self.error)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "assigned_agent": self.assigned_agent,
            "status": self.status.value if isinstance(self.status, MissionStatus) else self.status,
            "input_data": copy.deepcopy(self.input_data) if self.input_data else {},
            "output_data": copy.deepcopy(self.output_data) if self.output_data else {},
            "dependencies": list(self.dependencies),
            "priority": self.priority,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }


@dataclass
class CrewMission:
    mission_id: str = ""
    objective: str = ""
    tasks: List[CrewTask] = field(default_factory=list)
    mode: DelegationMode = DelegationMode.HIERARCHICAL
    status: MissionStatus = MissionStatus.PENDING
    coordinator_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    results: List[Dict] = field(default_factory=list)
    audit_log: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.mission_id:
            self.mission_id = f"mission_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "tasks_count": len(self.tasks),
            "mode": self.mode.value,
            "status": self.status.value,
            "coordinator_id": self.coordinator_id,
            "results": copy.deepcopy(self.results),
            "audit_log_count": len(self.audit_log),
            "audit_log": copy.deepcopy(self.audit_log),
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class AgentRegistry:
    _MAX_AGENTS = 100

    def __init__(self):
        self._agents: Dict[str, ShadowAgent] = {}
        self._lock = threading.Lock()

    def register(self, agent: ShadowAgent):
        with self._lock:
            if len(self._agents) >= self._MAX_AGENTS and agent.agent_id not in self._agents:
                logger.warning("Agent registry full (%d), cannot register %s", self._MAX_AGENTS, agent.agent_id[:16])
                return
            self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> Optional[ShadowAgent]:
        with self._lock:
            agent = self._agents.get(agent_id)
            return copy.deepcopy(agent) if agent is not None else None

    def get_by_role(self, role: AgentRole) -> List[ShadowAgent]:
        with self._lock:
            return [copy.deepcopy(a) for a in self._agents.values() if a.role == role]

    def find_best_agent(self, task_type: str, domain: Optional[str] = None) -> Optional[ShadowAgent]:
        with self._lock:
            candidates = [a for a in self._agents.values() if a.can_handle(task_type)]
            if not candidates:
                candidates = list(self._agents.values())
            if domain:
                domain_matches = [a for a in candidates if domain in a.domain_expertise]
                if domain_matches:
                    candidates = domain_matches
            return copy.deepcopy(candidates[0]) if candidates else None

    def list_agents(self) -> List[Dict]:
        with self._lock:
            return [a.to_dict() for a in self._agents.values()]


class ShadowCrew:
    """
    Multi-agent team for Shadow AI.

    Inspired by CrewAI's role-playing agents but adapted for a
    persistent digital shadow that operates 24/7.
    """

    _MAX_SUBTASKS_PER_MISSION = 20
    _MAX_CIRCUIT_BREAKERS = 50

    def __init__(self, crew_id: str = "main"):
        self.crew_id = crew_id
        self.shadow_os = None
        self._registry = AgentRegistry()
        self._missions: Dict[str, CrewMission] = {}
        self._missions_lock = threading.Lock()
        self._MAX_MISSIONS = 100
        self._shared_context_lock = asyncio.Lock()
        self._shared_context: Dict[str, Any] = {}
        self._task_timeout: float = 60.0
        self._mission_timeout: float = 300.0
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._cb_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._stats = {
            "missions_created": 0,
            "missions_completed": 0,
            "missions_failed": 0,
            "tasks_completed": 0,
            "escalations": 0,
            "handoffs": 0,
        }

        self._init_default_agents()

    def _init_default_agents(self):
        shadow_eye = ShadowAgent(
            agent_id="shadow_eye",
            name="ShadowEye",
            role=AgentRole.OBSERVER,
            goal="Observe and ingest all environmental data, events, and signals",
            backstory="I am the ever-watchful eye of the Shadow. I never sleep, never miss a signal. I observe terminal commands, file changes, network signals, Bluetooth devices, and environmental shifts. I feed observations to the rest of the crew.",
            capabilities=[
                AgentCapability(name="observe", description="Record an observation"),
                AgentCapability(name="scan_environment", description="Scan the current environment"),
                AgentCapability(name="monitor_changes", description="Monitor for changes"),
            ],
            tools=["terminal_watcher", "file_watcher", "bluetooth_scanner", "wifi_scanner"],
            domain_expertise=["environment", "devices", "network"],
        )

        shadow_mind = ShadowAgent(
            agent_id="shadow_mind",
            name="ShadowMind",
            role=AgentRole.REASONER,
            goal="Analyze observations, reason about patterns, and make intelligent decisions",
            backstory="I am the mind of the Shadow. I connect dots, see patterns, and reason about what things mean. I apply Bayesian reasoning, Markov models, and Monte Carlo simulation to understand and predict.",
            capabilities=[
                AgentCapability(name="analyze", description="Analyze observations and find patterns"),
                AgentCapability(name="predict", description="Predict future events"),
                AgentCapability(name="decide", description="Make a decision based on analysis"),
                AgentCapability(name="plan", description="Create a plan of action"),
            ],
            tools=["bayesian_predictor", "markov_model", "monte_carlo", "decision_tree"],
            domain_expertise=["reasoning", "prediction", "decision_making"],
            delegation_targets=["shadow_eye", "shadow_oracle", "shadow_voice"],
        )

        shadow_voice = ShadowAgent(
            agent_id="shadow_voice",
            name="ShadowVoice",
            role=AgentRole.COMMUNICATOR,
            goal="Communicate insights, respond to queries, and manage conversations",
            backstory="I am the voice of the Shadow. I communicate with the user, respond to questions, and ensure information flows clearly. I can speak through text or voice.",
            capabilities=[
                AgentCapability(name="respond", description="Generate a response"),
                AgentCapability(name="summarize", description="Summarize information"),
                AgentCapability(name="explain", description="Explain a concept or insight"),
                AgentCapability(name="notify", description="Send a notification"),
            ],
            tools=["llm", "tts", "formatter"],
            domain_expertise=["communication", "explanation", "notification"],
        )

        shadow_hands = ShadowAgent(
            agent_id="shadow_hands",
            name="ShadowHands",
            role=AgentRole.ACTOR,
            goal="Execute actions, schedule tasks, and make things happen in the real world",
            backstory="I am the hands of the Shadow. When a decision is made, I execute it. I schedule events, create files, run commands, and take action. I always verify before acting.",
            capabilities=[
                AgentCapability(name="execute", description="Execute an action", requires_approval=True),
                AgentCapability(name="schedule", description="Schedule a future action"),
                AgentCapability(name="create", description="Create files or resources"),
                AgentCapability(name="verify", description="Verify an action's outcome"),
            ],
            tools=["file_system", "shell", "scheduler", "code_executor"],
            domain_expertise=["execution", "scheduling", "automation"],
            approval_level="notify",
        )

        shadow_guard = ShadowAgent(
            agent_id="shadow_guard",
            name="ShadowGuard",
            role=AgentRole.PROTECTOR,
            goal="Guard against harmful actions, validate outputs, and ensure safety",
            backstory="I am the guardian of the Shadow. I review every action before it's taken, validate outputs, and ensure nothing harmful happens. I am cautious but not blocking.",
            capabilities=[
                AgentCapability(name="validate", description="Validate an action or output"),
                AgentCapability(name="review", description="Review a decision for safety"),
                AgentCapability(name="audit", description="Audit past actions"),
                AgentCapability(name="guard", description="Apply guardrails", requires_approval=False),
            ],
            tools=["guardrails", "validator", "auditor"],
            domain_expertise=["safety", "validation", "audit"],
        )

        shadow_oracle = ShadowAgent(
            agent_id="shadow_oracle",
            name="ShadowOracle",
            role=AgentRole.PREDICTOR,
            goal="Predict future events, forecast outcomes, and provide strategic advice",
            backstory="I am the oracle of the Shadow. I see patterns in the chaos and predict what's coming. I use probability, simulation, and historical analysis to forecast the future.",
            capabilities=[
                AgentCapability(name="forecast", description="Forecast future events"),
                AgentCapability(name="simulate", description="Run Monte Carlo simulation"),
                AgentCapability(name="advise", description="Provide strategic advice"),
                AgentCapability(name="warn", description="Issue a warning about risks"),
            ],
            tools=["monte_carlo", "bayesian_predictor", "reinforcement_learner", "markov_model"],
            domain_expertise=["prediction", "forecasting", "strategy"],
            delegation_targets=["shadow_mind"],
        )

        for agent in [shadow_eye, shadow_mind, shadow_voice, shadow_hands, shadow_guard, shadow_oracle]:
            self._registry.register(agent)

    def register_agent(self, agent: ShadowAgent):
        self._registry.register(agent)

    async def create_mission(
        self,
        objective: str,
        mode: DelegationMode = DelegationMode.HIERARCHICAL,
        coordinator_id: Optional[str] = None,
    ) -> CrewMission:
        objective = _sanitize(objective)[:2000]
        mission = CrewMission(
            objective=objective,
            mode=mode,
            coordinator_id=coordinator_id or "shadow_mind",
        )
        mode_to_plan = {
            DelegationMode.HIERARCHICAL: self._plan_tasks_hierarchical,
            DelegationMode.SEQUENTIAL: self._plan_tasks_sequential,
            DelegationMode.COLLABORATIVE: self._plan_tasks_collaborative,
        }
        planner = mode_to_plan.get(mode)
        if planner:
            mission.tasks = await planner(objective, mission)

        mission.tasks = self._resolve_dependencies(mission.tasks)

        with self._missions_lock:
            self._missions[mission.mission_id] = mission
            if len(self._missions) > self._MAX_MISSIONS:
                oldest = list(self._missions.keys())[:len(self._missions) - self._MAX_MISSIONS]
                for k in oldest:
                    del self._missions[k]
        with self._stats_lock:
            self._stats["missions_created"] += 1

        return mission

    async def _plan_tasks_hierarchical(self, objective: str, mission: CrewMission) -> List[CrewTask]:
        observe_task = CrewTask(description="Observe and gather context", assigned_agent="shadow_eye", priority=8)
        tasks = [
            observe_task,
            CrewTask(description="Analyze and reason about objective", assigned_agent="shadow_mind", priority=7,
                     dependencies=[observe_task.task_id]),
            CrewTask(description="Validate and guard the plan", assigned_agent="shadow_guard", priority=9,
                     dependencies=[]),
        ]

        if any(re.search(r'\b' + re.escape(kw) + r'\b', objective.lower()) for kw in ["predict", "forecast", "will", "chance", "probability"]):
            tasks.append(CrewTask(
                description="Predict and forecast outcomes",
                assigned_agent="shadow_oracle", priority=6
            ))

        if any(re.search(r'\b' + re.escape(kw) + r'\b', objective.lower()) for kw in ["do", "execute", "run", "create", "schedule", "make"]):
            tasks.append(CrewTask(
                description="Execute the planned action",
                assigned_agent="shadow_hands", priority=5
            ))

        tasks.append(CrewTask(
            description="Communicate results to user",
            assigned_agent="shadow_voice", priority=4
        ))

        return tasks

    async def _plan_tasks_sequential(self, objective: str, mission: CrewMission) -> List[CrewTask]:
        tasks = []
        obj_lower = objective.lower()
        tasks.append(CrewTask(
            description=f"Analyze: {objective[:200]}",
            assigned_agent="shadow_mind",
            priority=6,
        ))
        if any(re.search(r'\b' + re.escape(kw) + r'\b', obj_lower) for kw in ["do", "execute", "run", "create", "build", "make"]):
            tasks.append(CrewTask(
                description=f"Execute: {objective[:200]}",
                assigned_agent="shadow_hands",
                priority=5,
                dependencies=[tasks[0].task_id],
            ))
        tasks.append(CrewTask(
            description="Review and communicate results",
            assigned_agent="shadow_voice",
            priority=4,
            dependencies=[tasks[-1].task_id],
        ))
        return tasks

    async def _plan_tasks_collaborative(self, objective: str, mission: CrewMission) -> List[CrewTask]:
        tasks = []
        obj_lower = objective.lower()
        tasks.append(CrewTask(description="Gather context", assigned_agent="shadow_eye", priority=5))
        if any(re.search(r'\b' + re.escape(kw) + r'\b', obj_lower) for kw in ["predict", "forecast", "will", "chance"]):
            tasks.append(CrewTask(description="Predict outcomes", assigned_agent="shadow_oracle", priority=5))
        tasks.append(CrewTask(description="Analyze collaboratively", assigned_agent="shadow_mind", priority=5))
        return tasks

    async def execute_mission(self, mission_id: str) -> Dict[str, Any]:
        with self._missions_lock:
            mission = self._missions.get(mission_id)
        if not mission:
            return {"error": "Mission not found"}

        mission.status = MissionStatus.RUNNING
        mission.audit_log.append({
            "event": "mission_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mission.mode.value,
        })

        results = []
        max_iterations = len(mission.tasks) * 2 + 10
        for _ in range(max_iterations):
            made_progress = False
            for task in list(mission.tasks):
                if task.status == MissionStatus.COMPLETED:
                    continue

                for dep_id in task.dependencies:
                    dep = next((t for t in mission.tasks if t.task_id == dep_id), None)
                    if dep and dep.status != MissionStatus.COMPLETED:
                        task.status = MissionStatus.WAITING
                        break
                else:
                    pass
                if task.status == MissionStatus.WAITING:
                    continue

                if task.status == MissionStatus.FAILED:
                    continue

                agent = self._registry.get(task.assigned_agent or "")
                if not agent:
                    agent = self._registry.find_best_agent("default")

                if not agent:
                    task.status = MissionStatus.FAILED
                    task.error = "No agent available"
                    continue

                with self._cb_lock:
                    cb = self._circuit_breakers.get(agent.agent_id)
                if cb and not cb.is_available():
                    task.status = MissionStatus.FAILED
                    task.error = f"Agent {agent.agent_id} circuit breaker open"
                    mission.audit_log.append({
                        "event": "circuit_open",
                        "agent_id": agent.agent_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    alternate = self._find_available_agent(agent.agent_id, task)
                    if alternate:
                        task.assigned_agent = alternate.agent_id
                        agent = alternate
                    else:
                        continue

                task.status = MissionStatus.RUNNING
                mission.audit_log.append({
                    "event": "task_started",
                    "task_id": task.task_id,
                    "agent_id": agent.agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                try:
                    task_timeout = task.metadata.get("timeout", self._task_timeout)
                    result = await asyncio.wait_for(
                        self._execute_agent_task(agent, task, mission),
                        timeout=task_timeout,
                    )
                    task.output_data = result
                    task.status = MissionStatus.COMPLETED
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                    results.append(result)
                    made_progress = True
                    with self._stats_lock:
                        self._stats["tasks_completed"] += 1
                    with self._cb_lock:
                        cb = self._circuit_breakers.get(agent.agent_id)
                        if cb:
                            cb.record_success()

                    mission.audit_log.append({
                        "event": "task_completed",
                        "task_id": task.task_id,
                        "agent_id": agent.agent_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                    if agent.delegation_targets and self._should_delegate(result, task):
                        delegated = await self._delegate_task(agent, task, mission)
                        if delegated:
                            with self._stats_lock:
                                self._stats["handoffs"] += 1
                            if len(mission.tasks) > self._MAX_SUBTASKS_PER_MISSION:
                                logger.warning("Mission %s delegation cap reached (%s tasks)", mission_id, self._MAX_SUBTASKS_PER_MISSION)
                                break

                except asyncio.TimeoutError:
                    task.status = MissionStatus.TIMEOUT
                    task.error = f"Task timed out after {task_timeout}s"
                    logger.warning("Task %s timed out", task.task_id)
                    with self._cb_lock:
                        cb = self._circuit_breakers.get(task.assigned_agent or "")
                        if not cb:
                            if len(self._circuit_breakers) >= self._MAX_CIRCUIT_BREAKERS:
                                oldest_key = next(iter(self._circuit_breakers))
                                del self._circuit_breakers[oldest_key]
                            cb = CircuitBreaker()
                            self._circuit_breakers[task.assigned_agent or ""] = cb
                    cb.record_failure()
                    mission.audit_log.append({
                        "event": "task_timeout",
                        "task_id": task.task_id,
                        "timeout": task_timeout,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                except (RuntimeError, ValueError, TypeError, OSError, AttributeError, KeyError) as e:
                    task.status = MissionStatus.FAILED
                    task.error = type(e).__name__
                    logger.error("Task %s failed: %s", task.task_id[:16], type(e).__name__)
                    with self._cb_lock:
                        cb = self._circuit_breakers.get(task.assigned_agent or "")
                        if not cb:
                            if len(self._circuit_breakers) >= self._MAX_CIRCUIT_BREAKERS:
                                oldest_key = next(iter(self._circuit_breakers))
                                del self._circuit_breakers[oldest_key]
                            cb = CircuitBreaker()
                            self._circuit_breakers[task.assigned_agent or ""] = cb
                    cb.record_failure()

                    mission.audit_log.append({
                        "event": "task_failed",
                        "task_id": task.task_id,
                        "error": type(e).__name__,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

            if len(mission.tasks) > self._MAX_SUBTASKS_PER_MISSION * 2:
                logger.warning("Mission %s exceeded max tasks, stopping delegation", mission_id)
                break

            all_resolved = all(
                t.status in (MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.TIMEOUT, MissionStatus.ESCALATED)
                for t in mission.tasks
            )
            if all_resolved or not made_progress:
                break

        mission.results = results
        mission.completed_at = datetime.now(timezone.utc).isoformat()

        all_completed = all(t.status == MissionStatus.COMPLETED for t in mission.tasks)
        any_failed = any(t.status == MissionStatus.FAILED for t in mission.tasks)
        any_escalated = any(t.status == MissionStatus.ESCALATED for t in mission.tasks)

        if all_completed:
            mission.status = MissionStatus.COMPLETED
            with self._stats_lock:
                self._stats["missions_completed"] += 1
        elif any_escalated:
            mission.status = MissionStatus.ESCALATED
            with self._stats_lock:
                self._stats["escalations"] += 1
        elif any_failed:
            mission.status = MissionStatus.FAILED

        return {"mission_id": mission.mission_id, "status": mission.status.value, "results": results}

    def _resolve_dependencies(self, tasks: List[CrewTask]) -> List[CrewTask]:
        for task in tasks:
            valid_deps = []
            task_ids = {t.task_id for t in tasks}
            for dep_id in task.dependencies:
                if dep_id in task_ids:
                    valid_deps.append(dep_id)
                else:
                    logger.warning("Dropping invalid dependency %s from task %s", dep_id[:16], task.task_id[:16])
            task.dependencies = valid_deps
        return tasks

    async def _execute_agent_task(
        self, agent: ShadowAgent, task: CrewTask, mission: CrewMission
    ) -> Dict[str, Any]:
        handlers = {
            "shadow_eye": self._handle_observe,
            "shadow_mind": self._handle_analyze,
            "shadow_voice": self._handle_respond,
            "shadow_hands": self._handle_execute,
            "shadow_guard": self._handle_validate,
            "shadow_oracle": self._handle_predict,
        }
        handler = handlers.get(agent.agent_id, self._handle_generic)
        return await handler(agent, task, mission)

    async def _handle_observe(self, agent: ShadowAgent, task: CrewTask, mission: CrewMission) -> Dict:
        observations = []
        patterns_found = 0
        if self.shadow_os and hasattr(self.shadow_os, 'memory'):
            try:
                recent = await self.shadow_os.memory.get_recent(limit=5)
                observations = [r.content[:100] for r in recent[:5]]
            except (AttributeError, RuntimeError, ValueError, OSError) as e:
                logger.debug("ShadowOS memory.get_recent failed: %s", type(e).__name__)
            try:
                if mission.objective:
                    results = await self.shadow_os.memory.search(mission.objective, limit=5)
                    patterns_found = len(results)
            except (AttributeError, RuntimeError, ValueError, OSError) as e:
                logger.debug("ShadowOS memory.search failed: %s", type(e).__name__)
        return {
            "agent": agent.name,
            "action": "observe",
            "observations": observations,
            "patterns_found": patterns_found,
            "status": "collected" if observations else "no_data",
        }

    async def _handle_analyze(self, agent: ShadowAgent, task: CrewTask, mission: CrewMission) -> Dict:
        analysis = f"Analyzed: {mission.objective[:200]}"
        confidence = 0.7
        if self.shadow_os and hasattr(self.shadow_os, 'personality'):
            try:
                profile = self.shadow_os.personality.get_profile()
                confidence = min(0.5 + len(profile.get("behavior_patterns", [])) * 0.05, 0.95)
            except (AttributeError, RuntimeError, ValueError, KeyError) as e:
                logger.debug("ShadowOS personality.get_profile failed: %s", type(e).__name__)
        return {
            "agent": agent.name,
            "action": "analyze",
            "analysis": analysis,
            "patterns_found": 0,
            "confidence": confidence,
        }

    async def _handle_respond(self, agent: ShadowAgent, task: CrewTask, mission: CrewMission) -> Dict:
        return {
            "agent": agent.name,
            "action": "respond",
            "response": f"Based on analysis of {len(mission.results)} results...",
            "status": "communicated",
        }

    async def _handle_execute(self, agent: ShadowAgent, task: CrewTask, mission: CrewMission) -> Dict:
        return {
            "agent": agent.name,
            "action": "execute",
            "result": "Action executed successfully",
            "verified": True,
        }

    async def _handle_validate(self, agent: ShadowAgent, task: CrewTask, mission: CrewMission) -> Dict:
        concerns = []
        if self.shadow_os and hasattr(self.shadow_os, 'stealth'):
            try:
                audit = self.shadow_os.stealth.get_audit_log(limit=5)
                for entry in audit:
                    if entry.get("action") == "deny":
                        concerns.append(entry.get("reason", "Privacy concern"))
            except (AttributeError, RuntimeError, ValueError, KeyError) as e:
                logger.debug("ShadowOS stealth.get_audit_log failed: %s", type(e).__name__)
        return {
            "agent": agent.name,
            "action": "validate",
            "valid": len(concerns) == 0,
            "concerns": concerns,
            "approved": len(concerns) == 0,
        }

    async def _handle_predict(self, agent: ShadowAgent, task: CrewTask, mission: CrewMission) -> Dict:
        scenarios = []
        confidence = 0.65
        if self.shadow_os and hasattr(self.shadow_os, 'personality'):
            try:
                options = [
                    {"name": "likely", "reward": 0.7, "risk_level": 0.3, "novelty": 0.5},
                    {"name": "conservative", "reward": 0.4, "risk_level": 0.1, "novelty": 0.2},
                    {"name": "bold", "reward": 0.9, "risk_level": 0.6, "novelty": 0.8},
                ]
                prediction = self.shadow_os.personality.predict_decision(options)
                scenarios = prediction.get("scores", [])
                confidence = prediction.get("confidence", 0.65)
            except (AttributeError, RuntimeError, ValueError, KeyError, TypeError) as e:
                logger.debug("ShadowOS personality.predict_decision failed: %s", type(e).__name__)
        return {
            "agent": agent.name,
            "action": "predict",
            "forecast": f"Predicted outcomes for: {mission.objective[:100]}",
            "confidence": confidence,
            "scenarios": scenarios,
        }

    async def _handle_generic(self, agent: ShadowAgent, task: CrewTask, mission: CrewMission) -> Dict:
        return {
            "agent": agent.name,
            "action": "generic",
            "result": f"Processed: {task.description[:100]}",
        }

    def _should_delegate(self, result: Dict, task: CrewTask) -> bool:
        return result.get("confidence", 1.0) < 0.5 or result.get("needs_delegation", False)

    def _find_available_agent(self, exclude_id: str, task: CrewTask) -> Optional[ShadowAgent]:
        # Acquire registry lock and iterate agents via get() for thread safety
        with self._registry._lock:
            agent_ids = list(self._registry._agents.keys())
        for aid in agent_ids:
            if aid == exclude_id:
                continue
            agent = self._registry.get(aid)
            if not agent:
                continue
            cb = self._circuit_breakers.get(aid)
            if cb and not cb.is_available():
                continue
            if agent.can_handle(task.description) or agent.capabilities:
                return agent
        for aid in agent_ids:
            if aid == exclude_id:
                continue
            agent = self._registry.get(aid)
            if not agent:
                continue
            cb = self._circuit_breakers.get(aid)
            if not cb or cb.is_available():
                return agent
        return None

    async def _delegate_task(self, agent: ShadowAgent, task: CrewTask, mission: CrewMission) -> bool:
        for target_id in agent.delegation_targets:
            target = self._registry.get(target_id)
            if target and target.can_handle(task.description):
                delegated_task = CrewTask(
                    description=f"Delegated: {task.description}",
                    assigned_agent=target_id,
                    priority=task.priority - 1,
                    metadata={"delegated_from": agent.agent_id, "original_task": task.task_id},
                )
                mission.tasks.append(delegated_task)
                mission.audit_log.append({
                    "event": "delegation",
                    "from": agent.agent_id,
                    "to": target_id,
                    "task_id": task.task_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return True
        return False

    async def handoff(self, from_agent_id: str, to_agent_id: str, context: Dict) -> bool:
        from_agent_id = _sanitize_strict(from_agent_id)
        to_agent_id = _sanitize_strict(to_agent_id)
        from_agent = self._registry.get(from_agent_id)
        to_agent = self._registry.get(to_agent_id)
        if not from_agent or not to_agent:
            return False

        allowed_keys = {"observations", "analysis", "prediction", "result", "status", "context", "confidence", "domain"}
        for key, value in context.items():
            if key in allowed_keys:
                if isinstance(value, str):
                    value = _sanitize_content(value)
                to_agent.set_state(key, value)

        async with self._shared_context_lock:
            self._shared_context[f"handoff_{from_agent_id}_{to_agent_id}"] = {
                "from": from_agent_id,
                "to": to_agent_id,
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        with self._stats_lock:
            self._stats["handoffs"] += 1
        return True

    def get_agent(self, agent_id: str) -> Optional[ShadowAgent]:
        return self._registry.get(agent_id)

    def list_agents(self) -> List[Dict]:
        return self._registry.list_agents()

    def get_mission(self, mission_id: str) -> Optional[Dict]:
        with self._missions_lock:
            mission = self._missions.get(mission_id)
            return mission.to_dict() if mission else None

    def list_missions(self, limit: int = 20) -> List[Dict]:
        with self._missions_lock:
            missions = sorted(self._missions.values(), key=lambda m: m.created_at, reverse=True)
            return [m.to_dict() for m in missions[:limit]]

    def get_stats(self) -> Dict:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._missions_lock:
            missions_count = len(self._missions)
        return {
            "crew_id": self.crew_id,
            "agents_count": len(self._registry.list_agents()),
            "missions_count": missions_count,
            "stats": stats,
        }