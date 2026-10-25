#!/usr/bin/env python3
"""
ShadowArise — Universal Agent Creation Engine

Creates on-demand agents tailored to specific tasks. Each agent
is assembled from configurable capabilities, given a mission,
and can evolve through execution feedback.

Architecture:
  ShadowArise (Agent Factory)
  ├── AgentTemplate (blueprint for agent creation)
  ├── AgentInstance (running agent with state)
  ├── Capability (modular skill that agents can use)
  └── MissionContext (what the agent is trying to accomplish)

Agent Lifecycle:
  1. ARISE — User requests an agent for a task
  2. CONFIGURE — Capabilities are selected and tuned
  3. ACTIVATE — Agent begins executing its mission
  4. EXECUTE — Agent performs actions, gathers results
  5. EVOLVE — Agent adapts based on feedback
  6. COMPLETE — Agent finishes and reports results
  7. ARCHIVE — Agent is stored for future reference
"""

import copy
import re
import uuid
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("shadowd.cognition.shadow_arise")


_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _SANITIZE_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    """Strip control chars and <> for identifier fields."""
    if not isinstance(text, str):
        return ""
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 2000) -> str:
    """Strip control chars but preserve <> for content fields."""
    if not isinstance(text, str):
        return ""
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


class AgentLifecycle(str, Enum):
    ARISE = "arise"
    CONFIGURE = "configure"
    ACTIVATE = "activate"
    EXECUTE = "execute"
    EVOLVE = "evolve"
    COMPLETE = "complete"
    ARCHIVE = "archive"


class CapabilityType(str, Enum):
    REASON = "reason"
    OBSERVE = "observe"
    PLAN = "plan"
    EXECUTE_ACTION = "execute_action"
    COMMUNICATE = "communicate"
    LEARN = "learn"
    REFLECT = "reflect"
    CREATE = "create"
    ANALYZE = "analyze"
    ADAPT = "adapt"
    COORDINATE = "coordinate"
    REMEMBER = "remember"


class MissionPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class AgentStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class Capability:
    name: str
    capability_type: CapabilityType
    proficiency: float
    description: str
    configuration: Dict
    usage_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "capability_type": self.capability_type.value,
            "proficiency": round(self.proficiency, 3),
            "description": self.description,
            "configuration": copy.deepcopy(self.configuration),
            "usage_count": self.usage_count,
        }


@dataclass
class AgentTemplate:
    template_id: str
    name: str
    description: str
    capabilities: List[CapabilityType]
    default_proficiency: float
    mission_types: List[str]
    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "default_proficiency": round(self.default_proficiency, 3),
            "mission_types": list(self.mission_types),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class MissionContext:
    objective: str
    priority: MissionPriority
    constraints: List[str]
    success_criteria: List[str]
    deadline: Optional[str]
    context_data: Dict

    def to_dict(self) -> Dict:
        return {
            "objective": self.objective,
            "priority": self.priority.value,
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
            "deadline": self.deadline,
            "context_data": copy.deepcopy(self.context_data),
        }


@dataclass
class ExecutionStep:
    step_id: str
    description: str
    capability_used: CapabilityType
    input_data: Dict
    output_data: Dict
    confidence: float
    timestamp: str
    duration_ms: int

    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "capability_used": self.capability_used.value,
            "input_data": copy.deepcopy(self.input_data),
            "output_data": copy.deepcopy(self.output_data),
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentInstance:
    agent_id: str
    name: str
    template_id: str
    mission: MissionContext
    capabilities: List[Capability]
    lifecycle: AgentLifecycle
    status: AgentStatus
    created_at: str
    execution_log: List[ExecutionStep]
    evolution_count: int
    confidence_score: float
    result: Optional[Dict]
    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "template_id": self.template_id,
            "mission": self.mission.to_dict(),
            "capabilities": [c.to_dict() for c in self.capabilities],
            "lifecycle": self.lifecycle.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "execution_log": [s.to_dict() for s in self.execution_log],
            "evolution_count": self.evolution_count,
            "confidence_score": round(self.confidence_score, 3),
            "result": copy.deepcopy(self.result) if self.result else None,
            "metadata": copy.deepcopy(self.metadata),
        }


_BUILTIN_TEMPLATES = [
    AgentTemplate(
        template_id="analyst",
        name="Analyst",
        description="Deep analysis and research agent for complex problem-solving",
        capabilities=[CapabilityType.ANALYZE, CapabilityType.REASON, CapabilityType.REMEMBER, CapabilityType.REFLECT],
        default_proficiency=0.8,
        mission_types=["research", "analysis", "investigation", "evaluation"],
        metadata={"domain": "cognitive", "specialization": "analysis"},
    ),
    AgentTemplate(
        template_id="creator",
        name="Creator",
        description="Creative generation and ideation agent for producing novel content",
        capabilities=[CapabilityType.CREATE, CapabilityType.ADAPT, CapabilityType.COMMUNICATE, CapabilityType.REFLECT],
        default_proficiency=0.75,
        mission_types=["creation", "design", "writing", "ideation"],
        metadata={"domain": "creative", "specialization": "generation"},
    ),
    AgentTemplate(
        template_id="planner",
        name="Planner",
        description="Strategic planning and coordination agent for complex multi-step tasks",
        capabilities=[CapabilityType.PLAN, CapabilityType.COORDINATE, CapabilityType.REASON, CapabilityType.ANALYZE],
        default_proficiency=0.85,
        mission_types=["planning", "coordination", "scheduling", "organization"],
        metadata={"domain": "executive", "specialization": "planning"},
    ),
    AgentTemplate(
        template_id="executor",
        name="Executor",
        description="Action-oriented agent for executing tasks and implementing solutions",
        capabilities=[CapabilityType.EXECUTE_ACTION, CapabilityType.ADAPT, CapabilityType.LEARN, CapabilityType.COMMUNICATE],
        default_proficiency=0.7,
        mission_types=["execution", "implementation", "task_completion", "automation"],
        metadata={"domain": "operational", "specialization": "execution"},
    ),
    AgentTemplate(
        template_id="observer",
        name="Observer",
        description="Passive monitoring and insight agent for watching and reporting patterns",
        capabilities=[CapabilityType.OBSERVE, CapabilityType.REMEMBER, CapabilityType.ANALYZE, CapabilityType.REFLECT],
        default_proficiency=0.9,
        mission_types=["monitoring", "observation", "surveillance", "reporting"],
        metadata={"domain": "cognitive", "specialization": "observation"},
    ),
    AgentTemplate(
        template_id="learner",
        name="Learner",
        description="Adaptive learning agent that evolves from feedback and experience",
        capabilities=[CapabilityType.LEARN, CapabilityType.ADAPT, CapabilityType.REFLECT, CapabilityType.REMEMBER],
        default_proficiency=0.6,
        mission_types=["learning", "adaptation", "improvement", "skill_acquisition"],
        metadata={"domain": "adaptive", "specialization": "learning"},
    ),
    AgentTemplate(
        template_id="coordinator",
        name="Coordinator",
        description="Multi-agent coordination agent for managing teams of agents",
        capabilities=[CapabilityType.COORDINATE, CapabilityType.COMMUNICATE, CapabilityType.PLAN, CapabilityType.REASON],
        default_proficiency=0.8,
        mission_types=["coordination", "team_management", "delegation", "orchestration"],
        metadata={"domain": "social", "specialization": "coordination"},
    ),
    AgentTemplate(
        template_id="guardian",
        name="Guardian",
        description="Safety and compliance agent for risk assessment and protection",
        capabilities=[CapabilityType.ANALYZE, CapabilityType.REFLECT, CapabilityType.COORDINATE, CapabilityType.COMMUNICATE],
        default_proficiency=0.95,
        mission_types=["safety", "compliance", "risk_assessment", "protection"],
        metadata={"domain": "safety", "specialization": "guardianship"},
    ),
]


class ShadowArise:
    """
    Universal Agent Creation Engine.

    Creates on-demand agents with tailored capabilities,
    manages their lifecycle, and evolves them through feedback.
    """

    _MAX_AGENTS = 500
    _MAX_TEMPLATES = 100
    _MAX_EXECUTION_LOG = 200
    _MAX_METADATA_KEYS = 50
    _MAX_STEPS_PER_AGENT = 500
    _MAX_EVOLUTION_HISTORY = 200

    def __init__(self):
        self._lock = threading.Lock()
        self._agents: Dict[str, AgentInstance] = {}
        self._templates: Dict[str, AgentTemplate] = {}
        self._evolution_history: deque = deque(maxlen=self._MAX_EVOLUTION_HISTORY)
        self._stats = {
            "agents_created": 0,
            "agents_completed": 0,
            "agents_failed": 0,
            "agents_evolved": 0,
            "total_execution_steps": 0,
            "templates_used": {},
        }
        self._stats_lock = threading.Lock()

        for t in _BUILTIN_TEMPLATES:
            self._templates[t.template_id] = t

    def arise(
        self,
        name: str,
        objective: str,
        template_id: Optional[str] = None,
        capabilities: Optional[List[CapabilityType]] = None,
        priority: MissionPriority = MissionPriority.MEDIUM,
        constraints: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        deadline: Optional[str] = None,
        context_data: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> AgentInstance:
        """
        Create a new agent for a mission.

        Either use a template_id to base the agent on a predefined template,
        or provide explicit capabilities list for a custom agent.
        """
        name = _sanitize(name, 100)
        objective = _sanitize_content(objective, 2000)
        constraints = [_sanitize_content(c, 500) for c in (constraints or [])][:10]
        success_criteria = [_sanitize_content(c, 500) for c in (success_criteria or [])][:10]
        if deadline is not None:
            deadline = _sanitize(deadline, 50)
        context_data = dict(list((context_data or {}).items())[:self._MAX_METADATA_KEYS])
        metadata = dict(list((metadata or {}).items())[:self._MAX_METADATA_KEYS])

        agent_caps: List[Capability] = []

        if template_id and template_id in self._templates:
            template = self._templates[template_id]
            cap_types = capabilities if capabilities else template.capabilities
            proficiency = template.default_proficiency
        else:
            cap_types = capabilities if capabilities else [CapabilityType.REASON, CapabilityType.OBSERVE]
            proficiency = 0.5

        for ct in cap_types:
            agent_caps.append(Capability(
                name=ct.value,
                capability_type=ct,
                proficiency=min(1.0, max(0.0, proficiency + (int(uuid.uuid5(uuid.NAMESPACE_DNS, ct.value).hex[:8], 16) % 20 - 10) / 100.0)),
                description=f"Auto-generated {ct.value} capability",
                configuration={},
                usage_count=0,
            ))

        agent_id = uuid.uuid4().hex[:12]
        mission = MissionContext(
            objective=objective,
            priority=priority,
            constraints=constraints,
            success_criteria=success_criteria,
            deadline=deadline,
            context_data=context_data,
        )

        agent = AgentInstance(
            agent_id=agent_id,
            name=name,
            template_id=template_id or "custom",
            mission=mission,
            capabilities=agent_caps,
            lifecycle=AgentLifecycle.ARISE,
            status=AgentStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            execution_log=[],
            evolution_count=0,
            confidence_score=sum(c.proficiency for c in agent_caps) / max(len(agent_caps), 1),
            result=None,
            metadata=metadata,
        )

        with self._lock:
            if len(self._agents) >= self._MAX_AGENTS:
                oldest_id = next(iter(self._agents))
                old_agent = self._agents.pop(oldest_id)
                logger.warning("Evicting agent %s to make room for %s", oldest_id, agent_id)
            self._agents[agent_id] = agent
            with self._stats_lock:
                self._stats["agents_created"] += 1
                if template_id and template_id in self._templates:
                    self._stats["templates_used"][template_id] = self._stats["templates_used"].get(template_id, 0) + 1

        return agent

    def configure(
        self,
        agent_id: str,
        capabilities: Optional[List[CapabilityType]] = None,
        proficiency_overrides: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[AgentInstance]:
        """Configure an agent's capabilities before activation."""
        agent_id = _sanitize(agent_id, 12)
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return None
            if agent.lifecycle not in (AgentLifecycle.ARISE, AgentLifecycle.CONFIGURE):
                return None

            agent.lifecycle = AgentLifecycle.CONFIGURE

            if capabilities:
                for ct in capabilities:
                    if not any(c.capability_type == ct for c in agent.capabilities):
                        agent.capabilities.append(Capability(
                            name=ct.value,
                            capability_type=ct,
                            proficiency=0.5,
                            description=f"Added {ct.value} capability",
                            configuration={},
                            usage_count=0,
                        ))

            if proficiency_overrides:
                for cap_name, prof in proficiency_overrides.items():
                    prof = min(1.0, max(0.0, float(prof)))
                    for cap in agent.capabilities:
                        if cap.name == cap_name:
                            cap.proficiency = prof

            if metadata:
                for k, v in list(metadata.items())[:self._MAX_METADATA_KEYS]:
                    k = _sanitize(k, 50)
                    agent.metadata[k] = _sanitize_content(str(v), 500)

            agent.confidence_score = sum(c.proficiency for c in agent.capabilities) / max(len(agent.capabilities), 1)
            return copy.deepcopy(agent)

    def activate(self, agent_id: str) -> Optional[AgentInstance]:
        """Transition an agent to active status."""
        agent_id = _sanitize(agent_id, 12)
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return None
            if agent.lifecycle not in (AgentLifecycle.ARISE, AgentLifecycle.CONFIGURE):
                return None
            agent.lifecycle = AgentLifecycle.ACTIVATE
            agent.status = AgentStatus.ACTIVE
            return copy.deepcopy(agent)

    def execute_step(
        self,
        agent_id: str,
        description: str,
        capability_used: CapabilityType,
        input_data: Optional[Dict] = None,
        output_data: Optional[Dict] = None,
        confidence: float = 0.5,
        duration_ms: int = 0,
    ) -> Optional[ExecutionStep]:
        """Record an execution step for an active agent."""
        agent_id = _sanitize(agent_id, 12)
        description = _sanitize_content(description, 1000)
        input_data = dict(list((input_data or {}).items())[:self._MAX_METADATA_KEYS])
        output_data = dict(list((output_data or {}).items())[:self._MAX_METADATA_KEYS])
        confidence = min(1.0, max(0.0, float(confidence)))

        step = ExecutionStep(
            step_id=uuid.uuid4().hex[:12],
            description=description,
            capability_used=capability_used,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=max(0, int(duration_ms)),
        )

        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return None
            if agent.status != AgentStatus.ACTIVE:
                return None
            if agent.lifecycle not in (AgentLifecycle.ACTIVATE, AgentLifecycle.EXECUTE):
                return None

            agent.lifecycle = AgentLifecycle.EXECUTE
            if len(agent.execution_log) >= self._MAX_STEPS_PER_AGENT:
                agent.execution_log = agent.execution_log[-(self._MAX_STEPS_PER_AGENT - 1):]
            agent.execution_log.append(step)

            for cap in agent.capabilities:
                if cap.capability_type == capability_used:
                    cap.usage_count += 1
                    cap.proficiency = min(1.0, cap.proficiency + 0.01)

            agent.confidence_score = sum(c.proficiency for c in agent.capabilities) / max(len(agent.capabilities), 1)
            with self._stats_lock:
                self._stats["total_execution_steps"] += 1

            return step

    def evolve(
        self,
        agent_id: str,
        feedback: str,
        capability_adjustments: Optional[Dict[str, float]] = None,
        new_capabilities: Optional[List[CapabilityType]] = None,
    ) -> Optional[AgentInstance]:
        """Evolve an agent based on feedback."""
        agent_id = _sanitize(agent_id, 12)
        feedback = _sanitize_content(feedback, 2000)
        capability_adjustments = capability_adjustments or {}
        new_capabilities = new_capabilities or []

        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return None

            if agent.status in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.ARCHIVED):
                return None

            agent.lifecycle = AgentLifecycle.EVOLVE

            for cap_name, delta in capability_adjustments.items():
                delta = max(-0.5, min(0.5, float(delta)))
                for cap in agent.capabilities:
                    if cap.name == cap_name:
                        cap.proficiency = min(1.0, max(0.1, cap.proficiency + delta))

            for ct in new_capabilities:
                if not any(c.capability_type == ct for c in agent.capabilities):
                    agent.capabilities.append(Capability(
                        name=ct.value,
                        capability_type=ct,
                        proficiency=0.4,
                        description=f"Acquired via evolution: {ct.value}",
                        configuration={},
                        usage_count=0,
                    ))

            agent.evolution_count += 1
            agent.confidence_score = sum(c.proficiency for c in agent.capabilities) / max(len(agent.capabilities), 1)

            self._evolution_history.append({
                "agent_id": agent_id,
                "feedback": feedback[:500],
                "adjustments": capability_adjustments,
                "new_capabilities": [c.value for c in new_capabilities],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            with self._stats_lock:
                self._stats["agents_evolved"] += 1

            if agent.status == AgentStatus.ACTIVE:
                agent.lifecycle = AgentLifecycle.EXECUTE

            return copy.deepcopy(agent)

    def complete(self, agent_id: str, result: Optional[Dict] = None, status: AgentStatus = AgentStatus.COMPLETED) -> Optional[AgentInstance]:
        """Mark an agent as completed with an optional result."""
        agent_id = _sanitize(agent_id, 12)
        result = dict(list((result or {}).items())[:self._MAX_METADATA_KEYS]) if result else None

        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return None

            agent.lifecycle = AgentLifecycle.COMPLETE
            agent.status = status
            agent.result = result
            agent.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()

            if status == AgentStatus.COMPLETED:
                with self._stats_lock:
                    self._stats["agents_completed"] += 1
            elif status == AgentStatus.FAILED:
                with self._stats_lock:
                    self._stats["agents_failed"] += 1

            return copy.deepcopy(agent)

    def archive(self, agent_id: str) -> Optional[AgentInstance]:
        """Archive a completed agent."""
        agent_id = _sanitize(agent_id, 12)
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return None

            agent.lifecycle = AgentLifecycle.ARCHIVE
            agent.status = AgentStatus.ARCHIVED
            return copy.deepcopy(agent)

    def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Get an agent by ID."""
        agent_id = _sanitize(agent_id, 12)
        with self._lock:
            agent = self._agents.get(agent_id)
            return copy.deepcopy(agent) if agent else None

    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        lifecycle: Optional[AgentLifecycle] = None,
        limit: int = 50,
    ) -> List[AgentInstance]:
        """List agents, optionally filtered by status or lifecycle."""
        limit = min(max(1, limit), 200)
        with self._lock:
            agents = list(self._agents.values())
        if status:
            agents = [a for a in agents if a.status == status]
        if lifecycle:
            agents = [a for a in agents if a.lifecycle == lifecycle]
        agents.sort(key=lambda a: a.created_at, reverse=True)
        return [copy.deepcopy(a) for a in agents[:limit]]

    def add_template(self, template: AgentTemplate) -> AgentTemplate:
        """Add a custom agent template."""
        template.template_id = _sanitize(template.template_id, 50)
        template.name = _sanitize(template.name, 100)
        template.description = _sanitize_content(template.description, 500)
        template.metadata = dict(list(template.metadata.items())[:self._MAX_METADATA_KEYS])

        with self._lock:
            if len(self._templates) >= self._MAX_TEMPLATES:
                oldest_id = next(iter(self._templates))
                if oldest_id not in [t.template_id for t in _BUILTIN_TEMPLATES]:
                    del self._templates[oldest_id]
                    logger.warning("Evicting template %s", oldest_id)
            self._templates[template.template_id] = template
        return template

    def get_template(self, template_id: str) -> Optional[AgentTemplate]:
        """Get a template by ID."""
        template_id = _sanitize(template_id, 50)
        with self._lock:
            t = self._templates.get(template_id)
            return copy.deepcopy(t) if t else None

    def list_templates(self) -> List[AgentTemplate]:
        """List all available templates."""
        with self._lock:
            return [copy.deepcopy(t) for t in self._templates.values()]

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        with self._stats_lock:
            stats = dict(self._stats)
        stats["templates_used"] = dict(stats["templates_used"])
        with self._lock:
            active_agents = len(self._agents)
            templates_available = len(self._templates)
            evolution_history_size = len(self._evolution_history)
        return {
            **stats,
            "active_agents": active_agents,
            "templates_available": templates_available,
            "evolution_history_size": evolution_history_size,
        }

    def recommend_template(self, objective: str) -> Optional[AgentTemplate]:
        """Recommend the best template for a given objective."""
        objective_lower = _sanitize_content(objective.lower(), 2000)
        words = set(re.findall(r'\b[a-z]+\b', objective_lower))

        with self._lock:
            templates_snapshot = list(self._templates.values())

        best_template = None
        best_score = -1

        for template in templates_snapshot:
            score = 0.0
            for mtype in template.mission_types:
                if re.search(r'\b' + re.escape(mtype) + r'\b', objective_lower):
                    score += 3.0
                mtype_words = set(re.findall(r'\b[a-z]+\b', mtype.lower()))
                overlap = len(words & mtype_words)
                score += overlap * 1.0

            if re.search(r'\b' + re.escape(template.name.lower()) + r'\b', objective_lower):
                score += 5.0

            for cap in template.capabilities:
                if re.search(r'\b' + re.escape(cap.value) + r'\b', objective_lower):
                    score += 2.0

            if score > best_score:
                best_score = score
                best_template = template

        with self._lock:
            return copy.deepcopy(best_template) if best_template else None