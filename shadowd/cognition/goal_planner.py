#!/usr/bin/env python3
"""
CognitiveGoalPlanner - Goal Decomposition and Tracking for ShadowOS

Decomposes high-level goals into actionable steps, tracks progress,
and adapts plans based on outcomes. Integrates with:
- ShadowCrew for task delegation
- PersonalityEngine for preference-aware scheduling
- EscalationManager for autonomous action decisions
- DarkMemory for goal persistence and recall

Inspired by: STRIPS planning, hierarchical task networks (HTN),
and the concept of "cognitive load" from psychology.
"""

import copy
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..atomic_io import fsync_dir

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str) -> str:
    """Strip control chars but preserve <> for comparison expressions in content."""
    if text is None:
        return ""
    return _SANITIZE_RE.sub('', str(text)).strip()

def _sanitize_strict(text: str) -> str:
    """Strip control chars AND <> for identifiers and short fields."""
    if text is None:
        return ""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()

logger = logging.getLogger("shadowd.cognition.goal_planner")


class GoalStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class GoalPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PlanStep:
    step_id: str = ""
    description: str = ""
    order: int = 0
    status: StepStatus = StepStatus.PENDING
    assigned_to: Optional[str] = None
    estimated_minutes: float = 5.0
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.step_id:
            self.step_id = f"step_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "order": self.order,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "estimated_minutes": round(self.estimated_minutes, 2),
            "dependencies": list(self.dependencies),
            "result": copy.deepcopy(self.result) if self.result else None,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class Goal:
    goal_id: str = ""
    title: str = ""
    description: str = ""
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    domain: Optional[str] = None
    steps: List[PlanStep] = field(default_factory=list)
    parent_goal_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    progress: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    deadline: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.goal_id:
            self.goal_id = f"goal_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "goal_id": self.goal_id,
            "title": self.title.strip()[:200],
            "description": self.description.strip()[:500],
            "priority": self.priority.value,
            "status": self.status.value,
            "domain": self.domain,
            "steps": [s.to_dict() for s in self.steps],
            "parent_goal_id": self.parent_goal_id,
            "sub_goals": list(self.sub_goals),
            "progress": round(self.progress, 3),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deadline": self.deadline,
            "tags": self.tags[:10],
        }


class CognitiveGoalPlanner:
    """
    Goal decomposition and tracking engine for ShadowOS.

    Capabilities:
    - Create and decompose goals into steps
    - Track step progress and update goal status
    - Detect blocked goals and suggest alternative paths
    - Persist goals to disk for crash recovery
    - Integrate with ShadowCrew for delegation
    """

    _MAX_GOALS = 500
    _MAX_STEPS_PER_GOAL = 50
    _MAX_SUB_GOALS = 10

    def __init__(self, data_dir: Optional[str] = None):
        self._goals: Dict[str, Goal] = {}
        self._data_dir = data_dir
        self._lock = threading.Lock()
        if data_dir:
            os.makedirs(data_dir, mode=0o700, exist_ok=True)
            self._load()
        self._stats = {
            "goals_created": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "steps_completed": 0,
        }
        self._stats_lock = threading.Lock()

    def create_goal(
        self,
        title: str,
        description: str = "",
        priority: GoalPriority = GoalPriority.MEDIUM,
        domain: Optional[str] = None,
        deadline: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Goal]:
        """Create a new goal. Returns None if at max capacity after eviction."""
        title = _sanitize_strict(title).strip()[:200]
        description = _sanitize(description).strip()[:500]
        with self._lock:
            if len(self._goals) >= self._MAX_GOALS:
                self._evict_completed()

            if len(self._goals) >= self._MAX_GOALS:
                return None

            goal = Goal(
                title=title,
                description=description,
                priority=priority,
            domain=_sanitize_strict(domain).strip()[:100] if domain else None,
            deadline=_sanitize_strict(deadline).strip()[:50] if deadline else None,
            tags=([_sanitize_strict(t).strip()[:50] for t in tags] if tags else []),
            )
            self._goals[goal.goal_id] = goal
            with self._stats_lock:
                self._stats["goals_created"] += 1
            self._save()
            return goal

    def decompose_goal(self, goal_id: str, steps: List[Dict]) -> Optional[Goal]:
        """
        Decompose a goal into ordered steps.

        Each step dict should have: description, estimated_minutes (optional),
        dependencies (optional list of step indices).
        """
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            if len(steps) > self._MAX_STEPS_PER_GOAL:
                steps = steps[:self._MAX_STEPS_PER_GOAL]

            for i, step_data in enumerate(steps):
                try:
                    est_min = float(step_data.get("estimated_minutes", 5.0))
                except (TypeError, ValueError):
                    est_min = 5.0
                est_min = max(0, min(est_min, 9999))
                step = PlanStep(
                    description=_sanitize(str(step_data.get("description", ""))).strip()[:500],
                    order=i,
                    estimated_minutes=est_min,
                    dependencies=[str(d) for d in step_data.get("dependencies", []) if len(str(d)) < 100],
                )
                goal.steps.append(step)

            goal.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            return goal

    def add_sub_goal(self, parent_id: str, child_id: str) -> bool:
        """Add a sub-goal relationship."""
        with self._lock:
            parent = self._goals.get(parent_id)
            child = self._goals.get(child_id)
            if not parent or not child:
                return False
            if len(parent.sub_goals) >= self._MAX_SUB_GOALS:
                return False
            child.parent_goal_id = parent_id
            parent.sub_goals.append(child_id)
            self._save()
            return True

    def start_step(self, goal_id: str, step_id: str) -> bool:
        """Mark a step as active."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            for step in goal.steps:
                if step.step_id == step_id and step.status == StepStatus.PENDING:
                    pending_deps = [d for d in step.dependencies if d != step_id]
                    active_or_done = {
                        s.step_id for s in goal.steps
                        if s.status in (StepStatus.DONE, StepStatus.ACTIVE)
                    }
                    if all(d in active_or_done for d in pending_deps):
                        step.status = StepStatus.ACTIVE
                        step.started_at = datetime.now(timezone.utc).isoformat()
                        if goal.status == GoalStatus.PENDING:
                            goal.status = GoalStatus.IN_PROGRESS
                        goal.updated_at = datetime.now(timezone.utc).isoformat()
                        self._save()
                        return True
            return False

    def complete_step(self, goal_id: str, step_id: str, result: Optional[Dict] = None) -> bool:
        """Mark a step as done."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            for step in goal.steps:
                if step.step_id == step_id and step.status == StepStatus.ACTIVE:
                    step.status = StepStatus.DONE
                    step.completed_at = datetime.now(timezone.utc).isoformat()
                    step.result = result
                    with self._stats_lock:
                        self._stats["steps_completed"] += 1
                    self._update_progress(goal)
                    goal.updated_at = datetime.now(timezone.utc).isoformat()
                    self._save()
                    return True
            return False

    def fail_step(self, goal_id: str, step_id: str, reason: str = "") -> bool:
        """Mark a step as failed."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            for step in goal.steps:
                if step.step_id == step_id and step.status == StepStatus.ACTIVE:
                    step.status = StepStatus.FAILED
                    step.result = {"error": _sanitize(reason).strip()[:200]}
                    goal.updated_at = datetime.now(timezone.utc).isoformat()
                    self._check_goal_blocked(goal)
                    self._save()
                    return True
            return False

    def get_goal(self, goal_id: str) -> Optional[Dict]:
        """Get a goal by ID."""
        with self._lock:
            goal = self._goals.get(goal_id)
            return goal.to_dict() if goal else None

    def list_goals(
        self,
        status: Optional[GoalStatus] = None,
        domain: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """List goals with optional filtering."""
        with self._lock:
            goals = list(self._goals.values())
            if status:
                goals = [g for g in goals if g.status == status]
            if domain:
                goals = [g for g in goals if g.domain == domain]
            goals.sort(key=lambda g: g.created_at, reverse=True)
            return [g.to_dict() for g in goals[:limit]]

    def get_active_goals(self) -> List[Dict]:
        """Get all goals with status IN_PROGRESS."""
        return self.list_goals(status=GoalStatus.IN_PROGRESS)

    def get_blocked_goals(self) -> List[Dict]:
        """Get goals that are blocked due to failed steps."""
        return self.list_goals(status=GoalStatus.BLOCKED)

    def get_dependency_graph(self) -> Dict[str, Dict]:
        """
        Return a dict representing the goal dependency graph.
        {goal_id: {"goal": goal_dict, "depends_on": [parent_id_for_each_step], "depended_on_by": [child_goal_ids]}}
        """
        with self._lock:
            graph = {}
            for gid, goal in self._goals.items():
                depends_on = []
                if goal.parent_goal_id and goal.parent_goal_id in self._goals:
                    depends_on.append(goal.parent_goal_id)
                for step in goal.steps:
                    for dep in step.dependencies:
                        if dep not in depends_on:
                            depends_on.append(dep)
                depended_on_by = []
                for other_gid, other_goal in self._goals.items():
                    if other_goal.parent_goal_id == gid and other_gid not in depended_on_by:
                        depended_on_by.append(other_gid)
                graph[gid] = {
                    "goal": goal.to_dict(),
                    "depends_on": depends_on,
                    "depended_on_by": depended_on_by,
                }
            return graph

    def get_critical_path(self) -> List[Dict]:
        """
        Return the longest chain of incomplete goals (for prioritization).
        Uses BFS from root goals (no parent) to find the longest path
        of PENDING/IN_PROGRESS/BLOCKED goals.
        """
        with self._lock:
            incomplete = {
                gid for gid, g in self._goals.items()
                if g.status not in (GoalStatus.COMPLETED, GoalStatus.CANCELLED)
            }
            children: Dict[str, List[str]] = {}
            for gid in incomplete:
                goal = self._goals[gid]
                parent = goal.parent_goal_id
                if parent and parent in incomplete:
                    children.setdefault(parent, []).append(gid)

            roots = [gid for gid in incomplete if self._goals[gid].parent_goal_id not in incomplete]

            longest_path = []
            queue = [(gid, [gid]) for gid in roots]
            visited_edges = set()
            while queue:
                current, path = queue.pop(0)
                if len(path) > len(longest_path):
                    longest_path = path
                for child in children.get(current, []):
                    edge = (current, child)
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        queue.append((child, path + [child]))

            return [self._goals[gid].to_dict() for gid in longest_path if gid in self._goals]

    def get_next_step(self, goal_id: str) -> Optional[Dict]:
        """Get the next pending step for a goal."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal or goal.status not in (GoalStatus.PENDING, GoalStatus.IN_PROGRESS):
                return None
            active_or_done = {
                s.step_id for s in goal.steps
                if s.status in (StepStatus.DONE, StepStatus.ACTIVE)
            }
            for step in goal.steps:
                if step.status == StepStatus.PENDING:
                    if all(d in active_or_done for d in step.dependencies):
                        return step.to_dict()
            return None

    def cancel_goal(self, goal_id: str) -> bool:
        """Cancel a goal."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal or goal.status in (GoalStatus.COMPLETED, GoalStatus.CANCELLED):
                return False
            goal.status = GoalStatus.CANCELLED
            goal.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            return True

    def get_stats(self) -> Dict:
        with self._stats_lock:
            goals_created = self._stats["goals_created"]
            goals_completed = self._stats["goals_completed"]
            goals_failed = self._stats["goals_failed"]
            steps_completed = self._stats["steps_completed"]
        with self._lock:
            total_goals = len(self._goals)
            active_goals = sum(1 for g in self._goals.values() if g.status == GoalStatus.IN_PROGRESS)
            blocked_goals = sum(1 for g in self._goals.values() if g.status == GoalStatus.BLOCKED)
        return {
            "total_goals": total_goals,
            "goals_created": goals_created,
            "goals_completed": goals_completed,
            "goals_failed": goals_failed,
            "steps_completed": steps_completed,
            "active_goals": active_goals,
            "blocked_goals": blocked_goals,
        }

    def _update_progress(self, goal: Goal):
        """Recalculate goal progress based on step completion."""
        if not goal.steps:
            goal.progress = 0.0
            return
        done = sum(1 for s in goal.steps if s.status == StepStatus.DONE)
        goal.progress = done / len(goal.steps)
        if goal.progress >= 1.0:
            goal.status = GoalStatus.COMPLETED
            with self._stats_lock:
                self._stats["goals_completed"] += 1
        goal.updated_at = datetime.now(timezone.utc).isoformat()

    def _check_goal_blocked(self, goal: Goal):
        """Check if goal should be marked as blocked due to failed steps."""
        if goal.status not in (GoalStatus.PENDING, GoalStatus.IN_PROGRESS):
            return
        failed_steps = sum(1 for s in goal.steps if s.status == StepStatus.FAILED)
        if failed_steps > 0:
            goal.status = GoalStatus.BLOCKED
            with self._stats_lock:
                self._stats["goals_failed"] += 1

    def _evict_completed(self):
        """Remove completed/cancelled goals to make room, preserving those referenced as sub_goals."""
        active_sub_goals = set()
        for g in self._goals.values():
            if g.status not in (GoalStatus.COMPLETED, GoalStatus.CANCELLED, GoalStatus.FAILED):
                active_sub_goals.update(g.sub_goals)
        
        to_remove = [
            gid for gid, g in self._goals.items()
            if g.status in (GoalStatus.COMPLETED, GoalStatus.CANCELLED, GoalStatus.FAILED)
            and gid not in active_sub_goals
        ]
        for gid in to_remove[:len(self._goals) // 2]:
            del self._goals[gid]

    def _save(self):
        """Persist goals to disk (atomic)."""
        if not self._data_dir:
            return
        path = os.path.join(self._data_dir, "goals.json")
        try:
            data = {gid: g.to_dict() for gid, g in self._goals.items()}
            tmp = path + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                try:
                    f = os.fdopen(fd, 'w')
                except (OSError, ValueError):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    raise
                with f:
                    f.write(json.dumps(data, indent=2, default=str))
                    f.flush()
                    os.fsync(f.fileno())
                if self._data_dir:
                    fsync_dir(self._data_dir)
                os.replace(tmp, path)
            except (OSError, TypeError, ValueError):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save goals: %s", type(e).__name__)

    def _load(self):
        """Load goals from disk."""
        if not self._data_dir:
            return
        path = os.path.join(self._data_dir, "goals.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                raw = f.read()
            if len(raw) > 5 * 1024 * 1024:
                logger.warning("Goals file too large (%s bytes), skipping load", len(raw))
                return
            data = json.loads(raw)
            for gid, gdata in data.items():
                steps = []
                for sdata in gdata.get("steps", []):
                    raw_deps = sdata.get("dependencies", [])
                    if not isinstance(raw_deps, list):
                        raw_deps = []
                    steps.append(PlanStep(
                        step_id=sdata.get("step_id", ""),
                        description=sdata.get("description", ""),
                        order=sdata.get("order", 0),
                        status=StepStatus(sdata.get("status", "pending")),
                        assigned_to=sdata.get("assigned_to"),
                        estimated_minutes=sdata.get("estimated_minutes", 5.0),
                        dependencies=[str(d) for d in raw_deps if len(str(d)) < 100],
                    ))
                goal = Goal(
                    goal_id=gdata.get("goal_id", gid),
                    title=gdata.get("title", ""),
                    description=gdata.get("description", ""),
                    priority=GoalPriority(gdata.get("priority", "medium")),
                    status=GoalStatus(gdata.get("status", "pending")),
                    domain=gdata.get("domain"),
                    steps=steps,
                    parent_goal_id=gdata.get("parent_goal_id"),
                    sub_goals=gdata.get("sub_goals", []),
                    progress=gdata.get("progress", 0.0),
                    created_at=gdata.get("created_at", ""),
                    updated_at=gdata.get("updated_at", ""),
                    deadline=gdata.get("deadline"),
                    tags=gdata.get("tags", []),
                )
                self._goals[goal.goal_id] = goal
            logger.info("Loaded %s goals", len(self._goals))
        except (json.JSONDecodeError, OSError, ValueError, KeyError) as e:
            logger.warning("Failed to load goals: %s", type(e).__name__)