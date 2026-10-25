#!/usr/bin/env python3
"""
GoalReasoning - Strategic Goal Analysis and Adaptive Planning Engine

Provides strategic reasoning about goals beyond simple decomposition:
- Priority scheduling with urgency/deadline/effort scoring
- Obstacle detection and mitigation strategies
- Goal conflict analysis (resource, time, priority conflicts)
- Adaptive replanning when conditions change
- Goal-level strategic assessment and recommendations

Complements CognitiveGoalPlanner (which handles decomposition/tracking)
by adding meta-level reasoning about goal relationships and feasibility.

Architecture:
  - GoalAssessment: strategic evaluation of a goal's current state
  - Obstacle: detected barrier to goal progress
  - GoalConflict: identified conflict between two goals
  - ReplanAction: suggested change when replanning is needed
  - GoalReasoning: main engine managing assessments, obstacles, conflicts
  - Thread-safe, bounded collections, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("shadowd.cognition.goal_reasoning")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _SANITIZE_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


class ConflictType(str, Enum):
    RESOURCE = "resource"
    TIME = "time"
    PRIORITY = "priority"
    DEPENDENCY = "dependency"
    SCOPE = "scope"


class ObstacleSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReplanType(str, Enum):
    RESEQUENCE = "resequence"
    DECOMPOSE_FURTHER = "decompose_further"
    SUBSTITUTE_RESOURCE = "substitute_resource"
    DEFER = "defer"
    ABANDON = "abandon"
    ESCALATE = "escalate"


class AssessmentOutcome(str, Enum):
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


_REASSESS_PATTERNS = [
    re.compile(r'\b(delays?|postpone|defer|wait|hold)\b', re.IGNORECASE),
    re.compile(r'\b(blocked|stuck|can\'t|unable|impossible)\b', re.IGNORECASE),
    re.compile(r'\b(reconsider|rethink|change|pivot|shift)\b', re.IGNORECASE),
    re.compile(r'\b(urgent|critical|emergency|asap)\b', re.IGNORECASE),
    re.compile(r'\b(complete|done|finish|achieve|accomplish)\b', re.IGNORECASE),
]


@dataclass
class Obstacle:
    obstacle_id: str
    goal_id: str
    description: str
    severity: ObstacleSeverity
    detected_at: str
    mitigation: str
    resolved: bool = False
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "obstacle_id": self.obstacle_id,
            "goal_id": self.goal_id,
            "description": self.description,
            "severity": self.severity.value,
            "detected_at": self.detected_at,
            "mitigation": self.mitigation,
            "resolved": self.resolved,
            "metadata": copy.deepcopy(dict(list(self.metadata.items())[:20])),
        }


@dataclass
class GoalConflict:
    conflict_id: str
    goal_id_a: str
    goal_id_b: str
    conflict_type: ConflictType
    description: str
    resolution_hint: str
    detected_at: str

    def to_dict(self) -> Dict:
        return {
            "conflict_id": self.conflict_id,
            "goal_id_a": self.goal_id_a,
            "goal_id_b": self.goal_id_b,
            "conflict_type": self.conflict_type.value,
            "description": self.description,
            "resolution_hint": self.resolution_hint,
            "detected_at": self.detected_at,
        }


@dataclass
class ReplanAction:
    action_id: str
    goal_id: str
    replan_type: ReplanType
    reason: str
    suggestion: str
    created_at: str
    applied: bool = False

    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "goal_id": self.goal_id,
            "replan_type": self.replan_type.value,
            "reason": self.reason,
            "suggestion": self.suggestion,
            "created_at": self.created_at,
            "applied": self.applied,
        }


@dataclass
class GoalAssessment:
    assessment_id: str
    goal_id: str
    outcome: AssessmentOutcome
    confidence: float
    urgency_score: float
    effort_estimate: float
    risk_factors: List[str]
    recommendations: List[str]
    assessed_at: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "assessment_id": self.assessment_id,
            "goal_id": self.goal_id,
            "outcome": self.outcome.value,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
            "urgency_score": round(max(0.0, min(1.0, self.urgency_score)), 3),
            "effort_estimate": round(max(0.0, self.effort_estimate), 1),
            "risk_factors": list(self.risk_factors[:10]),
            "recommendations": list(self.recommendations[:5]),
            "assessed_at": self.assessed_at,
            "metadata": copy.deepcopy(dict(list(self.metadata.items())[:20])),
        }


class GoalReasoning:
    """
    Strategic goal analysis and adaptive planning engine.

    Provides meta-level reasoning about goals: priority scheduling,
    obstacle detection, conflict analysis, and replanning suggestions.
    """

    _MAX_ASSESSMENTS = 2000
    _MAX_OBSTACLES = 1000
    _MAX_CONFLICTS = 500
    _MAX_REPLAN_ACTIONS = 500
    _MAX_METADATA_KEYS = 20
    _MAX_GOAL_PRIORITIES = 2000

    def __init__(self):
        self._lock = threading.Lock()
        self._assessments: Dict[str, GoalAssessment] = {}
        self._obstacles: Dict[str, Obstacle] = {}
        self._conflicts: Dict[str, GoalConflict] = {}
        self._replan_actions: Dict[str, ReplanAction] = {}
        self._goal_priorities: Dict[str, float] = {}
        self._recent_assessments = deque(maxlen=100)

        self._stats = {
            "assessments_made": 0,
            "obstacles_detected": 0,
            "conflicts_found": 0,
            "replans_suggested": 0,
            "replans_applied": 0,
        }
        self._stats_lock = threading.Lock()

    def assess_goal(
        self,
        goal_id: str,
        progress: float,
        deadline: Optional[str] = None,
        effort_spent: float = 0.0,
        effort_total: float = 1.0,
        blockers: Optional[List[str]] = None,
        context: Optional[Dict] = None,
    ) -> GoalAssessment:
        """
        Assess a goal's current strategic state.

        Args:
            goal_id: Goal identifier
            progress: Progress percentage (0.0-1.0)
            deadline: ISO format deadline timestamp
            effort_spent: Effort units spent so far
            effort_total: Total estimated effort units
            blockers: List of blocking factor descriptions
            context: Additional context metadata
        """
        goal_id = _sanitize(goal_id, 64)
        progress = max(0.0, min(1.0, progress))
        effort_spent = max(0.0, effort_spent)
        effort_total = max(0.01, effort_total)
        blockers = [_sanitize_content(b, 200) for b in (blockers or [])][:10]
        context = dict(list((context or {}).items())[:self._MAX_METADATA_KEYS])
        context = {k: (_sanitize_content(v, 200) if isinstance(v, str) else v) for k, v in context.items()}

        urgency_score = self._calculate_urgency(progress, deadline)
        confidence = self._calculate_confidence(progress, effort_spent, effort_total, blockers)

        if blockers:
            outcome = AssessmentOutcome.BLOCKED
        elif progress >= 1.0:
            outcome = AssessmentOutcome.COMPLETED
        elif confidence < 0.4:
            outcome = AssessmentOutcome.AT_RISK
        else:
            outcome = AssessmentOutcome.ON_TRACK

        risk_factors = self._identify_risk_factors(progress, deadline, effort_spent, effort_total, blockers)
        recommendations = self._generate_recommendations(outcome, urgency_score, risk_factors, blockers)

        assessment = GoalAssessment(
            assessment_id=uuid.uuid4().hex[:12],
            goal_id=goal_id,
            outcome=outcome,
            confidence=confidence,
            urgency_score=urgency_score,
            effort_estimate=effort_total,
            risk_factors=risk_factors,
            recommendations=recommendations,
            assessed_at=datetime.now(timezone.utc).isoformat(),
            metadata=context,
        )

        with self._lock:
            if len(self._assessments) >= self._MAX_ASSESSMENTS:
                self._evict_assessments()
            if len(self._goal_priorities) >= self._MAX_GOAL_PRIORITIES:
                sorted_goals = sorted(self._goal_priorities.items(), key=lambda x: x[1])
                oldest_goals = [g[0] for g in sorted_goals[:len(sorted_goals) // 4]]
                for gid in oldest_goals:
                    del self._goal_priorities[gid]
                    logger.warning("Evicted goal priority for %s", gid)
            self._assessments[assessment.assessment_id] = assessment
            self._goal_priorities[goal_id] = urgency_score
            self._recent_assessments.append(assessment.assessment_id)
            with self._stats_lock:
                self._stats["assessments_made"] += 1

        return copy.deepcopy(assessment)

    def detect_obstacle(
        self,
        goal_id: str,
        description: str,
        severity: ObstacleSeverity = ObstacleSeverity.MEDIUM,
        mitigation: str = "",
        metadata: Optional[Dict] = None,
    ) -> Obstacle:
        """Detect and record an obstacle blocking goal progress."""
        goal_id = _sanitize(goal_id, 64)
        description = _sanitize_content(description, 500)
        mitigation = _sanitize_content(mitigation, 500)
        metadata = dict(list((metadata or {}).items())[:self._MAX_METADATA_KEYS])
        metadata = {k: (_sanitize_content(v, 200) if isinstance(v, str) else v) for k, v in metadata.items()}

        # Accept a plain string severity (callers often pass values from JSON)
        # while keeping the enum as the internal representation.
        if not isinstance(severity, ObstacleSeverity):
            try:
                severity = ObstacleSeverity(str(severity).lower())
            except (ValueError, TypeError):
                severity = ObstacleSeverity.MEDIUM

        obstacle = Obstacle(
            obstacle_id=uuid.uuid4().hex[:12],
            goal_id=goal_id,
            description=description,
            severity=severity,
            detected_at=datetime.now(timezone.utc).isoformat(),
            mitigation=mitigation,
            metadata=metadata,
        )

        with self._lock:
            if len(self._obstacles) >= self._MAX_OBSTACLES:
                self._evict_obstacles()
            self._obstacles[obstacle.obstacle_id] = obstacle
            with self._stats_lock:
                self._stats["obstacles_detected"] += 1

        return copy.deepcopy(obstacle)

    def resolve_obstacle(self, obstacle_id: str) -> bool:
        """Mark an obstacle as resolved."""
        with self._lock:
            obstacle = self._obstacles.get(obstacle_id)
            if not obstacle:
                return False
            obstacle.resolved = True
            return True

    def analyze_conflicts(
        self,
        goals: List[Dict],
    ) -> List[GoalConflict]:
        """
        Analyze a list of goals for potential conflicts.

        Each goal dict should have: goal_id, priority, deadline,
        domain, resource_needs, progress.
        """
        conflicts = []
        goals = goals[:100]

        for i in range(len(goals)):
            for j in range(i + 1, len(goals)):
                ga = goals[i]
                gb = goals[j]
                conflict = self._check_conflict(ga, gb)
                if conflict:
                    with self._lock:
                        if len(self._conflicts) >= self._MAX_CONFLICTS:
                            self._evict_conflicts()
                        self._conflicts[conflict.conflict_id] = conflict
                        with self._stats_lock:
                            self._stats["conflicts_found"] += 1
                    conflicts.append(conflict)

        return conflicts

    def suggest_replan(
        self,
        goal_id: str,
        reason: str,
        current_state: Optional[Dict] = None,
    ) -> ReplanAction:
        """Suggest a replanning action for a goal."""
        goal_id = _sanitize(goal_id, 64)
        reason = _sanitize_content(reason, 500)
        current_state = dict(list((current_state or {}).items())[:self._MAX_METADATA_KEYS])
        current_state = {k: (_sanitize_content(v, 200) if isinstance(v, str) else v) for k, v in current_state.items()}

        replan_type = self._classify_replan_reason(reason)
        suggestion = self._generate_replan_suggestion(replan_type, reason, current_state)

        action = ReplanAction(
            action_id=uuid.uuid4().hex[:12],
            goal_id=goal_id,
            replan_type=replan_type,
            reason=reason,
            suggestion=suggestion,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            if len(self._replan_actions) >= self._MAX_REPLAN_ACTIONS:
                self._evict_replan_actions()
            self._replan_actions[action.action_id] = action
            with self._stats_lock:
                self._stats["replans_suggested"] += 1

        return copy.deepcopy(action)

    def apply_replan(self, action_id: str) -> bool:
        """Mark a replan action as applied."""
        with self._lock:
            action = self._replan_actions.get(action_id)
            if not action or action.applied:
                return False
            action.applied = True
            with self._stats_lock:
                self._stats["replans_applied"] += 1
            return True

    def prioritize_goals(self, goals: List[Dict]) -> List[Dict]:
        """
        Sort goals by priority score considering urgency, importance,
        progress, and resource requirements.

        Each goal dict should have: goal_id, priority, progress, deadline.
        """
        goals = goals[:200]
        with self._lock:
            goal_priorities_snapshot = dict(self._goal_priorities)
        scored = []
        for g in goals:
            goal_id = g.get("goal_id", "")
            priority_val = self._priority_score(g.get("priority", "medium"))
            progress = max(0.0, min(1.0, float(g.get("progress", 0.0))))
            deadline = g.get("deadline")

            urgency = self._calculate_urgency(progress, deadline)
            stored_priority = goal_priorities_snapshot.get(goal_id, 0.5)

            composite_score = (
                priority_val * 0.3
                + urgency * 0.3
                + stored_priority * 0.2
                + (1.0 - progress) * 0.2
            )

            g_copy = copy.deepcopy(g)
            g_copy["_priority_score"] = round(composite_score, 3)
            scored.append(g_copy)

        scored.sort(key=lambda x: x["_priority_score"], reverse=True)
        return scored

    def get_assessment(self, assessment_id: str) -> Optional[Dict]:
        with self._lock:
            a = self._assessments.get(assessment_id)
            return copy.deepcopy(a.to_dict()) if a else None

    def list_assessments(self, goal_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        limit = max(1, min(limit, 200))
        with self._lock:
            assessments = list(self._assessments.values())
        if goal_id:
            assessments = [a for a in assessments if a.goal_id == goal_id]
        assessments.sort(key=lambda a: a.assessed_at, reverse=True)
        return [copy.deepcopy(a.to_dict()) for a in assessments[:limit]]

    def get_obstacle(self, obstacle_id: str) -> Optional[Dict]:
        with self._lock:
            o = self._obstacles.get(obstacle_id)
            return copy.deepcopy(o.to_dict()) if o else None

    def list_obstacles(self, goal_id: Optional[str] = None, resolved: Optional[bool] = None, limit: int = 50) -> List[Dict]:
        limit = max(1, min(limit, 200))
        with self._lock:
            obstacles = list(self._obstacles.values())
        if goal_id:
            obstacles = [o for o in obstacles if o.goal_id == goal_id]
        if resolved is not None:
            obstacles = [o for o in obstacles if o.resolved == resolved]
        obstacles.sort(key=lambda o: o.detected_at, reverse=True)
        return [copy.deepcopy(o.to_dict()) for o in obstacles[:limit]]

    def get_conflict(self, conflict_id: str) -> Optional[Dict]:
        with self._lock:
            c = self._conflicts.get(conflict_id)
            return copy.deepcopy(c.to_dict()) if c else None

    def list_conflicts(self, goal_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        limit = max(1, min(limit, 200))
        with self._lock:
            conflicts = list(self._conflicts.values())
        if goal_id:
            conflicts = [
                c for c in conflicts
                if c.goal_id_a == goal_id or c.goal_id_b == goal_id
            ]
        conflicts.sort(key=lambda c: c.detected_at, reverse=True)
        return [copy.deepcopy(c.to_dict()) for c in conflicts[:limit]]

    def get_replan_action(self, action_id: str) -> Optional[Dict]:
        with self._lock:
            a = self._replan_actions.get(action_id)
            return copy.deepcopy(a.to_dict()) if a else None

    def list_replan_actions(self, goal_id: Optional[str] = None, applied: Optional[bool] = None, limit: int = 50) -> List[Dict]:
        limit = max(1, min(limit, 200))
        with self._lock:
            actions = list(self._replan_actions.values())
        if goal_id:
            actions = [a for a in actions if a.goal_id == goal_id]
        if applied is not None:
            actions = [a for a in actions if a.applied == applied]
        actions.sort(key=lambda a: a.created_at, reverse=True)
        return [copy.deepcopy(a.to_dict()) for a in actions[:limit]]

    def get_stats(self) -> Dict:
        with self._lock:
            total_assessments = len(self._assessments)
            total_obstacles = len(self._obstacles)
            active_obstacles = sum(1 for o in self._obstacles.values() if not o.resolved)
            total_conflicts = len(self._conflicts)
            total_replan_actions = len(self._replan_actions)
            applied_replans = sum(1 for a in self._replan_actions.values() if a.applied)
            goal_priorities_tracked = len(self._goal_priorities)
        with self._stats_lock:
            stats_snapshot = dict(self._stats)
        return {
            "total_assessments": total_assessments,
            "total_obstacles": total_obstacles,
            "active_obstacles": active_obstacles,
            "total_conflicts": total_conflicts,
            "total_replan_actions": total_replan_actions,
            "applied_replans": applied_replans,
            **stats_snapshot,
            "goal_priorities_tracked": goal_priorities_tracked,
        }

    def _calculate_urgency(self, progress: float, deadline: Optional[str]) -> float:
        if not deadline:
            return 0.5
        try:
            dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if now >= dl:
                return 1.0
            time_remaining = (dl - now).total_seconds()
            total_time = max(time_remaining, 1.0)
            work_remaining = max(1.0 - progress, 0.01)
            urgency = min(1.0, work_remaining / (total_time / 86400.0 + 1.0))
            return round(urgency, 3)
        except (ValueError, TypeError):
            return 0.5

    def _calculate_confidence(
        self,
        progress: float,
        effort_spent: float,
        effort_total: float,
        blockers: List[str],
    ) -> float:
        if effort_total <= 0:
            effort_ratio = 1.0
        else:
            effort_ratio = effort_spent / effort_total

        if effort_ratio > 1.5 and progress < 0.5:
            effort_confidence = 0.15
        elif effort_ratio > 1.2 and progress < 0.7:
            effort_confidence = 0.35
        else:
            effort_confidence = min(progress + 0.2, 1.0)

        if blockers:
            severity_map = {
                1: 0.7,
                2: 0.4,
                3: 0.2,
            }
            num_blockers = min(len(blockers), 3)
            blocker_factor = severity_map.get(num_blockers, 0.2)
        else:
            blocker_factor = 1.0

        return round(max(0.0, min(1.0, effort_confidence * blocker_factor)), 3)

    def _identify_risk_factors(
        self,
        progress: float,
        deadline: Optional[str],
        effort_spent: float,
        effort_total: float,
        blockers: List[str],
    ) -> List[str]:
        risks = []

        if blockers:
            risks.append(f"{len(blockers)} blocker(s) active")

        if deadline:
            try:
                dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                remaining = (dl - now).total_seconds() / 86400.0
                if remaining < 0:
                    risks.append("deadline passed")
                elif remaining < 1 and progress < 0.8:
                    risks.append("approaching deadline with low progress")
                elif remaining < 7 and progress < 0.5:
                    risks.append("limited time remaining")
            except (ValueError, TypeError):
                pass

        if effort_total > 0:
            ratio = effort_spent / effort_total
            if ratio > 1.5 and progress < 0.5:
                risks.append("effort overrun exceeds 150% with low progress")
            elif ratio > 1.0 and progress < 0.7:
                risks.append("effort overrun without proportional progress")

        if progress < 0.1 and effort_spent > 0:
            risks.append("minimal progress despite effort")

        return risks[:10]

    def _generate_recommendations(
        self,
        outcome: AssessmentOutcome,
        urgency: float,
        risk_factors: List[str],
        blockers: List[str],
    ) -> List[str]:
        recs = []

        if outcome == AssessmentOutcome.BLOCKED:
            recs.append("Resolve blockers before continuing")
            for b in blockers[:2]:
                recs.append(f"Address: {b[:80]}")
            recs.append("Consider replanning if blockers persist")

        elif outcome == AssessmentOutcome.AT_RISK:
            recs.append("Review approach and adjust strategy")
            if urgency > 0.7:
                recs.append("Prioritize this goal due to urgency")
            recs.append("Consider breaking into smaller steps")

        elif outcome == AssessmentOutcome.ON_TRACK:
            if urgency > 0.8:
                recs.append("Monitor closely for emerging risks")
            else:
                recs.append("Continue current approach")

        if len(risk_factors) > 2:
            recs.append("High risk profile — increase review frequency")

        return recs[:5]

    def _check_conflict(self, ga: Dict, gb: Dict) -> Optional[GoalConflict]:
        conflicts_found = []

        ra = ga.get("resource_needs", {})
        rb = gb.get("resource_needs", {})
        if isinstance(ra, dict) and isinstance(rb, dict):
            shared = (set(ra.keys()) & set(rb.keys()))
            for r in shared:
                total = float(ra.get(r, 0)) + float(rb.get(r, 0))
                if total > 1.0:
                    conflicts_found.append(ConflictType.RESOURCE)

        da = ga.get("deadline")
        db = gb.get("deadline")
        if da and db:
            try:
                dta = datetime.fromisoformat(str(da).replace("Z", "+00:00"))
                dtb = datetime.fromisoformat(str(db).replace("Z", "+00:00"))
                overlap = abs((dta - dtb).total_seconds()) < 86400 * 3
                if overlap and ga.get("priority") in ("high", "critical") and gb.get("priority") in ("high", "critical"):
                    conflicts_found.append(ConflictType.TIME)
            except (ValueError, TypeError):
                pass

        pa = str(ga.get("priority", "medium")).lower()
        pb = str(gb.get("priority", "medium")).lower()
        if pa == "critical" and pb == "critical":
            pra = ga.get("progress", 0.0)
            prb = gb.get("progress", 0.0)
            if float(pra) > 0.3 and float(prb) > 0.3:
                conflicts_found.append(ConflictType.PRIORITY)

        if not conflicts_found:
            return None

        primary = conflicts_found[0]
        descriptions = {
            ConflictType.RESOURCE: "Resource contention between goals",
            ConflictType.TIME: "Deadline overlap for high-priority goals",
            ConflictType.PRIORITY: "Multiple critical-priority goals in progress",
            ConflictType.DEPENDENCY: "Circular or conflicting dependencies",
            ConflictType.SCOPE: "Overlapping scope between goals",
        }

        resolution_hints = {
            ConflictType.RESOURCE: "Stagger execution or allocate more resources",
            ConflictType.TIME: "Prioritize one goal and defer the other",
            ConflictType.PRIORITY: "Focus on the goal with higher urgency first",
            ConflictType.DEPENDENCY: "Break the dependency cycle",
            ConflictType.SCOPE: "Clarify goal boundaries and merge overlapping areas",
        }

        return GoalConflict(
            conflict_id=uuid.uuid4().hex[:12],
            goal_id_a=_sanitize(str(ga.get("goal_id", "")), 64),
            goal_id_b=_sanitize(str(gb.get("goal_id", "")), 64),
            conflict_type=primary,
            description=descriptions.get(primary, "Goal conflict detected"),
            resolution_hint=resolution_hints.get(primary, "Review and adjust priorities"),
            detected_at=datetime.now(timezone.utc).isoformat(),
        )

    def _classify_replan_reason(self, reason: str) -> ReplanType:
        reason_lower = reason.lower()
        delay_matches = sum(1 for p in _REASSESS_PATTERNS[:1] if p.search(reason_lower))
        block_matches = sum(1 for p in _REASSESS_PATTERNS[1:2] if p.search(reason_lower))
        change_matches = sum(1 for p in _REASSESS_PATTERNS[2:3] if p.search(reason_lower))
        urgent_matches = sum(1 for p in _REASSESS_PATTERNS[3:4] if p.search(reason_lower))

        if block_matches:
            return ReplanType.ESCALATE
        if delay_matches:
            return ReplanType.DEFER
        if urgent_matches:
            return ReplanType.RESEQUENCE
        if change_matches:
            return ReplanType.DECOMPOSE_FURTHER
        return ReplanType.RESEQUENCE

    def _generate_replan_suggestion(
        self,
        replan_type: ReplanType,
        reason: str,
        current_state: Optional[Dict],
    ) -> str:
        suggestions = {
            ReplanType.RESEQUENCE: "Reorder goal steps to maximize progress with current resources",
            ReplanType.DECOMPOSE_FURTHER: "Break the goal into smaller sub-goals for more manageable progress",
            ReplanType.SUBSTITUTE_RESOURCE: "Identify alternative resources to unblock progress",
            ReplanType.DEFER: "Postpone this goal until higher-priority goals are complete",
            ReplanType.ABANDON: "This goal may no longer be viable — consider cancelling",
            ReplanType.ESCALATE: "Escalate to stakeholders or decision-makers for guidance",
        }
        return suggestions.get(replan_type, "Review and adjust the plan")

    def _priority_score(self, priority: str) -> float:
        mapping = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.3,
        }
        return mapping.get(str(priority).lower(), 0.5)

    def _evict_assessments(self):
        completed = [
            aid for aid, a in self._assessments.items()
            if a.outcome == AssessmentOutcome.COMPLETED
        ]
        for aid in completed[:len(self._assessments) // 3]:
            del self._assessments[aid]
            logger.warning("Evicted completed assessment %s", aid)
        if len(self._assessments) >= self._MAX_ASSESSMENTS:
            oldest_ids = sorted(
                self._assessments.keys(),
                key=lambda k: self._assessments[k].assessed_at,
            )[:len(self._assessments) // 4]
            for aid in oldest_ids:
                del self._assessments[aid]
                logger.warning("Evicted old assessment %s", aid)

    def _evict_obstacles(self):
        resolved = [
            oid for oid, o in self._obstacles.items()
            if o.resolved
        ]
        for oid in resolved[:len(self._obstacles) // 3]:
            del self._obstacles[oid]
            logger.warning("Evicted resolved obstacle %s", oid)
        if len(self._obstacles) >= self._MAX_OBSTACLES:
            oldest = sorted(
                self._obstacles.keys(),
                key=lambda k: self._obstacles[k].detected_at,
            )[:len(self._obstacles) // 4]
            for oid in oldest:
                del self._obstacles[oid]
                logger.warning("Evicted old obstacle %s", oid)

    def _evict_conflicts(self):
        oldest = sorted(
            self._conflicts.keys(),
            key=lambda k: self._conflicts[k].detected_at,
        )[:len(self._conflicts) // 4]
        for cid in oldest:
            del self._conflicts[cid]
            logger.warning("Evicted old conflict %s", cid)

    def _evict_replan_actions(self):
        applied = [
            aid for aid, a in self._replan_actions.items()
            if a.applied
        ]
        for aid in applied[:len(self._replan_actions) // 3]:
            del self._replan_actions[aid]
            logger.warning("Evicted applied replan action %s", aid)
        if len(self._replan_actions) >= self._MAX_REPLAN_ACTIONS:
            oldest = sorted(
                self._replan_actions.keys(),
                key=lambda k: self._replan_actions[k].created_at,
            )[:len(self._replan_actions) // 4]
            for aid in oldest:
                del self._replan_actions[aid]
                logger.warning("Evicted old replan action %s", aid)