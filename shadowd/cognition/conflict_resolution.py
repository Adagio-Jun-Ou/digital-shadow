#!/usr/bin/env python3
"""
ConflictResolution - Mediation and Negotiation Engine

Analyzes interpersonal conflicts, suggests resolution strategies, models
perspectives of different parties, and tracks resolution outcomes.

Architecture:
  - Conflict: describes a dispute with involved parties and positions
  - Perspective: a party's viewpoint and needs in a conflict
  - ResolutionStrategy: suggested approach with expected outcomes
  - ConflictResolution: main engine managing conflicts and mediation
  - Thread-safe, bounded history, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("shadowd.cognition.conflict_resolution")


class ConflictType(str, Enum):
    INTEREST = "interest"
    VALUE = "value"
    RELATIONSHIP = "relationship"
    DATA = "data"
    PROCESS = "process"
    RESOURCE = "resource"


class ConflictIntensity(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class StrategyType(str, Enum):
    COLLABORATE = "collaborate"
    COMPROMISE = "compromise"
    ACCOMMODATE = "accommodate"
    AVOID = "avoid"
    COMPETE = "compete"


class ConflictStatus(str, Enum):
    ACTIVE = "active"
    MEDIATING = "mediating"
    RESOLVED = "resolved"
    ESCALATED = "escalated"



@dataclass
class Perspective:
    party: str
    position: str
    interests: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    flexibility: float = 0.5

    def to_dict(self) -> Dict:
        return {
            "party": self.party[:100],
            "position": self.position[:300],
            "interests": list(self.interests[:5]),
            "concerns": list(self.concerns[:5]),
            "flexibility": round(max(0.0, min(1.0, self.flexibility)), 3),
        }


@dataclass
class ResolutionStrategy:
    strategy_id: str
    conflict_id: str
    strategy_type: StrategyType
    description: str
    rationale: str
    expected_outcome: str
    difficulty: float = 0.5

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "conflict_id": self.conflict_id,
            "strategy_type": self.strategy_type.value,
            "description": self.description[:300],
            "rationale": self.rationale[:300],
            "expected_outcome": self.expected_outcome[:300],
            "difficulty": round(max(0.0, min(1.0, self.difficulty)), 3),
        }


@dataclass
class Conflict:
    conflict_id: str
    title: str
    conflict_type: ConflictType
    intensity: ConflictIntensity
    status: ConflictStatus = ConflictStatus.ACTIVE
    perspectives: List[Perspective] = field(default_factory=list)
    strategies: List[ResolutionStrategy] = field(default_factory=list)
    description: str = ""
    resolution: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "conflict_id": self.conflict_id,
            "title": self.title[:200],
            "conflict_type": self.conflict_type.value,
            "intensity": self.intensity.value,
            "status": self.status.value,
            "perspectives": [p.to_dict() for p in self.perspectives],
            "strategies": [s.to_dict() for s in self.strategies],
            "strategy_count": len(self.strategies),
            "description": self.description[:300],
            "resolution": self.resolution,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


_MAX_CONFLICTS = 500
_MAX_PERSPECTIVES = 10
_MAX_STRATEGIES = 20
_MAX_TITLE_LENGTH = 200
_MAX_TEXT_LENGTH = 2000


class ConflictResolution:
    _MAX_CONFLICTS = _MAX_CONFLICTS

    def __init__(self):
        self._lock = threading.Lock()
        self._conflicts: Dict[str, Conflict] = {}
        self._resolution_history: deque = deque(maxlen=1000)

    def _sanitize(self, text: str, max_len: int = _MAX_TEXT_LENGTH) -> str:
        text = re.sub(r"[<>\x00-\x1f\x7f]", "", text)
        return text.strip()[:max_len]

    @staticmethod
    def _sanitize_content(text: str, max_len: int = _MAX_TEXT_LENGTH) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()[:max_len]

    def create_conflict(
        self,
        title: str,
        conflict_type: str,
        intensity: int,
        description: str = "",
    ) -> Conflict:
        title = self._sanitize(title, _MAX_TITLE_LENGTH)
        description = self._sanitize_content(description)

        try:
            ct = ConflictType(conflict_type)
        except ValueError:
            raise ValueError(f"Invalid conflict_type: {conflict_type[:50]!r}")

        try:
            ci = ConflictIntensity(intensity)
        except ValueError:
            ci = ConflictIntensity.MEDIUM

        conflict = Conflict(
            conflict_id=uuid.uuid4().hex[:12],
            title=title,
            conflict_type=ct,
            intensity=ci,
            description=description,
        )

        with self._lock:
            if len(self._conflicts) >= self._MAX_CONFLICTS:
                oldest_id = next(iter(self._conflicts))
                logger.warning("Evicting oldest conflict %s to make room", oldest_id)
                del self._conflicts[oldest_id]
            self._conflicts[conflict.conflict_id] = conflict

        return conflict

    def add_perspective(
        self,
        conflict_id: str,
        party: str,
        position: str,
        interests: Optional[List[str]] = None,
        concerns: Optional[List[str]] = None,
        flexibility: float = 0.5,
    ) -> bool:
        party = self._sanitize(party, 100)
        position = self._sanitize_content(position, 300)
        interests = [self._sanitize_content(i, 100) for i in (interests or [])[:5]]
        concerns = [self._sanitize_content(c, 100) for c in (concerns or [])[:5]]
        flexibility = max(0.0, min(1.0, flexibility))

        perspective = Perspective(
            party=party,
            position=position,
            interests=interests,
            concerns=concerns,
            flexibility=flexibility,
        )

        with self._lock:
            conflict = self._conflicts.get(conflict_id)
            if conflict is None:
                return False
            if len(conflict.perspectives) >= _MAX_PERSPECTIVES:
                dropped = conflict.perspectives[0]
                logger.warning("Dropping oldest perspective from conflict %s: party=%s", conflict_id, dropped.party)
                conflict.perspectives = conflict.perspectives[1:]
            conflict.perspectives.append(perspective)
            if conflict.status == ConflictStatus.ACTIVE and len(conflict.perspectives) == 1:
                conflict.status = ConflictStatus.MEDIATING
            return True

    def suggest_strategies(self, conflict_id: str) -> List[ResolutionStrategy]:
        with self._lock:
            conflict = self._conflicts.get(conflict_id)
            if conflict is None:
                return []

            strategies = []
            perspectives = conflict.perspectives

            avg_flexibility = 0.5
            if perspectives:
                avg_flexibility = sum(p.flexibility for p in perspectives) / len(perspectives)

            if avg_flexibility > 0.7:
                strategies.append(self._make_strategy(
                    conflict, StrategyType.COLLABORATE,
                    "Find a solution that fully satisfies both parties' core needs",
                    "High flexibility on both sides enables creative win-win solutions",
                    "Mutual satisfaction with strengthened relationship",
                    0.3,
                ))
                strategies.append(self._make_strategy(
                    conflict, StrategyType.COMPROMISE,
                    "Both parties give up something to reach a middle ground",
                    "When collaboration is too time-consuming, compromise preserves progress",
                    "Partial satisfaction; relationship preserved",
                    0.4,
                ))
            elif avg_flexibility > 0.4:
                strategies.append(self._make_strategy(
                    conflict, StrategyType.COMPROMISE,
                    "Negotiate a middle ground respecting both positions",
                    "Moderate flexibility allows balanced trade-offs",
                    "Reasonable satisfaction for both parties",
                    0.5,
                ))
                strategies.append(self._make_strategy(
                    conflict, StrategyType.COLLABORATE,
                    "Explore deeper interests to find creative solutions",
                    "Underlying needs may align despite surface positions",
                    "Potential for high satisfaction if interests align",
                    0.6,
                ))
            else:
                strategies.append(self._make_strategy(
                    conflict, StrategyType.ACCOMMODATE,
                    "One party yields to preserve the relationship",
                    "Low flexibility means one party should prioritize the relationship",
                    "Relationship preserved; one party sacrifices",
                    0.7,
                ))
                strategies.append(self._make_strategy(
                    conflict, StrategyType.COMPETE,
                    "Assert your position firmly — stakes are too high to yield",
                    "Critical needs require firm advocacy",
                    "One party wins; relationship may strain",
                    0.8,
                ))

            if conflict.intensity.value >= 3:
                strategies.append(self._make_strategy(
                    conflict, StrategyType.AVOID,
                    "De-escalate by taking a cooling-off period before re-engaging",
                    "High intensity conflicts benefit from emotional regulation first",
                    "Reduced tension creating space for productive dialogue later",
                    0.3,
                ))

            conflict.strategies = strategies[:_MAX_STRATEGIES]
            return copy.deepcopy(strategies[:_MAX_STRATEGIES])

    def _make_strategy(
        self,
        conflict: Conflict,
        strategy_type: StrategyType,
        description: str,
        rationale: str,
        expected_outcome: str,
        difficulty: float,
    ) -> ResolutionStrategy:
        return ResolutionStrategy(
            strategy_id=uuid.uuid4().hex[:12],
            conflict_id=conflict.conflict_id,
            strategy_type=strategy_type,
            description=description,
            rationale=rationale,
            expected_outcome=expected_outcome,
            difficulty=difficulty,
        )

    def resolve_conflict(self, conflict_id: str, resolution: str) -> bool:
        resolution = self._sanitize_content(resolution, 500)
        with self._lock:
            conflict = self._conflicts.get(conflict_id)
            if conflict is None:
                return False
            if conflict.status == ConflictStatus.RESOLVED:
                logger.warning("Conflict %s already resolved, ignoring re-resolve attempt", conflict_id)
                return False
            conflict.resolution = resolution
            conflict.status = ConflictStatus.RESOLVED
            conflict.resolved_at = datetime.now(timezone.utc).isoformat()
            self._resolution_history.append({
                "conflict_id": conflict_id,
                "resolution": resolution,
                "resolved_at": conflict.resolved_at,
            })
            return True

    def escalate_conflict(self, conflict_id: str) -> bool:
        with self._lock:
            conflict = self._conflicts.get(conflict_id)
            if conflict is None:
                return False
            if conflict.status == ConflictStatus.RESOLVED:
                logger.warning("Cannot escalate resolved conflict %s", conflict_id)
                return False
            conflict.status = ConflictStatus.ESCALATED
            if conflict.intensity.value < 4:
                conflict.intensity = ConflictIntensity(conflict.intensity.value + 1)
            return True

    def get_conflict(self, conflict_id: str) -> Optional[Conflict]:
        with self._lock:
            c = self._conflicts.get(conflict_id)
            return copy.deepcopy(c) if c else None

    def list_conflicts(self, status: Optional[str] = None) -> List[Conflict]:
        with self._lock:
            conflicts = [copy.deepcopy(c) for c in self._conflicts.values()]
            if status:
                conflicts = [c for c in conflicts if c.status.value == status]
        return sorted(conflicts, key=lambda c: c.intensity.value, reverse=True)

    def analyze_common_ground(self, conflict_id: str) -> Dict:
        with self._lock:
            conflict = self._conflicts.get(conflict_id)
            if conflict is None:
                return {"common_interests": [], "common_concerns": [], "overlap_score": 0.0, "perspective_count": 0, "avg_flexibility": 0.0}
            if len(conflict.perspectives) < 2:
                return {"common_interests": [], "common_concerns": [], "overlap_score": 0.0, "perspective_count": len(conflict.perspectives), "avg_flexibility": 0.0}

            all_interests = [set(p.interests) for p in conflict.perspectives]
            all_concerns = [set(p.concerns) for p in conflict.perspectives]

            common_interests = set.intersection(*all_interests) if all_interests else set()
            common_concerns = set.intersection(*all_concerns) if all_concerns else set()

            total_items = sum(len(i) for i in all_interests) + sum(len(c) for c in all_concerns)
            overlap = len(common_interests) + len(common_concerns)
            overlap_score = overlap / max(total_items, 1)

            return {
                "common_interests": list(common_interests)[:10],
                "common_concerns": list(common_concerns)[:10],
                "overlap_score": round(min(1.0, overlap_score), 3),
                "perspective_count": len(conflict.perspectives),
                "avg_flexibility": round(
                    sum(p.flexibility for p in conflict.perspectives) / len(conflict.perspectives), 3
                ),
            }

    def get_resolution_history(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return [copy.deepcopy(d) for d in list(self._resolution_history)[:limit]]

    def remove_conflict(self, conflict_id: str) -> bool:
        with self._lock:
            if conflict_id not in self._conflicts:
                return False
            logger.warning("Removing conflict %s", conflict_id)
            del self._conflicts[conflict_id]
            return True

    def get_stats(self) -> Dict:
        with self._lock:
            conflicts = [c.status for c in self._conflicts.values()]
        active = sum(1 for s in conflicts if s == ConflictStatus.ACTIVE)
        resolved = sum(1 for s in conflicts if s == ConflictStatus.RESOLVED)
        escalated = sum(1 for s in conflicts if s == ConflictStatus.ESCALATED)
        mediating = sum(1 for s in conflicts if s == ConflictStatus.MEDIATING)
        return {
            "total_conflicts": len(conflicts),
            "active": active,
            "resolved": resolved,
            "escalated": escalated,
            "mediating": mediating,
            "resolution_rate": round(resolved / max(len(conflicts), 1), 3),
        }