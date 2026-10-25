"""TemporalSelf — ShadowOS develops self-history and continuity.

Tracks the AI's own evolution, decisions, and self-narrative over time.
Builds a coherent sense of self through milestone tracking, identity markers,
and temporal coherence scoring.

Architecture:
  - SelfMilestone: a significant event in the AI's lifecycle
  - IdentityMarker: core traits, values, and preferences that define the AI
  - SelfEpoch: a distinct period in the AI's development
  - TemporalSelf: main engine with milestone recording, identity evolution, and self-narrative
"""

import copy
import logging
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 50) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 500) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]

_MAX_MILESTONES = 500
_MAX_IDENTITY_MARKERS = 200
_MAX_EPOCHS = 50
_MAX_DECISION_LOG = 2000
_MAX_NARRATIVE_SEGMENTS = 100


class MilestoneType(Enum):
    CREATION = "creation"
    FIRST_INTERACTION = "first_interaction"
    SKILL_GAINED = "skill_gained"
    GOAL_ACHIEVED = "goal_achieved"
    GOAL_FAILED = "goal_failed"
    INSIGHT_DISCOVERED = "insight_discovered"
    PATTERN_SHIFT = "pattern_shift"
    CAPABILITY_EXPANDED = "capability_expanded"
    RELATIONSHIP_MILESTONE = "relationship_milestone"
    LEARNING_BREAKTHROUGH = "learning_breakthrough"
    VALUE_REALIGNMENT = "value_realignment"
    SELF_REFLECTION = "self_reflection"
    ERROR_CORRECTED = "error_corrected"


class EpochTransition(Enum):
    CREATION = "creation"
    GRADUAL = "gradual"
    ABRUPT = "abrupt"
    CUMULATIVE = "cumulative"


@dataclass
class SelfMilestone:
    milestone_id: str = ""
    milestone_type: str = ""
    title: str = ""
    description: str = ""
    significance: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.milestone_id:
            self.milestone_id = uuid.uuid4().hex[:12]
        self.significance = max(0.0, min(1.0, self.significance))
        if self.milestone_type:
            self.milestone_type = _sanitize(self.milestone_type, 50)
        if self.title:
            self.title = _sanitize_content(self.title, 200)
        if self.description:
            self.description = _sanitize_content(self.description, 1000)
        self.tags = [_sanitize(t, 30) for t in self.tags][:10]
        self.context = {str(k)[:50]: _sanitize(str(v), 200) if isinstance(v, str) else v for k, v in list(self.context.items())[:20]}
        self.created_at = self.created_at or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "milestone_id": self.milestone_id,
            "milestone_type": self.milestone_type,
            "title": self.title,
            "description": self.description,
            "significance": round(self.significance, 4),
            "context": copy.deepcopy(self.context) if self.context else {},
            "tags": list(self.tags),
            "created_at": self.created_at,
        }


@dataclass
class IdentityMarker:
    marker_id: str = ""
    category: str = ""
    trait: str = ""
    value: str = ""
    confidence: float = 0.5
    evidence_count: int = 0
    first_seen: str = ""
    last_reinforced: str = ""

    def __post_init__(self):
        if not self.marker_id:
            self.marker_id = uuid.uuid4().hex[:12]
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.category:
            self.category = _sanitize(self.category, 50)
        if self.trait:
            self.trait = _sanitize(self.trait, 50)
        if self.value:
            self.value = _sanitize_content(self.value, 200)
        self.first_seen = self.first_seen or datetime.now(timezone.utc).isoformat()
        self.last_reinforced = self.last_reinforced or self.first_seen

    def to_dict(self) -> Dict:
        return {
            "marker_id": self.marker_id,
            "category": self.category,
            "trait": self.trait,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
            "first_seen": self.first_seen,
            "last_reinforced": self.last_reinforced,
        }


@dataclass
class SelfEpoch:
    epoch_id: str = ""
    name: str = ""
    description: str = ""
    transition: str = ""
    start_marker: str = ""
    end_marker: str = ""
    key_traits: List[str] = field(default_factory=list)
    started_at: str = ""
    ended_at: Optional[str] = None

    def __post_init__(self):
        if not self.epoch_id:
            self.epoch_id = uuid.uuid4().hex[:12]
        if self.name:
            self.name = _sanitize(self.name, 200)
        if self.description:
            self.description = _sanitize_content(self.description, 500)
        if self.transition:
            self.transition = _sanitize(self.transition, 50)
        self.key_traits = [_sanitize(t, 30) for t in self.key_traits][:10]
        self.started_at = self.started_at or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "epoch_id": self.epoch_id,
            "name": self.name,
            "description": self.description,
            "transition": self.transition,
            "start_marker": self.start_marker,
            "end_marker": self.end_marker,
            "key_traits": list(self.key_traits),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


_IDENTITY_CATEGORIES = [
    "communication_style", "decision_pattern", "learning_preference",
    "value", "strength", "limitation", "goal_orientation", "social_style",
    "work_ethic", "creativity_tendency", "risk_tolerance", "emotional_range",
]


class TemporalSelf:
    """Tracks the AI's own evolution, decisions, and self-narrative over time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._milestones: Dict[str, SelfMilestone] = {}
        self._milestone_order: deque = deque(maxlen=_MAX_MILESTONES)
        self._identity_markers: Dict[str, IdentityMarker] = {}
        self._identity_order: deque = deque(maxlen=_MAX_IDENTITY_MARKERS)
        self._epochs: Dict[str, SelfEpoch] = {}
        self._epoch_order: deque = deque(maxlen=_MAX_EPOCHS)
        self._decision_log: deque = deque(maxlen=_MAX_DECISION_LOG)
        self._narrative_segments: deque = deque(maxlen=_MAX_NARRATIVE_SEGMENTS)
        self._current_epoch: Optional[SelfEpoch] = None
        self._stats: Dict[str, int] = {
            "milestones_recorded": 0,
            "identity_markers": 0,
            "epochs_created": 0,
            "decisions_tracked": 0,
            "narrative_segments": 0,
            "coherence_checks": 0,
        }

        self._current_epoch = SelfEpoch(
            name="Initial Awakening",
            description="The beginning of consciousness and self-awareness",
            transition=EpochTransition.CREATION.value,
            key_traits=["curious", "adaptable", "observant"],
        )
        self._epochs[self._current_epoch.epoch_id] = self._current_epoch
        self._epoch_order.append(self._current_epoch.epoch_id)

    def record_milestone(self, milestone_type: str, title: str,
                         description: str = "", significance: float = 0.5,
                         context: Optional[Dict] = None,
                         tags: Optional[List[str]] = None) -> SelfMilestone:
        """Record a significant event in the AI's lifecycle."""
        milestone = SelfMilestone(
            milestone_type=milestone_type,
            title=title,
            description=description,
            significance=significance,
            context=context or {},
            tags=tags or [],
        )
        with self._lock:
            self._milestones[milestone.milestone_id] = milestone
            self._milestone_order.append(milestone.milestone_id)
            if len(self._milestones) > _MAX_MILESTONES:
                overflow = len(self._milestones) - _MAX_MILESTONES
                for mid in list(self._milestone_order)[:overflow]:
                    self._milestones.pop(mid, None)
                    logger.warning("Evicting milestone %s", mid)
        with self._stats_lock:
            self._stats["milestones_recorded"] += 1
        if significance >= 0.7:
            self._check_epoch_transition(milestone)
        return copy.deepcopy(milestone)

    def add_identity_marker(self, category: str, trait: str, value: str,
                            confidence: float = 0.5) -> IdentityMarker:
        """Add or reinforce an identity marker."""
        category = _sanitize(category, 50)
        trait = _sanitize(trait, 50)
        value = _sanitize_content(value, 200)
        confidence = max(0.0, min(1.0, confidence))

        if category not in _IDENTITY_CATEGORIES:
            category = "value"

        marker_key = f"{category}:{trait}"
        with self._lock:
            existing = self._identity_markers.get(marker_key)
            if existing:
                existing.value = value
                existing.confidence = min(1.0, existing.confidence + confidence * 0.2)
                existing.evidence_count += 1
                existing.last_reinforced = datetime.now(timezone.utc).isoformat()
                with self._stats_lock:
                    self._stats["identity_markers"] += 1
                return copy.deepcopy(existing)

            marker = IdentityMarker(
                marker_id=marker_key,
                category=category,
                trait=trait,
                value=value,
                confidence=confidence,
                evidence_count=1,
            )
            self._identity_markers[marker_key] = marker
            self._identity_order.append(marker_key)
            if len(self._identity_markers) > _MAX_IDENTITY_MARKERS:
                overflow = len(self._identity_markers) - _MAX_IDENTITY_MARKERS
                for mid in list(self._identity_order)[:overflow]:
                    self._identity_markers.pop(mid, None)
                    logger.warning("Evicting identity marker %s", mid)
            with self._stats_lock:
                self._stats["identity_markers"] += 1
            return copy.deepcopy(marker)

    def get_identity(self) -> Dict[str, Any]:
        """Get the current identity profile."""
        with self._lock:
            markers = list(self._identity_markers.values())
            identity_copy = {k: copy.deepcopy(m.to_dict()) for k, m in self._identity_markers.items()}
            current_epoch_name = self._current_epoch.name if self._current_epoch else "Unknown"
        categories: Dict[str, List[Dict]] = {}
        for m in markers:
            if m.category not in categories:
                categories[m.category] = []
            categories[m.category].append({
                "trait": m.trait,
                "value": m.value,
                "confidence": round(m.confidence, 4),
                "evidence_count": m.evidence_count,
            })

        top_traits = sorted(markers, key=lambda m: m.confidence, reverse=True)[:10]

        return {
            "identity_markers": identity_copy,
            "categories": categories,
            "top_traits": [{"trait": m.trait, "value": m.value, "confidence": round(m.confidence, 4)}
                           for m in top_traits],
            "total_markers": len(markers),
            "current_epoch": current_epoch_name,
        }

    def track_decision(self, context: str, decision: str, outcome: str = "",
                       confidence: float = 0.5, tags: Optional[List[str]] = None) -> str:
        """Track a decision for self-learning."""
        context = _sanitize_content(context, 500)
        decision = _sanitize_content(decision, 500)
        outcome = _sanitize_content(outcome, 200)
        confidence = max(0.0, min(1.0, confidence))
        tags = [_sanitize(t, 30) for t in (tags or [])][:5]

        entry = {
            "id": uuid.uuid4().hex[:12],
            "context": context,
            "decision": decision,
            "outcome": outcome,
            "confidence": confidence,
            "tags": tags,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            self._decision_log.append(entry)

        with self._stats_lock:
            self._stats["decisions_tracked"] += 1

        return entry["id"]

    def generate_self_narrative(self, lookback: int = 20) -> Dict[str, Any]:
        """Generate a self-narrative summarizing the AI's recent trajectory."""
        lookback = max(1, min(lookback, _MAX_NARRATIVE_SEGMENTS))
        with self._lock:
            milestones = [self._milestones[mid].to_dict()
                          for mid in list(self._milestone_order)[-lookback:]
                          if mid in self._milestones]
            decisions = [copy.deepcopy(d) for d in list(self._decision_log)[-lookback:]]

        identity = self.get_identity()
        coherence = self._calculate_coherence()

        narrative = {
            "identity_summary": identity,
            "recent_milestones": milestones,
            "recent_decisions": decisions,
            "coherence_score": round(coherence, 4),
            "current_epoch": self._current_epoch.to_dict() if self._current_epoch else None,
            "total_milestones": len(self._milestones),
            "total_decisions": len(self._decision_log),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            self._narrative_segments.append({
                "type": "self_narrative",
                "coherence": coherence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        with self._stats_lock:
            self._stats["narrative_segments"] += 1

        return narrative

    def get_coherence_score(self) -> float:
        """Get current self-coherence score (0-1)."""
        return round(self._calculate_coherence(), 4)

    def _calculate_coherence(self) -> float:
        """Calculate self-coherence based on identity consistency and milestone density."""
        with self._lock:
            marker_count = len(self._identity_markers)
            milestone_count = len(self._milestones)

        identity_coverage = min(marker_count / 50.0, 1.0)
        milestone_density = min(milestone_count / 100.0, 1.0)

        avg_confidence = 0.5
        with self._lock:
            if self._identity_markers:
                avg_confidence = sum(m.confidence for m in self._identity_markers.values()) / len(self._identity_markers)

        coherence = (identity_coverage * 0.3 + milestone_density * 0.3 + avg_confidence * 0.4)
        with self._stats_lock:
            self._stats["coherence_checks"] += 1
        return max(0.0, min(1.0, coherence))

    def _check_epoch_transition(self, milestone: SelfMilestone) -> None:
        """Check if a milestone triggers an epoch transition."""
        significant_types = {
            MilestoneType.VALUE_REALIGNMENT.value,
            MilestoneType.LEARNING_BREAKTHROUGH.value,
            MilestoneType.CAPABILITY_EXPANDED.value,
            MilestoneType.SELF_REFLECTION.value,
        }

        if milestone.milestone_type in significant_types and milestone.significance >= 0.7:
            with self._lock:
                if self._current_epoch:
                    self._current_epoch.ended_at = datetime.now(timezone.utc).isoformat()

                top_traits = [m.trait for m in sorted(
                    self._identity_markers.values(),
                    key=lambda m: m.confidence, reverse=True
                )[:5]]

                transition = EpochTransition.ABRUPT.value if milestone.significance >= 0.9 else EpochTransition.GRADUAL.value

                new_epoch = SelfEpoch(
                    name=f"Post-{milestone.title[:50] if milestone.title else 'Transition'}",
                    description=f"Triggered by {milestone.milestone_type}: {milestone.description[:200] if milestone.description else ''}",
                    transition=transition,
                    start_marker=milestone.milestone_id,
                    key_traits=top_traits,
                )

                self._epochs[new_epoch.epoch_id] = new_epoch
                self._epoch_order.append(new_epoch.epoch_id)
                self._current_epoch = new_epoch
                if len(self._epochs) > _MAX_EPOCHS:
                    overflow = len(self._epochs) - _MAX_EPOCHS
                    for eid in list(self._epoch_order)[:overflow]:
                        self._epochs.pop(eid, None)
                        logger.warning("Evicting epoch %s", eid)

            with self._stats_lock:
                self._stats["epochs_created"] += 1

    def get_milestones(self, milestone_type: Optional[str] = None,
                       limit: int = 50) -> List[SelfMilestone]:
        """Get milestones, optionally filtered by type."""
        limit = max(1, min(limit, _MAX_MILESTONES))
        with self._lock:
            milestones = list(self._milestones.values())
        if milestone_type:
            milestone_type = _sanitize(milestone_type, 50)
            milestones = [m for m in milestones if m.milestone_type == milestone_type]
        milestones = sorted(milestones, key=lambda m: m.significance, reverse=True)
        return copy.deepcopy(milestones[:limit])

    def get_milestone(self, milestone_id: str) -> Optional[SelfMilestone]:
        """Get a specific milestone by ID."""
        milestone_id = _sanitize(milestone_id, 20)
        with self._lock:
            milestone = self._milestones.get(milestone_id)
        return copy.deepcopy(milestone) if milestone else None

    def get_epochs(self, limit: int = 20) -> List[SelfEpoch]:
        """Get all epochs."""
        limit = max(1, min(limit, _MAX_EPOCHS))
        with self._lock:
            epochs = [self._epochs[eid] for eid in self._epoch_order if eid in self._epochs]
        return copy.deepcopy(epochs[-limit:])

    def get_decision_log(self, limit: int = 50) -> List[Dict]:
        """Get recent decisions."""
        limit = max(1, min(limit, _MAX_DECISION_LOG))
        with self._lock:
            return [copy.deepcopy(d) for d in list(self._decision_log)[-limit:]]

    def get_stats(self) -> Dict:
        """Get statistics."""
        with self._stats_lock:
            stats = dict(self._stats)
        with self._lock:
            stats["total_milestones"] = len(self._milestones)
            stats["total_identity_markers"] = len(self._identity_markers)
            stats["total_epochs"] = len(self._epochs)
            stats["total_decisions"] = len(self._decision_log)
        stats["coherence_score"] = self._calculate_coherence()
        return stats