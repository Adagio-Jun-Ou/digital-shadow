#!/usr/bin/env python3
"""
EvolutionGuide - Personal Evolution Through Observation and Optimization

Guides the user to evolve by:
- Observing behavior patterns across domains (work, health, creativity, social)
- Identifying inefficiencies, friction points, and missed opportunities
- Generating optimization suggestions with clear rationale (WHY this matters)
- Tracking adoption of suggestions over time
- Measuring evolution progress with concrete metrics
- Cross-domain insight synthesis (e.g., sleep → productivity → mood)

Architecture:
  - ObservationPoint: A single observed behavior or pattern
  - FrictionPoint: Identified inefficiency or pain point
  - EvolutionSuggestion: Optimization suggestion with rationale
  - AdoptionRecord: Whether a suggestion was adopted/rejected/ignored
  - DomainProgress: Evolution progress per domain
  - EvolutionGuide: Main engine managing observations, friction, suggestions

  Thread-safe, bounded collections, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Set
from collections import deque, Counter

logger = logging.getLogger("shadowd.cognition.evolution_guide")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _SANITIZE_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_OBSERVATIONS = 5000
_MAX_FRICTION_POINTS = 500
_MAX_SUGGESTIONS = 1000
_MAX_ADOPTION_RECORDS = 2000
_MAX_DOMAIN_HISTORY = 200
_MAX_TEXT = 10000
_MAX_STRICT = 200
_MAX_DOMAINS = 30
_MAX_RATIONALE = 5000
_MAX_METADATA_KEYS = 50
_MAX_METADATA_VALUE = 500


def _sanitize(text: str, max_length: int = _MAX_STRICT) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = _MAX_TEXT) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_dict(d: Dict, max_keys: int = _MAX_METADATA_KEYS) -> Dict:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in list(d.items())[:max_keys]:
        ks = _sanitize(str(k), 100)
        if isinstance(v, str):
            out[ks] = _sanitize_content(v, _MAX_METADATA_VALUE)
        elif isinstance(v, (int, float, bool)):
            out[ks] = v
        else:
            out[ks] = _sanitize_content(str(v), _MAX_METADATA_VALUE)
    return out


class EvolutionDomain(str, Enum):
    WORK = "work"
    HEALTH = "health"
    CREATIVITY = "creativity"
    SOCIAL = "social"
    LEARNING = "learning"
    FINANCIAL = "financial"
    EMOTIONAL = "emotional"
    PRODUCTIVITY = "productivity"
    DECISION_MAKING = "decision_making"
    HABITS = "habits"


class FrictionSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SuggestionStatus(str, Enum):
    PROPOSED = "proposed"
    ADOPTED = "adopted"
    REJECTED = "rejected"
    IGNORED = "ignored"
    EXPIRED = "expired"


class AdoptionOutcome(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class ObservationType(str, Enum):
    BEHAVIOR = "behavior"
    PATTERN = "pattern"
    METRIC = "metric"
    EVENT = "event"
    FEEDBACK = "feedback"
    ANOMALY = "anomaly"


DOMAIN_CROSS_INFLUENCE = {
    (EvolutionDomain.HEALTH, EvolutionDomain.PRODUCTIVITY): 0.8,
    (EvolutionDomain.HEALTH, EvolutionDomain.EMOTIONAL): 0.75,
    (EvolutionDomain.HEALTH, EvolutionDomain.DECISION_MAKING): 0.6,
    (EvolutionDomain.HEALTH, EvolutionDomain.HABITS): 0.7,
    (EvolutionDomain.LEARNING, EvolutionDomain.CREATIVITY): 0.7,
    (EvolutionDomain.SOCIAL, EvolutionDomain.EMOTIONAL): 0.8,
    (EvolutionDomain.HABITS, EvolutionDomain.WORK): 0.65,
    (EvolutionDomain.EMOTIONAL, EvolutionDomain.DECISION_MAKING): 0.7,
    (EvolutionDomain.PRODUCTIVITY, EvolutionDomain.WORK): 0.85,
    (EvolutionDomain.FINANCIAL, EvolutionDomain.EMOTIONAL): 0.55,
}


@dataclass
class ObservationPoint:
    observation_id: str = ""
    observation_type: ObservationType = ObservationType.BEHAVIOR
    domain: EvolutionDomain = EvolutionDomain.WORK
    description: str = ""
    evidence: str = ""
    timestamp: str = ""
    confidence: float = 0.5
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.observation_id:
            self.observation_id = uuid.uuid4().hex[:12]
        self.observation_id = _sanitize(self.observation_id)
        if isinstance(self.observation_type, str):
            try:
                self.observation_type = ObservationType(self.observation_type)
            except ValueError:
                self.observation_type = ObservationType.BEHAVIOR
        if isinstance(self.domain, str):
            try:
                self.domain = EvolutionDomain(self.domain)
            except ValueError:
                self.domain = EvolutionDomain.WORK
        self.description = _sanitize_content(self.description, _MAX_TEXT)
        self.evidence = _sanitize_content(self.evidence, _MAX_RATIONALE)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.metadata = _sanitize_dict(self.metadata)
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "observation_id": self.observation_id,
            "observation_type": self.observation_type.value,
            "domain": self.domain.value,
            "description": self.description[:200],
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class FrictionPoint:
    friction_id: str = ""
    domain: EvolutionDomain = EvolutionDomain.WORK
    title: str = ""
    description: str = ""
    severity: FrictionSeverity = FrictionSeverity.MEDIUM
    evidence: str = ""
    impact_score: float = 0.5
    frequency: float = 0.0
    observation_ids: List[str] = field(default_factory=list)
    created_at: str = ""
    resolved_at: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.friction_id:
            self.friction_id = uuid.uuid4().hex[:12]
        self.friction_id = _sanitize(self.friction_id)
        if isinstance(self.domain, str):
            try:
                self.domain = EvolutionDomain(self.domain)
            except ValueError:
                self.domain = EvolutionDomain.WORK
        if isinstance(self.severity, str):
            try:
                self.severity = FrictionSeverity(self.severity)
            except ValueError:
                self.severity = FrictionSeverity.MEDIUM
        self.title = _sanitize(self.title, _MAX_STRICT)
        self.description = _sanitize_content(self.description, _MAX_TEXT)
        self.evidence = _sanitize_content(self.evidence, _MAX_RATIONALE)
        self.impact_score = max(0.0, min(1.0, float(self.impact_score)))
        self.frequency = max(0.0, min(1.0, float(self.frequency)))
        self.observation_ids = [_sanitize(oid, 12) for oid in (self.observation_ids or [])[:50]]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.metadata = _sanitize_dict(self.metadata)

    def to_dict(self) -> Dict:
        return {
            "friction_id": self.friction_id,
            "domain": self.domain.value,
            "title": self.title,
            "severity": self.severity.value,
            "impact_score": round(self.impact_score, 3),
            "frequency": round(self.frequency, 3),
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "observation_ids": list(self.observation_ids),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class EvolutionSuggestion:
    suggestion_id: str = ""
    domain: EvolutionDomain = EvolutionDomain.WORK
    title: str = ""
    description: str = ""
    rationale: str = ""
    expected_benefit: str = ""
    friction_ids: List[str] = field(default_factory=list)
    related_domains: List[str] = field(default_factory=list)
    confidence: float = 0.5
    priority: float = 0.5
    status: SuggestionStatus = SuggestionStatus.PROPOSED
    created_at: str = ""
    expires_at: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.suggestion_id:
            self.suggestion_id = uuid.uuid4().hex[:12]
        self.suggestion_id = _sanitize(self.suggestion_id)
        if isinstance(self.domain, str):
            try:
                self.domain = EvolutionDomain(self.domain)
            except ValueError:
                self.domain = EvolutionDomain.WORK
        if isinstance(self.status, str):
            try:
                self.status = SuggestionStatus(self.status)
            except ValueError:
                self.status = SuggestionStatus.PROPOSED
        self.title = _sanitize(self.title, _MAX_STRICT)
        self.description = _sanitize_content(self.description, _MAX_TEXT)
        self.rationale = _sanitize_content(self.rationale, _MAX_RATIONALE)
        self.expected_benefit = _sanitize_content(self.expected_benefit, _MAX_TEXT)
        self.friction_ids = [_sanitize(fid, 12) for fid in (self.friction_ids or [])[:50]]
        self.related_domains = [_sanitize(d, 50) for d in (self.related_domains or [])[:10]]
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.priority = max(0.0, min(1.0, float(self.priority)))
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.metadata = _sanitize_dict(self.metadata)

    def to_dict(self) -> Dict:
        return {
            "suggestion_id": self.suggestion_id,
            "domain": self.domain.value,
            "title": self.title,
            "rationale": self.rationale[:500],
            "expected_benefit": self.expected_benefit[:500],
            "friction_ids": list(self.friction_ids),
            "related_domains": list(self.related_domains),
            "confidence": round(self.confidence, 3),
            "priority": round(self.priority, 3),
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class AdoptionRecord:
    record_id: str = ""
    suggestion_id: str = ""
    outcome: AdoptionOutcome = AdoptionOutcome.UNKNOWN
    feedback: str = ""
    observed_impact: float = 0.0
    adopted_at: str = ""
    reviewed_at: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = uuid.uuid4().hex[:12]
        self.record_id = _sanitize(self.record_id)
        self.suggestion_id = _sanitize(self.suggestion_id, 12)
        if isinstance(self.outcome, str):
            try:
                self.outcome = AdoptionOutcome(self.outcome)
            except ValueError:
                self.outcome = AdoptionOutcome.UNKNOWN
        self.feedback = _sanitize_content(self.feedback, _MAX_TEXT)
        self.observed_impact = max(-1.0, min(1.0, float(self.observed_impact)))
        if not self.adopted_at:
            self.adopted_at = datetime.now(timezone.utc).isoformat()
        self.metadata = _sanitize_dict(self.metadata)

    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "suggestion_id": self.suggestion_id,
            "outcome": self.outcome.value,
            "observed_impact": round(self.observed_impact, 3),
            "adopted_at": self.adopted_at,
            "reviewed_at": self.reviewed_at,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class DomainProgress:
    domain: EvolutionDomain = EvolutionDomain.WORK
    current_score: float = 0.5
    baseline_score: float = 0.5
    trend: str = "stable"
    improvement_rate: float = 0.0
    friction_count: int = 0
    suggestion_count: int = 0
    adoption_rate: float = 0.0
    recorded_at: str = ""

    def __post_init__(self):
        if isinstance(self.domain, str):
            try:
                self.domain = EvolutionDomain(self.domain)
            except ValueError:
                self.domain = EvolutionDomain.WORK
        self.current_score = max(0.0, min(1.0, float(self.current_score)))
        self.baseline_score = max(0.0, min(1.0, float(self.baseline_score)))
        self.improvement_rate = max(-1.0, min(1.0, float(self.improvement_rate)))
        self.adoption_rate = max(0.0, min(1.0, float(self.adoption_rate)))
        if not self.recorded_at:
            self.recorded_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "domain": self.domain.value,
            "current_score": round(self.current_score, 3),
            "baseline_score": round(self.baseline_score, 3),
            "trend": self.trend,
            "improvement_rate": round(self.improvement_rate, 4),
            "friction_count": self.friction_count,
            "suggestion_count": self.suggestion_count,
            "adoption_rate": round(self.adoption_rate, 3),
            "recorded_at": self.recorded_at,
        }


_FRICTION_PATTERNS = {
    EvolutionDomain.WORK: [
        ("procrastination", r'\bprocrastinat\w+\b|\bputting\s+off\b|\bavoiding\b', 0.7),
        ("context_switching", r'\bcontext\s+switch\w*\b|\bmultitask\w*\b|\bjuggling\b', 0.6),
        ("perfectionism", r'\bperfect\w*\b|\bover\-?engineer\w*\b|\bpolish\w*\b', 0.5),
        ("scope_creep", r'\bscope\s+creep\b|\bfeature\s+creep\b|\bmission\s+creep\b', 0.7),
        ("unclear_priorities", r'\buncl\w*\s+prior\w*\b|\bconfus\w*\s+prior\w*\b|\btoo\s+many\s+things\b', 0.6),
    ],
    EvolutionDomain.HEALTH: [
        ("sleep_deprivation", r'\bsleep\s+dep\w*\b|\binsomn\w*\b|\btired\w*\b|\bfatigue\w*\b|\bnot\s+sleeping\b', 0.8),
        ("sedentary", r'\bsedent\w*\b|\bnot\s+exercis\w*\b|\bsitting\s+too\s+much\b', 0.6),
        ("poor_nutrition", r'\bpoor\s+diet\b|\bjunk\s+food\b|\bskip\w*\s+meal\w*\b|\bnot\s+eating\b', 0.5),
        ("stress_overload", r'\bstress\w*\b|\boverwhelm\w*\b|\bburn\w*\s*out\b|\bp\w*ress\w*\b', 0.7),
    ],
    EvolutionDomain.PRODUCTIVITY: [
        ("interruptions", r'\binterrupt\w*\b|\bdistract\w*\b|\bbroken\s+focus\b|\bcan\'?t\s+focus\b', 0.7),
        ("no_deep_work", r'\bno\s+deep\s+work\b|\bshallow\s+work\b|\bnever\s+uninterrupted\b', 0.8),
        ("meeting_overload", r'\btoo\s+many\s+meeting\w*\b|\bmeeting\s+over\w*\b|\bback\s+to\s+back\w*\b', 0.6),
        ("poor_planning", r'\bno\s+plan\w*\b|\bad\s+hoc\b|\breact\w*\b|\bfire\s+fight\w*\b', 0.6),
    ],
    EvolutionDomain.CREATIVITY: [
        ("creative_block", r'\bcreative\s+block\b|\bblocked\b|\bst\w*\s*uck\b|\bno\s+idea\w*\b', 0.6),
        ("routine_trapped", r'\broutine\b|\brut\b|\brepetitive\b|\bsame\s+thing\b', 0.4),
        ("no_inspiration", r'\bno\s+inspir\w*\b|\buninspir\w*\b|\bmotiv\w*\s*low\b', 0.5),
    ],
    EvolutionDomain.SOCIAL: [
        ("isolation", r'\bisol\w*\b|\blone\w*\b|\bno\s+social\b|\bdisconnect\w*\b', 0.7),
        ("people_pleasing", r'\bpeople\s+pleas\w*\b|\bcan\'?t\s+say\s+no\b|\bovercomm\w*\b', 0.5),
        ("conflict_avoidance", r'\bconflict\s+avoid\w*\b|\bavoid\w*\s+difficult\b|\bnot\s+speak\w*\s+up\b', 0.6),
    ],
    EvolutionDomain.DECISION_MAKING: [
        ("analysis_paralysis", r'\banalysis\s+paralysis\b|\bover\s*think\w*\b|\bcan\'?t\s+decide\b', 0.7),
        ("confirmation_bias", r'\bconfirm\w*\s+bias\b|\bonly\s+see\w*\b|\bignor\w*\s+evid\w*\b', 0.6),
        ("sunk_cost", r'\bsunk\s+cost\b|\balready\s+invest\w*\b|\bcan\'?t\s+abandon\b', 0.6),
    ],
    EvolutionDomain.HABITS: [
        ("habit_breaking", r'\bbreak\w*\s+habit\b|\brelaps\w*\b|\bfell\s+off\b|\bback\s+to\s+old\b', 0.5),
        ("no_routine", r'\bno\s+routine\b|\bno\s+structure\b|\bchaotic\b|\brandom\b', 0.5),
        ("willpower_depletion", r'\bwillpower\b|\bself\s+control\b|\bexhaust\w*\s+discipl\w*\b', 0.6),
    ],
}

_SUGGESTION_TEMPLATES = {
    EvolutionDomain.WORK: [
        ("Time-block deep work", "Deep work sessions of 90+ minutes produce 3x output vs fragmented time. Your context-switching pattern suggests you're losing 30-40% of productive capacity to switching overhead."),
        ("Set explicit priorities weekly", "Without written priorities, urgent tasks always crowd out important ones. The Eisenhower matrix can reduce decision fatigue by 50%."),
        ("Define done criteria upfront", "Scope creep thrives when 'done' is undefined. Pre-committing to deliverables prevents 60% of over-engineering."),
    ],
    EvolutionDomain.HEALTH: [
        ("Protect sleep window", "Even 30 minutes of extra sleep improves next-day cognitive performance by 15-25%. Your stress patterns correlate with sleep inconsistency."),
        ("Movement breaks every 90min", "Sedentary periods >90min reduce blood flow to prefrontal cortex by 15%. Brief walks restore executive function."),
        ("Prep meals on Sunday", "Decision fatigue from daily meal choices consumes ~5% of daily willpower. Pre-commitment eliminates this friction entirely."),
    ],
    EvolutionDomain.PRODUCTIVITY: [
        ("Institute focus time blocks", "Protected 2-hour focus blocks create 5x more value than equivalent fragmented time. Your interruption patterns suggest you average <15 min uninterrupted."),
        ("Batch similar tasks", "Task-switching costs 15-25 min of reorientation per switch. Batching emails, reviews, and admin into dedicated slots recovers 1-2 hours daily."),
        ("Review and plan daily", "A 10-minute evening review + 10-minute morning plan prevents 80% of 'what should I work on?' moments that open the door to distraction."),
    ],
    EvolutionDomain.CREATIVITY: [
        ("Schedule wander time", "Creative insight requires default-mode network activation. 20 min of unstructured thinking (walk, shower, doodle) generates more ideas than 2 hours of forced ideation."),
        ("Cross-pollinate domains", "Your best ideas come from domain intersections. Reading outside your field 30 min/day increases creative output by 40%."),
        ("Capture immediately, filter later", "Ideas have a 30-second half-life. A trusted capture system (notes app, voice memo) prevents 90% of 'I had a great idea but forgot it' losses."),
    ],
    EvolutionDomain.SOCIAL: [
        ("Schedule social connection", "Social bonds decay without maintenance. One meaningful conversation per week sustains relationships better than 10 shallow ones."),
        ("Practice assertive requests", "Unexpressed needs create resentment. Direct requests ('I need X') get 70% better outcomes than hints."),
        ("Set boundaries with empathy", "Boundaries aren't walls — they're guardrails. Saying 'no' to one thing is saying 'yes' to your priorities."),
    ],
    EvolutionDomain.DECISION_MAKING: [
        ("Set decide-by deadlines", "Most decisions are reversible. A time limit prevents analysis paralysis: 5 min for low-stakes, 1 hour for medium, 1 day for high."),
        ("Write the anti-recommendation", "Before committing, write 3 reasons this is wrong. Pre-mortems catch 30% of blind spots that confirmation bias hides."),
        ("Use 70% confidence rule", "Decide at 70% confidence rather than 90%. The extra 20% information rarely changes the decision but always costs more time."),
    ],
    EvolutionDomain.HABITS: [
        ("Stack on existing habits", "New habits attach to existing ones 3x more reliably than standalone intentions. 'After I [existing habit], I will [new habit]' is the most effective formula."),
        ("Make it 2 minutes", "Shrink any habit to a 2-minute version. 2 minutes of exercise beats 0 minutes every time, and consistency builds faster than intensity."),
        ("Design your environment", "Environment design is 10x more effective than willpower. Remove friction for good habits, add friction for bad ones."),
    ],
}


class EvolutionGuide:
    """
    Personal evolution engine that observes, identifies friction,
    generates suggestions with rationale, and tracks adoption.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._observations: Dict[str, ObservationPoint] = {}
        self._observation_order: deque = deque(maxlen=_MAX_OBSERVATIONS)
        self._friction_points: Dict[str, FrictionPoint] = {}
        self._friction_order: deque = deque(maxlen=_MAX_FRICTION_POINTS)
        self._suggestions: Dict[str, EvolutionSuggestion] = {}
        self._suggestion_order: deque = deque(maxlen=_MAX_SUGGESTIONS)
        self._adoptions: Dict[str, AdoptionRecord] = {}
        self._adoption_order: deque = deque(maxlen=_MAX_ADOPTION_RECORDS)
        self._domain_progress: Dict[str, DomainProgress] = {}
        self._domain_history: Dict[str, deque] = {}
        self._observed_patterns: Dict[str, int] = Counter()

        self._stats = {
            "observations_added": 0,
            "friction_detected": 0,
            "suggestions_generated": 0,
            "suggestions_adopted": 0,
            "suggestions_rejected": 0,
            "suggestions_expired": 0,
            "adoption_reviews": 0,
            "cross_domain_insights": 0,
        }

    def add_observation(
        self,
        description: str,
        domain: str = "work",
        observation_type: str = "behavior",
        evidence: str = "",
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> ObservationPoint:
        obs = ObservationPoint(
            observation_type=ObservationType(observation_type) if observation_type in [e.value for e in ObservationType] else ObservationType.BEHAVIOR,
            domain=EvolutionDomain(domain) if domain in [e.value for e in EvolutionDomain] else EvolutionDomain.WORK,
            description=description,
            evidence=evidence,
            confidence=confidence,
            metadata=metadata or {},
        )
        with self._lock:
            self._observations[obs.observation_id] = obs
            self._observation_order.append(obs.observation_id)
            if len(self._observations) > _MAX_OBSERVATIONS:
                evict_id = self._observation_order.popleft()
                self._observations.pop(evict_id, None)
                logger.warning("Observation evicted: %s", evict_id)
        with self._stats_lock:
            self._stats["observations_added"] += 1

        self._detect_friction(obs)
        return obs

    def _detect_friction(self, observation: ObservationPoint) -> None:
        patterns = _FRICTION_PATTERNS.get(observation.domain, [])
        desc_lower = observation.description.lower()
        for friction_name, pattern_re, base_severity in patterns:
            if re.search(pattern_re, desc_lower):
                with self._lock:
                    self._observed_patterns[f"{observation.domain.value}:{friction_name}"] += 1
                self._create_friction_from_observation(
                    observation, friction_name, base_severity
                )

    def _create_friction_from_observation(
        self,
        observation: ObservationPoint,
        friction_name: str,
        base_severity: float,
    ) -> FrictionPoint:
        with self._lock:
            count = self._observed_patterns.get(f"{observation.domain.value}:{friction_name}", 1)
        freq = min(1.0, count / 5.0)
        severity = FrictionSeverity.CRITICAL if base_severity * freq > 0.7 else (
            FrictionSeverity.HIGH if base_severity * freq > 0.5 else (
                FrictionSeverity.MEDIUM if base_severity * freq > 0.3 else FrictionSeverity.LOW
            )
        )
        fp = FrictionPoint(
            domain=observation.domain,
            title=_sanitize(friction_name.replace("_", " ").title()),
            description=f"Detected from observation: {observation.description[:200]}",
            severity=severity,
            evidence=observation.evidence[:500] if observation.evidence else f"Pattern matched in observation {observation.observation_id}",
            impact_score=round(base_severity * freq, 3),
            frequency=freq,
            observation_ids=[observation.observation_id],
        )
        with self._lock:
            self._friction_points[fp.friction_id] = fp
            self._friction_order.append(fp.friction_id)
            if len(self._friction_points) > _MAX_FRICTION_POINTS:
                evict_id = self._friction_order.popleft()
                evicted = self._friction_points.pop(evict_id, None)
                if evicted:
                    logger.warning("Friction point evicted: %s", evict_id)
        with self._stats_lock:
            self._stats["friction_detected"] += 1
        return fp

    def generate_suggestions(
        self,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> List[EvolutionSuggestion]:
        with self._lock:
            friction_snapshot = dict(self._friction_points)
            suggestion_snapshot = dict(self._suggestions)

        domain_friction = {}
        for fp in friction_snapshot.values():
            if fp.resolved_at:
                continue
            if domain and fp.domain.value != domain:
                continue
            d = fp.domain.value
            if d not in domain_friction:
                domain_friction[d] = []
            domain_friction[d].append(fp)

        existing_titles = {s.title for s in suggestion_snapshot.values() if s.status == SuggestionStatus.PROPOSED}

        new_suggestions = []
        for d, frictions in domain_friction.items():
            templates = _SUGGESTION_TEMPLATES.get(EvolutionDomain(d), [])
            for title_base, rationale in templates:
                if title_base in existing_titles:
                    continue
                avg_impact = sum(fp.impact_score for fp in frictions) / max(1, len(frictions))
                related = self._find_related_domains(d)
                suggestion = EvolutionSuggestion(
                    domain=EvolutionDomain(d),
                    title=title_base,
                    rationale=rationale,
                    expected_benefit=f"Addresses {len(frictions)} friction point(s) in {d} domain",
                    friction_ids=[fp.friction_id for fp in frictions[:10]],
                    related_domains=related,
                    confidence=min(1.0, avg_impact + 0.2),
                    priority=round(avg_impact * len(frictions) / max(1, len(frictions)), 3),
                )
                with self._lock:
                    self._suggestions[suggestion.suggestion_id] = suggestion
                    self._suggestion_order.append(suggestion.suggestion_id)
                    if len(self._suggestions) > _MAX_SUGGESTIONS:
                        evict_id = self._suggestion_order.popleft()
                        self._suggestions.pop(evict_id, None)
                        logger.warning("Suggestion evicted: %s", evict_id)
                with self._stats_lock:
                    self._stats["suggestions_generated"] += 1
                new_suggestions.append(suggestion)
                if len(new_suggestions) >= limit:
                    break
            if len(new_suggestions) >= limit:
                break

        return [copy.deepcopy(s) for s in new_suggestions]

    def _find_related_domains(self, domain: str) -> List[str]:
        related = []
        for (d1, d2), influence in DOMAIN_CROSS_INFLUENCE.items():
            if d1.value == domain and influence > 0.5:
                related.append(d2.value)
            elif d2.value == domain and influence > 0.5:
                related.append(d1.value)
        return related[:5]

    def adopt_suggestion(self, suggestion_id: str, feedback: str = "") -> AdoptionRecord:
        suggestion_id = _sanitize(suggestion_id, 12)
        with self._lock:
            suggestion = self._suggestions.get(suggestion_id)
            if not suggestion:
                return AdoptionRecord(suggestion_id=suggestion_id, outcome=AdoptionOutcome.UNKNOWN)
            if suggestion.status != SuggestionStatus.PROPOSED:
                return AdoptionRecord(suggestion_id=suggestion_id, outcome=AdoptionOutcome.UNKNOWN)
            suggestion.status = SuggestionStatus.ADOPTED

            record = AdoptionRecord(
                suggestion_id=suggestion_id,
                outcome=AdoptionOutcome.UNKNOWN,
                feedback=feedback,
            )
            self._adoptions[record.record_id] = record
            self._adoption_order.append(record.record_id)
            if len(self._adoptions) > _MAX_ADOPTION_RECORDS:
                evict_id = self._adoption_order.popleft()
                self._adoptions.pop(evict_id, None)
                logger.warning("Adoption record evicted: %s", evict_id)
        with self._stats_lock:
            self._stats["suggestions_adopted"] += 1
        return copy.deepcopy(record)

    def reject_suggestion(self, suggestion_id: str, feedback: str = "") -> bool:
        suggestion_id = _sanitize(suggestion_id, 12)
        with self._lock:
            suggestion = self._suggestions.get(suggestion_id)
            if not suggestion or suggestion.status != SuggestionStatus.PROPOSED:
                return False
            suggestion.status = SuggestionStatus.REJECTED
        with self._stats_lock:
            self._stats["suggestions_rejected"] += 1
        return True

    def review_adoption(
        self,
        suggestion_id: str,
        outcome: str = "unknown",
        observed_impact: float = 0.0,
        feedback: str = "",
    ) -> Optional[AdoptionRecord]:
        suggestion_id = _sanitize(suggestion_id, 12)
        try:
            outcome_enum = AdoptionOutcome(outcome)
        except ValueError:
            outcome_enum = AdoptionOutcome.UNKNOWN

        with self._lock:
            found = None
            for record in self._adoptions.values():
                if record.suggestion_id == suggestion_id and record.outcome == AdoptionOutcome.UNKNOWN:
                    record.outcome = outcome_enum
                    record.observed_impact = max(-1.0, min(1.0, float(observed_impact)))
                    record.feedback = _sanitize_content(feedback, _MAX_TEXT)
                    record.reviewed_at = datetime.now(timezone.utc).isoformat()
                    found = copy.deepcopy(record)
                    break
            if not found:
                return None
        with self._stats_lock:
            self._stats["adoption_reviews"] += 1

        self._update_domain_progress(suggestion_id, outcome_enum, observed_impact)
        return found

    def _update_domain_progress(
        self,
        suggestion_id: str,
        outcome: AdoptionOutcome,
        impact: float,
    ) -> None:
        with self._lock:
            suggestion = self._suggestions.get(suggestion_id)
            if not suggestion:
                return
            domain_key = suggestion.domain.value
            if domain_key not in self._domain_progress:
                self._domain_progress[domain_key] = DomainProgress(domain=suggestion.domain)
            progress = self._domain_progress[domain_key]
            delta = impact * 0.1 if outcome == AdoptionOutcome.POSITIVE else (-impact * 0.05 if outcome == AdoptionOutcome.NEGATIVE else 0)
            progress.current_score = max(0.0, min(1.0, progress.current_score + delta))
            if progress.current_score > progress.baseline_score:
                progress.trend = "improving"
            elif progress.current_score < progress.baseline_score:
                progress.trend = "declining"
            else:
                progress.trend = "stable"
            progress.improvement_rate = round(progress.current_score - progress.baseline_score, 4)
            progress.recorded_at = datetime.now(timezone.utc).isoformat()

            if domain_key not in self._domain_history:
                self._domain_history[domain_key] = deque(maxlen=_MAX_DOMAIN_HISTORY)
            self._domain_history[domain_key].append(progress.current_score)

    def get_friction_points(
        self,
        domain: Optional[str] = None,
        include_resolved: bool = False,
    ) -> List[FrictionPoint]:
        with self._lock:
            result = []
            for fp in self._friction_points.values():
                if not include_resolved and fp.resolved_at:
                    continue
                if domain and fp.domain.value != domain:
                    continue
                result.append(copy.deepcopy(fp))
        return result

    def get_suggestions(
        self,
        domain: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[EvolutionSuggestion]:
        with self._lock:
            result = []
            for s in self._suggestions.values():
                if domain and s.domain.value != domain:
                    continue
                if status and s.status.value != status:
                    continue
                result.append(copy.deepcopy(s))
        return result

    def get_domain_progress(self, domain: Optional[str] = None) -> Dict:
        with self._lock:
            if domain:
                dp = self._domain_progress.get(domain)
                return dp.to_dict() if dp else {}
            return {k: v.to_dict() for k, v in self._domain_progress.items()}

    def get_evolution_summary(self) -> Dict:
        with self._lock:
            domain_progress = {k: v.to_dict() for k, v in self._domain_progress.items()}
            active_friction = sum(1 for fp in self._friction_points.values() if not fp.resolved_at)
            open_suggestions = sum(1 for s in self._suggestions.values() if s.status == SuggestionStatus.PROPOSED)
            adopted = sum(1 for s in self._suggestions.values() if s.status == SuggestionStatus.ADOPTED)
            total = max(1, len(self._suggestions))
            with self._stats_lock:
                stats_copy = dict(self._stats)
        return {
            "overall_adoption_rate": round(adopted / total, 3) if total > 0 else 0.0,
            "active_friction_points": active_friction,
            "open_suggestions": open_suggestions,
            "domains_tracked": len(domain_progress),
            "domain_progress": domain_progress,
            "cross_domain_insights": stats_copy.get("cross_domain_insights", 0),
        }

    def find_cross_domain_insights(self) -> List[Dict]:
        insights = []
        with self._lock:
            friction_by_domain = {}
            for fp in self._friction_points.values():
                if fp.resolved_at:
                    continue
                d = fp.domain.value
                if d not in friction_by_domain:
                    friction_by_domain[d] = []
                friction_by_domain[d].append(fp)

        for (d1, d2), influence in DOMAIN_CROSS_INFLUENCE.items():
            if influence < 0.5:
                continue
            f1 = friction_by_domain.get(d1.value, [])
            f2 = friction_by_domain.get(d2.value, [])
            if f1 and f2:
                insight_text = f"Friction in {d1.value} likely amplifies {d2.value} issues (influence: {round(influence, 2)}). Addressing {d1.value} friction first may improve {d2.value} by {round(influence * 30, 1)}%."
                insights.append({
                    "insight_id": uuid.uuid4().hex[:12],
                    "source_domain": d1.value,
                    "target_domain": d2.value,
                    "influence": round(influence, 3),
                    "description": insight_text,
                    "friction_count_source": len(f1),
                    "friction_count_target": len(f2),
                })
        with self._stats_lock:
            self._stats["cross_domain_insights"] += len(insights)
        return insights

    def get_stats(self) -> Dict:
        with self._stats_lock:
            return dict(self._stats)

    def ingest_device_context(self, context) -> None:
        """
        Ingest observations from DeviceContextEngine.
        Translates device health/activity data into EvolutionGuide observations.
        """
        try:
            unified = context if isinstance(context, dict) else {}
            if hasattr(context, 'to_dict'):
                unified = context.to_dict()
            health = unified.get("health", {})
            activity = unified.get("activity", {})
            focus = unified.get("focus", {})
            stress_level = health.get("stress_level")
            if stress_level is not None:
                if isinstance(stress_level, str):
                    if stress_level not in ("high", "critical"):
                        stress_level = None
                elif isinstance(stress_level, (int, float)):
                    if stress_level < 0.7:
                        stress_level = None
            if stress_level is not None:
                desc = f"Elevated stress level: {round(float(stress_level) if isinstance(stress_level, str) else stress_level, 2)}"
                self.add_observation(
                    observation_type="metric",
                    domain="health",
                    description=desc,
                    confidence=min(0.9, float(stress_level) if isinstance(stress_level, str) else float(stress_level)),
                )
            heart_rate = health.get("heart_rate")
            if heart_rate is not None and heart_rate > 100:
                desc = f"Elevated heart rate: {round(heart_rate, 1)} bpm"
                self.add_observation(
                    observation_type="metric",
                    domain="health",
                    description=desc,
                    confidence=0.7,
                )
            sleep_hours = health.get("sleep_hours")
            if sleep_hours is not None and sleep_hours < 5:
                desc = f"Sleep deficit: {round(sleep_hours, 1)} hours"
                self.add_observation(
                    observation_type="pattern",
                    domain="health",
                    description=desc,
                    confidence=0.8,
                )
            focus_state = focus.get("focus_state")
            if focus_state == "distracted":
                self.add_observation(
                    observation_type="pattern",
                    domain="productivity",
                    description="Device context indicates distracted state",
                    confidence=0.6,
                )
            activity_type = activity.get("activity_type")
            if activity_type and activity_type not in ("resting", "unknown"):
                desc = f"Activity detected: {activity_type}"
                self.add_observation(
                    observation_type="event",
                    domain="work",
                    description=desc,
                    confidence=0.4,
                )
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.warning("device context ingest error: %s", type(e).__name__)

    def get_briefing_evolution_items(self, limit: int = 5) -> List[Dict]:
        """
        Return top evolution insights for DailyBriefing integration.
        Prioritizes unresolved friction points and unreviewed suggestions.
        """
        items = []
        with self._lock:
            for fp in self._friction_points.values():
                if fp.resolved_at:
                    continue
                items.append({
                    "type": "friction",
                    "domain": fp.domain.value,
                    "description": fp.description,
                    "severity": fp.severity.value,
                    "urgency": "high" if fp.severity.value in ("high", "critical") else "medium",
                })
        suggestions = self.get_suggestions(status="proposed")
        for s in suggestions[:limit]:
            items.append({
                "type": "suggestion",
                "domain": s.domain.value,
                "description": s.description,
                "rationale": s.rationale,
                "urgency": "low",
            })
        items.sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.get("urgency", "low"), 0), reverse=True)
        return items[:limit]