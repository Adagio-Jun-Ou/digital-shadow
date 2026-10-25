#!/usr/bin/env python3
"""
ChronoSynthesis — Past/Future Into Present Decisions.

Synthesizes temporal information from past patterns and future predictions
into present-moment decisions with weighted confidence scoring.

Thread-safe, bounded, sanitized, deepcopy returns.
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
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.chrono_synthesis")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_PATTERNS = 5000
_MAX_PROJECTIONS = 2000
_MAX_DECISIONS = 500
_MAX_CONTEXTS = 500
_MAX_METADATA_KEYS = 20


class TemporalLayer(str, Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    SYNTHESIZED = "synthesized"


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0, decimals: int = 4) -> float:
    return round(max(lo, min(hi, value)), decimals)


@dataclass
class PastPattern:
    pattern_id: str
    event_type: str
    frequency: int
    recency_weight: float
    domain: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = uuid.uuid4().hex[:12]
        self.pattern_id = _sanitize(self.pattern_id, 50)
        self.event_type = _sanitize(self.event_type, 100)
        self.frequency = max(1, int(self.frequency))
        self.recency_weight = _clamp(self.recency_weight)
        self.domain = _sanitize(self.domain, 100)
        self.confidence = _clamp(self.confidence)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "event_type": self.event_type,
            "frequency": self.frequency,
            "recency_weight": round(self.recency_weight, 4),
            "domain": self.domain,
            "confidence": round(self.confidence, 4),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class FutureProjection:
    projection_id: str
    event_type: str
    probability: float
    timeframe: str
    confidence: float
    source_patterns: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.projection_id:
            self.projection_id = uuid.uuid4().hex[:12]
        self.projection_id = _sanitize(self.projection_id, 50)
        self.event_type = _sanitize(self.event_type, 100)
        self.probability = _clamp(self.probability)
        self.timeframe = _sanitize(self.timeframe, 100)
        self.confidence = _clamp(self.confidence)
        self.source_patterns = [_sanitize(p, 50) for p in (self.source_patterns or [])[:20]]
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "projection_id": self.projection_id,
            "event_type": self.event_type,
            "probability": round(self.probability, 4),
            "timeframe": self.timeframe,
            "confidence": round(self.confidence, 4),
            "source_patterns": list(self.source_patterns),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class SynthesisDecision:
    decision_id: str
    context: str
    past_weight: float
    present_weight: float
    future_weight: float
    synthesized_score: float
    contributing_patterns: List[str]
    contributing_projections: List[str]
    rationale: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.decision_id:
            self.decision_id = uuid.uuid4().hex[:12]
        self.decision_id = _sanitize(self.decision_id, 50)
        self.context = _sanitize_content(self.context, 1000)
        self.past_weight = _clamp(self.past_weight)
        self.present_weight = _clamp(self.present_weight)
        self.future_weight = _clamp(self.future_weight)
        self.synthesized_score = _clamp(self.synthesized_score)
        self.contributing_patterns = [_sanitize(p, 50) for p in (self.contributing_patterns or [])[:20]]
        self.contributing_projections = [_sanitize(p, 50) for p in (self.contributing_projections or [])[:20]]
        self.rationale = _sanitize_content(self.rationale, 1000)
        self.confidence = _clamp(self.confidence)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "decision_id": self.decision_id,
            "context": self.context,
            "past_weight": round(self.past_weight, 4),
            "present_weight": round(self.present_weight, 4),
            "future_weight": round(self.future_weight, 4),
            "synthesized_score": round(self.synthesized_score, 4),
            "contributing_patterns": list(self.contributing_patterns),
            "contributing_projections": list(self.contributing_projections),
            "rationale": self.rationale,
            "confidence": round(self.confidence, 4),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class TemporalContext:
    context_id: str
    past_patterns: List[str]
    present_factors: Dict[str, float]
    future_projections: List[str]
    domain: str
    urgency: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.context_id:
            self.context_id = uuid.uuid4().hex[:12]
        self.context_id = _sanitize(self.context_id, 50)
        self.past_patterns = [_sanitize(p, 50) for p in (self.past_patterns or [])[:10]]
        self.present_factors = {k: _clamp(v) for k, v in list(self.present_factors.items())[:20]}
        self.future_projections = [_sanitize(p, 50) for p in (self.future_projections or [])[:10]]
        self.domain = _sanitize(self.domain, 100)
        self.urgency = _clamp(self.urgency)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "context_id": self.context_id,
            "past_patterns": list(self.past_patterns),
            "present_factors": {k: round(v, 4) for k, v in self.present_factors.items()},
            "future_projections": list(self.future_projections),
            "domain": self.domain,
            "urgency": round(self.urgency, 4),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


class ChronoSynthesis:
    """Temporal synthesis engine: combines past patterns and future
    projections into weighted present-moment decisions."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._past_patterns: Dict[str, PastPattern] = {}
        self._future_projections: Dict[str, FutureProjection] = {}
        self._decisions: Dict[str, SynthesisDecision] = {}
        self._contexts: Dict[str, TemporalContext] = {}
        self._pattern_order: deque = deque(maxlen=_MAX_PATTERNS)
        self._projection_order: deque = deque(maxlen=_MAX_PROJECTIONS)
        self._decision_order: deque = deque(maxlen=_MAX_DECISIONS)
        self._context_order: deque = deque(maxlen=_MAX_CONTEXTS)
        self._stats = {
            "patterns_recorded": 0,
            "projections_added": 0,
            "decisions_synthesized": 0,
            "contexts_created": 0,
            "avg_synthesized_score": 0.0,
            "avg_confidence": 0.0,
        }

    def record_past_pattern(
        self,
        event_type: str,
        domain: str,
        frequency: int,
        metadata: Optional[Dict] = None,
    ) -> PastPattern:
        recency = _clamp(1.0 / max(1, frequency * 0.1))
        pattern = PastPattern(
            pattern_id=uuid.uuid4().hex[:12],
            event_type=event_type,
            frequency=frequency,
            recency_weight=recency,
            domain=domain,
            confidence=_clamp(0.5 + 0.5 * min(1.0, frequency / 10.0)),
            metadata=metadata or {},
        )
        with self._lock:
            if len(self._past_patterns) >= _MAX_PATTERNS:
                evicted_id = self._pattern_order.popleft()
                self._past_patterns.pop(evicted_id, None)
                logger.warning("Evicted past pattern %s", evicted_id)
            self._past_patterns[pattern.pattern_id] = pattern
            self._pattern_order.append(pattern.pattern_id)
            with self._stats_lock:
                self._stats["patterns_recorded"] += 1
        return copy.deepcopy(pattern)

    def add_future_projection(
        self,
        event_type: str,
        probability: float,
        timeframe: str,
        confidence: float,
        source_patterns: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> FutureProjection:
        projection = FutureProjection(
            projection_id=uuid.uuid4().hex[:12],
            event_type=event_type,
            probability=probability,
            timeframe=timeframe,
            confidence=confidence,
            source_patterns=source_patterns or [],
            metadata=metadata or {},
        )
        with self._lock:
            if len(self._future_projections) >= _MAX_PROJECTIONS:
                evicted_id = self._projection_order.popleft()
                self._future_projections.pop(evicted_id, None)
                logger.warning("Evicted future projection %s", evicted_id)
            self._future_projections[projection.projection_id] = projection
            self._projection_order.append(projection.projection_id)
            with self._stats_lock:
                self._stats["projections_added"] += 1
        return copy.deepcopy(projection)

    def create_temporal_context(
        self,
        past_pattern_ids: List[str],
        present_factors: Dict[str, float],
        future_projection_ids: List[str],
        domain: str,
        urgency: float,
        metadata: Optional[Dict] = None,
    ) -> TemporalContext:
        context = TemporalContext(
            context_id=uuid.uuid4().hex[:12],
            past_patterns=past_pattern_ids,
            present_factors=present_factors,
            future_projections=future_projection_ids,
            domain=domain,
            urgency=urgency,
            metadata=metadata or {},
        )
        with self._lock:
            if len(self._contexts) >= _MAX_CONTEXTS:
                evicted_id = self._context_order.popleft()
                self._contexts.pop(evicted_id, None)
                logger.warning("Evicted temporal context %s", evicted_id)
            self._contexts[context.context_id] = context
            self._context_order.append(context.context_id)
            with self._stats_lock:
                self._stats["contexts_created"] += 1
        return copy.deepcopy(context)

    def synthesize(self, context_id: str) -> SynthesisDecision:
        cid = _sanitize(context_id, 50)
        with self._lock:
            context = self._contexts.get(cid)
            if not context:
                return SynthesisDecision(
                    decision_id=uuid.uuid4().hex[:12],
                    context=cid,
                    past_weight=0.33,
                    present_weight=0.34,
                    future_weight=0.33,
                    synthesized_score=0.0,
                    contributing_patterns=[],
                    contributing_projections=[],
                    rationale="No context found",
                    confidence=0.0,
                )
            past_patterns = [self._past_patterns.get(pid) for pid in context.past_patterns]
            past_patterns = [p for p in past_patterns if p is not None]
            future_proj = [self._future_projections.get(pid) for pid in context.future_projections]
            future_proj = [p for p in future_proj if p is not None]
            present_weight = _clamp(sum(context.present_factors.values()) / max(1, len(context.present_factors)))
            past_score = 0.0
            past_contributing: List[str] = []
            for p in past_patterns:
                weighted = p.recency_weight * p.confidence * min(1.0, p.frequency / 5.0)
                past_score += weighted
                past_contributing.append(p.pattern_id)
            past_score = _clamp(past_score / max(1, len(past_patterns)))
            future_score = 0.0
            future_contributing: List[str] = []
            for proj in future_proj:
                weighted = proj.probability * proj.confidence
                future_score += weighted
                future_contributing.append(proj.projection_id)
            future_score = _clamp(future_score / max(1, len(future_proj)))
            urgency = context.urgency
            past_w = _clamp(0.3 + 0.1 * (1.0 - urgency))
            present_w = _clamp(0.4)
            future_w = _clamp(0.3 + 0.1 * urgency)
            total_w = past_w + present_w + future_w
            if total_w > 0:
                past_w /= total_w
                present_w /= total_w
                future_w /= total_w
            synthesized = _clamp(
                past_w * past_score + present_w * present_weight + future_w * future_score
            )
            confidence = _clamp(0.5 + 0.3 * min(1.0, len(past_patterns) / 3.0) + 0.2 * min(1.0, len(future_proj) / 2.0))
            if synthesized >= 0.7:
                rationale = f"Strong synthesis: past({len(past_patterns)}) + present({len(context.present_factors)}) + future({len(future_proj)}). High confidence decision."
            elif synthesized >= 0.4:
                rationale = f"Moderate synthesis: insufficient evidence from one or more temporal layers. Consider gathering more data."
            else:
                rationale = f"Weak synthesis: limited temporal data. Past({len(past_patterns)}), future({len(future_proj)}). Proceed with caution."
            decision = SynthesisDecision(
                decision_id=uuid.uuid4().hex[:12],
                context=cid,
                past_weight=past_w,
                present_weight=present_w,
                future_weight=future_w,
                synthesized_score=synthesized,
                contributing_patterns=past_contributing[:20],
                contributing_projections=future_contributing[:20],
                rationale=rationale,
                confidence=confidence,
            )
            if len(self._decisions) >= _MAX_DECISIONS:
                evicted_id = self._decision_order.popleft()
                self._decisions.pop(evicted_id, None)
                logger.warning("Evicted synthesis decision %s", evicted_id)
            self._decisions[decision.decision_id] = decision
            self._decision_order.append(decision.decision_id)
            with self._stats_lock:
                n = self._stats["decisions_synthesized"] + 1
                self._stats["decisions_synthesized"] = n
                self._stats["avg_synthesized_score"] = round(
                    (self._stats["avg_synthesized_score"] * (n - 1) + synthesized) / n, 4
                )
                self._stats["avg_confidence"] = round(
                    (self._stats["avg_confidence"] * (n - 1) + confidence) / n, 4
                )
        return copy.deepcopy(decision)

    def get_past_pattern(self, pattern_id: str) -> Optional[PastPattern]:
        with self._lock:
            p = self._past_patterns.get(_sanitize(pattern_id, 50))
            return copy.deepcopy(p) if p else None

    def get_future_projection(self, projection_id: str) -> Optional[FutureProjection]:
        with self._lock:
            p = self._future_projections.get(_sanitize(projection_id, 50))
            return copy.deepcopy(p) if p else None

    def get_decision(self, decision_id: str) -> Optional[SynthesisDecision]:
        with self._lock:
            d = self._decisions.get(_sanitize(decision_id, 50))
            return copy.deepcopy(d) if d else None

    def list_decisions(self, limit: int = 100, offset: int = 0) -> List[SynthesisDecision]:
        limit = max(1, min(500, limit))
        offset = max(0, offset)
        with self._lock:
            dids = list(self._decision_order)[offset:offset + limit]
            return [copy.deepcopy(self._decisions[did]) for did in dids if did in self._decisions]

    def list_past_patterns(self, domain: Optional[str] = None, limit: int = 100) -> List[PastPattern]:
        limit = max(1, min(500, limit))
        with self._lock:
            if domain:
                dom = _sanitize(domain, 100)
                results = [p for p in self._past_patterns.values() if p.domain == dom]
            else:
                results = list(self._past_patterns.values())
            return [copy.deepcopy(r) for r in results[:limit]]

    def list_future_projections(self, event_type: Optional[str] = None, limit: int = 100) -> List[FutureProjection]:
        limit = max(1, min(500, limit))
        with self._lock:
            if event_type:
                et = _sanitize(event_type, 100)
                results = [p for p in self._future_projections.values() if p.event_type == et]
            else:
                results = list(self._future_projections.values())
            return [copy.deepcopy(r) for r in results[:limit]]

    def compute_past_weight(self, pattern_id: str, current_time: Optional[str] = None) -> float:
        pid = _sanitize(pattern_id, 50)
        with self._lock:
            pattern = self._past_patterns.get(pid)
            if not pattern:
                return 0.0
            now = datetime.now(timezone.utc)
            if current_time:
                try:
                    now = datetime.fromisoformat(current_time)
                except (ValueError, TypeError):
                    pass
            try:
                created = datetime.fromisoformat(pattern.created_at)
                age_hours = max(0.0, (now - created).total_seconds() / 3600.0)
            except (ValueError, TypeError):
                age_hours = 0.0
            decay = _clamp(1.0 / (1.0 + 0.01 * age_hours))
            return round(pattern.confidence * pattern.recency_weight * decay, 4)

    def compute_future_weight(self, projection_id: str, current_time: Optional[str] = None) -> float:
        prid = _sanitize(projection_id, 50)
        with self._lock:
            proj = self._future_projections.get(prid)
            if not proj:
                return 0.0
            timeframe_map = {
                "immediate": 0.5,
                "short_term": 1.0,
                "medium_term": 2.0,
                "long_term": 5.0,
            }
            hours = timeframe_map.get(proj.timeframe, 24.0)
            try:
                proximity = _clamp(hours / (hours + 24.0))
            except (ZeroDivisionError, ValueError):
                proximity = 0.5
            return round(proj.probability * proj.confidence * proximity, 4)

    def get_stats(self) -> Dict:
        with self._lock:
            stats = dict(self._stats)
            stats["past_pattern_count"] = len(self._past_patterns)
            stats["future_projection_count"] = len(self._future_projections)
            stats["decision_count"] = len(self._decisions)
            stats["context_count"] = len(self._contexts)
            return stats