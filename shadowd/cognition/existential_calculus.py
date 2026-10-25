#!/usr/bin/env python3
"""
ExistentialCalculus — Meaning-Making and Significance.

Computes significance and meaning of events, decisions, and patterns
through multi-axial evaluation across purpose, impact, legacy,
authenticity, contribution, growth, and connection axes.

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

logger = logging.getLogger("shadowd.cognition.existential_calculus")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_FRAMES = 3000
_MAX_ALIGNMENTS = 500
_MAX_THEMES = 200
_MAX_RESULTS = 2000
_MAX_METADATA_KEYS = 20
_MAX_REASON_LENGTH = 500
_MAX_FRAMES_PER_EVENT = 50


class SignificanceAxis(str, Enum):
    PURPOSE = "purpose"
    IMPACT = "impact"
    LEGACY = "legacy"
    AUTHENTICITY = "authenticity"
    CONTRIBUTION = "contribution"
    GROWTH = "growth"
    CONNECTION = "connection"


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
class MeaningFrame:
    frame_id: str
    event_id: str
    axis_scores: Dict[str, float]
    overall_significance: float
    narrative: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.frame_id:
            self.frame_id = uuid.uuid4().hex[:12]
        self.frame_id = _sanitize(self.frame_id, 50)
        self.event_id = _sanitize(self.event_id, 100)
        cleaned: Dict[str, float] = {}
        for k, v in list(self.axis_scores.items())[:len(SignificanceAxis)]:
            cleaned[_sanitize(k, 50)] = _clamp(v)
        self.axis_scores = cleaned
        self.overall_significance = _clamp(self.overall_significance)
        self.narrative = _sanitize_content(self.narrative, _MAX_REASON_LENGTH)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "event_id": self.event_id,
            "axis_scores": {k: round(v, 4) for k, v in self.axis_scores.items()},
            "overall_significance": round(self.overall_significance, 4),
            "narrative": self.narrative,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class PurposeAlignment:
    alignment_id: str
    purpose_statement: str
    current_activities: List[str]
    alignment_score: float
    gap_areas: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.alignment_id:
            self.alignment_id = uuid.uuid4().hex[:12]
        self.alignment_id = _sanitize(self.alignment_id, 50)
        self.purpose_statement = _sanitize_content(self.purpose_statement, 1000)
        self.current_activities = [_sanitize_content(a, 300) for a in (self.current_activities or [])[:20]]
        self.alignment_score = _clamp(self.alignment_score)
        self.gap_areas = [_sanitize_content(g, 200) for g in (self.gap_areas or [])[:10]]
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "alignment_id": self.alignment_id,
            "purpose_statement": self.purpose_statement,
            "current_activities": list(self.current_activities),
            "alignment_score": round(self.alignment_score, 4),
            "gap_areas": list(self.gap_areas),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class LifeTheme:
    theme_id: str
    theme: str
    frequency: int
    significance: float
    related_axes: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.theme_id:
            self.theme_id = uuid.uuid4().hex[:12]
        self.theme_id = _sanitize(self.theme_id, 50)
        self.theme = _sanitize_content(self.theme, 500)
        self.frequency = max(1, int(self.frequency))
        self.significance = _clamp(self.significance)
        self.related_axes = [_sanitize(a, 50) for a in (self.related_axes or [])[:len(SignificanceAxis)]]
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "theme_id": self.theme_id,
            "theme": self.theme,
            "frequency": self.frequency,
            "significance": round(self.significance, 4),
            "related_axes": list(self.related_axes),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class CalculusResult:
    result_id: str
    event_id: str
    significance: float
    meaning: str
    axis_breakdown: Dict[str, float]
    recommended_action: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.result_id:
            self.result_id = uuid.uuid4().hex[:12]
        self.result_id = _sanitize(self.result_id, 50)
        self.event_id = _sanitize(self.event_id, 100)
        self.significance = _clamp(self.significance)
        self.meaning = _sanitize_content(self.meaning, _MAX_REASON_LENGTH)
        self.axis_breakdown = {k: _clamp(v) for k, v in list(self.axis_breakdown.items())[:len(SignificanceAxis)]}
        self.recommended_action = _sanitize_content(self.recommended_action, 500)
        self.confidence = _clamp(self.confidence)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "event_id": self.event_id,
            "significance": round(self.significance, 4),
            "meaning": self.meaning,
            "axis_breakdown": {k: round(v, 4) for k, v in self.axis_breakdown.items()},
            "recommended_action": self.recommended_action,
            "confidence": round(self.confidence, 4),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


class ExistentialCalculus:
    """Meaning-making engine: evaluates significance across multiple axes
    and detects recurring life themes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._frames: Dict[str, MeaningFrame] = {}
        self._frames_by_event: Dict[str, List[str]] = {}
        self._alignments: Dict[str, PurposeAlignment] = {}
        self._themes: Dict[str, LifeTheme] = {}
        self._results: Dict[str, CalculusResult] = {}
        self._frame_order: deque = deque(maxlen=_MAX_FRAMES)
        self._alignment_order: deque = deque(maxlen=_MAX_ALIGNMENTS)
        self._theme_order: deque = deque(maxlen=_MAX_THEMES)
        self._result_order: deque = deque(maxlen=_MAX_RESULTS)
        self._stats = {
            "frames_created": 0,
            "alignments_created": 0,
            "themes_detected": 0,
            "results_computed": 0,
            "avg_significance": 0.0,
            "avg_confidence": 0.0,
        }

    def evaluate_significance(
        self,
        event_id: str,
        description: str,
        axis_weights: Optional[Dict[SignificanceAxis, float]] = None,
    ) -> CalculusResult:
        eid = _sanitize(event_id, 100)
        desc = _sanitize_content(description, _MAX_REASON_LENGTH)
        weights: Dict[str, float] = {}
        if axis_weights:
            for axis, w in axis_weights.items():
                key = axis.value if isinstance(axis, SignificanceAxis) else str(axis).lower()
                weights[key] = _clamp(w)
        if not weights:
            weights = {a.value: 1.0 / len(SignificanceAxis) for a in SignificanceAxis}
        axis_scores: Dict[str, float] = {}
        desc_lower = desc.lower()
        purpose_score = self._score_axis(desc_lower, "purpose", [
            "purpose", "meaning", "mission", "calling", "destiny",
            "why", "reason for being", "life's work",
        ])
        impact_score = self._score_axis(desc_lower, "impact", [
            "impact", "affect", "influence", "change", "difference",
            "consequence", "ripple", "transform",
        ])
        legacy_score = self._score_axis(desc_lower, "legacy", [
            "legacy", "remember", "lasting", "enduring", "future",
            "generation", "leave behind", "inheritance",
        ])
        authenticity_score = self._score_axis(desc_lower, "authenticity", [
            "authentic", "genuine", "true self", "real", "honest",
            "integrity", "values", "beliefs",
        ])
        contribution_score = self._score_axis(desc_lower, "contribution", [
            "contribute", "give", "help", "serve", "support",
            "volunteer", "donate", "share",
        ])
        growth_score = self._score_axis(desc_lower, "growth", [
            "growth", "learn", "develop", "improve", "evolve",
            "become", "progress", "overcome",
        ])
        connection_score = self._score_axis(desc_lower, "connection", [
            "connect", "together", "relationship", "community", "belong",
            "bond", "friendship", "love",
        ])
        axis_scores = {
            SignificanceAxis.PURPOSE.value: purpose_score,
            SignificanceAxis.IMPACT.value: impact_score,
            SignificanceAxis.LEGACY.value: legacy_score,
            SignificanceAxis.AUTHENTICITY.value: authenticity_score,
            SignificanceAxis.CONTRIBUTION.value: contribution_score,
            SignificanceAxis.GROWTH.value: growth_score,
            SignificanceAxis.CONNECTION.value: connection_score,
        }
        weighted_sum = sum(
            axis_scores.get(a.value, 0.0) * weights.get(a.value, 1.0 / len(SignificanceAxis))
            for a in SignificanceAxis
        )
        weight_total = sum(weights.get(a.value, 1.0 / len(SignificanceAxis)) for a in SignificanceAxis)
        overall = _clamp(weighted_sum / weight_total) if weight_total > 0 else 0.0
        confidence = _clamp(0.5 + 0.5 * min(1.0, len(desc) / 200.0))
        meaningful_axes = [a for a, s in axis_scores.items() if s >= 0.4]
        if overall >= 0.7:
            meaning = f"Highly significant across {', '.join(meaningful_axes[:3])}"
            recommended = "Deepen engagement and invest resources"
        elif overall >= 0.4:
            meaning = f"Moderately significant for {', '.join(meaningful_axes[:2])}"
            recommended = "Consider strategic alignment with core values"
        else:
            meaning = "Limited significance in current context"
            recommended = "Monitor for pattern emergence"
        result = CalculusResult(
            result_id=uuid.uuid4().hex[:12],
            event_id=eid,
            significance=overall,
            meaning=meaning,
            axis_breakdown=axis_scores,
            recommended_action=recommended,
            confidence=confidence,
        )
        with self._lock:
            if len(self._results) >= _MAX_RESULTS:
                evicted_id = self._result_order.popleft()
                self._results.pop(evicted_id, None)
                logger.warning("Evicted calculus result %s", evicted_id)
            self._results[result.result_id] = result
            self._result_order.append(result.result_id)
            with self._stats_lock:
                self._stats["results_computed"] += 1
                n = self._stats["results_computed"]
                self._stats["avg_significance"] = round(
                    (self._stats["avg_significance"] * (n - 1) + overall) / n, 4
                )
                self._stats["avg_confidence"] = round(
                    (self._stats["avg_confidence"] * (n - 1) + confidence) / n, 4
                )
        return copy.deepcopy(result)

    def compute_purpose_alignment(
        self,
        purpose: str,
        activities: List[str],
    ) -> PurposeAlignment:
        purpose_clean = _sanitize_content(purpose, 1000)
        activities_clean = [_sanitize_content(a, 300) for a in (activities or [])[:20]]
        purpose_lower = purpose_clean.lower()
        purpose_words = set(re.findall(r'[a-z]+', purpose_lower))
        if not purpose_words:
            purpose_words = {"purpose"}
        alignment_scores: List[float] = []
        gap_areas: List[str] = []
        for activity in activities_clean:
            act_words = set(re.findall(r'[a-z]+', activity.lower()))
            overlap = len(purpose_words & act_words)
            total = max(1, len(purpose_words))
            score = _clamp(overlap / total)
            alignment_scores.append(score)
            if score < 0.2:
                gap_areas.append(activity)
        alignment_score = _clamp(
            sum(alignment_scores) / max(1, len(alignment_scores))
        ) if alignment_scores else 0.0
        alignment = PurposeAlignment(
            alignment_id=uuid.uuid4().hex[:12],
            purpose_statement=purpose_clean,
            current_activities=activities_clean,
            alignment_score=alignment_score,
            gap_areas=gap_areas[:10],
        )
        with self._lock:
            if len(self._alignments) >= _MAX_ALIGNMENTS:
                evicted_id = self._alignment_order.popleft()
                self._alignments.pop(evicted_id, None)
                logger.warning("Evicted purpose alignment %s", evicted_id)
            self._alignments[alignment.alignment_id] = alignment
            self._alignment_order.append(alignment.alignment_id)
            with self._stats_lock:
                self._stats["alignments_created"] += 1
        return copy.deepcopy(alignment)

    def detect_life_themes(self, min_frequency: int = 2) -> List[LifeTheme]:
        min_freq = max(1, int(min_frequency))
        event_themes: Dict[str, int] = {}
        event_axes: Dict[str, set] = {}
        with self._lock:
            for frame in self._frames.values():
                theme_words = re.findall(r'[a-z]+', frame.narrative.lower())
                for word in theme_words:
                    if len(word) > 3:
                        event_themes[word] = event_themes.get(word, 0) + 1
                        if word not in event_axes:
                            event_axes[word] = set()
                        for axis_name in frame.axis_scores:
                            if frame.axis_scores[axis_name] >= 0.3:
                                event_axes[word].add(axis_name)
        themes: List[LifeTheme] = []
        for word, freq in sorted(event_themes.items(), key=lambda x: -x[1]):
            if freq >= min_freq and len(themes) < _MAX_THEMES:
                sig = _clamp(freq / 10.0)
                themes.append(
                    LifeTheme(
                        theme_id=uuid.uuid4().hex[:12],
                        theme=word,
                        frequency=freq,
                        significance=sig,
                        related_axes=list(event_axes.get(word, set()))[:len(SignificanceAxis)],
                    )
                )
        with self._lock:
            for theme in themes:
                if len(self._themes) >= _MAX_THEMES:
                    evicted_id = self._theme_order.popleft()
                    self._themes.pop(evicted_id, None)
                    logger.warning("Evicted life theme %s", evicted_id)
                self._themes[theme.theme_id] = theme
                self._theme_order.append(theme.theme_id)
            with self._stats_lock:
                self._stats["themes_detected"] = len(self._themes)
        return [copy.deepcopy(t) for t in themes]

    def add_meaning_frame(
        self,
        event_id: str,
        axis_scores: Dict[str, float],
        narrative: str,
        metadata: Optional[Dict] = None,
    ) -> MeaningFrame:
        frame = MeaningFrame(
            frame_id=uuid.uuid4().hex[:12],
            event_id=event_id,
            axis_scores=axis_scores,
            overall_significance=sum(axis_scores.values()) / max(1, len(axis_scores)),
            narrative=narrative,
            metadata=metadata or {},
        )
        with self._lock:
            if len(self._frames) >= _MAX_FRAMES:
                evicted_id = self._frame_order.popleft()
                self._frames.pop(evicted_id, None)
                evicted_event = self._frames_by_event
                for eid_key, fids in list(evicted_event.items()):
                    if evicted_id in fids:
                        fids.remove(evicted_id)
                        if not fids:
                            del evicted_event[eid_key]
                logger.warning("Evicted meaning frame %s", evicted_id)
            self._frames[frame.frame_id] = frame
            self._frame_order.append(frame.frame_id)
            self._frames_by_event.setdefault(frame.event_id, []).append(frame.frame_id)
            if len(self._frames_by_event[frame.event_id]) > _MAX_FRAMES_PER_EVENT:
                self._frames_by_event[frame.event_id] = self._frames_by_event[frame.event_id][-_MAX_FRAMES_PER_EVENT:]
            with self._stats_lock:
                self._stats["frames_created"] += 1
        return copy.deepcopy(frame)

    def get_meaning_frame(self, frame_id: str) -> Optional[MeaningFrame]:
        with self._lock:
            f = self._frames.get(_sanitize(frame_id, 50))
            return copy.deepcopy(f) if f else None

    def get_meaning_frames(self, event_id: str) -> List[MeaningFrame]:
        eid = _sanitize(event_id, 100)
        with self._lock:
            fids = self._frames_by_event.get(eid, [])
            return [copy.deepcopy(self._frames[fid]) for fid in fids if fid in self._frames]

    def get_purpose_alignments(self, limit: int = 100) -> List[PurposeAlignment]:
        limit = max(1, min(500, limit))
        with self._lock:
            aids = list(self._alignment_order)[:limit]
            return [copy.deepcopy(self._alignments[aid]) for aid in aids if aid in self._alignments]

    def get_themes(self, limit: int = 100) -> List[LifeTheme]:
        limit = max(1, min(500, limit))
        with self._lock:
            tids = list(self._theme_order)[:limit]
            return [copy.deepcopy(self._themes[tid]) for tid in tids if tid in self._themes]

    def get_result(self, result_id: str) -> Optional[CalculusResult]:
        with self._lock:
            r = self._results.get(_sanitize(result_id, 50))
            return copy.deepcopy(r) if r else None

    def list_results(self, limit: int = 100, offset: int = 0) -> List[CalculusResult]:
        limit = max(1, min(500, limit))
        offset = max(0, offset)
        with self._lock:
            rids = list(self._result_order)[offset:offset + limit]
            return [copy.deepcopy(self._results[rid]) for rid in rids if rid in self._results]

    def recommend_action(self, event_id: str) -> str:
        eid = _sanitize(event_id, 100)
        with self._lock:
            best_result: Optional[CalculusResult] = None
            best_sig = 0.0
            for result in self._results.values():
                if result.event_id == eid and result.significance > best_sig:
                    best_sig = result.significance
                    best_result = result
        if best_result:
            return best_result.recommended_action
        return "Monitor for pattern emergence"

    def get_stats(self) -> Dict:
        with self._lock:
            stats = dict(self._stats)
            stats["frame_count"] = len(self._frames)
            stats["alignment_count"] = len(self._alignments)
            stats["theme_count"] = len(self._themes)
            stats["result_count"] = len(self._results)
            return stats

    def _score_axis(self, text_lower: str, axis_name: str, keywords: List[str]) -> float:
        matches = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
        if matches == 0:
            return 0.1
        ratio = min(1.0, matches / len(keywords))
        return _clamp(0.3 + 0.5 * ratio + 0.2 * min(1.0, matches / 3.0))