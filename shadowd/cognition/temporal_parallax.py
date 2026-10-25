#!/usr/bin/env python3
"""
TemporalParallax -- Multi-Perspective Temporal Analysis Engine.

Views events from multiple temporal "parallax angles" -- different time
perspectives (immediate, short-term, medium-term, long-term, generational).
Measures how event significance changes with temporal distance.

Key concepts:
  - TemporalEvent: an event with raw impact score
  - PerspectiveView: how an event appears from a specific temporal distance
  - ParallaxMeasurement: aggregate analysis across all temporal perspectives
  - TemporalInsight: discovered patterns about significance shifts

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
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.temporal_parallax")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_EVENTS = 2000
_MAX_INSIGHTS = 500
_MAX_VIEWS_PER_EVENT = 5
_MAX_TAGS = 20

_TEMPORAL_DECAY = {
    1: 1.0,
    2: 0.85,
    3: 0.65,
    4: 0.45,
    5: 0.25,
}

_EMERGENCE_BOOST = {
    1: 0.0,
    2: 0.05,
    3: 0.15,
    4: 0.30,
    5: 0.50,
}

_IMPACT_THRESHOLDS = {
    "insignificant": (0.0, 0.2),
    "minor": (0.2, 0.4),
    "moderate": (0.4, 0.6),
    "significant": (0.6, 0.8),
    "transformative": (0.8, 1.0),
}


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _classify_impact(impact: float) -> str:
    if impact < 0.2:
        return "insignificant"
    elif impact < 0.4:
        return "minor"
    elif impact < 0.6:
        return "moderate"
    elif impact < 0.8:
        return "significant"
    else:
        return "transformative"


def _determine_shift(immediate_impact: float, distant_impact: float) -> str:
    if (immediate_impact < 0.3 and distant_impact > 0.5):
        return "emergent"
    elif (immediate_impact > 0.5 and distant_impact < 0.3):
        return "reversed"
    elif distant_impact > immediate_impact + 0.1:
        return "amplified"
    elif distant_impact < immediate_impact - 0.1:
        return "diminished"
    else:
        return "maintained"


class TemporalDistance(IntEnum):
    IMMEDIATE = 1
    SHORT_TERM = 2
    MEDIUM_TERM = 3
    LONG_TERM = 4
    GENERATIONAL = 5


class ImpactCategory(str, Enum):
    INSIGNIFICANT = "insignificant"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    TRANSFORMATIVE = "transformative"


class PerspectiveShift(str, Enum):
    AMPLIFIED = "amplified"
    MAINTAINED = "maintained"
    DIMINISHED = "diminished"
    REVERSED = "reversed"
    EMERGENT = "emergent"


@dataclass
class TemporalEvent:
    event_id: str = ""
    event_type: str = ""
    description: str = ""
    domain: str = ""
    raw_impact: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.event_id:
            self.event_id = "te_%s" % uuid.uuid4().hex[:12]
        self.event_id = _sanitize(self.event_id, 64)
        self.event_type = _sanitize(self.event_type, 100)
        self.description = _sanitize_content(self.description, 10000)
        self.domain = _sanitize(self.domain, 100)
        self.raw_impact = _clamp(self.raw_impact)

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "description": self.description,
            "domain": self.domain,
            "raw_impact": round(self.raw_impact, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class PerspectiveView:
    view_id: str = ""
    event_id: str = ""
    temporal_distance: int = 1
    perceived_impact: float = 0.5
    impact_category: str = "moderate"
    confidence: float = 0.5
    reasoning: str = ""

    def __post_init__(self):
        if not self.view_id:
            self.view_id = "pv_%s" % uuid.uuid4().hex[:12]
        self.view_id = _sanitize(self.view_id, 64)
        self.event_id = _sanitize(self.event_id, 64)
        if isinstance(self.temporal_distance, TemporalDistance):
            self.temporal_distance = int(self.temporal_distance)
        self.temporal_distance = max(1, min(5, int(self.temporal_distance)))
        self.perceived_impact = _clamp(self.perceived_impact)
        if isinstance(self.impact_category, ImpactCategory):
            self.impact_category = self.impact_category.value
        self.impact_category = _sanitize(self.impact_category, 50)
        self.confidence = _clamp(self.confidence)
        self.reasoning = _sanitize_content(self.reasoning, 2000)

    def to_dict(self) -> Dict:
        return {
            "view_id": self.view_id,
            "event_id": self.event_id,
            "temporal_distance": self.temporal_distance,
            "perceived_impact": round(self.perceived_impact, 4),
            "impact_category": self.impact_category,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
        }


@dataclass
class ParallaxMeasurement:
    measurement_id: str = ""
    event_id: str = ""
    views: Dict[int, Any] = field(default_factory=dict)
    parallax_score: float = 0.0
    consensus_impact: float = 0.5
    dominant_shift: str = "maintained"

    def __post_init__(self):
        if not self.measurement_id:
            self.measurement_id = "pm_%s" % uuid.uuid4().hex[:12]
        self.measurement_id = _sanitize(self.measurement_id, 64)
        self.event_id = _sanitize(self.event_id, 64)
        self.parallax_score = _clamp(self.parallax_score)
        self.consensus_impact = _clamp(self.consensus_impact)
        if isinstance(self.dominant_shift, PerspectiveShift):
            self.dominant_shift = self.dominant_shift.value
        self.dominant_shift = _sanitize(self.dominant_shift, 50)

    def to_dict(self) -> Dict:
        return {
            "measurement_id": self.measurement_id,
            "event_id": self.event_id,
            "views": {str(k): v.to_dict() if hasattr(v, 'to_dict') else copy.deepcopy(v)
                      for k, v in self.views.items()},
            "parallax_score": round(self.parallax_score, 4),
            "consensus_impact": round(self.consensus_impact, 4),
            "dominant_shift": self.dominant_shift,
        }


@dataclass
class TemporalInsight:
    insight_id: str = ""
    event_id: str = ""
    insight_type: str = ""
    description: str = ""
    confidence: float = 0.5
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.insight_id:
            self.insight_id = "ti_%s" % uuid.uuid4().hex[:12]
        self.insight_id = _sanitize(self.insight_id, 64)
        self.event_id = _sanitize(self.event_id, 64)
        self.insight_type = _sanitize(self.insight_type, 100)
        self.description = _sanitize_content(self.description, 5000)
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> Dict:
        return {
            "insight_id": self.insight_id,
            "event_id": self.event_id,
            "insight_type": self.insight_type,
            "description": self.description,
            "confidence": round(self.confidence, 4),
            "discovered_at": self.discovered_at,
        }


class TemporalParallax:
    """Multi-Perspective Temporal Analysis Engine.

    Views events from multiple temporal parallax angles -- immediate,
    short-term, medium-term, long-term, and generational perspectives.
    Measures how event significance shifts with temporal distance.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._events: Dict[str, TemporalEvent] = {}
        self._event_order: deque = deque(maxlen=_MAX_EVENTS)
        self._views: Dict[str, Dict[int, PerspectiveView]] = {}
        self._measurements: Dict[str, ParallaxMeasurement] = {}
        self._insights: Dict[str, TemporalInsight] = {}
        self._insight_order: deque = deque(maxlen=_MAX_INSIGHTS)
        self._stats = {
            "events_registered": 0,
            "parallax_computed": 0,
            "perspectives_computed": 0,
            "insights_discovered": 0,
            "shifts_identified": 0,
        }

    def register_event(self, event_type: str, description: str,
                       domain: str = "", raw_impact: float = 0.5) -> TemporalEvent:
        event_type = _sanitize(event_type, 100)
        description = _sanitize_content(description, 10000)
        domain = _sanitize(domain, 100)
        raw_impact = _clamp(raw_impact)

        event = TemporalEvent(
            event_id="te_%s" % uuid.uuid4().hex[:12],
            event_type=event_type,
            description=description,
            domain=domain,
            raw_impact=raw_impact,
        )

        with self._lock:
            if len(self._events) >= _MAX_EVENTS:
                evict_id = self._event_order.popleft()
                self._events.pop(evict_id, None)
                self._views.pop(evict_id, None)
                self._measurements.pop(evict_id, None)
                logger.warning("Evicting event %s (cap=%d)", evict_id, _MAX_EVENTS)
            self._events[event.event_id] = event
            self._event_order.append(event.event_id)
            with self._stats_lock:
                self._stats["events_registered"] += 1

        return copy.deepcopy(event)

    def compute_parallax(self, event_id: str) -> Optional[ParallaxMeasurement]:
        eid = _sanitize(event_id, 64)

        with self._lock:
            event = self._events.get(eid)
            if not event:
                return None

            views = {}
            for distance in TemporalDistance:
                d = int(distance)
                decay_factor = _TEMPORAL_DECAY.get(d, 1.0)
                emergence_boost = _EMERGENCE_BOOST.get(d, 0.0)

                base_perception = event.raw_impact * decay_factor
                domain_keywords = re.findall(r"[a-z][a-z'\-]*", event.description.lower()) if event.description else []
                domain_factor = 0.0
                if any(kw in ("war", "pandemic", "revolution", "discovery",
                              "invention", "breakthrough", "catastrophe",
                              "collapse", "founding", "extinction")
                       for kw in domain_keywords):
                    domain_factor = 0.1

                generational_relevance = emergence_boost if event.raw_impact > 0.3 else emergence_boost * 0.5
                perceived = base_perception + generational_relevance + domain_factor

                if d >= 3 and event.raw_impact < 0.3:
                    perceived = max(perceived, event.raw_impact * 0.4)

                perceived = _clamp(perceived)
                category = _classify_impact(perceived)
                confidence = _clamp(1.0 - (0.05 * d) + (event.raw_impact * 0.2))

                if d == TemporalDistance.IMMEDIATE:
                    reasoning = "Direct, unfiltered impact assessment"
                elif d == TemporalDistance.SHORT_TERM:
                    reasoning = "Near-term consequences begin to emerge"
                elif d == TemporalDistance.MEDIUM_TERM:
                    reasoning = "Medium-range implications shape broader understanding"
                elif d == TemporalDistance.LONG_TERM:
                    reasoning = "Long-term structural effects become visible"
                else:
                    reasoning = "Generational significance emerges across eras"

                view = PerspectiveView(
                    view_id="pv_%s" % uuid.uuid4().hex[:12],
                    event_id=eid,
                    temporal_distance=d,
                    perceived_impact=perceived,
                    impact_category=category,
                    confidence=confidence,
                    reasoning=reasoning,
                )
                views[d] = view

            if len(views) < 2:
                return None

            impacts = [v.perceived_impact for v in views.values()]
            min_impact = min(impacts)
            max_impact = max(impacts)
            parallax_score = _clamp(max_impact - min_impact)

            confidence_weighted_sum = sum(
                v.perceived_impact * v.confidence for v in views.values()
            )
            confidence_sum = sum(v.confidence for v in views.values())
            consensus_impact = _clamp(
                confidence_weighted_sum / max(confidence_sum, 0.01)
            )

            immediate_view = views.get(int(TemporalDistance.IMMEDIATE))
            generational_view = views.get(int(TemporalDistance.GENERATIONAL))

            if immediate_view and generational_view:
                dominant_shift = _determine_shift(
                    immediate_view.perceived_impact,
                    generational_view.perceived_impact,
                )
            else:
                dominant_shift = "maintained"

            measurement = ParallaxMeasurement(
                measurement_id="pm_%s" % uuid.uuid4().hex[:12],
                event_id=eid,
                views=views,
                parallax_score=parallax_score,
                consensus_impact=consensus_impact,
                dominant_shift=dominant_shift,
            )
            self._views[eid] = views
            self._measurements[eid] = measurement

            if parallax_score > 0.3:
                insight_type = "emergence" if dominant_shift in ("amplified", "emergent") else \
                              "diminution" if dominant_shift in ("diminished", "reversed") else \
                              "shift"
                insight = TemporalInsight(
                    insight_id="ti_%s" % uuid.uuid4().hex[:12],
                    event_id=eid,
                    insight_type=insight_type,
                    description="Event '%s' shows %s temporal parallax (score=%.3f)" % (
                        event.event_type, dominant_shift, parallax_score
                    ),
                    confidence=_clamp(parallax_score * 0.8 + 0.1),
                )
                if len(self._insights) >= _MAX_INSIGHTS:
                    evict_id = self._insight_order.popleft()
                    self._insights.pop(evict_id, None)
                    logger.warning("Evicting insight %s (cap=%d)", evict_id, _MAX_INSIGHTS)
                self._insights[insight.insight_id] = insight
                self._insight_order.append(insight.insight_id)
                with self._stats_lock:
                    self._stats["insights_discovered"] += 1

            with self._stats_lock:
                self._stats["parallax_computed"] += 1
                self._stats["perspectives_computed"] += len(views)
                self._stats["shifts_identified"] += 1

        return copy.deepcopy(measurement)

    def compare_perspectives(self, event_id: str, distance_a: int,
                            distance_b: int) -> Optional[Dict]:
        eid = _sanitize(event_id, 64)
        distance_a = max(1, min(5, int(distance_a)))
        distance_b = max(1, min(5, int(distance_b)))

        with self._lock:
            views = self._views.get(eid)
            if not views:
                return None

            view_a = views.get(distance_a)
            view_b = views.get(distance_b)

            if not view_a or not view_b:
                return None

            impact_diff = abs(view_a.perceived_impact - view_b.perceived_impact)
            confidence_diff = abs(view_a.confidence - view_b.confidence)

            if view_a.perceived_impact > view_b.perceived_impact + 0.1:
                direction = "a_higher"
            elif view_b.perceived_impact > view_a.perceived_impact + 0.1:
                direction = "b_higher"
            else:
                direction = "similar"

            return {
                "distance_a": distance_a,
                "distance_b": distance_b,
                "impact_a": round(view_a.perceived_impact, 4),
                "impact_b": round(view_b.perceived_impact, 4),
                "impact_difference": round(impact_diff, 4),
                "confidence_difference": round(confidence_diff, 4),
                "category_a": view_a.impact_category,
                "category_b": view_b.impact_category,
                "direction": direction,
            }

    def identify_shifts(self, min_parallax_score: float = 0.2) -> List[Dict]:
        min_parallax_score = _clamp(min_parallax_score, 0.0, 1.0)

        with self._lock:
            shifts = []
            for eid, measurement in self._measurements.items():
                if measurement.parallax_score >= min_parallax_score:
                    event = self._events.get(eid)
                    if event:
                        shifts.append({
                            "event_id": eid,
                            "event_type": event.event_type,
                            "domain": event.domain,
                            "parallax_score": round(measurement.parallax_score, 4),
                            "dominant_shift": measurement.dominant_shift,
                            "consensus_impact": round(measurement.consensus_impact, 4),
                        })

            shifts.sort(key=lambda x: x["parallax_score"], reverse=True)
            return [copy.deepcopy(s) for s in shifts[:100]]

    def get_dominant_perspective(self, event_id: str) -> Optional[Dict]:
        eid = _sanitize(event_id, 64)

        with self._lock:
            views = self._views.get(eid)
            if not views:
                return None

            best_view = max(views.values(), key=lambda v: v.confidence)

            return {
                "event_id": eid,
                "dominant_distance": best_view.temporal_distance,
                "perceived_impact": round(best_view.perceived_impact, 4),
                "confidence": round(best_view.confidence, 4),
                "impact_category": best_view.impact_category,
                "reasoning": best_view.reasoning,
            }

    def find_emergent_significance(self, min_impact: float = 0.5) -> List[Dict]:
        min_impact = _clamp(min_impact, 0.0, 1.0)

        with self._lock:
            results = []
            for eid, views in self._views.items():
                if not views:
                    continue

                immediate_view = views.get(int(TemporalDistance.IMMEDIATE))
                long_term_views = [v for v in views.values()
                                    if v.temporal_distance >= int(TemporalDistance.LONG_TERM)]

                if not immediate_view or not long_term_views:
                    continue

                max_long_impact = max(v.perceived_impact for v in long_term_views)

                if immediate_view.perceived_impact < 0.3 and max_long_impact >= min_impact:
                    event = self._events.get(eid)
                    if event:
                        results.append({
                            "event_id": eid,
                            "event_type": event.event_type,
                            "domain": event.domain,
                            "immediate_impact": round(immediate_view.perceived_impact, 4),
                            "long_term_impact": round(max_long_impact, 4),
                            "emergence_factor": round(
                                max_long_impact - immediate_view.perceived_impact, 4
                            ),
                        })

            results.sort(key=lambda x: x["emergence_factor"], reverse=True)
            return [copy.deepcopy(r) for r in results[:100]]

    def find_diminishing_events(self, min_impact: float = 0.5) -> List[Dict]:
        min_impact = _clamp(min_impact, 0.0, 1.0)

        with self._lock:
            results = []
            for eid, views in self._views.items():
                if not views:
                    continue

                immediate_view = views.get(int(TemporalDistance.IMMEDIATE))
                long_term_views = [v for v in views.values()
                                    if v.temporal_distance >= int(TemporalDistance.LONG_TERM)]

                if not immediate_view or not long_term_views:
                    continue

                max_long_impact = max(v.perceived_impact for v in long_term_views)

                if immediate_view.perceived_impact >= min_impact and max_long_impact < 0.3:
                    event = self._events.get(eid)
                    if event:
                        results.append({
                            "event_id": eid,
                            "event_type": event.event_type,
                            "domain": event.domain,
                            "immediate_impact": round(immediate_view.perceived_impact, 4),
                            "long_term_impact": round(max_long_impact, 4),
                            "diminution_factor": round(
                                immediate_view.perceived_impact - max_long_impact, 4
                            ),
                        })

            results.sort(key=lambda x: x["diminution_factor"], reverse=True)
            return [copy.deepcopy(r) for r in results[:100]]

    def get_event(self, event_id: str) -> Optional[Dict]:
        eid = _sanitize(event_id, 64)

        with self._lock:
            event = self._events.get(eid)
            if not event:
                return None

            views = self._views.get(eid, {})
            measurement = self._measurements.get(eid)

            return {
                "event": copy.deepcopy(event).to_dict(),
                "views": {str(k): copy.deepcopy(v).to_dict()
                          for k, v in views.items()},
                "measurement": copy.deepcopy(measurement).to_dict() if measurement else None,
            }

    def list_events(self, domain: str = "", limit: int = 50) -> List[Dict]:
        domain = _sanitize(domain, 100)
        limit = max(1, min(500, limit))

        with self._lock:
            events = []
            for eid in reversed(list(self._event_order)):
                event = self._events.get(eid)
                if not event:
                    continue
                if domain and event.domain != domain:
                    continue
                events.append(copy.deepcopy(event).to_dict())
                if len(events) >= limit:
                    break

        return events

    def get_stats(self) -> Dict:
        with self._lock:
            total_events = len(self._events)
            total_views = sum(len(v) for v in self._views.values())
            total_measurements = len(self._measurements)
            total_insights = len(self._insights)

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_events"] = total_events
        stats["total_views"] = total_views
        stats["total_measurements"] = total_measurements
        stats["total_insights"] = total_insights
        return stats