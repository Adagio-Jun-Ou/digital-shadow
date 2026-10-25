#!/usr/bin/env python3
"""
CognitiveReflection — Self-Assessment, Hindsight Analysis, and Learning Journaling

Enables the cognitive system to reflect on past decisions, assess what went
well/poorly, identify learning opportunities, and build a reflection journal
that guides future behavior.

Architecture:
  - ReflectionEntry: a single reflection with before/after confidence, lessons
  - LearningInsight: extracted insight from reflection entries
  - ReflectionPattern: recurring themes detected across reflections
  - CognitiveReflection: main engine managing reflections, insights, patterns
  - Thread-safe, bounded collections, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.cognitive_reflection")


class ReflectionType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MILESTONE = "milestone"
    POST_DECISION = "post_decision"
    RETROSPECTIVE = "retrospective"
    LESSON_LEARNED = "lesson_learned"


_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')  # alias for content-safe

_LESSON_PATTERNS = [
    re.compile(r'\b(should have|could have|would have|need to)\b', re.IGNORECASE),
    re.compile(r'\b(mistake|error|wrong|failed|failure)\b', re.IGNORECASE),
    re.compile(r'\b(improved|learned|discovered|realized)\b', re.IGNORECASE),
    re.compile(r'\b(better|worse|stronger|weaker)\b', re.IGNORECASE),
    re.compile(r'\b(next time|going forward|in the future)\b', re.IGNORECASE),
]


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


@dataclass
class ReflectionEntry:
    entry_id: str
    topic: str
    description: str
    reflection_type: ReflectionType
    what_went_well: List[str]
    what_could_improve: List[str]
    lessons_learned: List[str]
    confidence_before: float
    confidence_after: float
    domain: str
    linked_decision_ids: List[str]
    sentiment: float
    created_at: str
    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            "entry_id": self.entry_id,
            "topic": self.topic,
            "description": self.description,
            "reflection_type": self.reflection_type.value,
            "what_went_well": list(self.what_went_well),
            "what_could_improve": list(self.what_could_improve),
            "lessons_learned": list(self.lessons_learned),
            "confidence_before": round(self.confidence_before, 3),
            "confidence_after": round(self.confidence_after, 3),
            "domain": self.domain,
            "linked_decision_ids": list(self.linked_decision_ids),
            "sentiment": round(self.sentiment, 3),
            "created_at": self.created_at,
            "metadata": copy.deepcopy(dict(list(self.metadata.items())[:20])),
        }


@dataclass
class LearningInsight:
    insight_id: str
    title: str
    description: str
    category: str
    source_reflection_ids: List[str]
    frequency: int
    confidence: float
    applicable_domains: List[str]
    created_at: str

    def to_dict(self) -> Dict:
        return {
            "insight_id": self.insight_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "source_reflection_ids": list(self.source_reflection_ids),
            "frequency": self.frequency,
            "confidence": round(self.confidence, 3),
            "applicable_domains": list(self.applicable_domains),
            "created_at": self.created_at,
        }


@dataclass
class ReflectionPattern:
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    first_seen: str
    last_seen: str

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


class CognitiveReflection:
    _MAX_REFLECTIONS = 5000
    _MAX_INSIGHTS = 1000
    _MAX_PATTERNS = 500
    _MAX_METADATA_KEYS = 20
    _MAX_ITEMS = 10
    _MAX_LINKED_DECISIONS = 5
    _MAX_APPLICABLE_DOMAINS = 5
    _MAX_SOURCE_REFLECTION_IDS = 5

    def __init__(self):
        self._lock = threading.Lock()
        self._reflections: Dict[str, ReflectionEntry] = {}
        self._reflection_order: deque = deque(maxlen=self._MAX_REFLECTIONS)
        self._insights: Dict[str, LearningInsight] = {}
        self._insight_order: deque = deque(maxlen=self._MAX_INSIGHTS)
        self._patterns: Dict[str, ReflectionPattern] = {}
        self._pattern_order: deque = deque(maxlen=self._MAX_PATTERNS)
        self._stats = {
            "reflections_added": 0,
            "insights_extracted": 0,
            "patterns_detected": 0,
        }
        self._stats_lock = threading.Lock()

    def add_reflection(
        self,
        topic: str,
        description: str = "",
        reflection_type: str = "daily",
        what_went_well: Optional[List[str]] = None,
        what_could_improve: Optional[List[str]] = None,
        lessons_learned: Optional[List[str]] = None,
        confidence_before: float = 0.5,
        confidence_after: float = 0.5,
        domain: str = "",
        linked_decision_ids: Optional[List[str]] = None,
        sentiment: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> ReflectionEntry:
        topic = _sanitize(topic, 200)
        description = _sanitize_content(description, 2000)
        domain = _sanitize(domain, 100)

        try:
            rt_enum = ReflectionType(reflection_type)
        except ValueError:
            rt_enum = ReflectionType.DAILY

        if what_went_well is None:
            what_went_well = []
        what_went_well = [_sanitize_content(w, 200) for w in what_went_well[:self._MAX_ITEMS]]

        if what_could_improve is None:
            what_could_improve = []
        what_could_improve = [_sanitize_content(c, 200) for c in what_could_improve[:self._MAX_ITEMS]]

        if lessons_learned is None:
            lessons_learned = []
        lessons_learned = [_sanitize_content(l, 200) for l in lessons_learned[:self._MAX_ITEMS]]

        confidence_before = max(0.0, min(1.0, float(confidence_before)))
        confidence_after = max(0.0, min(1.0, float(confidence_after)))
        sentiment = max(-1.0, min(1.0, float(sentiment)))

        if linked_decision_ids is None:
            linked_decision_ids = []
        linked_decision_ids = [_sanitize(str(d), 64) for d in linked_decision_ids[:self._MAX_LINKED_DECISIONS]]

        if metadata is None:
            metadata = {}
        clean_meta = {}
        for k, v in list(metadata.items())[:self._MAX_METADATA_KEYS]:
            k_clean = _sanitize(str(k), 50)
            if not k_clean:
                continue
            if isinstance(v, str):
                clean_meta[k_clean] = _sanitize_content(v, 500)
            elif isinstance(v, (int, float, bool)):
                clean_meta[k_clean] = v
            else:
                clean_meta[k_clean] = _sanitize_content(str(v), 500)
        metadata = clean_meta

        entry_id = f"ref_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        entry = ReflectionEntry(
            entry_id=entry_id,
            topic=topic,
            description=description,
            reflection_type=rt_enum,
            what_went_well=what_went_well,
            what_could_improve=what_could_improve,
            lessons_learned=lessons_learned,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            domain=domain,
            linked_decision_ids=linked_decision_ids,
            sentiment=sentiment,
            created_at=now,
            metadata=metadata,
        )

        with self._lock:
            if len(self._reflections) >= self._MAX_REFLECTIONS:
                evict_id = self._reflection_order.popleft()
                self._reflections.pop(evict_id, None)
                logger.warning("Evicting reflection %s (cap=%d)", evict_id, self._MAX_REFLECTIONS)
            self._reflections[entry_id] = entry
            self._reflection_order.append(entry_id)
            with self._stats_lock:
                self._stats["reflections_added"] += 1

        return copy.deepcopy(entry)

    def extract_insights(self, reflection_id: str) -> List[LearningInsight]:
        reflection_id = _sanitize(reflection_id, 64)
        with self._lock:
            entry = self._reflections.get(reflection_id)
            if entry is None:
                return []
            entry_copy = copy.deepcopy(entry)

        insights = []
        all_lessons = list(entry_copy.what_went_well) + list(entry_copy.what_could_improve) + list(entry_copy.lessons_learned)

        for lesson in all_lessons:
            if not lesson.strip():
                continue
            category = self._classify_insight(lesson)
            confidence = min(1.0, 0.3 + len(lesson.split()) * 0.03)
            insight_id = f"learn_{uuid.uuid4().hex[:12]}"
            now = datetime.now(timezone.utc).isoformat()
            insight = LearningInsight(
                insight_id=insight_id,
                title=_sanitize_content(lesson, 200),
                description=_sanitize_content(lesson, 1000),
                category=category,
                source_reflection_ids=[reflection_id][:self._MAX_SOURCE_REFLECTION_IDS],
                frequency=1,
                confidence=round(confidence, 4),
                applicable_domains=[entry_copy.domain][:self._MAX_APPLICABLE_DOMAINS] if entry_copy.domain else [],
                created_at=now,
            )

            with self._lock:
                merged = False
                for existing_id, existing in list(self._insights.items()):
                    if existing.title == insight.title and existing.category == category:
                        existing.frequency += 1
                        existing.confidence = min(1.0, existing.confidence + 0.05)
                        if reflection_id not in existing.source_reflection_ids:
                            existing.source_reflection_ids.append(reflection_id)
                            existing.source_reflection_ids = existing.source_reflection_ids[:self._MAX_SOURCE_REFLECTION_IDS]
                        merged = True
                        break
                if not merged:
                    if len(self._insights) >= self._MAX_INSIGHTS:
                        evict_id = self._insight_order.popleft()
                        self._insights.pop(evict_id, None)
                        logger.warning("Evicting insight %s (cap=%d)", evict_id, self._MAX_INSIGHTS)
                    self._insights[insight_id] = insight
                    self._insight_order.append(insight_id)
                with self._stats_lock:
                    self._stats["insights_extracted"] += 1

                insights.append(copy.deepcopy(existing if merged else insight))

        return insights

    def _classify_insight(self, text: str) -> str:
        text_lower = text.lower()
        for pattern in _LESSON_PATTERNS:
            if pattern.search(text_lower):
                if pattern is _LESSON_PATTERNS[0]:
                    return "missed_opportunity"
                elif pattern is _LESSON_PATTERNS[1]:
                    return "error_pattern"
                elif pattern is _LESSON_PATTERNS[2]:
                    return "discovery"
                elif pattern is _LESSON_PATTERNS[3]:
                    return "comparative"
                elif pattern is _LESSON_PATTERNS[4]:
                    return "forward_looking"
        return "general"

    def detect_patterns(self, min_frequency: int = 2) -> List[ReflectionPattern]:
        min_frequency = max(1, min_frequency)
        patterns = []

        with self._lock:
            entries = list(self._reflections.values())

        domain_sentiment: Dict[str, List[float]] = {}
        improve_items: Dict[str, int] = {}
        well_items: Dict[str, int] = {}
        domain_entries: Dict[str, List[ReflectionEntry]] = {}

        for entry in entries:
            domain_key = entry.domain or "unknown"
            domain_sentiment.setdefault(domain_key, []).append(entry.sentiment)
            domain_entries.setdefault(domain_key, []).append(entry)
            for item in entry.what_could_improve[:5]:
                improve_items[item] = improve_items.get(item, 0) + 1
            for item in entry.what_went_well[:5]:
                well_items[item] = well_items.get(item, 0) + 1

        for domain_key, sentiments in domain_sentiment.items():
            if len(sentiments) < min_frequency:
                continue
            avg = sum(sentiments) / len(sentiments)
            if avg < -0.3:
                pid = f"rpat_{uuid.uuid4().hex[:12]}"
                first = domain_entries[domain_key][0].created_at
                last = domain_entries[domain_key][-1].created_at
                patterns.append(ReflectionPattern(
                    pattern_id=pid,
                    pattern_type="recurring_mistake",
                    description=f"Negative sentiment trend in domain '{domain_key}': avg {avg:.2f}",
                    frequency=len(sentiments),
                    first_seen=first,
                    last_seen=last,
                ))
            elif avg > 0.3:
                pid = f"rpat_{uuid.uuid4().hex[:12]}"
                first = domain_entries[domain_key][0].created_at
                last = domain_entries[domain_key][-1].created_at
                patterns.append(ReflectionPattern(
                    pattern_id=pid,
                    pattern_type="strength_pattern",
                    description=f"Positive sentiment trend in domain '{domain_key}': avg {avg:.2f}",
                    frequency=len(sentiments),
                    first_seen=first,
                    last_seen=last,
                ))

        for item, count in improve_items.items():
            if count >= min_frequency and item.strip():
                pid = f"rpat_{uuid.uuid4().hex[:12]}"
                first_matching = ""
                last_matching = ""
                for entry in entries:
                    if item in entry.what_could_improve:
                        if not first_matching:
                            first_matching = entry.created_at
                        last_matching = entry.created_at
                patterns.append(ReflectionPattern(
                    pattern_id=pid,
                    pattern_type="improvement_trend",
                    description=f"Repeated improvement area: '{item}' ({count} times)",
                    frequency=count,
                    first_seen=first_matching,
                    last_seen=last_matching,
                ))

        for item, count in well_items.items():
            if count >= min_frequency and item.strip():
                pid = f"rpat_{uuid.uuid4().hex[:12]}"
                first_matching = ""
                last_matching = ""
                for entry in entries:
                    if item in entry.what_went_well:
                        if not first_matching:
                            first_matching = entry.created_at
                        last_matching = entry.created_at
                patterns.append(ReflectionPattern(
                    pattern_id=pid,
                    pattern_type="strength_pattern",
                    description=f"Repeated strength: '{item}' ({count} times)",
                    frequency=count,
                    first_seen=first_matching,
                    last_seen=last_matching,
                ))

        with self._lock:
            for p in patterns:
                if len(self._patterns) >= self._MAX_PATTERNS:
                    evict_id = self._pattern_order.popleft()
                    self._patterns.pop(evict_id, None)
                    logger.warning("Evicting pattern %s (cap=%d)", evict_id, self._MAX_PATTERNS)
                self._patterns[p.pattern_id] = p
                self._pattern_order.append(p.pattern_id)
            with self._stats_lock:
                self._stats["patterns_detected"] = len(self._patterns)

        return [copy.deepcopy(p) for p in patterns]

    def get_confidence_trend(self, domain: Optional[str] = None, limit: int = 20) -> Dict:
        limit = max(1, min(100, limit))
        if domain:
            domain = _sanitize(domain, 100)

        with self._lock:
            entries = list(self._reflections.values())

        if domain:
            entries = [e for e in entries if e.domain == domain]

        entries.sort(key=lambda e: e.created_at)
        entries = entries[-limit:]

        if not entries:
            return {"trend": [], "average_delta": 0.0, "direction": "neutral"}

        trend = []
        deltas = []
        for entry in entries:
            delta = entry.confidence_after - entry.confidence_before
            deltas.append(delta)
            trend.append({
                "entry_id": entry.entry_id,
                "confidence_before": entry.confidence_before,
                "confidence_after": entry.confidence_after,
                "delta": round(delta, 4),
                "domain": entry.domain,
                "created_at": entry.created_at,
            })

        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
        if avg_delta > 0.05:
            direction = "learning"
        elif avg_delta < -0.05:
            direction = "overconfidence"
        else:
            direction = "neutral"

        return {
            "trend": trend,
            "average_delta": round(avg_delta, 4),
            "direction": direction,
        }

    def get_reflection(self, entry_id: str) -> Optional[ReflectionEntry]:
        with self._lock:
            entry = self._reflections.get(entry_id)
            if entry is None:
                return None
            return copy.deepcopy(entry)

    def list_reflections(
        self,
        reflection_type: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 50,
    ) -> List[ReflectionEntry]:
        limit = max(1, min(200, limit))
        if domain:
            domain = _sanitize(domain, 100)
        try:
            rt_filter = ReflectionType(reflection_type) if reflection_type else None
        except ValueError:
            rt_filter = None

        with self._lock:
            results = []
            for eid in reversed(self._reflection_order):
                entry = self._reflections.get(eid)
                if entry is None:
                    continue
                if rt_filter and entry.reflection_type != rt_filter:
                    continue
                if domain and entry.domain != domain:
                    continue
                results.append(copy.deepcopy(entry))
                if len(results) >= limit:
                    break
            return results

    def get_learning_insights(
        self,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[LearningInsight]:
        limit = max(1, min(200, limit))
        if category:
            category = _sanitize(category, 100)

        with self._lock:
            insights = list(self._insights.values())

        if category:
            insights = [i for i in insights if i.category == category]

        insights.sort(key=lambda i: i.frequency, reverse=True)

        return [copy.deepcopy(i) for i in insights[:limit]]

    def get_stats(self) -> Dict:
        with self._stats_lock:
            reflections_added = self._stats["reflections_added"]
            insights_extracted = self._stats["insights_extracted"]
            patterns_detected = self._stats["patterns_detected"]
        with self._lock:
            current_reflections = len(self._reflections)
            current_insights = len(self._insights)
            current_patterns = len(self._patterns)
        return {
            "reflections_added": reflections_added,
            "insights_extracted": insights_extracted,
            "patterns_detected": patterns_detected,
            "current_reflections": current_reflections,
            "current_insights": current_insights,
            "current_patterns": current_patterns,
        }