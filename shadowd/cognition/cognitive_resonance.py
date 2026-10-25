"""CognitiveResonanceDetector — Discover hidden cross-module connections.

Analyzes patterns across emotional, cognitive, social, temporal, and domain
activities to find non-obvious correlations. When multiple modules show
related activity patterns, the detector surfaces them as resonance findings.

Architecture:
  - ResonanceLink: a discovered connection between two modules/domains
  - ResonancePattern: recurring cross-module correlation
  - ResonanceSource: where the signal came from (module, domain, emotion)
  - ResonanceType: category of connection (temporal, causal, thematic, amplifying)
  - CognitiveResonanceDetector: main engine with detection, learning, and validation
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
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 50) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 500) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]

_MAX_LINKS = 1000
_MAX_PATTERNS = 500
_MAX_ACTIVITY_LOG = 5000
_MAX_FINDINGS = 200


class ResonanceType(Enum):
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    THEMATIC = "thematic"
    AMPLIFYING = "amplifying"
    DAMPENING = "dampening"
    SEQUENTIAL = "sequential"
    FEEDBACK = "feedback"


class ResonanceStrength(Enum):
    WEAK = 0.2
    MODERATE = 0.5
    STRONG = 0.75
    VERY_STRONG = 0.9


@dataclass
class ResonanceSource:
    module: str = ""
    domain: str = ""
    emotion: str = ""
    metric: str = ""
    value: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        self.module = _sanitize(self.module, 50)
        self.domain = _sanitize(self.domain, 50)
        self.emotion = _sanitize(self.emotion, 30)
        self.metric = _sanitize(self.metric, 50)
        self.value = max(-1.0, min(1.0, self.value))
        self.timestamp = self.timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "module": self.module,
            "domain": self.domain,
            "emotion": self.emotion,
            "metric": self.metric,
            "value": round(self.value, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class ResonanceLink:
    link_id: str = ""
    source_a: Optional[ResonanceSource] = None
    source_b: Optional[ResonanceSource] = None
    resonance_type: str = ""
    strength: float = 0.3
    description: str = ""
    evidence_count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    validated: bool = False

    def __post_init__(self):
        if not self.link_id:
            self.link_id = uuid.uuid4().hex[:12]
        self.strength = max(0.0, min(1.0, self.strength))
        if self.resonance_type:
            self.resonance_type = _sanitize(self.resonance_type, 50)
        if self.description:
            self.description = _sanitize_content(self.description, 500)
        self.first_seen = self.first_seen or datetime.now(timezone.utc).isoformat()
        self.last_seen = self.last_seen or self.first_seen

    def to_dict(self) -> Dict:
        return {
            "link_id": self.link_id,
            "source_a": self.source_a.to_dict() if self.source_a else None,
            "source_b": self.source_b.to_dict() if self.source_b else None,
            "resonance_type": self.resonance_type,
            "strength": round(self.strength, 4),
            "description": self.description,
            "evidence_count": self.evidence_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "validated": self.validated,
        }


@dataclass
class ResonancePattern:
    pattern_id: str = ""
    pattern_type: str = ""
    modules_involved: List[str] = field(default_factory=list)
    frequency: int = 0
    avg_strength: float = 0.0
    description: str = ""
    last_seen: str = ""

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = uuid.uuid4().hex[:12]
        if self.pattern_type:
            self.pattern_type = _sanitize(self.pattern_type, 50)
        self.modules_involved = [_sanitize(m, 30) for m in self.modules_involved][:15]
        self.avg_strength = max(0.0, min(1.0, self.avg_strength))
        if self.description:
            self.description = _sanitize_content(self.description, 500)
        self.last_seen = self.last_seen or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "modules_involved": list(self.modules_involved),
            "frequency": self.frequency,
            "avg_strength": round(self.avg_strength, 4),
            "description": self.description,
            "last_seen": self.last_seen,
        }


_ACTIVITY_MODULES = [
    "emotional_intelligence", "thought_sense", "attention_engine",
    "goal_planner", "narrative_memory", "creative_synthesis",
    "social_intelligence", "temporal_reasoning", "recall_synthesis",
    "evolution_guide", "device_context", "procedural_memory",
]

_EMOTION_CAUSAL_MAP: Dict[str, List[str]] = {
    "joy": ["creative_synthesis", "social_intelligence", "attention_engine"],
    "sadness": ["recall_synthesis", "narrative_memory", "evolution_guide"],
    "anger": ["conflict_resolution", "attention_engine", "moral_reasoning"],
    "fear": ["attention_engine", "goal_planner", "device_context"],
    "surprise": ["thought_sense", "recall_synthesis", "creative_synthesis"],
    "disgust": ["moral_reasoning", "conflict_resolution"],
    "anticipation": ["goal_planner", "temporal_reasoning", "creative_synthesis"],
    "trust": ["social_intelligence", "evolution_guide"],
}

_DOMAIN_ASSOCIATIONS: Dict[str, List[str]] = {
    "work": ["temporal_reasoning", "goal_planner", "attention_engine", "procedural_memory"],
    "health": ["device_context", "emotional_intelligence", "evolution_guide"],
    "creative": ["creative_synthesis", "thought_sense", "narrative_memory"],
    "social": ["social_intelligence", "empathy_mapping", "conflict_resolution"],
    "learning": ["recall_synthesis", "attention_engine", "evolution_guide"],
    "financial": ["goal_planner", "temporal_reasoning", "attention_engine"],
}


class CognitiveResonanceDetector:
    """Discover hidden cross-module connections and correlations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._activity_log: deque = deque(maxlen=_MAX_ACTIVITY_LOG)
        self._links: Dict[str, ResonanceLink] = {}
        self._link_order: deque = deque(maxlen=_MAX_LINKS)
        self._patterns: Dict[str, ResonancePattern] = {}
        self._pattern_order: deque = deque(maxlen=_MAX_PATTERNS)
        self._findings: List[Dict] = []
        self._module_activity: Dict[str, deque] = {
            m: deque(maxlen=200) for m in _ACTIVITY_MODULES
        }
        self._cross_correlations: Dict[str, float] = {}
        self._stats: Dict[str, int] = {
            "activities_logged": 0,
            "links_discovered": 0,
            "patterns_detected": 0,
            "resonances_found": 0,
            "validations": 0,
        }

    def log_activity(self, module: str, domain: str = "",
                     emotion: str = "", metric: str = "",
                     value: float = 0.0) -> None:
        """Log an activity from a module."""
        module = _sanitize(module, 50)
        domain = _sanitize(domain, 50)
        emotion = _sanitize(emotion, 30)
        metric = _sanitize(metric, 50)
        value = max(-1.0, min(1.0, value))

        source = ResonanceSource(
            module=module, domain=domain, emotion=emotion,
            metric=metric, value=value,
        )

        with self._lock:
            self._activity_log.append({
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if module in self._module_activity:
                self._module_activity[module].append({
                    "domain": domain,
                    "emotion": emotion,
                    "value": value,
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
            activity_count = len(self._activity_log)

        with self._stats_lock:
            self._stats["activities_logged"] += 1

        if activity_count % 10 == 0:
            self._auto_detect()

    def detect(self) -> List[ResonanceLink]:
        """Run resonance detection across all logged activities."""
        with self._lock:
            activity_snapshot = list(self._activity_log)
            module_snapshot = {k: list(v) for k, v in self._module_activity.items()}
            existing_keys = {f"{l.source_a.module}:{l.source_b.module}:{l.resonance_type}"
                             for l in self._links.values()}

        links: List[ResonanceLink] = []
        links.extend(self._detect_emotional_causal(activity_snapshot, module_snapshot))
        links.extend(self._detect_temporal_correlations(module_snapshot))
        links.extend(self._detect_domain_associations(activity_snapshot))
        links.extend(self._detect_feedback_loops(activity_snapshot))
        links.extend(self._detect_amplification(activity_snapshot))

        new_links = []
        seen_keys: set = set()
        with self._lock:
            for link in links:
                key = f"{link.source_a.module}:{link.source_b.module}:{link.resonance_type}"
                if key not in seen_keys and key not in existing_keys:
                    seen_keys.add(key)
                    self._links[link.link_id] = link
                    self._link_order.append(link.link_id)
                    new_links.append(link)

            if len(self._links) > _MAX_LINKS:
                overflow = len(self._links) - _MAX_LINKS
                for lid in list(self._link_order)[:overflow]:
                    self._links.pop(lid, None)
                    logger.warning("Evicting resonance link %s (cap=%d)", lid, _MAX_LINKS)

        with self._stats_lock:
            self._stats["links_discovered"] += len(new_links)

        return copy.deepcopy(new_links[:20])

    def _detect_emotional_causal(self, activity_snapshot: List, module_snapshot: Dict) -> List[ResonanceLink]:
        """Detect causal links between emotions and module activation."""
        links: List[ResonanceLink] = []

        for emotion, modules in _EMOTION_CAUSAL_MAP.items():
            emotion_activities = [
                a for a in activity_snapshot
                if a["source"].emotion == emotion
            ]

            if len(emotion_activities) < 3:
                continue

            for module in modules:
                module_activities = list(module_snapshot.get(module, []))

                if len(module_activities) < 3:
                    continue

                co_occurrence = min(len(emotion_activities), len(module_activities))
                strength = min(co_occurrence / 10.0, 0.9)

                if strength >= 0.3:
                    links.append(ResonanceLink(
                        source_a=ResonanceSource(module="emotional_intelligence", emotion=emotion),
                        source_b=ResonanceSource(module=module, domain=""),
                        resonance_type=ResonanceType.CAUSAL.value,
                        strength=round(strength, 4),
                        description=f"Emotion '{emotion}' causally linked to {module} activation",
                        evidence_count=co_occurrence,
                    ))

        return links

    def _detect_temporal_correlations(self, module_snapshot: Dict) -> List[ResonanceLink]:
        """Detect modules that activate at similar times."""
        links: List[ResonanceLink] = []
        modules = list(module_snapshot.keys())

        for i in range(0, len(modules)):
            for j in range(i + 1, min(i + 3, len(modules))):
                m1, m2 = modules[i], modules[j]
                a1 = list(module_snapshot.get(m1, []))
                a2 = list(module_snapshot.get(m2, []))

                if len(a1) < 2 or len(a2) < 2:
                    continue

                overlap = 0.1 * min(len(a1), len(a2), 5)
                strength = min(overlap / 3.0, 0.8)

                if strength >= 0.3:
                    links.append(ResonanceLink(
                        source_a=ResonanceSource(module=m1),
                        source_b=ResonanceSource(module=m2),
                        resonance_type=ResonanceType.TEMPORAL.value,
                        strength=round(strength, 4),
                        description=f"{m1} and {m2} show temporal co-activation",
                        evidence_count=min(len(a1), len(a2)),
                    ))

        return links

    def _detect_domain_associations(self, activity_snapshot: List) -> List[ResonanceLink]:
        """Detect modules associated with active domains."""
        links: List[ResonanceLink] = []

        for domain, modules in _DOMAIN_ASSOCIATIONS.items():
            domain_count = sum(
                1 for a in activity_snapshot
                if a["source"].domain == domain
            )

            if domain_count < 2:
                continue

            for i in range(min(len(modules) - 1, 2)):
                m1 = modules[i]
                m2 = modules[i + 1] if i + 1 < len(modules) else modules[0]
                strength = min(domain_count / 5.0, 0.7)

                links.append(ResonanceLink(
                    source_a=ResonanceSource(module=m1, domain=domain),
                    source_b=ResonanceSource(module=m2, domain=domain),
                    resonance_type=ResonanceType.THEMATIC.value,
                    strength=round(strength, 4),
                    description=f"{m1} and {m2} co-activate in '{domain}' domain",
                    evidence_count=domain_count,
                ))

        return links

    def _detect_feedback_loops(self, activity_snapshot: List) -> List[ResonanceLink]:
        """Detect feedback loops where module A activates B which activates A."""
        links: List[ResonanceLink] = []
        seen_pairs: set = set()

        with self._lock:
            activities = list(activity_snapshot)

        window = min(50, len(activities))
        for i in range(max(0, len(activities) - window), min(len(activities) - 1, max(0, len(activities) - window) + 30)):
            a = activities[i]
            b = activities[i + 1] if i + 1 < len(activities) else None
            if not b:
                continue

            m_a = a["source"].module
            m_b = b["source"].module

            pair_key = f"{min(m_a, m_b)}:{max(m_a, m_b)}"
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            for j in range(max(0, len(activities) - window), min(len(activities) - 1, max(0, len(activities) - window) + 30)):
                if j == i:
                    continue
                c = activities[j]
                d = activities[j + 1] if j + 1 < len(activities) else None
                if not d:
                    continue
                if c["source"].module == m_b and d["source"].module == m_a:
                    links.append(ResonanceLink(
                        source_a=ResonanceSource(module=m_a),
                        source_b=ResonanceSource(module=m_b),
                        resonance_type=ResonanceType.FEEDBACK.value,
                        strength=0.5,
                        description=f"Feedback loop: {m_a} ↔ {m_b}",
                        evidence_count=1,
                    ))
                    break

        return links[:5]

    def _detect_amplification(self, activity_snapshot: List) -> List[ResonanceLink]:
        """Detect amplifying relationships where one module boosts another's activity."""
        links: List[ResonanceLink] = []

        for domain, modules in _DOMAIN_ASSOCIATIONS.items():
            if len(modules) < 2:
                continue

            domain_activities = [
                a for a in activity_snapshot
                if a["source"].domain == domain
            ]

            high_value = [a for a in domain_activities if a["source"].value >= 0.7]
            if len(high_value) >= 2:
                for i in range(min(len(modules) - 1, 3)):
                    m = modules[i]
                    links.append(ResonanceLink(
                        source_a=ResonanceSource(module="emotional_intelligence", domain=domain),
                        source_b=ResonanceSource(module=m, domain=domain, value=0.7),
                        resonance_type=ResonanceType.AMPLIFYING.value,
                        strength=0.6,
                        description=f"High activity in '{domain}' amplifies {m}",
                        evidence_count=len(high_value),
                    ))

        return links[:5]

    def _auto_detect(self) -> None:
        """Auto-detect correlations on every 10th activity."""
        try:
            self.detect()
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.debug("Auto-detect failed: %s", type(e).__name__)

    def find_resonance(self, module: str = "", domain: str = "",
                       resonance_type: str = "", min_strength: float = 0.3,
                       limit: int = 20) -> List[Dict]:
        """Find resonance links matching criteria."""
        limit = max(1, min(limit, _MAX_FINDINGS))
        min_strength = max(0.0, min(1.0, min_strength))
        module = _sanitize(module, 50)
        domain = _sanitize(domain, 50)

        with self._lock:
            links = list(self._links.values())

        results = []
        for link in links:
            if min_strength > 0 and link.strength < min_strength:
                continue
            if module and link.source_a and link.source_a.module != module and link.source_b.module != module:
                continue
            if domain and link.source_a and link.source_a.domain != domain and link.source_b.domain != domain:
                continue
            if resonance_type and link.resonance_type != resonance_type:
                continue
            results.append(link.to_dict())

        results.sort(key=lambda x: x.get("strength", 0), reverse=True)
        with self._lock: self._findings = results[:_MAX_FINDINGS]

        with self._stats_lock:
            self._stats["resonances_found"] = len(self._findings)

        return copy.deepcopy(results[:limit])

    def get_links(self, resonance_type: Optional[str] = None,
                  limit: int = 50) -> List[ResonanceLink]:
        """Get discovered resonance links."""
        limit = max(1, min(limit, _MAX_LINKS))
        with self._lock:
            links = list(self._links.values())
        if resonance_type:
            links = [l for l in links if l.resonance_type == resonance_type]
        links = sorted(links, key=lambda l: l.strength, reverse=True)
        return copy.deepcopy(links[:limit])

    def get_patterns(self, limit: int = 50) -> List[ResonancePattern]:
        """Get detected resonance patterns."""
        limit = max(1, min(limit, _MAX_PATTERNS))
        with self._lock:
            patterns = list(self._patterns.values())
        patterns = sorted(patterns, key=lambda p: p.frequency, reverse=True)
        return copy.deepcopy(patterns[:limit])

    def validate_link(self, link_id: str, confirmed: bool) -> bool:
        """Validate or reject a resonance link."""
        link_id = _sanitize(link_id, 20)
        with self._lock:
            link = self._links.get(link_id)
            if not link:
                return False
            link.validated = True

        with self._stats_lock:
            self._stats["validations"] += 1

        if confirmed and link:
            key = f"{link.source_a.module}:{link.source_b.module}:{link.resonance_type}"
            with self._lock:
                if key not in self._patterns:
                    self._patterns[key] = ResonancePattern(
                        pattern_type=link.resonance_type,
                        modules_involved=[link.source_a.module, link.source_b.module],
                        frequency=1,
                        avg_strength=link.strength,
                        description=link.description,
                    )
                    self._pattern_order.append(key)
                else:
                    self._patterns[key].frequency += 1
                    old_avg = self._patterns[key].avg_strength
                    self._patterns[key].avg_strength = round(
                        (old_avg * (self._patterns[key].frequency - 1) + link.strength) /
                        self._patterns[key].frequency, 4
                    )
                    self._patterns[key].last_seen = datetime.now(timezone.utc).isoformat()

                if len(self._patterns) > _MAX_PATTERNS:
                    overflow = len(self._patterns) - _MAX_PATTERNS
                    for pid in list(self._pattern_order)[:overflow]:
                        self._patterns.pop(pid, None)
                        logger.warning("Evicting resonance pattern %s (cap=%d)", pid, _MAX_PATTERNS)

        return True

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        with self._lock:
            total_links = len(self._links)
            total_patterns = len(self._patterns)
            total_findings = len(self._findings)
        with self._stats_lock:
            stats = dict(self._stats)
        stats["total_links"] = total_links
        stats["total_patterns"] = total_patterns
        stats["total_findings"] = total_findings
        return stats