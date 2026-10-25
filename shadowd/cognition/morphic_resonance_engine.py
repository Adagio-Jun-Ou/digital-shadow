#!/usr/bin/env python3
"""
MorphicResonanceEngine — Cross-Context Pattern Propagation.

Patterns that succeed in one context propagate morphic influence to similar
contexts. Enables the system to recognize and propagate successful strategies,
behaviors, and structures across domains through resonance fields.

Thread-safe, bounded, sanitized, deepcopy returns.
"""

import copy
import hashlib
import logging
import math
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.morphic_resonance_engine")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_FIELDS = 200
_MAX_PATTERNS = 3000
_MAX_INFLUENCES = 5000
_MAX_TRAIL = 20
_MAX_METADATA_KEYS = 20


class ResonanceType(str, Enum):
    DIRECT = "direct"
    TRANSFORMED = "transformed"
    EMERGENT = "emergent"
    DAMPENED = "dampened"
    AMPLIFIED = "amplified"


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
class MorphicField:
    field_id: str
    domain: str
    pattern_signature: str = ""
    resonance_count: int = 0
    propagation_count: int = 0
    strength: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.field_id:
            self.field_id = uuid.uuid4().hex[:12]
        self.field_id = _sanitize(self.field_id, 50)
        self.domain = _sanitize(self.domain, 100)
        self.pattern_signature = _sanitize_content(self.pattern_signature, 500)
        self.resonance_count = max(0, self.resonance_count)
        self.propagation_count = max(0, self.propagation_count)
        self.strength = _clamp(self.strength)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "field_id": self.field_id,
            "domain": self.domain,
            "pattern_signature": self.pattern_signature,
            "resonance_count": self.resonance_count,
            "propagation_count": self.propagation_count,
            "strength": round(self.strength, 4),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class ResonancePattern:
    pattern_id: str
    source_domain: str
    target_domain: str
    pattern: str
    confidence: float
    resonance_type: ResonanceType
    propagation_trail: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = uuid.uuid4().hex[:12]
        self.pattern_id = _sanitize(self.pattern_id, 50)
        self.source_domain = _sanitize(self.source_domain, 100)
        self.target_domain = _sanitize(self.target_domain, 100)
        self.pattern = _sanitize_content(self.pattern, 1000)
        self.confidence = _clamp(self.confidence)
        self.propagation_trail = [_sanitize(t, 100) for t in self.propagation_trail[:_MAX_TRAIL]]
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "pattern": self.pattern,
            "confidence": round(self.confidence, 4),
            "resonance_type": self.resonance_type.value,
            "propagation_trail": list(self.propagation_trail),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class MorphicInfluence:
    influence_id: str
    source_pattern_id: str
    target_domain: str
    influence_strength: float
    decay_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.influence_id:
            self.influence_id = uuid.uuid4().hex[:12]
        self.influence_id = _sanitize(self.influence_id, 50)
        self.source_pattern_id = _sanitize(self.source_pattern_id, 50)
        self.target_domain = _sanitize(self.target_domain, 100)
        self.influence_strength = _clamp(self.influence_strength)
        self.decay_rate = _clamp(self.decay_rate, 0.0, 1.0)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "influence_id": self.influence_id,
            "source_pattern_id": self.source_pattern_id,
            "target_domain": self.target_domain,
            "influence_strength": round(self.influence_strength, 4),
            "decay_rate": round(self.decay_rate, 4),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


class MorphicResonanceEngine:
    """Cross-context pattern propagation through morphic resonance fields.

    Patterns that succeed in one domain generate resonance fields that
    propagate influence to similar domains, enabling knowledge transfer
    and emergent pattern recognition.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._fields: Dict[str, MorphicField] = {}
        self._patterns: Dict[str, ResonancePattern] = {}
        self._influences: Dict[str, MorphicInfluence] = {}
        self._field_order: deque = deque(maxlen=_MAX_FIELDS)
        self._pattern_order: deque = deque(maxlen=_MAX_PATTERNS)
        self._influence_order: deque = deque(maxlen=_MAX_INFLUENCES)
        self._domain_fields: Dict[str, str] = {}
        self._stats = {
            "patterns_registered": 0,
            "influences_propagated": 0,
            "resonances_detected": 0,
            "fields_created": 0,
            "influences_applied": 0,
            "influences_decayed": 0,
        }

    def register_pattern(self, domain: str, pattern: str,
                         confidence: float,
                         metadata: Optional[Dict] = None) -> ResonancePattern:
        dom = _sanitize(domain, 100)
        pat = _sanitize_content(pattern, 1000)

        rp = ResonancePattern(
            pattern_id=uuid.uuid4().hex[:12],
            source_domain=dom,
            target_domain="",
            pattern=pat,
            confidence=confidence,
            resonance_type=ResonanceType.DIRECT,
            metadata=metadata or {},
        )
        with self._lock:
            self._ensure_field_unlocked(dom)
            if len(self._patterns) >= _MAX_PATTERNS:
                evicted_id = self._pattern_order.popleft()
                self._patterns.pop(evicted_id, None)
                logger.warning("Evicted resonance pattern %s", evicted_id)
            self._patterns[rp.pattern_id] = rp
            self._pattern_order.append(rp.pattern_id)
            if dom in self._domain_fields:
                field = self._fields.get(self._domain_fields[dom])
                if field:
                    field.resonance_count += 1
                    sig = self._compute_signature(pat)
                    field.pattern_signature = sig
            with self._stats_lock:
                self._stats["patterns_registered"] += 1
        return copy.deepcopy(rp)

    def _ensure_field_unlocked(self, domain: str):
        if domain in self._domain_fields:
            return
        if len(self._fields) >= _MAX_FIELDS:
            evicted_id = self._field_order.popleft()
            evicted_field = self._fields.pop(evicted_id, None)
            if evicted_field:
                self._domain_fields.pop(evicted_field.domain, None)
            logger.warning("Evicted morphic field %s", evicted_id)
        mf = MorphicField(
            field_id=uuid.uuid4().hex[:12],
            domain=domain,
            strength=0.5,
        )
        self._fields[mf.field_id] = mf
        self._field_order.append(mf.field_id)
        self._domain_fields[domain] = mf.field_id
        with self._stats_lock:
            self._stats["fields_created"] += 1

    def _ensure_field(self, domain: str):
        with self._lock:
            self._ensure_field_unlocked(domain)

    def propagate(self, source_domain: str, target_domain: str,
                  min_confidence: float = 0.3) -> List[MorphicInfluence]:
        src = _sanitize(source_domain, 100)
        tgt = _sanitize(target_domain, 100)
        with self._lock:
            if src not in self._domain_fields:
                return []
            self._ensure_field_unlocked(tgt)
            source_fid = self._domain_fields.get(src)
            target_fid = self._domain_fields.get(tgt)
            if not source_fid or not target_fid:
                return []
            source_patterns = [p for p in self._patterns.values()
                               if p.source_domain == src and p.confidence >= min_confidence]
            source_field = self._fields.get(source_fid)
            target_field = self._fields.get(target_fid)
            if not source_field or not target_field:
                return []
            domain_similarity = self._domain_similarity(src, tgt)
            influences: List[MorphicInfluence] = []
            for pat in source_patterns:
                influence_strength = pat.confidence * source_field.strength * domain_similarity
                if influence_strength < 0.05:
                    continue
                resonance_type = self._determine_resonance_type(
                    pat.confidence, domain_similarity
                )
                inf = MorphicInfluence(
                    influence_id=uuid.uuid4().hex[:12],
                    source_pattern_id=pat.pattern_id,
                    target_domain=tgt,
                    influence_strength=influence_strength,
                    decay_rate=0.05,
                    metadata={"resonance_type": resonance_type.value,
                              "domain_similarity": round(domain_similarity, 4),
                              "source_confidence": round(pat.confidence, 4)},
                )
                if len(self._influences) >= _MAX_INFLUENCES:
                    evicted_id = self._influence_order.popleft()
                    self._influences.pop(evicted_id, None)
                    logger.warning("Evicted morphic influence %s", evicted_id)
                self._influences[inf.influence_id] = inf
                self._influence_order.append(inf.influence_id)
                target_field.resonance_count += 1
                source_field.propagation_count += 1
                influences.append(inf)
            with self._stats_lock:
                self._stats["influences_propagated"] += len(influences)
        return [copy.deepcopy(i) for i in influences]

    def detect_resonance(self, domain: str,
                         pattern: str) -> List[ResonancePattern]:
        dom = _sanitize(domain, 100)
        pat = _sanitize_content(pattern, 1000)
        with self._lock:
            fid = self._domain_fields.get(dom)
            if not fid:
                return []
            results = []
            for p in self._patterns.values():
                if p.source_domain == dom:
                    continue
                similarity = self._compute_word_overlap(pat, p.pattern)
                if similarity >= 0.3:
                    rp = ResonancePattern(
                        pattern_id=p.pattern_id,
                        source_domain=p.source_domain,
                        target_domain=dom,
                        pattern=p.pattern,
                        confidence=_clamp(p.confidence * similarity),
                        resonance_type=self._determine_resonance_type(
                            p.confidence, similarity
                        ),
                        propagation_trail=p.propagation_trail + [dom][:_MAX_TRAIL],
                        metadata=dict(p.metadata),
                    )
                    results.append(rp)
            results.sort(key=lambda x: x.confidence, reverse=True)
            with self._stats_lock:
                self._stats["resonances_detected"] += len(results)
            return [copy.deepcopy(r) for r in results[:20]]

    def get_field(self, domain: str) -> Optional[MorphicField]:
        dom = _sanitize(domain, 100)
        with self._lock:
            fid = self._domain_fields.get(dom)
            if not fid:
                return None
            field = self._fields.get(fid)
            return copy.deepcopy(field) if field else None

    def get_influences(self, target_domain: str = "",
                       limit: int = 100) -> list:
        limit = max(1, min(500, limit))
        with self._lock:
            if target_domain:
                dom = _sanitize(target_domain, 100)
                results = [inf for inf in self._influences.values()
                           if inf.target_domain == dom]
            else:
                inf_ids = list(self._influence_order)[:limit]
                results = [self._influences[iid] for iid in inf_ids
                           if iid in self._influences]
            results.sort(key=lambda x: x.influence_strength, reverse=True)
            return [copy.deepcopy(r) for r in results[:limit]]

    def apply_influence(self, influence_id: str) -> bool:
        iid = _sanitize(influence_id, 50)
        with self._lock:
            inf = self._influences.get(iid)
            if not inf:
                return False
            target_fid = self._domain_fields.get(inf.target_domain)
            if target_fid and target_fid in self._fields:
                field = self._fields[target_fid]
                field.strength = _clamp(
                    field.strength + inf.influence_strength * 0.1
                )
                field.propagation_count += 1
            source_pattern = self._patterns.get(inf.source_pattern_id)
            if source_pattern:
                source_pattern.propagation_trail = (
                    source_pattern.propagation_trail + [inf.target_domain]
                )[:_MAX_TRAIL]
            with self._stats_lock:
                self._stats["influences_applied"] += 1
            return True

    def compute_resonance_strength(self, pattern_id_1: str,
                                    pattern_id_2: str) -> float:
        pid1 = _sanitize(pattern_id_1, 50)
        pid2 = _sanitize(pattern_id_2, 50)
        with self._lock:
            p1 = self._patterns.get(pid1)
            p2 = self._patterns.get(pid2)
            if not p1 or not p2:
                return 0.0
            return self._compute_word_overlap(p1.pattern, p2.pattern)

    def decay_influences(self) -> int:
        with self._lock:
            to_remove = []
            for iid, inf in self._influences.items():
                inf.influence_strength *= (1.0 - inf.decay_rate)
                inf.influence_strength = round(inf.influence_strength, 4)
                if inf.influence_strength < 0.01:
                    to_remove.append(iid)
            for iid in to_remove:
                self._influences.pop(iid, None)
                try:
                    self._influence_order.remove(iid)
                except ValueError:
                    pass
            with self._stats_lock:
                self._stats["influences_decayed"] += len(to_remove)
            return len(to_remove)

    def get_pattern(self, pattern_id: str) -> Optional[ResonancePattern]:
        pid = _sanitize(pattern_id, 50)
        with self._lock:
            p = self._patterns.get(pid)
            return copy.deepcopy(p) if p else None

    def get_stats(self) -> Dict:
        with self._lock:
            stats = dict(self._stats)
            stats["field_count"] = len(self._fields)
            stats["pattern_count"] = len(self._patterns)
            stats["influence_count"] = len(self._influences)
            total_strengths = [f.strength for f in self._fields.values()]
            stats["avg_field_strength"] = (
                round(sum(total_strengths) / len(total_strengths), 4)
                if total_strengths else 0.0
            )
            return stats

    @staticmethod
    def _compute_signature(text: str) -> str:
        words = re.findall(r'[a-zA-Z0-9]+', text.lower())
        if not words:
            return ""
        return hashlib.sha256(" ".join(sorted(set(words))).encode()).hexdigest()[:16]

    @staticmethod
    def _compute_word_overlap(text1: str, text2: str) -> float:
        words1 = set(re.findall(r'[a-zA-Z0-9]+', text1.lower()))
        words2 = set(re.findall(r'[a-zA-Z0-9]+', text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    @staticmethod
    def _determine_resonance_type(confidence: float,
                                   similarity: float) -> ResonanceType:
        if similarity >= 0.8:
            return ResonanceType.DIRECT
        elif similarity >= 0.5:
            if confidence >= 0.7:
                return ResonanceType.AMPLIFIED
            else:
                return ResonanceType.TRANSFORMED
        elif similarity >= 0.3:
            if confidence >= 0.7:
                return ResonanceType.DAMPENED
            else:
                return ResonanceType.TRANSFORMED
        else:
            return ResonanceType.EMERGENT

    def _domain_similarity(self, domain1: str, domain2: str) -> float:
        similarity_map = {
            frozenset({"cognitive", "reasoning"}): 0.8,
            frozenset({"cognitive", "emotion"}): 0.6,
            frozenset({"cognitive", "social"}): 0.5,
            frozenset({"cognitive", "physical"}): 0.3,
            frozenset({"reasoning", "logic"}): 0.9,
            frozenset({"reasoning", "emotion"}): 0.4,
            frozenset({"reasoning", "social"}): 0.5,
            frozenset({"emotion", "social"}): 0.8,
            frozenset({"emotion", "physical"}): 0.4,
            frozenset({"social", "language"}): 0.7,
            frozenset({"language", "cognitive"}): 0.7,
        }
        d1 = _sanitize(domain1, 100).lower()
        d2 = _sanitize(domain2, 100).lower()
        if d1 == d2:
            return 1.0
        key = frozenset({d1, d2})
        if key in similarity_map:
            return similarity_map[key]
        overlap = self._compute_word_overlap(d1, d2)
        return max(0.05, overlap)