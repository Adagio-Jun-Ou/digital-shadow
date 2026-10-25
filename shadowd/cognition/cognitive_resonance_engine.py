#!/usr/bin/env python3
"""
CognitiveResonanceEngine -- Harmonic Pattern Detection and Amplification.

Detects when cognitive patterns align harmonically — like musical resonance
where frequencies create constructive interference. When multiple cognitive
modules produce similar patterns simultaneously, they amplify each other,
creating emergent insights that wouldn't exist in any single module.

Key concepts:
  - CognitivePattern: a pattern emitted by a cognitive module
  - ResonanceEvent: detected alignment between patterns
  - HarmonicSpectrum: frequency analysis of a domain
  - ResonanceInsight: emergent insight from resonant patterns

Thread-safe, bounded, sanitized, deepcopy returns.
"""

import copy
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

logger = logging.getLogger("shadowd.cognition.cognitive_resonance_engine")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_PATTERNS = 3000
_MAX_EVENTS = 2000
_MAX_INSIGHTS = 500
_MAX_CONTENT_LENGTH = 5000
_MAX_STRICT_LENGTH = 200
_MAX_LIST_ITEMS = 100


class ResonanceType(str, Enum):
    CONSTRUCTIVE = "constructive"
    DESTRUCTIVE = "destructive"
    PARTIAL = "partial"
    SYNCHRONIZED = "synchronized"
    ANTI_PHASE = "anti_phase"


class HarmonicLevel(int, Enum):
    FUNDAMENTAL = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
    FIFTH = 5


class PatternDomain(str, Enum):
    REASONING = "reasoning"
    EMOTIONAL = "emotional"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SOCIAL = "social"
    CREATIVE = "creative"
    INTUITIVE = "intuitive"
    ANALYTICAL = "analytical"


def _sanitize(text: str, max_len: int = _MAX_STRICT_LENGTH) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = _MAX_CONTENT_LENGTH) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _phase_coherence(phase1: float, phase2: float) -> float:
    diff = abs(phase1 - phase2)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return 1.0 - diff / math.pi


@dataclass
class CognitivePattern:
    pattern_id: str = ""
    domain: str = ""
    frequency: float = 0.5
    amplitude: float = 0.5
    phase: float = 0.0
    content: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = "cp_%s" % uuid.uuid4().hex[:12]
        self.pattern_id = _sanitize(self.pattern_id, 64)
        if isinstance(self.domain, PatternDomain):
            pass
        elif isinstance(self.domain, str):
            try:
                self.domain = PatternDomain(self.domain)
            except ValueError:
                self.domain = PatternDomain.REASONING
        else:
            self.domain = PatternDomain.REASONING
        self.frequency = _clamp(self.frequency)
        self.amplitude = _clamp(self.amplitude)
        self.phase = _clamp(self.phase, 0.0, 2 * math.pi)
        self.content = _sanitize_content(self.content)

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "domain": self.domain.value if isinstance(self.domain, PatternDomain) else self.domain,
            "frequency": round(self.frequency, 4),
            "amplitude": round(self.amplitude, 4),
            "phase": round(self.phase, 4),
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class ResonanceEvent:
    event_id: str = ""
    pattern_ids: List[str] = field(default_factory=list)
    resonance_type: str = ResonanceType.PARTIAL.value
    harmonic_level: int = HarmonicLevel.FUNDAMENTAL.value
    combined_amplitude: float = 0.5
    coherence: float = 0.5
    insight: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.event_id:
            self.event_id = "re_%s" % uuid.uuid4().hex[:12]
        self.event_id = _sanitize(self.event_id, 64)
        self.pattern_ids = [_sanitize(pid, 64) for pid in (self.pattern_ids or [])[:_MAX_LIST_ITEMS]]
        if isinstance(self.resonance_type, ResonanceType):
            pass
        elif isinstance(self.resonance_type, str):
            try:
                self.resonance_type = ResonanceType(self.resonance_type)
            except ValueError:
                self.resonance_type = ResonanceType.PARTIAL
        else:
            self.resonance_type = ResonanceType.PARTIAL
        if isinstance(self.harmonic_level, HarmonicLevel):
            pass
        elif isinstance(self.harmonic_level, int):
            try:
                self.harmonic_level = HarmonicLevel(self.harmonic_level)
            except ValueError:
                self.harmonic_level = HarmonicLevel.FUNDAMENTAL
        else:
            self.harmonic_level = HarmonicLevel.FUNDAMENTAL
        self.combined_amplitude = _clamp(self.combined_amplitude)
        self.coherence = _clamp(self.coherence)
        self.insight = _sanitize_content(self.insight, 1000)

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "pattern_ids": list(self.pattern_ids),
            "resonance_type": self.resonance_type.value if isinstance(self.resonance_type, ResonanceType) else self.resonance_type,
            "harmonic_level": self.harmonic_level.value if isinstance(self.harmonic_level, HarmonicLevel) else self.harmonic_level,
            "combined_amplitude": round(self.combined_amplitude, 4),
            "coherence": round(self.coherence, 4),
            "insight": self.insight,
            "timestamp": self.timestamp,
        }


@dataclass
class HarmonicSpectrum:
    spectrum_id: str = ""
    fundamental_frequency: float = 0.5
    harmonics: Dict[int, float] = field(default_factory=dict)
    total_energy: float = 0.5
    peak_harmonic: int = 1

    def __post_init__(self):
        if not self.spectrum_id:
            self.spectrum_id = "hs_%s" % uuid.uuid4().hex[:12]
        self.spectrum_id = _sanitize(self.spectrum_id, 64)
        self.fundamental_frequency = _clamp(self.fundamental_frequency)
        self.harmonics = {
            int(k): _clamp(v) for k, v in list(self.harmonics.items())[:5]
        }
        self.total_energy = _clamp(self.total_energy)
        if not isinstance(self.peak_harmonic, int):
            try:
                self.peak_harmonic = int(self.peak_harmonic)
            except (ValueError, TypeError):
                self.peak_harmonic = 1
        self.peak_harmonic = max(1, min(5, self.peak_harmonic))

    def to_dict(self) -> Dict:
        return {
            "spectrum_id": self.spectrum_id,
            "fundamental_frequency": round(self.fundamental_frequency, 4),
            "harmonics": {int(k): round(v, 4) for k, v in self.harmonics.items()},
            "total_energy": round(self.total_energy, 4),
            "peak_harmonic": self.peak_harmonic,
        }


@dataclass
class ResonanceInsight:
    insight_id: str = ""
    source_domains: List[str] = field(default_factory=list)
    insight_type: str = ""
    content: str = ""
    confidence: float = 0.5
    amplitude: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.insight_id:
            self.insight_id = "ri_%s" % uuid.uuid4().hex[:12]
        self.insight_id = _sanitize(self.insight_id, 64)
        self.source_domains = [
            d.value if isinstance(d, PatternDomain) else _sanitize(str(d), 100)
            for d in (self.source_domains or [])[:_MAX_LIST_ITEMS]
        ]
        self.insight_type = _sanitize(self.insight_type, 100)
        self.content = _sanitize_content(self.content)
        self.confidence = _clamp(self.confidence)
        self.amplitude = _clamp(self.amplitude)

    def to_dict(self) -> Dict:
        return {
            "insight_id": self.insight_id,
            "source_domains": list(self.source_domains),
            "insight_type": self.insight_type,
            "content": self.content,
            "confidence": round(self.confidence, 4),
            "amplitude": round(self.amplitude, 4),
            "timestamp": self.timestamp,
        }


class CognitiveResonanceEngine:
    """Harmonic Pattern Detection and Amplification Engine.

    Detects when cognitive patterns align harmonically. When multiple
    modules produce similar patterns, constructive interference amplifies
    insight quality while destructive interference cancels noise.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._patterns: Dict[str, CognitivePattern] = {}
        self._pattern_order: deque = deque(maxlen=_MAX_PATTERNS)
        self._events: Dict[str, ResonanceEvent] = {}
        self._event_order: deque = deque(maxlen=_MAX_EVENTS)
        self._insights: Dict[str, ResonanceInsight] = {}
        self._insight_order: deque = deque(maxlen=_MAX_INSIGHTS)
        self._stats = {
            "patterns_emitted": 0,
            "resonance_events": 0,
            "constructive_events": 0,
            "destructive_events": 0,
            "insights_synthesized": 0,
            "amplifications": 0,
            "dampenings": 0,
            "harmonic_spectra_computed": 0,
        }

    def emit_pattern(self, domain: str, frequency: float = 0.5,
                     amplitude: float = 0.5, phase: float = 0.0,
                     content: str = "") -> CognitivePattern:
        if isinstance(domain, str):
            try:
                domain_enum = PatternDomain(domain)
            except ValueError:
                domain_enum = PatternDomain.REASONING
        elif isinstance(domain, PatternDomain):
            domain_enum = domain
        else:
            domain_enum = PatternDomain.REASONING
        frequency = _clamp(frequency)
        amplitude = _clamp(amplitude)
        phase = _clamp(phase, 0.0, 2 * math.pi)
        content = _sanitize_content(content)
        pattern = CognitivePattern(
            domain=domain_enum,
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            content=content,
        )
        with self._lock:
            if len(self._patterns) >= _MAX_PATTERNS:
                evict_id = self._pattern_order.popleft()
                self._patterns.pop(evict_id, None)
                logger.warning("Evicting pattern %s (cap=%d)", evict_id, _MAX_PATTERNS)
            self._patterns[pattern.pattern_id] = pattern
            self._pattern_order.append(pattern.pattern_id)
            with self._stats_lock:
                self._stats["patterns_emitted"] += 1
        return copy.deepcopy(pattern)

    def detect_resonance(self, min_amplitude: float = 0.3,
                         min_coherence: float = 0.5) -> List[ResonanceEvent]:
        min_amplitude = _clamp(min_amplitude)
        min_coherence = _clamp(min_coherence)
        events = []
        with self._lock:
            patterns = list(self._patterns.values())
        if len(patterns) < 2:
            return []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                p1 = patterns[i]
                p2 = patterns[j]
                if p1.amplitude < min_amplitude or p2.amplitude < min_amplitude:
                    continue
                coherence = _phase_coherence(p1.phase, p2.phase)
                freq_ratio = p1.frequency / max(p2.frequency, 0.001)
                harmonic_level = self._classify_harmonic(freq_ratio)
                combined_amp = self._combine_amplitudes(
                    p1.amplitude, p2.amplitude, coherence, freq_ratio
                )
                if coherence < min_coherence:
                    continue
                resonance_type = self._classify_resonance(coherence, freq_ratio)
                if harmonic_level > HarmonicLevel.FIFTH.value:
                    continue
                insight_text = self._generate_insight_text(p1, p2, resonance_type, combined_amp)
                event = ResonanceEvent(
                    pattern_ids=[p1.pattern_id, p2.pattern_id],
                    resonance_type=resonance_type,
                    harmonic_level=harmonic_level,
                    combined_amplitude=combined_amp,
                    coherence=coherence,
                    insight=insight_text,
                )
                events.append(event)
        with self._lock:
            for event in events:
                if len(self._events) >= _MAX_EVENTS:
                    evict_id = self._event_order.popleft()
                    self._events.pop(evict_id, None)
                    logger.warning("Evicting resonance event %s (cap=%d)", evict_id, _MAX_EVENTS)
                self._events[event.event_id] = event
                self._event_order.append(event.event_id)
            with self._stats_lock:
                self._stats["resonance_events"] += len(events)
                for event in events:
                    if event.resonance_type in (ResonanceType.CONSTRUCTIVE, ResonanceType.SYNCHRONIZED):
                        self._stats["constructive_events"] += 1
                    elif event.resonance_type in (ResonanceType.DESTRUCTIVE, ResonanceType.ANTI_PHASE):
                        self._stats["destructive_events"] += 1
        return [copy.deepcopy(e) for e in events]

    def compute_harmonics(self, domain: str) -> HarmonicSpectrum:
        if isinstance(domain, str):
            try:
                domain_enum = PatternDomain(domain)
            except ValueError:
                domain_enum = PatternDomain.REASONING
        else:
            domain_enum = domain
        with self._lock:
            domain_patterns = [
                p for p in self._patterns.values()
                if p.domain == domain_enum
            ]
            if not domain_patterns:
                domain_patterns = [
                    p for p in self._patterns.values()
                    if p.domain == PatternDomain.REASONING
                ]
        if not domain_patterns:
            return HarmonicSpectrum()
        avg_freq = sum(p.frequency for p in domain_patterns) / len(domain_patterns)
        harmonics: Dict[int, float] = {}
        for level in range(1, 6):
            freq = avg_freq * level
            amp = sum(
                p.amplitude for p in domain_patterns
                if abs(p.frequency - freq) < 0.15 * freq
            )
            if amp > 0:
                harmonics[level] = min(amp, 1.0)
        if not harmonics:
            harmonics = {1: sum(p.amplitude for p in domain_patterns) / len(domain_patterns)}
        total_energy = sum(harmonics.values()) / max(len(harmonics), 1)
        peak = max(harmonics, key=harmonics.get) if harmonics else 1
        spectrum = HarmonicSpectrum(
            fundamental_frequency=avg_freq,
            harmonics=harmonics,
            total_energy=total_energy,
            peak_harmonic=peak,
        )
        with self._stats_lock:
            self._stats["harmonic_spectra_computed"] += 1
        return copy.deepcopy(spectrum)

    def find_constructive(self, min_coherence: float = 0.6) -> List[ResonanceEvent]:
        min_coherence = _clamp(min_coherence)
        with self._lock:
            events = [
                copy.deepcopy(e) for e in self._events.values()
                if e.resonance_type in (ResonanceType.CONSTRUCTIVE, ResonanceType.SYNCHRONIZED)
                and e.coherence >= min_coherence
            ]
        return events

    def find_destructive(self) -> List[ResonanceEvent]:
        with self._lock:
            events = [
                copy.deepcopy(e) for e in self._events.values()
                if e.resonance_type in (ResonanceType.DESTRUCTIVE, ResonanceType.ANTI_PHASE)
            ]
        return events

    def amplify(self, pattern_id: str, factor: float = 1.5) -> Optional[CognitivePattern]:
        pid = _sanitize(pattern_id, 64)
        factor = max(0.1, min(factor, 5.0))
        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                return None
            pattern.amplitude = _clamp(pattern.amplitude * factor)
            with self._stats_lock:
                self._stats["amplifications"] += 1
            return copy.deepcopy(pattern)

    def dampen(self, pattern_id: str, factor: float = 0.5) -> Optional[CognitivePattern]:
        pid = _sanitize(pattern_id, 64)
        factor = max(0.01, min(factor, 0.99))
        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                return None
            pattern.amplitude = _clamp(pattern.amplitude * factor)
            with self._stats_lock:
                self._stats["dampenings"] += 1
            return copy.deepcopy(pattern)

    def synthesize_insight(self, pattern_ids: Optional[List[str]] = None) -> Optional[ResonanceInsight]:
        if pattern_ids is None:
            with self._lock:
                pattern_ids = list(self._patterns.keys())[:10]
        if not pattern_ids:
            return None
        sanitized_ids = [_sanitize(pid, 64) for pid in pattern_ids[:_MAX_LIST_ITEMS]]
        with self._lock:
            patterns = [
                self._patterns[pid]
                for pid in sanitized_ids
                if pid in self._patterns
            ]
        if len(patterns) < 2:
            return None
        domains = list(set(
            p.domain.value if isinstance(p.domain, PatternDomain) else str(p.domain)
            for p in patterns
        ))
        coherence_scores = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                coherence_scores.append(
                    _phase_coherence(patterns[i].phase, patterns[j].phase)
                )
        avg_coherence = sum(coherence_scores) / max(len(coherence_scores), 1)
        combined_amp = sum(p.amplitude for p in patterns) / len(patterns)
        if all(c > 0.7 for c in coherence_scores):
            insight_type = "convergent"
        elif any(c < 0.3 for c in coherence_scores):
            insight_type = "divergent"
        elif avg_coherence > 0.5:
            insight_type = "harmonic"
        else:
            insight_type = "partial"
        content_parts = [p.content[:200] for p in patterns if p.content]
        content = " | ".join(content_parts)
        insight = ResonanceInsight(
            source_domains=domains,
            insight_type=insight_type,
            content=content,
            confidence=_clamp(avg_coherence),
            amplitude=_clamp(combined_amp),
        )
        with self._lock:
            if len(self._insights) >= _MAX_INSIGHTS:
                evict_id = self._insight_order.popleft()
                self._insights.pop(evict_id, None)
                logger.warning("Evicting insight %s (cap=%d)", evict_id, _MAX_INSIGHTS)
            self._insights[insight.insight_id] = insight
            self._insight_order.append(insight.insight_id)
            with self._stats_lock:
                self._stats["insights_synthesized"] += 1
        return copy.deepcopy(insight)

    def get_pattern(self, pattern_id: str) -> Optional[CognitivePattern]:
        pid = _sanitize(pattern_id, 64)
        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                return None
            return copy.deepcopy(pattern)

    def get_resonance_events(self, domain: Optional[str] = None,
                             limit: int = 50) -> List[ResonanceEvent]:
        limit = max(1, min(500, limit))
        domain_enum = None
        if domain:
            if isinstance(domain, str):
                try:
                    domain_enum = PatternDomain(domain)
                except ValueError:
                    domain_enum = None
            elif isinstance(domain, PatternDomain):
                domain_enum = domain
        with self._lock:
            events = list(self._events.values())
        if domain_enum is not None:
            domain_patterns = {
                pid for pid, p in self._patterns.items()
                if p.domain == domain_enum
            }
            events = [
                e for e in events
                if any(pid in domain_patterns for pid in e.pattern_ids)
            ]
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return [copy.deepcopy(e) for e in events[:limit]]

    def get_insights(self, domain: Optional[str] = None,
                     limit: int = 50) -> List[ResonanceInsight]:
        limit = max(1, min(500, limit))
        domain_key = None
        if domain:
            if isinstance(domain, PatternDomain):
                domain_key = domain.value
            elif isinstance(domain, str):
                domain_key = domain
            else:
                domain_key = None
        with self._lock:
            insights = list(self._insights.values())
        if domain_key is not None:
            insights = [i for i in insights if domain_key in i.source_domains]
        insights.sort(key=lambda i: i.timestamp, reverse=True)
        return [copy.deepcopy(i) for i in insights[:limit]]

    def get_stats(self) -> Dict:
        with self._lock:
            total_patterns = len(self._patterns)
            total_events = len(self._events)
            total_insights = len(self._insights)
            domain_counts = {}
            for p in self._patterns.values():
                d = p.domain.value if isinstance(p.domain, PatternDomain) else str(p.domain)
                domain_counts[d] = domain_counts.get(d, 0) + 1
        with self._stats_lock:
            stats = dict(self._stats)
        stats["total_patterns"] = total_patterns
        stats["total_events"] = total_events
        stats["total_insights"] = total_insights
        stats["domain_distribution"] = dict(domain_counts)
        return stats

    def _classify_harmonic(self, freq_ratio: float) -> int:
        inv = 1.0 / max(freq_ratio, 0.001)
        ratio = min(freq_ratio, inv)
        harmonics = [
            (1.0, HarmonicLevel.FUNDAMENTAL.value, 0.05),
            (0.5, HarmonicLevel.SECOND.value, 0.05),
            (1.0 / 3.0, HarmonicLevel.THIRD.value, 0.04),
            (0.25, HarmonicLevel.FOURTH.value, 0.03),
            (0.2, HarmonicLevel.FIFTH.value, 0.02),
        ]
        for target, level, tolerance in harmonics:
            if abs(ratio - target) < tolerance:
                return level
        if freq_ratio >= 0.8:
            return HarmonicLevel.FUNDAMENTAL.value
        return HarmonicLevel.SECOND.value

    def _combine_amplitudes(self, a1: float, a2: float,
                            coherence: float, freq_ratio: float) -> float:
        inv = 1.0 / max(freq_ratio, 0.001)
        ratio = min(freq_ratio, inv)
        if abs(ratio - 1.0) < 0.05:
            combined = math.sqrt(a1 ** 2 + a2 ** 2 + 2 * a1 * a2 * coherence)
        elif abs(ratio - 0.5) < 0.05:
            combined = math.sqrt(a1 ** 2 + a2 ** 2 + a1 * a2 * coherence)
        else:
            combined = math.sqrt(a1 ** 2 + a2 ** 2) * (0.5 + 0.5 * coherence)
        return _clamp(combined)

    def _classify_resonance(self, coherence: float, freq_ratio: float) -> ResonanceType:
        inv = 1.0 / max(freq_ratio, 0.001)
        ratio = min(freq_ratio, inv)
        if abs(ratio - 1.0) < 0.1 and coherence > 0.8:
            return ResonanceType.SYNCHRONIZED
        if coherence > 0.6 and abs(ratio - 1.0) < 0.2:
            return ResonanceType.CONSTRUCTIVE
        if coherence < 0.3:
            return ResonanceType.ANTI_PHASE
        if coherence < 0.5:
            return ResonanceType.DESTRUCTIVE
        return ResonanceType.PARTIAL

    def _generate_insight_text(self, p1: CognitivePattern, p2: CognitivePattern,
                                resonance_type: ResonanceType,
                                combined_amp: float) -> str:
        d1 = p1.domain.value if isinstance(p1.domain, PatternDomain) else str(p1.domain)
        d2 = p2.domain.value if isinstance(p2.domain, PatternDomain) else str(p2.domain)
        rt = resonance_type.value if isinstance(resonance_type, ResonanceType) else str(resonance_type)
        parts = [p.content[:100] for p in (p1, p2) if p.content]
        if resonance_type in (ResonanceType.CONSTRUCTIVE, ResonanceType.SYNCHRONIZED):
            return "Amplified alignment between %s and %s (%s, amplitude=%.2f): %s" % (
                d1, d2, rt, combined_amp, " | ".join(parts)
            )
        elif resonance_type in (ResonanceType.DESTRUCTIVE, ResonanceType.ANTI_PHASE):
            return "Interference between %s and %s (%s): %s" % (
                d1, d2, rt, " | ".join(parts)
            )
        else:
            return "Partial resonance between %s and %s: %s" % (
                d1, d2, " | ".join(parts)
            )