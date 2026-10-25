"""
Feedback Loop Engine — Cross-Module Resonance and Emotional Contagion.

When Module A produces output, FeedLoop routes it to Modules B, C, D that need it.
Emotions detected by EmotionalIntelligence propagate to CognitiveTrinity (affecting
decision weighting), AttentionEngine (boosting priority), and PersonalityEngine
(recording behavioral patterns). Anomalies from AnomalyDetector feed into GoalReasoning
(obstacle detection) and EvolutionGuide (friction detection).

This is the "nervous system" connecting all 38 cognitive modules — not just data
passing, but weighted signal amplification, emotional contagion, and cascading
feedback loops that make the whole greater than the sum of its parts.
"""

import copy
import logging
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 500) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_RE.sub('', text)
    return text.strip()[:max_length]

_MAX_FEEDBACK_HISTORY = 2000
_MAX_ACTIVE_LOOPS = 500
_MAX_SIGNAL_AGE_SECONDS = 3600  # 1 hour
_MAX_RESONANCE_SIGNALS = 200


class SignalType(Enum):
    EMOTION = "emotion"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    GOAL = "goal"
    ATTENTION = "attention"
    SOCIAL = "social"
    COGNITIVE = "cognitive"
    HEALTH = "health"
    BEHAVIOR = "behavior"
    PREDICTION = "prediction"


class ContagionMode(Enum):
    AMPLIFY = "amplify"
    DAMPEN = "dampen"
    BROADCAST = "broadcast"
    TARGETED = "targeted"
    THRESHOLD = "threshold"


@dataclass
class FeedbackSignal:
    signal_type: str
    source_module: str
    target_modules: List[str]
    intensity: float
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signal_id: str = field(default_factory=lambda: "")

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = uuid.uuid4().hex[:12]
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.source_module = _SANITIZE_STRICT_RE.sub('', str(self.source_module))[:50]
        self.target_modules = [_SANITIZE_STRICT_RE.sub('', str(m))[:50] for m in (self.target_modules or [])[:20]]
        self.content = {_sanitize(str(k), 50): _sanitize_content(str(v), 500) for k, v in list(self.content.items())[:30]}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "source_module": self.source_module,
            "target_modules": list(self.target_modules),
            "intensity": round(self.intensity, 3),
            "content": copy.deepcopy(self.content) if isinstance(self.content, dict) else self.content,
            "timestamp": self.timestamp.isoformat(),
            "age_seconds": round((datetime.now(timezone.utc) - self.timestamp).total_seconds(), 1),
        }


@dataclass
class ResonanceEvent:
    event_id: str
    signals: List[FeedbackSignal]
    resonance_score: float
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        self.description = _SANITIZE_RE.sub('', str(self.description))[:500]
        self.resonance_score = max(0.0, min(1.0, self.resonance_score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "signal_count": len(self.signals),
            "resonance_score": round(self.resonance_score, 3),
            "description": self.description[:500],
            "timestamp": self.timestamp.isoformat(),
        }


# Routing table: source module → list of (target module, signal_type, weight)
_FEEDBACK_ROUTES: Dict[str, List[Tuple[str, str, float]]] = {
    "emotional_intelligence": [
        ("cognitive_trinity", "emotion", 0.8),
        ("attention_engine", "emotion", 0.6),
        ("personality", "behavior", 0.7),
        ("dream_state", "pattern", 0.3),
        ("empathy_mapping", "social", 0.9),
        ("conflict_resolution", "social", 0.5),
    ],
    "thought_sense": [
        ("cognitive_trinity", "cognitive", 0.9),
        ("attention_engine", "cognitive", 0.7),
        ("goal_reasoning", "cognitive", 0.4),
        ("moral_reasoning", "cognitive", 0.5),
        ("decision_journal", "cognitive", 0.6),
    ],
    "anomaly_detector": [
        ("goal_reasoning", "anomaly", 0.8),
        ("evolution_guide", "anomaly", 0.7),
        ("attention_engine", "anomaly", 0.9),
        ("autonomous_daemon", "anomaly", 0.6),
        ("temporal_reasoning", "anomaly", 0.4),
    ],
    "attention_engine": [
        ("cognitive_trinity", "attention", 0.7),
        ("personality", "behavior", 0.3),
        ("dream_state", "pattern", 0.2),
    ],
    "recall_synthesis": [
        ("creative_synthesis", "pattern", 0.6),
        ("narrative_memory", "pattern", 0.7),
        ("evolution_guide", "pattern", 0.4),
    ],
    "temporal_reasoning": [
        ("goal_reasoning", "prediction", 0.8),
        ("oracle_bridge", "prediction", 0.5),
        ("procedural_memory", "pattern", 0.4),
        ("attention_engine", "prediction", 0.3),
    ],
    "goal_reasoning": [
        ("attention_engine", "goal", 0.7),
        ("evolution_guide", "goal", 0.8),
        ("personality", "behavior", 0.3),
        ("conflict_resolution", "goal", 0.5),
    ],
    "device_context": [
        ("emotional_intelligence", "health", 0.8),
        ("attention_engine", "health", 0.9),
        ("cognitive_trinity", "health", 0.5),
        ("daily_briefing", "health", 0.7),
        ("personality", "behavior", 0.3),
    ],
    "presence_detection": [
        ("empathy_mapping", "social", 0.9),
        ("attention_engine", "social", 0.5),
        ("social_intelligence", "social", 0.8),
    ],
    "personality": [
        ("cognitive_trinity", "behavior", 0.5),
        ("goal_reasoning", "behavior", 0.4),
        ("evolution_guide", "behavior", 0.6),
    ],
    "moral_reasoning": [
        ("conflict_resolution", "cognitive", 0.7),
        ("decision_journal", "cognitive", 0.6),
        ("cognitive_trinity", "cognitive", 0.4),
    ],
    "dream_state": [
        ("recall_synthesis", "pattern", 0.5),
        ("creative_synthesis", "pattern", 0.6),
        ("narrative_memory", "pattern", 0.4),
    ],
    "creative_synthesis": [
        ("goal_reasoning", "cognitive", 0.3),
        ("evolution_guide", "cognitive", 0.4),
        ("narrative_memory", "pattern", 0.5),
    ],
    "social_intelligence": [
        ("empathy_mapping", "social", 0.8),
        ("conflict_resolution", "social", 0.7),
        ("personality", "behavior", 0.4),
    ],
    "decision_journal": [
        ("metacognition_engine", "cognitive", 0.7),
        ("evolution_guide", "cognitive", 0.5),
        ("personality", "behavior", 0.3),
    ],
    "cognitive_reflection": [
        ("metacognition_engine", "cognitive", 0.8),
        ("evolution_guide", "cognitive", 0.5),
        ("personality", "behavior", 0.4),
    ],
    "procedural_memory": [
        ("evolution_guide", "pattern", 0.6),
        ("goal_reasoning", "pattern", 0.3),
    ],
    "narrative_memory": [
        ("evolution_guide", "pattern", 0.5),
        ("creative_synthesis", "pattern", 0.4),
        ("dream_state", "pattern", 0.3),
    ],
}

# Reverse map: target module → what feeds into it (computed at init)
_REVERSE_ROUTES: Dict[str, List[Tuple[str, str, float]]] = {}
_REVERSE_ROUTES_LOCK = threading.Lock()


class FeedLoop:
    """
    Cross-Module Feedback Loop Engine.

    Routes signals between 38 cognitive modules based on a weighted routing table.
    Supports emotional contagion (emotions cascade), anomaly amplification (urgency
    propagation), and resonance detection (multiple signals combining).

    Usage:
        floop = FeedLoop()
        floop.emit("emotional_intelligence", "emotion", 0.8, {"primary": "stress"})
        signals = floop.poll("attention_engine")
        resonance = flook.detect_resonance()
    """

    def __init__(self):
        self._signals: deque = deque(maxlen=_MAX_FEEDBACK_HISTORY)
        self._active_loops: Dict[str, FeedbackSignal] = {}
        self._resonance_events: deque = deque(maxlen=_MAX_ACTIVE_LOOPS)
        self._signal_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "signals_emitted": 0,
            "signals_delivered": 0,
            "resonance_events": 0,
            "loops_created": 0,
            "loops_completed": 0,
        }
        self._contagion_thresholds: Dict[str, float] = {
            "emotion": 0.3,
            "anomaly": 0.5,
            "cognitive": 0.2,
            "social": 0.3,
            "attention": 0.4,
            "health": 0.6,
        }
        self._build_reverse_routes()

    def _build_reverse_routes(self):
        global _REVERSE_ROUTES
        with _REVERSE_ROUTES_LOCK:
            _REVERSE_ROUTES.clear()
            for source, targets in _FEEDBACK_ROUTES.items():
                for target, signal_type, weight in targets:
                    if target not in _REVERSE_ROUTES:
                        _REVERSE_ROUTES[target] = []
                    if (source, signal_type, weight) not in _REVERSE_ROUTES[target]:
                        _REVERSE_ROUTES[target].append((source, signal_type, weight))

    def emit(self, source_module: str, signal_type: str, intensity: float,
             content: Optional[Dict[str, Any]] = None,
             mode: str = "broadcast") -> List[str]:
        """Emit a signal from a source module. Returns list of signal IDs created."""
        source_module = _sanitize(source_module, 50)
        signal_type = _sanitize(signal_type, 30)
        mode = _sanitize(mode, 20)
        intensity = max(0.0, min(1.0, float(intensity)))
        if isinstance(content, dict):
            content = {_sanitize(str(k), 50): _sanitize_content(str(v), 500) for k, v in list(content.items())[:30]}
        else:
            content = {}
        with _REVERSE_ROUTES_LOCK:
            routes = _FEEDBACK_ROUTES.get(source_module, [])
            if not routes and mode == "broadcast":
                routes = [(m, signal_type, 0.3) for m in _REVERSE_ROUTES.get(source_module, [])]

        created_ids = []
        with self._lock:
            threshold = self._contagion_thresholds.get(signal_type, 0.2)

        for target_module, routed_type, weight in routes:
            if mode == "threshold" and intensity < threshold:
                continue

            effective_intensity = intensity * weight

            if mode == "amplify":
                effective_intensity = min(1.0, effective_intensity * 1.5)
            elif mode == "dampen":
                effective_intensity = effective_intensity * 0.5

            signal = FeedbackSignal(
                signal_type=routed_type,
                source_module=source_module,
                target_modules=[target_module],
                intensity=round(effective_intensity, 4),
                content=content,
            )

            with self._lock:
                self._signals.append(signal)
                with self._stats_lock:
                    self._stats["signals_emitted"] += 1

            created_ids.append(signal.signal_id)

        return created_ids

    def poll(self, target_module: str, signal_type: Optional[str] = None,
             limit: int = 10) -> List[FeedbackSignal]:
        """Poll for signals targeting a specific module."""
        target_module = _sanitize(target_module, 50)
        if signal_type:
            signal_type = _sanitize(signal_type, 30)
        limit = max(1, min(limit, 100))
        now = datetime.now(timezone.utc)

        with self._lock:
            relevant = []
            for signal in reversed(self._signals):
                age = (now - signal.timestamp).total_seconds()
                if age > _MAX_SIGNAL_AGE_SECONDS:
                    continue
                if target_module in signal.target_modules:
                    if signal_type is None or signal.signal_type == signal_type:
                        relevant.append(signal)
                        if len(relevant) >= limit:
                            break

        with self._stats_lock:
            self._stats["signals_delivered"] += len(relevant)

        return [copy.deepcopy(s) for s in relevant]

    def snapshot_signals(self, limit: int = _MAX_FEEDBACK_HISTORY,
                         max_age_seconds: float = _MAX_SIGNAL_AGE_SECONDS) -> List[FeedbackSignal]:
        """Return a deep-copied snapshot of recent signals, oldest first.

        Unlike poll(), this does not filter by target and does not touch
        delivery stats — it exists for the ShadowOS dispatcher to consume
        signals exactly once and route them into real module calls.
        """
        limit = max(1, min(limit, _MAX_FEEDBACK_HISTORY))
        max_age_seconds = max(1.0, min(max_age_seconds, _MAX_SIGNAL_AGE_SECONDS))
        now = datetime.now(timezone.utc)
        with self._lock:
            signals = [
                s for s in self._signals
                if (now - s.timestamp).total_seconds() <= max_age_seconds
            ]
        return [copy.deepcopy(s) for s in signals[-limit:]]


    def create_loop(self, source_module: str, target_module: str,
                    signal_type: str, intensity: float,
                    content: Optional[Dict[str, Any]] = None) -> str:
        """Create a persistent feedback loop between two modules."""
        source_module = _sanitize(source_module, 50)
        target_module = _sanitize(target_module, 50)
        signal_type = _sanitize(signal_type, 30)
        intensity = max(0.0, min(1.0, float(intensity)))
        if content and isinstance(content, dict):
            content = {str(k)[:50]: _sanitize_content(str(v), 500) for k, v in list(content.items())[:30]}
        else:
            content = content or {}

        signal = FeedbackSignal(
            signal_type=signal_type,
            source_module=source_module,
            target_modules=[target_module],
            intensity=intensity,
            content=content or {},
        )

        with self._lock:
            if len(self._active_loops) >= _MAX_ACTIVE_LOOPS:
                oldest_key = next(iter(self._active_loops))
                logger.warning("Evicting loop %s to stay within max_active_loops=%d", oldest_key, _MAX_ACTIVE_LOOPS)
                del self._active_loops[oldest_key]
            self._active_loops[signal.signal_id] = signal
            with self._stats_lock:
                self._stats["loops_created"] += 1

        return signal.signal_id

    def complete_loop(self, loop_id: str, outcome_intensity: float,
                      outcome_content: Optional[Dict[str, Any]] = None) -> Optional[FeedbackSignal]:
        """Complete a feedback loop with outcome data."""
        loop_id = _sanitize(loop_id, 12)
        with self._lock:
            loop = self._active_loops.pop(loop_id, None)

        if not loop:
            return None

        outcome = FeedbackSignal(
            signal_type=f"loop_outcome_{loop.signal_type}",
            source_module=loop.target_modules[0] if loop.target_modules else "unknown",
            target_modules=[loop.source_module],
            intensity=max(0.0, min(1.0, outcome_intensity)),
            content=outcome_content if isinstance(outcome_content, dict) else loop.content,
        )

        with self._lock:
            self._signals.append(outcome)
        with self._stats_lock:
            self._stats["loops_completed"] += 1

        return copy.deepcopy(outcome)

    def detect_resonance(self, min_signals: int = 3, time_window_seconds: float = 300.0,
                         min_intensity: float = 0.3) -> List[ResonanceEvent]:
        """Detect when multiple signals converge on similar themes within a time window."""
        min_signals = max(2, min(min_signals, 20))
        time_window_seconds = max(1.0, min(time_window_seconds, 86400.0))
        min_intensity = max(0.0, min(1.0, min_intensity))
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - time_window_seconds

        with self._lock:
            recent = [s for s in self._signals if s.timestamp.timestamp() >= cutoff]
            recent = copy.deepcopy(recent)

        if len(recent) < min_signals:
            return []

        source_groups: Dict[str, List[FeedbackSignal]] = {}
        for signal in recent:
            if signal.intensity < min_intensity:
                continue
            key = signal.source_module
            if key not in source_groups:
                source_groups[key] = []
            source_groups[key].append(signal)

        resonance_events = []
        for source, signals in source_groups.items():
            if len(signals) < min_signals:
                continue

            avg_intensity = sum(s.intensity for s in signals) / len(signals)

            unique_targets = set()
            for s in signals:
                unique_targets.update(s.target_modules)

            if len(unique_targets) >= 2 and avg_intensity >= min_intensity:
                event = ResonanceEvent(
                    event_id=uuid.uuid4().hex[:12],
                    signals=signals[:_MAX_RESONANCE_SIGNALS],
                    resonance_score=round(min(1.0, avg_intensity * len(unique_targets) / 5.0), 3),
                    description=f"Resonance in {source}: {len(signals)} signals across {len(unique_targets)} targets",
                )
                resonance_events.append(event)

        resonance_events.sort(key=lambda e: e.resonance_score, reverse=True)

        for event in resonance_events[:10]:
            with self._lock:
                self._resonance_events.append(event)
        with self._stats_lock:
            self._stats["resonance_events"] += len(resonance_events)

        return [copy.deepcopy(e) for e in resonance_events[:10]]

    def get_active_loops(self) -> List[Dict[str, Any]]:
        """Get all active feedback loops."""
        with self._lock:
            return [copy.deepcopy(loop).to_dict() for loop in self._active_loops.values()]

    def get_signals_by_source(self, source_module: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent signals from a specific source module."""
        source_module = _sanitize(source_module, 50)
        limit = max(1, min(limit, 100))

        with self._lock:
            signals = [s for s in self._signals if s.source_module == source_module]

        return [copy.deepcopy(s).to_dict() for s in signals[-limit:]]

    def get_signal_counts(self) -> Dict[str, int]:
        """Get signal counts by type."""
        with self._lock:
            counts: Dict[str, int] = {}
            for signal in self._signals:
                counts[signal.signal_type] = counts.get(signal.signal_type, 0) + 1

        return dict(counts)

    def get_routes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the routing table."""
        return {
            source: [
                {"target": t, "signal_type": st, "weight": round(w, 4)}
                for t, st, w in targets
            ]
            for source, targets in _FEEDBACK_ROUTES.items()
        }

    def get_reverse_routes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the reverse routing table (what feeds into each module)."""
        return {
            target: [
                {"source": s, "signal_type": st, "weight": round(w, 4)}
                for s, st, w in sources
            ]
            for target, sources in _REVERSE_ROUTES.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback loop statistics."""
        with self._stats_lock:
            return dict(self._stats)

    def get_resonance_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent resonance events."""
        limit = max(1, min(limit, 100))
        with self._lock:
            return [copy.deepcopy(e).to_dict() for e in list(self._resonance_events)[-limit:]]

    def set_contagion_threshold(self, signal_type: str, threshold: float) -> None:
        signal_type = _sanitize(signal_type, 30)
        threshold = max(0.0, min(1.0, float(threshold)))
        with self._lock:
            self._contagion_thresholds[signal_type] = threshold
            if len(self._contagion_thresholds) > 100:
                oldest = next(iter(self._contagion_thresholds))
                del self._contagion_thresholds[oldest]

    def clear(self) -> None:
        """Clear all signals and active loops."""
        with self._lock:
            self._signals.clear()
            self._active_loops.clear()
            self._resonance_events.clear()