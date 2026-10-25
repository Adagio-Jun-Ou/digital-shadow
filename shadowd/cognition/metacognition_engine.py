#!/usr/bin/env python3
"""
MetacognitionEngine - Thinking About Thinking

Monitors and regulates the cognitive system's own reasoning processes:
confidence calibration, bias detection, strategy selection, and 
cognitive load management.

Architecture:
  - ThinkingEpisode: tracked reasoning episode with confidence and outcome
  - CognitiveBias: detected bias pattern with mitigation strategy
  - MetacognitiveAlert: system alert about reasoning quality
  - MetacognitionEngine: main engine for metacognitive monitoring
  - Thread-safe, bounded history, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("shadowd.cognition.metacognition_engine")

_MAX_EPISODES = 5000
_MAX_BIASES = 200
_MAX_ALERTS = 500
_MAX_HISTORY = 1000

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 5000) -> str:
    text = _SANITIZE_STRICT_RE.sub('', text).strip()
    return text[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    text = _SANITIZE_CONTENT_RE.sub('', text).strip()
    return text[:max_len]


def _word_match(keyword: str, text: str, compiled: "Optional[re.Pattern]" = None) -> bool:
    if compiled is not None:
        return bool(compiled.search(text))
    if ' ' in keyword:
        return keyword in text
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))


def _count_word_matches(keyword: str, text: str, compiled: "Optional[re.Pattern]" = None) -> int:
    if compiled is not None:
        return len(compiled.findall(text))
    if ' ' in keyword:
        return text.count(keyword)
    return len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))


class ReasoningType(str, Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    INTUITIVE = "intuitive"
    SYSTEMATIC = "systematic"
    CREATIVE = "creative"


class BiasType(str, Enum):
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    OVERCONFIDENCE = "overconfidence"
    RECENCY = "recency"
    SUNK_COST = "sunk_cost"
    FRAMING = "framing"
    DUNNING_KRUGER = "dunning_kruger"
    BANDWAGON = "bandwagon"


class AlertSeverity(int, Enum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2


class CognitiveState(str, Enum):
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    OVERLOADED = "overloaded"
    REFLECTIVE = "reflective"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


@dataclass
class ThinkingEpisode:
    episode_id: str = ""
    reasoning_type: ReasoningType = ReasoningType.SYSTEMATIC
    topic: str = ""
    confidence: float = 0.5
    calibration_offset: float = 0.0
    outcome: Optional[str] = None
    outcome_accuracy: Optional[float] = None
    biases_detected: List[str] = field(default_factory=list)
    strategies_used: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    cognitive_load: float = 0.3
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.episode_id:
            self.episode_id = f"think_{uuid.uuid4().hex[:12]}"
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.cognitive_load = max(0.0, min(1.0, self.cognitive_load))
        if self.metadata:
            self.metadata = {k: _sanitize_content(str(v), 500) for k, v in self.metadata.items()}
        if len(self.metadata) > 50:
            self.metadata = dict(list(self.metadata.items())[:50])

    def resolve(self, outcome: str, accuracy: float):
        self.outcome = outcome
        self.outcome_accuracy = max(0.0, min(1.0, accuracy))
        self.calibration_offset = self.outcome_accuracy - self.confidence
        self.resolved_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "reasoning_type": self.reasoning_type.value,
            "topic": self.topic[:200],
            "confidence": round(self.confidence, 3),
            "calibration_offset": round(self.calibration_offset, 3),
            "outcome_accuracy": round(self.outcome_accuracy, 3) if self.outcome_accuracy is not None else None,
            "biases_detected": list(self.biases_detected[:5]),
            "strategies_used": list(self.strategies_used[:5]),
            "cognitive_load": round(self.cognitive_load, 3),
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }


@dataclass
class CognitiveBias:
    bias_id: str = ""
    bias_type: BiasType = BiasType.CONFIRMATION
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    mitigation: str = ""
    frequency: int = 1
    severity: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.bias_id:
            self.bias_id = f"bias_{uuid.uuid4().hex[:12]}"
        self.severity = max(0.0, min(1.0, self.severity))

    def to_dict(self) -> Dict:
        return {
            "bias_id": self.bias_id,
            "bias_type": self.bias_type.value,
            "description": self.description[:200],
            "evidence": list(self.evidence[:5]),
            "mitigation": self.mitigation[:200],
            "frequency": self.frequency,
            "severity": round(self.severity, 3),
        }


@dataclass
class MetacognitiveAlert:
    alert_id: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    title: str = ""
    description: str = ""
    recommendation: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dismissed: bool = False

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"meta_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title[:100],
            "description": self.description[:500],
            "recommendation": self.recommendation[:200],
            "created_at": self.created_at,
            "dismissed": self.dismissed,
        }


_BIAS_DETECTION_KEYWORDS = {
    BiasType.CONFIRMATION: ["only looking for", "confirms my belief", "i knew it all along", "ignores evidence"],
    BiasType.ANCHORING: ["starting from", "based on initial", "locked onto", "first impression"],
    BiasType.AVAILABILITY: ["recent example", "comes to mind", "easily remembered", "salient case"],
    BiasType.OVERCONFIDENCE: ["definitely", "certain", "impossible to be wrong", "guaranteed"],
    BiasType.RECENCY: ["just happened", "latest result", "most recent", "right now"],
    BiasType.SUNK_COST: ["already invested", "too much effort", "can't give up now", "committed"],
    BiasType.FRAMING: ["framed as", "presented as", "looks different when", "perspective change"],
    BiasType.DUNNING_KRUGER: ["simple really", "anyone can", "obvious", "no expertise needed"],
    BiasType.BANDWAGON: ["everyone thinks", "popular opinion", "most people", "consensus"],
}

_BIAS_COMPILED: Dict[BiasType, List[Tuple[str, "re.Pattern"]]] = {}
for _bt, _keywords in _BIAS_DETECTION_KEYWORDS.items():
    _BIAS_COMPILED[_bt] = []
    for _kw in _keywords:
        if ' ' in _kw:
            _BIAS_COMPILED[_bt].append((_kw, None))
        else:
            _BIAS_COMPILED[_bt].append((_kw, re.compile(r'\b' + re.escape(_kw) + r'\b')))

_BIAS_MITIGATIONS = {
    BiasType.CONFIRMATION: "Actively seek disconfirming evidence. List reasons you might be wrong.",
    BiasType.ANCHORING: "Consider multiple starting points. Generate independent estimates before comparing.",
    BiasType.AVAILABILITY: "Use systematic data rather than memorable examples. Check base rates.",
    BiasType.OVERCONFIDENCE: "Assign wider confidence intervals. Ask what would change your mind.",
    BiasType.RECENCY: "Look at longer time horizons. Weight historical data more heavily.",
    BiasType.SUNK_COST: "Evaluate future costs and benefits independently of past investment.",
    BiasType.FRAMING: "Reframe from multiple perspectives. Ask what the opposite framing would suggest.",
    BiasType.DUNNING_KRUGER: "Consult domain experts. Acknowledge the limits of your knowledge.",
    BiasType.BANDWAGON: "Evaluate evidence independently. Consider minority viewpoints.",
}


class MetacognitionEngine:
    """
    Monitors and regulates the cognitive system's own reasoning.

    Tracks thinking episodes, detects cognitive biases, calibrates confidence,
    manages cognitive load, and generates alerts about reasoning quality.
    """

    def __init__(self):
        self._episodes: Dict[str, ThinkingEpisode] = {}
        self._open_episodes: Dict[str, ThinkingEpisode] = {}
        self._biases: Dict[str, CognitiveBias] = {}
        self._alerts: deque = deque(maxlen=_MAX_ALERTS)
        self._cognitive_state = CognitiveState.FOCUSED
        self._cognitive_load: float = 0.3
        self._lock = threading.Lock()
        self._stats = {
            "episodes_tracked": 0,
            "episodes_resolved": 0,
            "biases_detected": 0,
            "alerts_generated": 0,
            "avg_confidence": 0.5,
            "avg_calibration_offset": 0.0,
            "avg_cognitive_load": 0.3,
        }
        self._stats_lock = threading.Lock()

    def start_thinking(self, reasoning_type: ReasoningType, topic: str,
                        confidence: float = 0.5, cognitive_load: float = 0.3) -> ThinkingEpisode:
        topic = _sanitize(topic, max_len=500)
        confidence = max(0.0, min(1.0, confidence))
        cognitive_load = max(0.0, min(1.0, cognitive_load))

        episode = ThinkingEpisode(
            reasoning_type=reasoning_type,
            topic=topic,
            confidence=confidence,
            cognitive_load=cognitive_load,
        )

        with self._lock:
            self._episodes[episode.episode_id] = episode
            self._open_episodes[episode.episode_id] = episode
            if len(self._episodes) > _MAX_EPISODES:
                oldest = list(self._episodes.keys())[0]
                logger.warning("Evicting episode %s (may be referenced by biases)", oldest)
                self._episodes.pop(oldest, None)
                self._open_episodes.pop(oldest, None)
            if len(self._open_episodes) > _MAX_HISTORY:
                oldest_open = list(self._open_episodes.keys())[0]
                logger.warning("Evicting open episode %s", oldest_open)
                self._open_episodes.pop(oldest_open, None)
            self._cognitive_load = cognitive_load
            with self._stats_lock:
                self._stats["episodes_tracked"] += 1
                n = self._stats["episodes_tracked"]
                self._stats["avg_confidence"] = self._stats["avg_confidence"] + (confidence - self._stats["avg_confidence"]) / n
                self._stats["avg_cognitive_load"] = self._stats["avg_cognitive_load"] + (cognitive_load - self._stats["avg_cognitive_load"]) / n

        self._update_cognitive_state()

        return copy.deepcopy(episode)

    def end_thinking(self, episode_id: Optional[str] = None,
                     outcome: str = "completed", confidence_after: float = 0.5) -> Optional[ThinkingEpisode]:
        if episode_id is None:
            with self._lock:
                if not self._open_episodes:
                    return None
                episode_id = next(iter(self._open_episodes))
        accuracy = max(0.0, min(1.0, confidence_after))
        return self.resolve_thinking(episode_id, outcome, accuracy)

    def resolve_thinking(self, episode_id: str, outcome: str, accuracy: float) -> Optional[ThinkingEpisode]:
        accuracy = max(0.0, min(1.0, accuracy))
        outcome = _sanitize_content(outcome, max_len=1000)

        with self._lock:
            episode = self._episodes.get(episode_id)
            if not episode:
                return None
            episode.resolve(outcome, accuracy)
            self._open_episodes.pop(episode_id, None)
            with self._stats_lock:
                self._stats["episodes_resolved"] += 1
                total_resolved = self._stats["episodes_resolved"]
                old_offset = self._stats["avg_calibration_offset"]
                self._stats["avg_calibration_offset"] = old_offset + (episode.calibration_offset - old_offset) / total_resolved

        self._check_calibration_alert(episode)

        return copy.deepcopy(episode)

    def detect_biases(self, text: str = "", episode_id: Optional[str] = None) -> List[CognitiveBias]:
        text = _sanitize_content(text, max_len=5000)
        lower = text.lower()
        detected = []
        seen_bias_types = set()

        with self._lock:
            episode = self._episodes.get(episode_id) if episode_id else None

            for bias_type, compiled_kws in _BIAS_COMPILED.items():
                if bias_type in seen_bias_types:
                    continue
                for keyword, pat in compiled_kws:
                    if _word_match(keyword, lower, pat):
                        match_count = _count_word_matches(keyword, lower, pat)
                        bias = CognitiveBias(
                            bias_type=bias_type,
                            description=f"Detected {bias_type.value} bias in reasoning",
                            evidence=[keyword],
                            mitigation=_BIAS_MITIGATIONS.get(bias_type, "Review reasoning carefully"),
                            severity=0.5 + 0.1 * match_count,
                        )
                        detected.append(bias)
                        seen_bias_types.add(bias_type)

                        if episode:
                            episode.biases_detected.append(bias_type.value)

                        existing = next(
                            (b for b in self._biases.values() if b.bias_type == bias_type),
                            None
                        )
                        if existing:
                            existing.frequency += 1
                            existing.evidence.extend([keyword])
                            if len(existing.evidence) > 20:
                                existing.evidence = existing.evidence[-20:]
                            existing.severity = min(1.0, existing.severity + 0.05)
                        else:
                            self._biases[bias.bias_id] = bias
                            if len(self._biases) > _MAX_BIASES:
                                oldest = list(self._biases.keys())[0]
                                del self._biases[oldest]
                        with self._stats_lock:
                            self._stats["biases_detected"] += 1
                        break

        return copy.deepcopy(detected)

    def _check_calibration_alert(self, episode: ThinkingEpisode):
        if abs(episode.calibration_offset) > 0.3:
            direction = "overconfident" if episode.calibration_offset < 0 else "underconfident"
            alert = MetacognitiveAlert(
                severity=AlertSeverity.WARNING if abs(episode.calibration_offset) < 0.5 else AlertSeverity.CRITICAL,
                title=f"Calibration alert: {direction}",
                description=f"Confidence was {episode.confidence:.1%} but accuracy was {episode.outcome_accuracy:.1%}. Offset: {episode.calibration_offset:+.3f}",
                recommendation="Adjust confidence estimates and seek more evidence before high-confidence claims.",
            )
            with self._lock:
                self._alerts.append(alert)
                with self._stats_lock:
                    self._stats["alerts_generated"] += 1

    def _update_cognitive_state(self):
        with self._lock:
            if self._cognitive_load > 0.8:
                self._cognitive_state = CognitiveState.OVERLOADED
            elif self._cognitive_load >= 0.6:
                self._cognitive_state = CognitiveState.DISTRACTED
            elif self._cognitive_load >= 0.3:
                self._cognitive_state = CognitiveState.ANALYTICAL
            else:
                self._cognitive_state = CognitiveState.FOCUSED

    def assess_confidence(self, base_confidence: float, evidence_count: int = 0,
                           disconfirming_count: int = 0) -> float:
        base_confidence = max(0.0, min(1.0, base_confidence))
        adjustment = 0.0

        adjustment += min(evidence_count * 0.05, 0.2)
        adjustment += min(disconfirming_count * 0.03, 0.1)

        with self._lock:
            avg_offset = self._stats["avg_calibration_offset"]

        calibrated = base_confidence + adjustment + avg_offset * 0.3
        return max(0.0, min(1.0, calibrated))

    def get_cognitive_load(self) -> Dict:
        with self._lock:
            current_load = self._cognitive_load
            state = self._cognitive_state.value
            open_eps = len(self._open_episodes)
            if current_load > 0.8:
                recommendation = "Cognitive overload detected. Delegate tasks or take a break."
            elif current_load > 0.6:
                recommendation = "Moderate load. Prioritize and avoid multitasking."
            else:
                recommendation = "Cognitive capacity available for complex tasks."
        return {
            "current_load": round(current_load, 3),
            "state": state,
            "open_episodes": open_eps,
            "recommendation": recommendation,
        }

    def get_calibration_report(self) -> Dict:
        with self._lock:
            resolved = [e for e in self._episodes.values() if e.outcome_accuracy is not None]

        if not resolved:
            return {"status": "insufficient_data", "episodes_resolved": 0}

        avg_confidence = sum(e.confidence for e in resolved) / len(resolved)
        avg_accuracy = sum(e.outcome_accuracy for e in resolved) / len(resolved)
        avg_offset = sum(e.calibration_offset for e in resolved) / len(resolved)

        overconfident = sum(1 for e in resolved if e.calibration_offset < -0.1)
        underconfident = sum(1 for e in resolved if e.calibration_offset > 0.1)

        return {
            "total_resolved": len(resolved),
            "avg_confidence": round(avg_confidence, 3),
            "avg_accuracy": round(avg_accuracy, 3),
            "avg_calibration_offset": round(avg_offset, 3),
            "overconfident_count": overconfident,
            "underconfident_count": underconfident,
            "well_calibrated_count": len(resolved) - overconfident - underconfident,
        }

    def get_biases(self, bias_type: Optional[BiasType] = None) -> List[Dict]:
        with self._lock:
            biases = list(self._biases.values())
        if bias_type:
            biases = [b for b in biases if b.bias_type == bias_type]
        biases.sort(key=lambda b: b.severity, reverse=True)
        return [b.to_dict() for b in biases[:30]]

    def get_alerts(self, include_dismissed: bool = False, limit: int = 20) -> List[Dict]:
        with self._lock:
            alerts = list(self._alerts)
        if not include_dismissed:
            alerts = [a for a in alerts if not a.dismissed]
        return [a.to_dict() for a in alerts[-limit:]]

    def dismiss_alert(self, alert_id: str) -> bool:
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.dismissed = True
                    return True
        return False

    def get_stats(self) -> Dict:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._lock:
            cognitive_state = self._cognitive_state.value
            cognitive_load = round(self._cognitive_load, 3)
            episodes_total = len(self._episodes)
            open_episodes = len(self._open_episodes)
            biases_tracked = len(self._biases)
            alerts_active = sum(1 for a in self._alerts if not a.dismissed)
        return {
            "stats": stats,
            "cognitive_state": cognitive_state,
            "cognitive_load": cognitive_load,
            "episodes_total": episodes_total,
            "open_episodes": open_episodes,
            "biases_tracked": biases_tracked,
            "alerts_active": alerts_active,
        }