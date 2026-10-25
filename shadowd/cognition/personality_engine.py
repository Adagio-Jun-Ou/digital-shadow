#!/usr/bin/env python3
"""
PersonalityEngine - User Personality Distillation for Shadow AI

Implements personality modeling and persona distillation from behavioral data.
The shadow doesn't just have a fixed personality - it learns yours and can
reflect your decision-making patterns.

Architecture:
  PersonalityEngine
  ├── BigFiveProfile (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
  ├── DecisionStyle (How you make decisions: analytical, intuitive, cautious, bold)
  ├── CommunicationStyle (How you write/speak: formal, casual, terse, verbose)
  ├── ValueSystem (What you prioritize: speed, quality, relationships, money)
  ├── BehaviorPattern (Observed behavioral patterns with confidence)
  └── PersonaDistiller (LLM-powered persona distillation from observations)

Inspired by:
- Big Five personality traits (OCEAN model)
- Myers-Briggs decision styles
- Letta's self-editing memory blocks
- The concept of "personality fingerprinting" for digital twins
"""

import asyncio
import copy
import json
import hashlib
import math
import os
import re
import threading
import time
import uuid
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from pathlib import Path

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()


def _sanitize_content(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_CONTENT_RE.sub('', str(text)).strip()

logger = logging.getLogger("shadowd.cognition.personality_engine")


class TraitDimension(str, Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class DecisionStyle(str, Enum):
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    CAUTIOUS = "cautious"
    BOLD = "bold"
    DELIBERATIVE = "deliberative"
    SPONTANEOUS = "spontaneous"


class CommunicationStyle(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    TERSE = "terse"
    VERBOSE = "verbose"
    TECHNICAL = "technical"
    NARRATIVE = "narrative"


class ValuePriority(str, Enum):
    SPEED = "speed"
    QUALITY = "quality"
    RELATIONSHIPS = "relationships"
    MONEY = "money"
    LEARNING = "learning"
    SECURITY = "security"
    FREEDOM = "freedom"
    CREATIVITY = "creativity"


@dataclass
class BehaviorPattern:
    pattern_id: str = ""
    name: str = ""
    description: str = ""
    evidence_count: int = 0
    confidence: float = 0.0
    first_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    examples: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = f"pat_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "evidence_count": self.evidence_count,
            "confidence": round(self.confidence, 3),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "examples": list(self.examples[:5]),
            "implications": list(self.implications[:5]),
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }

    def reinforce(self, example: str = ""):
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        self.last_seen = datetime.now(timezone.utc).isoformat()
        if example and len(self.examples) < 20:
            self.examples.append(example)


@dataclass
class BigFiveProfile:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    evidence_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "openness": round(self.openness, 3),
            "conscientiousness": round(self.conscientiousness, 3),
            "extraversion": round(self.extraversion, 3),
            "agreeableness": round(self.agreeableness, 3),
            "neuroticism": round(self.neuroticism, 3),
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated,
        }

    def update(self, dimension: TraitDimension, delta: float):
        """Apply a trait update with evidence-based diminishing returns."""
        BASELINE = 0.5
        DECAY_RATE = 0.001
        current = getattr(self, dimension.value)
        decayed = current + DECAY_RATE * (BASELINE - current)
        weight = 1.0 / (1.0 + self.evidence_count / 50.0)
        updated = decayed + delta * weight
        setattr(self, dimension.value, max(0.0, min(1.0, updated)))
        self.evidence_count += 1
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def to_prompt_fragment(self) -> str:
        dominant = []
        if self.openness > 0.6:
            dominant.append("curious and open to new ideas")
        if self.conscientiousness > 0.6:
            dominant.append("organized and detail-oriented")
        if self.extraversion > 0.6:
            dominant.append("socially energetic and outgoing")
        if self.agreeableness > 0.6:
            dominant.append("cooperative and considerate")
        if self.neuroticism > 0.6:
            dominant.append("sensitive to stress and uncertainty")

        if self.openness < 0.4:
            dominant.append("practical and conventional")
        if self.conscientiousness < 0.4:
            dominant.append("flexible and spontaneous")
        if self.extraversion < 0.4:
            dominant.append("prefers solitude and reflection")
        if self.agreeableness < 0.4:
            dominant.append("direct and competitive")
        if self.neuroticism < 0.4:
            dominant.append("emotionally stable and calm")

        return "The user tends to be " + ", ".join(dominant) + "." if dominant else "Personality profile still forming."


@dataclass
class PersonalityProfile:
    profile_id: str = ""
    user_name: str = ""
    big_five: BigFiveProfile = field(default_factory=BigFiveProfile)
    decision_style: DecisionStyle = DecisionStyle.DELIBERATIVE
    communication_style: CommunicationStyle = CommunicationStyle.CASUAL
    value_priorities: List[ValuePriority] = field(default_factory=lambda: [ValuePriority.LEARNING])
    behavior_patterns: List[BehaviorPattern] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    decision_history: List[Dict] = field(default_factory=list)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1

    def __post_init__(self):
        if not self.profile_id:
            self.profile_id = f"pers_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "profile_id": self.profile_id,
            "user_name": self.user_name,
            "big_five": self.big_five.to_dict(),
            "decision_style": self.decision_style.value,
            "communication_style": self.communication_style.value,
            "value_priorities": [v.value for v in self.value_priorities],
            "behavior_patterns_count": len(self.behavior_patterns),
            "preferences": copy.deepcopy(self.preferences),
            "decision_history_count": len(self.decision_history),
            "interaction_patterns": copy.deepcopy(self.interaction_patterns),
            "version": self.version,
            "updated_at": self.updated_at,
        }

    def to_system_prompt(self) -> str:
        parts = [
            f"<personality_profile>",
            f"User tends to make {self.decision_style.value} decisions.",
            f"User communicates in a {self.communication_style.value} style.",
            f"User values: {', '.join(v.value for v in self.value_priorities[:3])}.",
            self.big_five.to_prompt_fragment(),
        ]
        if self.preferences:
            parts.append(f"Known preferences: {json.dumps(self.preferences)[:500]}")
        parts.append("</personality_profile>")
        return "\n".join(parts)

    _MAX_PREFERENCES = 500
    _MAX_INTERACTION_PATTERNS = 500

    def add_decision(self, decision: Dict):
        sanitized = {_sanitize(str(k)): _sanitize(str(v))[:500] if isinstance(v, str) else v for k, v in decision.items()}
        self.decision_history.append({
            **sanitized,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.decision_history) > 500:
            self.decision_history = self.decision_history[-500:]
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def find_pattern(self, pattern_name: str) -> Optional[BehaviorPattern]:
        for p in self.behavior_patterns:
            if p.name == pattern_name:
                return p
        return None

    def add_or_reinforce_pattern(self, name: str, description: str, example: str = "") -> BehaviorPattern:
        name = _sanitize(name)[:200]
        description = _sanitize_content(description)[:500]
        existing = self.find_pattern(name)
        if existing:
            existing.reinforce(example)
            return existing
        if len(self.behavior_patterns) >= PersonalityEngine.MAX_BEHAVIOR_PATTERNS:
            self.behavior_patterns.sort(key=lambda p: p.confidence)
            evicted = self.behavior_patterns.pop(0)
            logger.warning("Evicted behavior pattern '%s' (confidence=%.3f) to make room for '%s'", evicted.name, evicted.confidence, name)
        pattern = BehaviorPattern(name=name, description=description)
        if example:
            pattern.examples.append(example)
            pattern.evidence_count = 1
            pattern.confidence = 0.1
        self.behavior_patterns.append(pattern)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return pattern


class PersonaDistiller:
    def __init__(self):
        self._behavior_signals: deque = deque(maxlen=5000)
        self._signal_timestamps: deque = deque(maxlen=100)
        self._max_signals_per_second: int = 10

    def record_behavior(self, event_type: str, details: Dict[str, Any]):
        import time as _time
        now = _time.monotonic()
        self._signal_timestamps = deque(
            (t for t in self._signal_timestamps if now - t < 1.0),
            maxlen=100,
        )
        if len(self._signal_timestamps) >= self._max_signals_per_second:
            return
        self._signal_timestamps.append(now)
        safe_details = {k: str(v)[:200] for k, v in details.items()}
        signal = {
            "event_type": event_type,
            "details": safe_details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._behavior_signals.append(signal)

    def distill_trait_updates(self) -> List[Dict]:
        updates = []

        for signal in list(self._behavior_signals)[-100:]:
            event_type = signal.get("event_type", "")
            details = signal.get("details", {})

            if event_type == "question_asked":
                complexity = details.get("complexity", "medium")
                if complexity == "complex":
                    updates.append({"dimension": "openness", "delta": 0.02})
                elif complexity == "trivial":
                    updates.append({"dimension": "openness", "delta": -0.01})

            elif event_type == "task_completed":
                on_time = details.get("on_time", True)
                quality = details.get("quality", "medium")
                if on_time and quality == "high":
                    updates.append({"dimension": "conscientiousness", "delta": 0.02})
                elif not on_time:
                    updates.append({"dimension": "conscientiousness", "delta": -0.02})

            elif event_type == "social_interaction":
                mode = details.get("mode", "one_on_one")
                if mode == "group":
                    updates.append({"dimension": "extraversion", "delta": 0.02})
                elif mode == "async":
                    updates.append({"dimension": "extraversion", "delta": -0.01})

            elif event_type == "conflict":
                resolution = details.get("resolution", "compromise")
                if resolution == "compromise":
                    updates.append({"dimension": "agreeableness", "delta": 0.02})
                elif resolution == "stand_firm":
                    updates.append({"dimension": "agreeableness", "delta": -0.01})

            elif event_type == "stress_event":
                severity = details.get("severity", "low")
                if severity == "high":
                    updates.append({"dimension": "neuroticism", "delta": 0.01})
                elif severity == "handled_well":
                    updates.append({"dimension": "neuroticism", "delta": -0.02})

            elif event_type == "decision_made":
                speed = details.get("speed", "moderate")
                risk = details.get("risk_level", "low")
                if speed == "fast" and risk == "high":
                    updates.append({"action": "set_decision_style", "style": "bold"})
                elif speed == "slow" and risk == "high":
                    updates.append({"action": "set_decision_style", "style": "cautious"})
                elif speed == "moderate" and risk == "low":
                    updates.append({"action": "set_decision_style", "style": "deliberative"})

            elif event_type == "communication":
                length = details.get("length", "medium")
                formality = details.get("formality", "casual")
                if length == "short" and formality == "casual":
                    updates.append({"action": "set_communication_style", "style": "terse"})
                elif length == "long" and formality == "formal":
                    updates.append({"action": "set_communication_style", "style": "verbose"})

        return updates


class PersonalityEngine:
    """
    Personality Engine for Shadow AI.

    Learns the user's personality from behavioral observations and
    builds a profile that the shadow can use to:
    - Predict what the user would decide
    - Communicate in the user's preferred style
    - Respect the user's values and boundaries
    - Self-edit its persona to match the user's expectations
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / "digital-shadow" / "data" / "personality"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.profile = PersonalityProfile()
        self.distiller = PersonaDistiller()
        self._lock = threading.Lock()

        self._observers: List[Callable] = []
        self._stats = {
            "behaviors_recorded": 0,
            "traits_updated": 0,
            "patterns_discovered": 0,
            "profile_versions": 1,
            "distillation_runs": 0,
        }
        self._stats_lock = threading.Lock()

        self._load_profile()

    def _load_profile(self):
        profile_file = self.data_dir / "personality_profile.json"
        if profile_file.exists():
            try:
                data = json.loads(profile_file.read_text())
                self.profile = PersonalityProfile(
                    profile_id=data.get("profile_id", self.profile.profile_id),
                    user_name=data.get("user_name", ""),
                    big_five=BigFiveProfile(
                        openness=data.get("big_five", {}).get("openness", 0.5),
                        conscientiousness=data.get("big_five", {}).get("conscientiousness", 0.5),
                        extraversion=data.get("big_five", {}).get("extraversion", 0.5),
                        agreeableness=data.get("big_five", {}).get("agreeableness", 0.5),
                        neuroticism=data.get("big_five", {}).get("neuroticism", 0.5),
                        evidence_count=data.get("big_five", {}).get("evidence_count", 0),
                    ),
                    decision_style=DecisionStyle(data.get("decision_style", "deliberative")),
                    communication_style=CommunicationStyle(data.get("communication_style", "casual")),
                    value_priorities=[],
                )
                for v in data.get("value_priorities", ["learning"]):
                    try:
                        self.profile.value_priorities.append(ValuePriority(v))
                    except (ValueError, KeyError):
                        continue
                logger.info("Loaded personality profile from disk")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.error("Failed to load profile (corrupted): %s", type(e).__name__)
            except (OSError, AttributeError) as e:
                logger.error("Unexpected error loading profile: %s", type(e).__name__)

    def _save_profile(self):
        profile_file = self.data_dir / "personality_profile.json"
        try:
            data = self.profile.to_dict()
            data["big_five"] = self.profile.big_five.to_dict()
            tmp = profile_file.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(data, indent=2))
            fd = os.open(str(tmp), os.O_WRONLY)
            os.fsync(fd)
            os.close(fd)
            tmp.replace(profile_file)
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save profile: %s", type(e).__name__)

    def record_behavior(self, event_type: str, details: Dict[str, Any]):
        event_type = _sanitize(event_type)[:200]
        details = {k: _sanitize_content(str(v))[:500] if isinstance(v, str) else v for k, v in details.items()}
        if len(details) > 20:
            details = dict(list(details.items())[:20])
        with self._lock:
            self.distiller.record_behavior(event_type, details)
            with self._stats_lock:
                self._stats["behaviors_recorded"] += 1

        self._auto_detect_pattern(event_type, details)

    def _auto_detect_pattern(self, event_type: str, details: Dict[str, Any]):
        with self._lock:
            self.detect_creative_patterns(event_type, details)
            self.detect_collaboration_patterns(event_type, details)
            self.detect_urgency_patterns(event_type, details)

            if event_type == "decision_made":
                decisiveness = details.get("speed", "moderate")
                if decisiveness == "fast":
                    self.profile.add_or_reinforce_pattern(
                        "quick_decider",
                        "Tends to make decisions quickly",
                        str(details.get("context", ""))[:100],
                    )
                elif decisiveness == "slow":
                    self.profile.add_or_reinforce_pattern(
                        "careful_decider",
                        "Takes time to make decisions",
                        str(details.get("context", ""))[:100],
                    )
                risk_level = details.get("risk", "medium")
                if risk_level == "high":
                    self.profile.add_or_reinforce_pattern(
                        "risk_taker",
                        "Willing to take risks",
                        str(details.get("context", ""))[:100],
                    )
                elif risk_level == "low":
                    self.profile.add_or_reinforce_pattern(
                        "risk_averse",
                        "Prefers safe options",
                        str(details.get("context", ""))[:100],
                    )

            elif event_type == "communication":
                time_of_day = details.get("hour", 12)
                if 6 <= time_of_day <= 10:
                    self.profile.add_or_reinforce_pattern(
                        "early_communicator",
                        "Communicates actively in the morning",
                    )
                elif 21 <= time_of_day <= 23 or 0 <= time_of_day <= 2:
                    self.profile.add_or_reinforce_pattern(
                        "night_owl_communicator",
                        "Communicates actively at night",
                    )
                tone = details.get("tone", "neutral")
                if tone == "formal":
                    self.profile.add_or_reinforce_pattern(
                        "formal_communicator",
                        "Uses formal language",
                    )
                elif tone == "casual":
                    self.profile.add_or_reinforce_pattern(
                        "casual_communicator",
                        "Uses casual language",
                    )

            elif event_type == "task_completed":
                task_type = details.get("type", "general")
                self.profile.add_or_reinforce_pattern(
                    f"completes_{task_type}_tasks",
                    f"Regularly completes {task_type} tasks",
                )
                if details.get("perfectionist", False):
                    self.profile.add_or_reinforce_pattern(
                        "perfectionist",
                        "Pays extra attention to quality and details",
                    )

            elif event_type == "social_interaction":
                group_size = details.get("group_size", 1)
                if group_size > 3:
                    self.profile.add_or_reinforce_pattern(
                        "group_socializer",
                        "Comfortable in group settings",
                    )
                elif group_size == 1:
                    self.profile.add_or_reinforce_pattern(
                        "one_on_one_preference",
                        "Prefers one-on-one interactions",
                    )

            elif event_type == "learning":
                learning_style = details.get("style", "mixed")
                self.profile.add_or_reinforce_pattern(
                    f"learns_via_{learning_style}",
                    f"Prefers {learning_style} learning approach",
                )

    def detect_creative_patterns(self, event_type: str, details: Dict[str, Any]) -> Dict:
        creative_keywords = ["create", "design", "build", "imagine", "novel", "invent", "innovation", "creative"]
        evidence = []
        text_fields = [str(v) for v in details.values() if isinstance(v, (str, int, float))]
        combined_text = " ".join(text_fields).lower()
        event_text = event_type.lower()
        for kw in creative_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', combined_text) or re.search(r'\b' + re.escape(kw) + r'\b', event_text):
                evidence.append(f"keyword: {kw}")
        confidence = min(0.9, 0.3 + len(evidence) * 0.15) if evidence else 0.0
        result = {
            "pattern": "creative_behavior",
            "confidence": round(confidence, 3),
            "evidence": evidence,
        }
        if evidence:
            self.profile.add_or_reinforce_pattern(
                "creative_behavior",
                "Shows creative and innovative tendencies",
                combined_text[:100],
            )
        return result

    def detect_collaboration_patterns(self, event_type: str, details: Dict[str, Any]) -> Dict:
        collaboration_keywords = ["help", "together", "team", "share", "support", "collaborate", "pair", "group"]
        evidence = []
        text_fields = [str(v) for v in details.values() if isinstance(v, (str, int, float))]
        combined_text = " ".join(text_fields).lower()
        event_text = event_type.lower()
        for kw in collaboration_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', combined_text) or re.search(r'\b' + re.escape(kw) + r'\b', event_text):
                evidence.append(f"keyword: {kw}")
        if event_type == "social_interaction" and details.get("group_size", 1) > 2:
            evidence.append("group_size > 2")
        confidence = min(0.9, 0.3 + len(evidence) * 0.15) if evidence else 0.0
        result = {
            "pattern": "collaborative_behavior",
            "confidence": round(confidence, 3),
            "evidence": evidence,
        }
        if evidence:
            self.profile.add_or_reinforce_pattern(
                "collaborative_behavior",
                "Shows collaborative and team-oriented tendencies",
                combined_text[:100],
            )
        return result

    def detect_urgency_patterns(self, event_type: str, details: Dict[str, Any]) -> Dict:
        urgency_keywords = ["urgent", "asap", "deadline", "rush", "hurry", "emergency", "critical", "immediately"]
        evidence = []
        text_fields = [str(v) for v in details.values() if isinstance(v, (str, int, float))]
        combined_text = " ".join(text_fields).lower()
        event_text = event_type.lower()
        for kw in urgency_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', combined_text) or re.search(r'\b' + re.escape(kw) + r'\b', event_text):
                evidence.append(f"keyword: {kw}")
        if details.get("urgency") is not None:
            try:
                urgency_val = float(details["urgency"])
            except (TypeError, ValueError):
                urgency_val = 0.0
            if urgency_val >= 0.8:
                evidence.append("urgency_level >= 0.8")
        if details.get("priority") in ("high", "critical", "urgent"):
            evidence.append(f"priority: {details.get('priority')}")
        confidence = min(0.9, 0.3 + len(evidence) * 0.15) if evidence else 0.0
        result = {
            "pattern": "urgency_driven",
            "confidence": round(confidence, 3),
            "evidence": evidence,
        }
        if evidence:
            self.profile.add_or_reinforce_pattern(
                "urgency_driven",
                "Operates under time pressure and urgency",
                combined_text[:100],
            )
        return result

    async def distill(self) -> Dict[str, Any]:
        with self._lock:
            updates = self.distiller.distill_trait_updates()
            with self._stats_lock:
                self._stats["distillation_runs"] += 1

            trait_updates = 0
            style_updates = 0

            for update in updates:
                if "dimension" in update:
                    dimension = TraitDimension(update["dimension"])
                    delta = update["delta"]
                    self.profile.big_five.update(dimension, delta)
                    trait_updates += 1
                    with self._stats_lock:
                        self._stats["traits_updated"] += 1

                elif "action" in update:
                    action = update["action"]
                    if action == "set_decision_style":
                        new_style = DecisionStyle(update.get("style", "deliberative"))
                        if new_style != self.profile.decision_style:
                            self.profile.decision_style = new_style
                            style_updates += 1
                    elif action == "set_communication_style":
                        new_style = CommunicationStyle(update.get("style", "casual"))
                        if new_style != self.profile.communication_style:
                            self.profile.communication_style = new_style
                            style_updates += 1

            self.profile.updated_at = datetime.now(timezone.utc).isoformat()

            if trait_updates > 0 or style_updates > 0:
                self.profile.version += 1
                self._stats["profile_versions"] = self.profile.version

            self._save_profile()

            observers_snapshot = list(self._observers)

        for observer in observers_snapshot:
            try:
                observer(self.profile)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                pass

        return {
            "trait_updates": trait_updates,
            "style_updates": style_updates,
            "profile_version": self.profile.version,
            "big_five": self.profile.big_five.to_dict(),
        }

    def predict_decision(self, options: List[Dict], context: Optional[Dict] = None, device_context=None) -> Dict:
        if not options:
            return {"error": "No options provided"}

        with self._lock:
            bf_openness = self.profile.big_five.openness
            bf_conscientiousness = self.profile.big_five.conscientiousness
            bf_extraversion = self.profile.big_five.extraversion
            bf_agreeableness = self.profile.big_five.agreeableness
            bf_neuroticism = self.profile.big_five.neuroticism
            bf_evidence_count = self.profile.big_five.evidence_count
            decision_style = self.profile.decision_style
            profile_version = self.profile.version
        style_weights = {
            DecisionStyle.ANALYTICAL: {"risk": -0.4, "reward": 0.3, "novelty": -0.1, "speed": -0.2},
            DecisionStyle.INTUITIVE: {"risk": 0.0, "reward": 0.2, "novelty": 0.3, "speed": 0.3},
            DecisionStyle.CAUTIOUS: {"risk": -0.5, "reward": 0.1, "novelty": -0.2, "speed": -0.1},
            DecisionStyle.BOLD: {"risk": 0.2, "reward": 0.3, "novelty": 0.3, "speed": 0.2},
            DecisionStyle.DELIBERATIVE: {"risk": -0.3, "reward": 0.2, "novelty": 0.0, "speed": -0.1},
            DecisionStyle.SPONTANEOUS: {"risk": 0.1, "reward": 0.1, "novelty": 0.4, "speed": 0.4},
        }
        weights = style_weights.get(decision_style, style_weights[DecisionStyle.DELIBERATIVE])

        weights["novelty"] += (bf_openness - 0.5) * 0.3
        weights["risk"] += (bf_neuroticism - 0.5) * 0.2
        weights["reward"] += (bf_conscientiousness - 0.5) * 0.2

        if device_context is not None:
            ctx = device_context.get_context()
            if ctx is not None:
                sl = ctx.health.stress_level.value
                if sl in ("critical", "high"):
                    weights["risk"] -= 0.2
                    weights["speed"] -= 0.1
                if ctx.focus.value == "deep_work":
                    weights["speed"] -= 0.1
                elif ctx.focus.value in ("distracted", "fatigued"):
                    weights["speed"] += 0.15

        scores = []
        for i, option in enumerate(options):
            risk = option.get("risk_level", 0.5)
            reward = option.get("reward", 0.5)
            novelty = option.get("novelty", 0.5)
            speed = option.get("speed", 0.5)

            r_weight = max(0.1, 1 + weights.get("reward", 0))
            n_weight = max(0.1, 1 + weights.get("novelty", 0))
            k_weight = max(0.1, 1 + weights.get("risk", 0))
            s_weight = max(0.1, 1 + weights.get("speed", 0))

            score = (reward * r_weight + (1 - risk) * k_weight +
                     novelty * n_weight + speed * s_weight) / 4.0

            confidence = min(0.99, 0.3 + bf_evidence_count / 200)
            scores.append({
                "option_index": i,
                "score": round(score, 3),
                "confidence": round(confidence, 3),
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return {
            "predicted_choice": scores[0]["option_index"],
            "scores": scores,
            "confidence": scores[0]["confidence"],
                "based_on": f"v{profile_version} profile with {bf_evidence_count} evidence points",
            }

    def get_response_style(self) -> Dict:
        with self._lock:
            return {
            "communication_style": self.profile.communication_style.value,
            "decision_style": self.profile.decision_style.value,
            "formality": "formal" if self.profile.big_five.conscientiousness > 0.6 else "casual",
            "verbosity": "verbose" if self.profile.big_five.extraversion > 0.6 else "concise",
            "technical_depth": "high" if self.profile.big_five.openness > 0.6 else "moderate",
        }

    def generate_persona_block(self) -> str:
        return self.profile.to_system_prompt()

    MAX_OBSERVERS = 20
    MAX_BEHAVIOR_PATTERNS = 200

    def on_profile_update(self, observer: Callable):
        with self._lock:
            if len(self._observers) >= self.MAX_OBSERVERS:
                evicted = self._observers.pop(0)
                logger.warning("Evicted observer to make room for new one")
            self._observers.append(observer)

    def _cap_profile_collections(self):
        if len(self.profile.preferences) > self._MAX_PREFERENCES if hasattr(self, '_MAX_PREFERENCES') else 500:
            max_prefs = 500
            keys = list(self.profile.preferences.keys())[:len(self.profile.preferences) - max_prefs]
            for k in keys:
                del self.profile.preferences[k]
        if len(self.profile.interaction_patterns) > (self._MAX_INTERACTION_PATTERNS if hasattr(self, '_MAX_INTERACTION_PATTERNS') else 500):
            max_ip = 500
            keys = list(self.profile.interaction_patterns.keys())[:len(self.profile.interaction_patterns) - max_ip]
            for k in keys:
                del self.profile.interaction_patterns[k]

    def get_profile(self) -> Dict:
        with self._lock:
            return self.profile.to_dict()

    def get_patterns(self, min_confidence: float = 0.0) -> List[Dict]:
        with self._lock:
            patterns = [p for p in self.profile.behavior_patterns if p.confidence >= min_confidence]
            patterns.sort(key=lambda p: p.confidence, reverse=True)
            return [p.to_dict() for p in patterns]

    def get_stats(self) -> Dict:
        with self._stats_lock:
            behaviors_recorded = self._stats["behaviors_recorded"]
            traits_updated = self._stats["traits_updated"]
            distillation_runs = self._stats["distillation_runs"]
        with self._lock:
            profile_version = self.profile.version
            patterns_discovered = len(self.profile.behavior_patterns)
            decisions_recorded = len(self.profile.decision_history)
            big_five = self.profile.big_five.to_dict()
            decision_style = self.profile.decision_style.value
            communication_style = self.profile.communication_style.value
        return {
            "profile_version": profile_version,
            "behaviors_recorded": behaviors_recorded,
            "traits_updated": traits_updated,
            "patterns_discovered": patterns_discovered,
            "decisions_recorded": decisions_recorded,
            "distillation_runs": distillation_runs,
            "big_five": big_five,
            "decision_style": decision_style,
            "communication_style": communication_style,
        }

    def export_profile(self) -> str:
        return json.dumps(self.profile.to_dict(), indent=2, default=str)

    def import_profile(self, data: str):
        data = _SANITIZE_RE.sub('', data)
        profile_data = json.loads(data)
        with self._lock:
            VALID_BIG_FIVE = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
            if "big_five" in profile_data:
                bf = profile_data["big_five"]
                for attr in VALID_BIG_FIVE:
                    if attr in bf:
                        try:
                            val = max(0.0, min(1.0, float(bf[attr])))
                        except (TypeError, ValueError):
                            continue
                        try:
                            dim = TraitDimension(attr)
                            current = getattr(self.profile.big_five, attr, 0.5)
                            self.profile.big_five.update(dim, val - current)
                        except (ValueError, KeyError):
                            setattr(self.profile.big_five, attr, val)
            if "decision_style" in profile_data:
                try:
                    self.profile.decision_style = DecisionStyle(profile_data["decision_style"])
                except ValueError:
                    pass
            if "communication_style" in profile_data:
                try:
                    self.profile.communication_style = CommunicationStyle(profile_data["communication_style"])
                except ValueError:
                    pass
            self._save_profile()