#!/usr/bin/env python3
"""
EmotionalIntelligence - Emotional Awareness and Regulation Engine

Analyzes emotional content in interactions, tracks emotional patterns over time,
provides regulation strategies, and models emotional dynamics in social contexts.

Architecture:
  - EmotionState: describes a discrete emotional state with intensity and context
  - EmotionalPattern: recurring emotional pattern with triggers and outcomes
  - RegulationStrategy: suggested approach for emotional regulation
  - EmotionalIntelligence: main engine managing emotional awareness
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

logger = logging.getLogger("shadowd.cognition.emotional_intelligence")

_MAX_EMOTION_HISTORY = 5000
_MAX_PATTERNS = 500
_MAX_REGULATION_STRATEGIES = 200
_MAX_METADATA_KEYS = 50

_PRIMARY_EMOTIONS = {
    "joy": ["happy", "glad", "delighted", "pleased", "cheerful", "excited", "thrilled", "elated", "content", "grateful", "optimistic", "hopeful", "proud", "confident", "amused"],
    "sadness": ["sad", "unhappy", "depressed", "down", "blue", "miserable", "grief", "lonely", "disappointed", "heartbroken", "mournful", "melancholy", "gloomy"],
    "anger": ["angry", "frustrated", "irritated", "annoyed", "furious", "outraged", "hostile", "agitated", "resentful", "irate", "enraged", "livid"],
    "fear": ["afraid", "anxious", "worried", "nervous", "terrified", "panicked", "dread", "uneasy", "apprehensive", "scared", "alarmed", "stressed", "tense"],
    "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned", "startled", "unexpected", "unbelievable"],
    "disgust": ["disgusted", "revolted", "repulsed", "nauseated", "appalled", "offended", "disturbed"],
    "trust": ["trusting", "comfortable", "safe", "reliable", "dependable", "loyal", "faithful", "confident"],
    "anticipation": ["expecting", "eager", "anticipating", "looking forward", "ready", "prepared", "hopeful"],
}

_EMOTION_COMPILED = {}
for _emo, _keywords in _PRIMARY_EMOTIONS.items():
    _EMOTION_COMPILED[_emo] = []
    for _kw in _keywords:
        if ' ' in _kw:
            _EMOTION_COMPILED[_emo].append((_kw, None))
        else:
            _EMOTION_COMPILED[_emo].append((_kw, re.compile(r'\b' + re.escape(_kw) + r'\b')))

_VALENCE_MAP = {
    "joy": 0.8, "sadness": -0.7, "anger": -0.8, "fear": -0.6,
    "surprise": 0.1, "disgust": -0.7, "trust": 0.6, "anticipation": 0.4,
}

_AROUSAL_MAP = {
    "joy": 0.6, "sadness": 0.3, "anger": 0.9, "fear": 0.8,
    "surprise": 0.7, "disgust": 0.5, "trust": 0.2, "anticipation": 0.5,
}

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 5000) -> str:
    if text is None:
        return ""
    text = _SANITIZE_STRICT_RE.sub('', str(text)).strip()
    if max_len > 0:
        text = text[:max_len]
    return text


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if text is None:
        return ""
    text = _SANITIZE_RE.sub('', str(text)).strip()
    if max_len > 0:
        text = text[:max_len]
    return text


def _emotion_keyword_match(keyword: str, text: str, compiled: "Optional[re.Pattern]" = None) -> bool:
    if ' ' in keyword:
        return keyword in text
    if compiled is not None:
        return bool(compiled.search(text))
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))


class EmotionCategory(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class RegulationApproach(str, Enum):
    COGNITIVE_REAPPRAISAL = "cognitive_reappraisal"
    MINDFULNESS = "mindfulness"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    SOCIAL_SUPPORT = "social_support"
    ACCEPTANCE = "acceptance"
    PROBLEM_SOLVING = "problem_solving"
    DISTRACTION = "distraction"
    SUPPRESSION = "suppression"


class PatternType(str, Enum):
    TRIGGER = "trigger"
    CYCLE = "cycle"
    ESCALATION = "escalation"
    RECOVERY = "recovery"
    AMPLIFICATION = "amplification"


@dataclass
class EmotionState:
    state_id: str = ""
    primary: EmotionCategory = EmotionCategory.NEUTRAL
    secondary: Optional[EmotionCategory] = None
    intensity: float = 0.5
    valence: float = 0.0
    arousal: float = 0.5
    context: str = ""
    triggers: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.state_id:
            self.state_id = f"emo_{uuid.uuid4().hex[:12]}"
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.triggers = self.triggers[:5]
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])
        self.metadata = {k: _sanitize(str(v), 500) for k, v in self.metadata.items()}

    def to_dict(self) -> Dict:
        return {
            "state_id": self.state_id,
            "primary": self.primary.value,
            "secondary": self.secondary.value if self.secondary else None,
            "intensity": round(self.intensity, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "context": self.context[:200],
            "triggers": list(self.triggers[:5]),
            "timestamp": self.timestamp,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class EmotionalPattern:
    pattern_id: str = ""
    pattern_type: PatternType = PatternType.TRIGGER
    name: str = ""
    description: str = ""
    emotions: List[EmotionCategory] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    frequency: int = 0
    last_occurrence: Optional[str] = None
    average_intensity: float = 0.5
    outcomes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = f"pat_{uuid.uuid4().hex[:12]}"
        self.average_intensity = max(0.0, min(1.0, self.average_intensity))

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "name": self.name[:100],
            "description": self.description[:500],
            "emotions": [e.value for e in self.emotions],
            "triggers": list(self.triggers[:10]),
            "frequency": self.frequency,
            "last_occurrence": self.last_occurrence,
            "average_intensity": round(self.average_intensity, 3),
            "outcomes": list(self.outcomes[:5]),
        }


@dataclass
class RegulationStrategy:
    strategy_id: str = ""
    approach: RegulationApproach = RegulationApproach.COGNITIVE_REAPPRAISAL
    target_emotion: EmotionCategory = EmotionCategory.NEUTRAL
    target_intensity: float = 0.3
    description: str = ""
    steps: List[str] = field(default_factory=list)
    effectiveness: float = 0.5
    conditions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.strategy_id:
            self.strategy_id = f"reg_{uuid.uuid4().hex[:12]}"
        self.effectiveness = max(0.0, min(1.0, self.effectiveness))
        self.target_intensity = max(0.0, min(1.0, self.target_intensity))
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])
        self.metadata = {k: _sanitize(str(v), 500) for k, v in self.metadata.items()}

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "approach": self.approach.value,
            "target_emotion": self.target_emotion.value,
            "target_intensity": round(self.target_intensity, 3),
            "description": self.description[:500],
            "steps": list(self.steps[:10]),
            "effectiveness": round(self.effectiveness, 3),
            "conditions": list(self.conditions[:5]),
            "metadata": copy.deepcopy(self.metadata),
        }


_REGULATION_GUIDANCE = {
    (EmotionCategory.ANGER, RegulationApproach.COGNITIVE_REAPPRAISAL): [
        "Pause before responding",
        "Consider alternative interpretations of the trigger",
        "Reframe the situation as a problem to solve",
        "Ask yourself what the other person might be experiencing",
    ],
    (EmotionCategory.ANGER, RegulationApproach.MINDFULNESS): [
        "Take 3 deep breaths",
        "Notice physical sensations without judgment",
        "Observe the anger as a temporary state",
        "Ground yourself in the present moment",
    ],
    (EmotionCategory.SADNESS, RegulationApproach.BEHAVIORAL_ACTIVATION): [
        "Engage in a small pleasant activity",
        "Connect with someone you trust",
        "Take a brief walk outside",
        "Set a small achievable goal",
    ],
    (EmotionCategory.SADNESS, RegulationApproach.SOCIAL_SUPPORT): [
        "Reach out to a friend or family member",
        "Share your feelings with someone who listens",
        "Join a supportive community or group",
        "Consider professional counseling if persistent",
    ],
    (EmotionCategory.FEAR, RegulationApproach.ACCEPTANCE): [
        "Acknowledge the fear without trying to eliminate it",
        "Name the specific fear you are experiencing",
        "Remind yourself that fear is a natural response",
        "Allow the feeling to exist while maintaining perspective",
    ],
    (EmotionCategory.FEAR, RegulationApproach.PROBLEM_SOLVING): [
        "Identify the specific threat you are facing",
        "List concrete steps you can take",
        "Focus on what is within your control",
        "Break the feared situation into smaller, manageable parts",
    ],
    (EmotionCategory.ANGER, RegulationApproach.DISTRACTION): [
        "Shift focus to a completely different task",
        "Engage in physical activity",
        "Watch or listen to something absorbing",
        "Work on a hobby or creative project",
    ],
    (EmotionCategory.SADNESS, RegulationApproach.COGNITIVE_REAPPRAISAL): [
        "Consider alternative interpretations of the situation",
        "Remind yourself that feelings are temporary",
        "Look for any positive aspects in the situation",
        "Reframe the experience as a growth opportunity",
    ],
}


class EmotionalIntelligence:
    """
    Emotional Awareness and Regulation Engine.

    Detects emotions in text, tracks emotional patterns over time,
    suggests regulation strategies, and provides emotional intelligence insights.
    """

    def __init__(self):
        self._emotion_history: deque = deque(maxlen=_MAX_EMOTION_HISTORY)
        self._patterns: Dict[str, EmotionalPattern] = {}
        self._strategies: Dict[str, RegulationStrategy] = {}
        self._emotion_counts: Dict[str, int] = {}
        self._states_counted = 0
        self._lock = threading.Lock()
        self._stats = {
            "states_recorded": 0,
            "patterns_detected": 0,
            "regulations_suggested": 0,
            "average_valence": 0.0,
            "average_arousal": 0.0,
        }
        self._stats_lock = threading.Lock()

    def detect_emotion(self, text: str, context: str = "", device_context=None) -> EmotionState:
        text = _sanitize(text)
        context = _sanitize(context, max_len=500)
        lower = text.lower()
        emotion_scores: Dict[str, float] = {}

        for emotion, compiled_kws in _EMOTION_COMPILED.items():
            score = sum(1 for kw, pat in compiled_kws if _emotion_keyword_match(kw, lower, pat))
            if score > 0:
                emotion_scores[emotion] = score / len(compiled_kws)

        physiological_boost = 0.0
        stress_biased_primary = None
        if device_context is not None:
            ctx = device_context.get_context()
            if ctx is not None:
                sl = ctx.health.stress_level.value
                if sl == "critical":
                    physiological_boost = 0.3
                    stress_biased_primary = EmotionCategory.FEAR
                elif sl == "high":
                    physiological_boost = 0.15
                    stress_biased_primary = EmotionCategory.ANGER
                elif sl == "moderate":
                    physiological_boost = 0.05
                if ctx.health.heart_rate is not None and ctx.health.heart_rate > 100:
                    physiological_boost += 0.1
                if ctx.health.sleep_minutes_last_24h > 0 and ctx.health.sleep_minutes_last_24h < 300:
                    if "sadness" not in emotion_scores:
                        emotion_scores.setdefault("sadness", 0.05)

        if not emotion_scores and stress_biased_primary:
            state = EmotionState(
                primary=stress_biased_primary,
                intensity=0.3 + physiological_boost,
                valence=-0.3,
                arousal=0.5 + physiological_boost,
                context=context[:200],
            )
        elif not emotion_scores:
            state = EmotionState(
                primary=EmotionCategory.NEUTRAL,
                context=context[:200],
            )
        else:
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            primary = EmotionCategory(sorted_emotions[0][0])
            secondary = EmotionCategory(sorted_emotions[1][0]) if len(sorted_emotions) > 1 else None

            intensity = min(1.0, sum(emotion_scores.values()) * 2 + physiological_boost)
            valence = _VALENCE_MAP.get(sorted_emotions[0][0], 0.0) * intensity
            arousal = _AROUSAL_MAP.get(sorted_emotions[0][0], 0.5) * intensity

            top_compiled = _EMOTION_COMPILED.get(sorted_emotions[0][0], [])
            triggers = [kw for kw, pat in top_compiled if _emotion_keyword_match(kw, lower, pat)][:5]

            state = EmotionState(
                primary=primary,
                secondary=secondary,
                intensity=intensity,
                valence=valence,
                arousal=arousal,
                context=context[:200],
                triggers=triggers,
            )

        with self._lock:
            self._emotion_history.append(state)
            with self._stats_lock:
                self._stats["states_recorded"] += 1
            self._states_counted += 1
            primary_key = state.primary.value
            self._emotion_counts[primary_key] = self._emotion_counts.get(primary_key, 0) + 1
            total = self._states_counted
            if total > 0:
                running_val = self._stats["average_valence"]
                running_ars = self._stats["average_arousal"]
                self._stats["average_valence"] = running_val + (state.valence - running_val) / total
                self._stats["average_arousal"] = running_ars + (state.arousal - running_ars) / total

        return copy.deepcopy(state)

    def analyze_emotional_trajectory(self, limit: int = 20) -> Dict:
        with self._lock:
            history = list(self._emotion_history)[-limit:]

        if len(history) < 2:
            return {"direction": "stable", "trend": "insufficient_data", "states": len(history)}

        valences = [s.valence for s in history]
        arousals = [s.arousal for s in history]

        val_change = valences[-1] - valences[0]
        ars_change = arousals[-1] - arousals[0]

        if val_change > 0.1:
            direction = "improving"
        elif val_change < -0.1:
            direction = "declining"
        else:
            direction = "stable"

        intensity_avg = sum(s.intensity for s in history) / len(history)

        return {
            "direction": direction,
            "valence_change": round(val_change, 3),
            "arousal_change": round(ars_change, 3),
            "average_intensity": round(intensity_avg, 3),
            "dominant_emotion": max(set(s.primary.value for s in history), key=lambda e: sum(1 for s in history if s.primary.value == e)),
            "trend": "positive" if val_change > 0 else "negative" if val_change < 0 else "stable",
            "states": len(history),
        }

    def suggest_regulation(self, emotion: EmotionCategory, intensity: float = 0.5, device_context=None) -> RegulationStrategy:
        intensity = max(0.0, min(1.0, intensity))
        best_approach = self._best_regulation_approach(emotion, intensity)

        steps = self._get_regulation_steps(emotion, best_approach)

        if device_context is not None:
            ctx = device_context.get_context()
            if ctx is not None:
                sl = ctx.health.stress_level.value
                if sl in ("critical", "high"):
                    if best_approach == RegulationApproach.COGNITIVE_REAPPRAISAL:
                        best_approach = RegulationApproach.MINDFULNESS
                        steps = self._get_regulation_steps(emotion, best_approach)
                    steps.append("Consider breathing exercises — physiological stress detected")
                if ctx.health.sleep_minutes_last_24h > 0 and ctx.health.sleep_minutes_last_24h < 300:
                    steps.append(f"Sleep deficit ({ctx.health.sleep_minutes_last_24h // 60}h) — prioritize rest")

        description = f"Regulation strategy for {emotion.value} at intensity {intensity:.1f}"

        strategy = RegulationStrategy(
            approach=best_approach,
            target_emotion=emotion,
            target_intensity=max(0.0, intensity * 0.5),
            description=description,
            steps=steps,
            effectiveness=min(1.0, 0.3 + (1.0 - intensity) * 0.5),
        )

        with self._lock:
            self._strategies[strategy.strategy_id] = strategy
            if len(self._strategies) > _MAX_REGULATION_STRATEGIES:
                oldest = list(self._strategies.keys())[0]
                logger.warning("Evicting strategy %s", oldest)
                del self._strategies[oldest]
            with self._stats_lock:
                self._stats["regulations_suggested"] += 1

        return copy.deepcopy(strategy)

    def _best_regulation_approach(self, emotion: EmotionCategory, intensity: float) -> RegulationApproach:
        high_intensity = intensity > 0.7
        negative_valence = _VALENCE_MAP.get(emotion.value, 0.0) < 0

        if emotion == EmotionCategory.ANGER:
            return RegulationApproach.COGNITIVE_REAPPRAISAL if not high_intensity else RegulationApproach.MINDFULNESS
        elif emotion == EmotionCategory.SADNESS:
            return RegulationApproach.BEHAVIORAL_ACTIVATION if not high_intensity else RegulationApproach.SOCIAL_SUPPORT
        elif emotion == EmotionCategory.FEAR:
            return RegulationApproach.ACCEPTANCE if not high_intensity else RegulationApproach.PROBLEM_SOLVING
        elif emotion == EmotionCategory.DISGUST:
            return RegulationApproach.COGNITIVE_REAPPRAISAL
        elif emotion == EmotionCategory.SURPRISE:
            return RegulationApproach.ACCEPTANCE
        elif negative_valence and high_intensity:
            return RegulationApproach.MINDFULNESS
        else:
            return RegulationApproach.COGNITIVE_REAPPRAISAL

    def _get_regulation_steps(self, emotion: EmotionCategory, approach: RegulationApproach) -> List[str]:
        key = (emotion, approach)
        if key in _REGULATION_GUIDANCE:
            return _REGULATION_GUIDANCE[key]
        fallback_key = (emotion, RegulationApproach.COGNITIVE_REAPPRAISAL)
        if fallback_key in _REGULATION_GUIDANCE:
            return _REGULATION_GUIDANCE[fallback_key]
        return ["Pause and notice what you are feeling", "Consider alternative perspectives", "Choose a constructive response"]

    def record_pattern(self, pattern_type: PatternType, name: str, emotions: List[EmotionCategory],
                        triggers: List[str], description: str = "") -> EmotionalPattern:
        name = _sanitize(name, max_len=100)
        description = _sanitize(description, max_len=500)
        triggers = [_sanitize(t, max_len=200) for t in triggers[:10]]

        pattern = EmotionalPattern(
            pattern_type=pattern_type,
            name=name,
            description=description,
            emotions=emotions[:5],
            triggers=triggers,
        )

        with self._lock:
            self._patterns[pattern.pattern_id] = pattern
            if len(self._patterns) > _MAX_PATTERNS:
                oldest = list(self._patterns.keys())[0]
                logger.warning("Evicting pattern %s", oldest)
                del self._patterns[oldest]
            with self._stats_lock:
                self._stats["patterns_detected"] += 1

        return copy.deepcopy(pattern)

    def detect_patterns(self, min_frequency: int = 2) -> List[EmotionalPattern]:
        with self._lock:
            history = list(self._emotion_history)

        if len(history) < 3:
            return []

        detected = []
        emotion_seq = [s.primary for s in history]
        trigger_map: Dict[str, int] = {}

        for state in history:
            for trigger in state.triggers:
                trigger_map[trigger] = trigger_map.get(trigger, 0) + 1

        common_triggers = sorted(trigger_map.items(), key=lambda x: x[1], reverse=True)[:5]

        if len(common_triggers) > 0:
            for trigger, freq in common_triggers:
                if freq >= min_frequency:
                    related_emotions = [s.primary for s in history if trigger in s.triggers]
                    pattern = EmotionalPattern(
                        pattern_type=PatternType.TRIGGER,
                        name=f"Trigger: {trigger}",
                        description=f"Trigger '{trigger}' appears {freq} times",
                        emotions=list(set(related_emotions))[:5],
                        triggers=[trigger],
                        frequency=freq,
                        average_intensity=sum(s.intensity for s in history if trigger in s.triggers) / max(1, freq),
                    )
                    detected.append(pattern)

        emotion_groups = []
        for emotion in set(emotion_seq):
            run_start = None
            run_count = 0
            for i, e in enumerate(emotion_seq):
                if e == emotion:
                    if run_start is None:
                        run_start = i
                    run_count += 1
                else:
                    if run_count >= min_frequency:
                        emotion_groups.append((emotion, run_start, run_count))
                    run_start = None
                    run_count = 0
            if run_count >= min_frequency:
                emotion_groups.append((emotion, run_start, run_count))

        for emotion, start_idx, count in emotion_groups:
            intensity_sum = sum(s.intensity for i, s in enumerate(history) if i >= start_idx and i < start_idx + count and s.primary == emotion)
            actual_count = sum(1 for i, s in enumerate(history) if i >= start_idx and i < start_idx + count and s.primary == emotion)
            avg_intensity = intensity_sum / max(actual_count, 1)
            pattern = EmotionalPattern(
                pattern_type=PatternType.CYCLE,
                name=f"Emotional cycle: {emotion.value}",
                description=f"{emotion.value} appears {count} consecutive times in recent history",
                emotions=[emotion],
                frequency=count,
                average_intensity=avg_intensity,
            )
            detected.append(pattern)

        with self._lock:
            for p in detected:
                self._patterns[p.pattern_id] = p
            if len(self._patterns) > _MAX_PATTERNS:
                excess = len(self._patterns) - _MAX_PATTERNS
                for key in list(self._patterns.keys())[:excess]:
                    del self._patterns[key]

        return copy.deepcopy(detected)

    def get_emotional_balance(self) -> Dict:
        with self._lock:
            history = list(self._emotion_history)
            recent = history[-10:]

        if not history:
            return {"balance": 0.5, "dominant": "neutral", "diversity": 0.0, "stability": 1.0}

        emotion_counts: Dict[str, int] = {}
        for s in history:
            key = s.primary.value
            emotion_counts[key] = emotion_counts.get(key, 0) + 1

        total = sum(emotion_counts.values())
        probs = [c / total for c in emotion_counts.values()]
        diversity = 1.0 - max(probs) if probs else 0.0

        if len(recent) >= 2:
            val_changes = [abs(recent[i].valence - recent[i-1].valence) for i in range(1, len(recent))]
            stability = max(0.0, 1.0 - sum(val_changes) / len(val_changes))
        else:
            stability = 1.0

        positive = sum(emotion_counts.get(e, 0) for e in ["joy", "trust", "anticipation"])
        negative = sum(emotion_counts.get(e, 0) for e in ["sadness", "anger", "fear", "disgust"])
        balance = positive / max(1, positive + negative) if (positive + negative) > 0 else 0.5

        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"

        return {
            "balance": round(balance, 3),
            "dominant": dominant,
            "diversity": round(diversity, 3),
            "stability": round(stability, 3),
            "emotion_counts": emotion_counts,
        }

    def get_empathy_insight(self, emotion: EmotionCategory, intensity: float = 0.5) -> Dict:
        intensity = max(0.0, min(1.0, intensity))
        valence = _VALENCE_MAP.get(emotion.value, 0.0)
        arousal = _AROUSAL_MAP.get(emotion.value, 0.5)

        needs = []
        if valence < -0.3:
            needs.extend(["validation", "support", "understanding"])
        if valence > 0.3:
            needs.extend(["celebration", "sharing", "connection"])
        if arousal > 0.7:
            needs.extend(["calm_presence", "grounding", "patience"])
        if arousal < 0.3:
            needs.extend(["encouragement", "motivation", "engagement"])
        if not needs:
            needs = ["listening", "presence"]

        response_style = "reflective" if intensity > 0.7 else "supportive" if valence < 0 else "enthusiastic"

        return {
            "emotion": emotion.value,
            "intensity": round(intensity, 3),
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "likely_needs": needs,
            "recommended_response_style": response_style,
            "empathy_score": round(min(1.0, 0.5 + abs(valence) * 0.3 + intensity * 0.2), 3),
        }

    def get_stats(self) -> Dict:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._lock:
            emotion_counts = dict(self._emotion_counts)
            patterns_count = len(self._patterns)
            strategies_count = len(self._strategies)
            history_size = len(self._emotion_history)
        return {
            "stats": stats,
            "emotion_counts": emotion_counts,
            "patterns_tracked": patterns_count,
            "strategies_available": strategies_count,
            "history_size": history_size,
        }