"""AnticipatoryEngine — Predict what the user needs before they ask.

Learns from temporal patterns, emotional trajectories, domain activity, and
cross-module signals to generate proactive suggestions and predictions.

Architecture:
  - AnticipationContext: snapshot of user state at prediction time
  - AnticipationSignal: a predicted need with confidence and reasoning
  - SignalType: category of anticipated need (information, action, resource, reminder, warning)
  - TriggerCondition: what triggers a signal (time, emotion, domain, device, goal)
  - AnticipatoryEngine: main engine with learning, prediction, and validation
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
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

_MAX_SIGNALS = 500
_MAX_HISTORY = 2000
_MAX_PATTERNS = 300
_MAX_CONTEXT_AGE_HOURS = 168  # 7 days
_MAX_SIGNAL_AGE_HOURS = 72
_MAX_VALIDATION_KEYS = 500
_MAX_SEQUENCE_MEMORY = 500


def _sanitize(text: Any, max_len: int = 200) -> str:
    """Strip control chars AND <>, then whitespace, then truncate."""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()[:max_len]


def _sanitize_content(text: Any, max_len: int = 2000) -> str:
    """Strip control chars only (preserve <>), then whitespace, then truncate."""
    return _SANITIZE_CONTENT_RE.sub('', str(text)).strip()[:max_len]


class SignalType(Enum):
    INFORMATION = "information"
    ACTION = "action"
    RESOURCE = "resource"
    REMINDER = "reminder"
    WARNING = "warning"
    SOCIAL = "social"
    CREATIVE = "creative"


class TriggerCondition(Enum):
    TIME = "time"
    EMOTION = "emotion"
    DOMAIN = "domain"
    DEVICE = "device"
    GOAL = "goal"
    PATTERN = "pattern"
    SEQUENCE = "sequence"


@dataclass
class AnticipationContext:
    user_state: Dict[str, Any] = field(default_factory=dict)
    emotional_state: Optional[str] = None
    active_domains: List[str] = field(default_factory=list)
    time_of_day: int = 0
    day_of_week: int = 0
    recent_actions: List[str] = field(default_factory=list)
    goal_progress: Dict[str, float] = field(default_factory=dict)
    device_context: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __post_init__(self):
        if self.emotional_state:
            self.emotional_state = _sanitize(self.emotional_state, 50)
        self.active_domains = [_sanitize(d, 50) for d in self.active_domains][:10]
        self.recent_actions = [_sanitize_content(a, 200) for a in self.recent_actions][:20]
        self.timestamp = self.timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "user_state": copy.deepcopy(self.user_state) if self.user_state else {},
            "emotional_state": self.emotional_state,
            "active_domains": list(self.active_domains),
            "time_of_day": self.time_of_day,
            "day_of_week": self.day_of_week,
            "recent_actions": list(self.recent_actions),
            "goal_progress": {k: round(v, 4) for k, v in self.goal_progress.items()} if self.goal_progress else {},
            "device_context": copy.deepcopy(self.device_context) if self.device_context else None,
            "timestamp": self.timestamp,
        }


@dataclass
class AnticipationSignal:
    signal_id: str = ""
    signal_type: str = ""
    trigger: str = ""
    prediction: str = ""
    confidence: float = 0.5
    reasoning: str = ""
    suggested_action: str = ""
    context_snapshot: Optional[AnticipationContext] = None
    related_modules: List[str] = field(default_factory=list)
    priority: float = 0.5
    validated: bool = False
    validation_outcome: Optional[str] = None
    created_at: str = ""

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = uuid.uuid4().hex[:12]
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.priority = max(0.0, min(1.0, self.priority))
        if self.signal_type:
            self.signal_type = _sanitize(self.signal_type, 50)
        if self.trigger:
            self.trigger = _sanitize(self.trigger, 50)
        if self.prediction:
            self.prediction = _sanitize_content(self.prediction, 500)
        if self.reasoning:
            self.reasoning = _sanitize_content(self.reasoning, 1000)
        if self.suggested_action:
            self.suggested_action = _sanitize_content(self.suggested_action, 500)
        self.related_modules = [_sanitize(m, 30) for m in self.related_modules][:10]
        self.created_at = self.created_at or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "trigger": self.trigger,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "suggested_action": self.suggested_action,
            "context_snapshot": self.context_snapshot.to_dict() if self.context_snapshot else None,
            "related_modules": list(self.related_modules),
            "priority": round(self.priority, 4),
            "validated": self.validated,
            "validation_outcome": self.validation_outcome,
            "created_at": self.created_at,
        }


_TIME_DOMAIN_MAP: Dict[str, List[str]] = {
    "morning": ["health", "planning", "communication"],
    "afternoon": ["work", "productivity", "collaboration"],
    "evening": ["reflection", "learning", "social"],
    "night": ["rest", "consolidation", "automation"],
}

_EMOTION_PREDICTION_MAP: Dict[str, List[Tuple[str, str, float]]] = {
    "joy": [
        ("social", "Share positive experience", 0.7),
        ("creative", "Channel creativity", 0.5),
        ("information", "Learn something new", 0.4),
    ],
    "sadness": [
        ("social", "Reach out to someone", 0.8),
        ("resource", "Self-care resources", 0.7),
        ("reminder", "Review what's going well", 0.5),
    ],
    "anger": [
        ("action", "Physical activity", 0.7),
        ("warning", "Avoid impulsive decisions", 0.8),
        ("information", "Understand the trigger", 0.5),
    ],
    "fear": [
        ("warning", "Assess actual risk", 0.8),
        ("action", "Prepare contingency plan", 0.7),
        ("resource", "Support resources", 0.6),
    ],
    "surprise": [
        ("information", "Investigate further", 0.7),
        ("action", "Adapt quickly", 0.6),
    ],
    "disgust": [
        ("action", "Remove from situation", 0.6),
        ("warning", "Identify boundary violation", 0.5),
    ],
    "anticipation": [
        ("planning", "Prepare for upcoming event", 0.7),
        ("information", "Research what's coming", 0.6),
    ],
    "trust": [
        ("social", "Deepen relationship", 0.6),
        ("resource", "Share resources", 0.5),
    ],
}

_SEQUENCE_PATTERNS: Dict[str, List[Tuple[str, List[str]]]] = {
    "work_start": [
        ("morning_routine", ["check_messages", "review_schedule", "prioritize_tasks"]),
        ("deep_work_prep", ["close_notifications", "set_timer", "focus_task"]),
    ],
    "work_end": [
        ("wrap_up", ["review_completed", "note_tomorrow", "archive_files"]),
        ("transition", ["personal_time", "exercise", "social_call"]),
    ],
    "research": [
        ("deep_dive", ["search_topic", "read_articles", "take_notes", "synthesize"]),
    ],
    "creative": [
        ("ideation", ["brainstorm", "sketch_ideas", "combine_concepts", "refine"]),
    ],
}


class AnticipatoryEngine:
    """Predicts what the user needs before they ask.

    Combines temporal patterns, emotional trajectories, domain activity,
    and cross-module signals to generate proactive anticipation signals.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._signals: Dict[str, AnticipationSignal] = {}
        self._signal_order: deque = deque(maxlen=_MAX_SIGNALS)
        self._history: deque = deque(maxlen=_MAX_HISTORY)
        self._patterns: Dict[str, Dict] = {}
        self._pattern_order: deque = deque(maxlen=_MAX_PATTERNS)
        self._sequence_memory: Dict[str, List[str]] = {}
        self._sequence_order: deque = deque(maxlen=_MAX_SEQUENCE_MEMORY)
        self._context_history: deque = deque(maxlen=500)
        self._validation_accuracy: Dict[str, List[float]] = {}
        self._stats: Dict[str, int] = {
            "signals_generated": 0,
            "signals_validated": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "patterns_learned": 0,
            "anticipations_fired": 0,
        }

    def predict(self, context: AnticipationContext) -> List[AnticipationSignal]:
        """Generate anticipation signals based on current context."""
        if not context or not isinstance(context, AnticipationContext):
            return []

        with self._lock:
            self._context_history.append(context)
        signals: List[AnticipationSignal] = []

        signals.extend(self._predict_from_time(context))
        signals.extend(self._predict_from_emotion(context))
        signals.extend(self._predict_from_domains(context))
        signals.extend(self._predict_from_sequence(context))
        signals.extend(self._predict_from_goals(context))

        seen_ids: set = set()
        unique: List[AnticipationSignal] = []
        for s in signals:
            if s.prediction not in seen_ids:
                seen_ids.add(s.prediction)
                unique.append(s)

        with self._lock:
            for s in unique[:20]:
                self._signals[s.signal_id] = s
                self._signal_order.append(s.signal_id)
            self._history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal_count": len(unique),
                "context_domains": list(context.active_domains),
            })
        with self._stats_lock:
            self._stats["signals_generated"] += len(unique)
            self._stats["anticipations_fired"] += 1

        return unique[:20]

    def validate(self, signal_id: str, outcome: str) -> bool:
        """Validate a prediction with actual outcome.

        Args:
            signal_id: The signal that was validated
            outcome: "correct", "incorrect", or "partial"

        Returns:
            True if signal was found and validated
        """
        outcome = _sanitize(outcome, 50)
        with self._lock:
            signal = self._signals.get(signal_id)
            if not signal:
                return False

            signal.validated = True
            signal.validation_outcome = outcome

        with self._stats_lock:
            self._stats["signals_validated"] += 1
            if outcome == "correct":
                self._stats["correct_predictions"] += 1
            elif outcome == "incorrect":
                self._stats["incorrect_predictions"] += 1

        key = signal.trigger
        with self._lock:
            if key not in self._validation_accuracy:
                if len(self._validation_accuracy) >= _MAX_VALIDATION_KEYS:
                    oldest_key = next(iter(self._validation_accuracy))
                    del self._validation_accuracy[oldest_key]
                    logger.warning("Evicting validation accuracy key %s (cap=%d)", oldest_key, _MAX_VALIDATION_KEYS)
                self._validation_accuracy[key] = []
            self._validation_accuracy[key].append(1.0 if outcome == "correct" else 0.0)
            if len(self._validation_accuracy[key]) > 100:
                self._validation_accuracy[key] = self._validation_accuracy[key][-100:]

        if outcome == "correct" and signal.trigger and signal.prediction:
            self._reinforce_pattern(signal.trigger, signal.prediction)

        return True

    def get_signals(self, signal_type: Optional[str] = None, limit: int = 50) -> List[AnticipationSignal]:
        """Get anticipation signals, optionally filtered by type."""
        limit = max(1, min(limit, _MAX_SIGNALS))
        with self._lock:
            signals = list(self._signals.values())
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        signals = sorted(signals, key=lambda s: s.priority, reverse=True)
        return copy.deepcopy(signals[:limit])

    def get_signal(self, signal_id: str) -> Optional[AnticipationSignal]:
        """Get a specific signal by ID."""
        with self._lock:
            signal = self._signals.get(signal_id)
        return copy.deepcopy(signal) if signal else None

    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get anticipation history."""
        limit = max(1, min(limit, _MAX_HISTORY))
        with self._lock:
            return [copy.deepcopy(h) for h in list(self._history)[-limit:]]

    def get_accuracy(self, trigger: Optional[str] = None) -> Dict[str, Any]:
        """Get prediction accuracy, optionally filtered by trigger type."""
        if trigger:
            trigger = _sanitize(trigger, 50)
            with self._lock:
                scores = list(self._validation_accuracy.get(trigger, []))
            return {
                "trigger": trigger,
                "accuracy": round(sum(scores) / max(len(scores), 1), 4),
                "samples": len(scores),
            }

        with self._lock:
            total = sum(len(v) for v in self._validation_accuracy.values())
            correct = sum(sum(v) for v in self._validation_accuracy.values())
            tracked = len(self._validation_accuracy)
        return {
            "overall_accuracy": round(correct / max(total, 1), 4),
            "triggers_tracked": tracked,
            "total_validations": total,
        }

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        with self._stats_lock:
            return dict(self._stats)

    # --- Internal prediction methods ---

    def _predict_from_time(self, context: AnticipationContext) -> List[AnticipationSignal]:
        """Predict needs based on time of day and day of week."""
        signals: List[AnticipationSignal] = []
        hour = context.time_of_day
        tod = self._time_of_day(hour)
        day = context.day_of_week

        for domain in _TIME_DOMAIN_MAP.get(tod, []):
            priority = 0.6 if domain in context.active_domains else 0.4
            signals.append(AnticipationSignal(
                signal_type=self._domain_to_signal_type(domain),
                trigger=TriggerCondition.TIME.value,
                prediction=f"Likely need for {domain} based on {tod} routine",
                confidence=round(0.5 + (0.1 if day < 6 else -0.1), 4),
                reasoning=f"It's {tod} on {'weekday' if day < 5 else 'weekend'}, typical {domain} time",
                suggested_action=f"Prepare {domain} resources",
                related_modules=[domain],
                priority=round(priority, 4),
                context_snapshot=context,
            ))

        return signals

    def _predict_from_emotion(self, context: AnticipationContext) -> List[AnticipationSignal]:
        """Predict needs based on emotional state."""
        signals: List[AnticipationSignal] = []
        emotion = context.emotional_state
        if not emotion:
            return signals

        predictions = _EMOTION_PREDICTION_MAP.get(emotion, [])
        for signal_type, prediction, confidence in predictions[:3]:
            base_conf = confidence
            if any(d in context.active_domains for d in _EMOTION_PREDICTION_MAP.get(emotion, [])):
                base_conf = min(confidence + 0.1, 1.0)
            signals.append(AnticipationSignal(
                signal_type=signal_type,
                trigger=TriggerCondition.EMOTION.value,
                prediction=prediction,
                confidence=round(base_conf, 4),
                reasoning=f"Emotional state '{emotion}' typically precedes {signal_type} needs",
                suggested_action=f"Proactively {prediction.lower()}",
                related_modules=["emotional_intelligence", "attention_engine"],
                priority=round(confidence, 4),
                context_snapshot=context,
            ))

        return signals

    def _predict_from_domains(self, context: AnticipationContext) -> List[AnticipationSignal]:
        """Predict needs based on active domain patterns."""
        signals: List[AnticipationSignal] = []
        domains = context.active_domains

        if "work" in domains and "health" not in domains:
            signals.append(AnticipationSignal(
                signal_type=SignalType.REMINDER.value,
                trigger=TriggerCondition.DOMAIN.value,
                prediction="Take a break after extended work",
                confidence=0.6,
                reasoning="Extended work domain activity often precedes fatigue",
                suggested_action="Suggest 5-minute break",
                related_modules=["attention_engine", "evolution_guide"],
                priority=0.5,
                context_snapshot=context,
            ))

        if "learning" in domains and "creative" not in domains:
            signals.append(AnticipationSignal(
                signal_type=SignalType.CREATIVE.value,
                trigger=TriggerCondition.DOMAIN.value,
                prediction="Creative synthesis may boost learning retention",
                confidence=0.4,
                reasoning="Learning + creative synthesis is a powerful combination",
                suggested_action="Try a creative exercise related to learning topic",
                related_modules=["creative_synthesis", "evolution_guide"],
                priority=0.3,
                context_snapshot=context,
            ))

        if len(domains) >= 3:
            signals.append(AnticipationSignal(
                signal_type=SignalType.WARNING.value,
                trigger=TriggerCondition.DOMAIN.value,
                prediction="Cognitive overload risk from multi-domain activity",
                confidence=0.5,
                reasoning=f"Active in {len(domains)} domains simultaneously",
                suggested_action="Consider focusing on top priority domain",
                related_modules=["attention_engine", "metacognition_engine"],
                priority=0.6,
                context_snapshot=context,
            ))

        return signals

    def _predict_from_sequence(self, context: AnticipationContext) -> List[AnticipationSignal]:
        """Predict needs based on action sequences."""
        signals: List[AnticipationSignal] = []
        recent = context.recent_actions[-5:]

        if len(recent) < 2:
            return signals

        action_key = ";".join(a[:30] for a in recent[-3:])
        if action_key in self._sequence_memory:
            stored = self._sequence_memory[action_key]
            if len(stored) > 0 and len(stored) <= 5:
                predicted_next = stored[0] if stored else ""
                if predicted_next:
                    signals.append(AnticipationSignal(
                        signal_type=SignalType.ACTION.value,
                        trigger=TriggerCondition.SEQUENCE.value,
                        prediction=f"Likely next action: {predicted_next}",
                        confidence=0.55,
                        reasoning=f"Sequence pattern detected: {' → '.join(recent[-2:])}",
                        suggested_action=f"Prepare for {predicted_next}",
                        related_modules=["procedural_memory", "attention_engine"],
                        priority=0.5,
                        context_snapshot=context,
                    ))

        for category, patterns in _SEQUENCE_PATTERNS.items():
            for pattern_name, steps in patterns:
                if len(recent) >= 2:
                    last_two = [a[:30] for a in recent[-2:]]
                    pattern_start = [s[:30] for s in steps[:2]]
                    match_count = sum(1 for a, p in zip(last_two, pattern_start)
                                    if a.lower() == p.lower() or re.search(r'\b' + re.escape(p.lower()) + r'\b', a.lower()))
                    if match_count >= 1:
                        remaining = steps[len(last_two):]
                        if remaining:
                            signals.append(AnticipationSignal(
                                signal_type=SignalType.ACTION.value,
                                trigger=TriggerCondition.PATTERN.value,
                                prediction=f"Sequence '{pattern_name}' in progress, next: {remaining[0]}",
                                confidence=round(0.4 + match_count * 0.15, 4),
                                reasoning=f"Detected start of '{pattern_name}' pattern ({match_count}/2 steps match)",
                                suggested_action=f"Prepare for {remaining[0]}",
                                related_modules=["procedural_memory"],
                                priority=round(0.4 + match_count * 0.1, 4),
                                context_snapshot=context,
                            ))

        return signals[:5]

    def _predict_from_goals(self, context: AnticipationContext) -> List[AnticipationSignal]:
        """Predict needs based on goal progress."""
        signals: List[AnticipationSignal] = []
        goals = context.goal_progress

        if not goals:
            return signals

        for goal_name, progress in list(goals.items())[:5]:
            if progress < 0.3:
                signals.append(AnticipationSignal(
                    signal_type=SignalType.RESOURCE.value,
                    trigger=TriggerCondition.GOAL.value,
                    prediction=f"Goal '{goal_name}' needs more resources or support",
                    confidence=0.5,
                    reasoning=f"Goal at {round(progress * 100)}% progress — may be stuck",
                    suggested_action=f"Review obstacles for {goal_name}",
                    related_modules=["goal_planner", "goal_reasoning"],
                    priority=0.6,
                    context_snapshot=context,
                ))
            elif progress > 0.8:
                signals.append(AnticipationSignal(
                    signal_type=SignalType.REMINDER.value,
                    trigger=TriggerCondition.GOAL.value,
                    prediction=f"Goal '{goal_name}' is nearly complete",
                    confidence=0.6,
                    reasoning=f"Goal at {round(progress * 100)}% — final push possible",
                    suggested_action=f"Focus on completing {goal_name}",
                    related_modules=["goal_planner", "attention_engine"],
                    priority=0.4,
                    context_snapshot=context,
                ))

        return signals[:5]

    # --- Learning ---

    def _reinforce_pattern(self, trigger: str, prediction: str) -> None:
        """Reinforce a successful prediction pattern."""
        key = f"{trigger}:{prediction[:100]}"
        with self._lock:
            if len(self._patterns) >= _MAX_PATTERNS:
                orphaned = [k for k in self._patterns if k not in set(self._pattern_order)]
                for ok in orphaned:
                    del self._patterns[ok]
                    logger.warning("Evicting orphaned pattern key %s", ok)
                if len(self._patterns) >= _MAX_PATTERNS and self._pattern_order:
                    oldest_key = self._pattern_order.popleft()
                    self._patterns.pop(oldest_key, None)
                    logger.warning("Evicting pattern key %s (cap=%d)", oldest_key, _MAX_PATTERNS)
            if key not in self._patterns:
                self._patterns[key] = {
                    "trigger": trigger,
                    "prediction": prediction[:500],
                    "success_count": 0,
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                }
                self._pattern_order.append(key)
            self._patterns[key]["success_count"] = self._patterns[key].get("success_count", 0) + 1
            self._patterns[key]["last_seen"] = datetime.now(timezone.utc).isoformat()

    def record_sequence(self, actions: List[str]) -> None:
        """Record an action sequence for future prediction."""
        actions = [_sanitize_content(a, 200) for a in actions[:10]]
        if len(actions) < 2:
            return
        key = ";".join(a[:30] for a in actions[:-1])
        value = actions[-1]
        with self._lock:
            if key not in self._sequence_memory:
                self._sequence_memory[key] = []
            self._sequence_memory[key].append(value)
            if len(self._sequence_memory[key]) > 10:
                self._sequence_memory[key] = self._sequence_memory[key][-10:]
            if len(self._sequence_memory) > _MAX_SEQUENCE_MEMORY:
                if key not in self._sequence_memory:
                    oldest_key = self._sequence_order.popleft() if self._sequence_order else next(iter(self._sequence_memory))
                    self._sequence_memory.pop(oldest_key, None)
                    logger.warning("Evicting sequence memory key %s (cap=%d)", oldest_key, _MAX_SEQUENCE_MEMORY)
                else:
                    self._sequence_order.append(key)

    # --- Helpers ---

    @staticmethod
    def _time_of_day(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    @staticmethod
    def _domain_to_signal_type(domain: str) -> str:
        domain_signal_map = {
            "health": SignalType.REMINDER.value,
            "planning": SignalType.ACTION.value,
            "communication": SignalType.SOCIAL.value,
            "work": SignalType.ACTION.value,
            "productivity": SignalType.ACTION.value,
            "collaboration": SignalType.SOCIAL.value,
            "reflection": SignalType.INFORMATION.value,
            "learning": SignalType.INFORMATION.value,
            "social": SignalType.SOCIAL.value,
            "rest": SignalType.REMINDER.value,
            "consolidation": SignalType.INFORMATION.value,
            "automation": SignalType.RESOURCE.value,
            "creative": SignalType.CREATIVE.value,
        }
        return domain_signal_map.get(domain, SignalType.INFORMATION.value)