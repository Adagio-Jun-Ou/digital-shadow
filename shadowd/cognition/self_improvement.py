#!/usr/bin/env python3
"""
Self-Improvement Engine - Shadow AI Learns From Outcomes

Implements the core insight from Letta and 12-Factor Agents:
the shadow doesn't just observe - it learns from the OUTCOMES of its
own decisions and improves over time.

Architecture:
  SelfImprovementEngine
  ├── OutcomeTracker (record what happened after decisions)
  ├── AccuracyCalibrator (compare predictions vs reality)
  ├── StrategyEvolver (adjust decision strategies based on accuracy)
  ├── ConfidenceCalibrator (learn when to trust yourself)
  ├── FeedbackAggregator (collect explicit + implicit feedback)
  └── ImprovementCycle (periodic self-review and updates)

Key Insight: The shadow should be a "stateless reducer":
  state_out = agent(state_in, event)
  
But the IMPROVEMENT happens by updating the agent's internal models
based on outcome comparison, not by modifying the current state.

Inspired by:
- Letta's self-editing memory blocks
- OpenAI Agents SDK's guardrails with feedback
- 12-Factor principle: compact errors into context
- Multi-Armed Bandit exploration-exploitation
"""

import asyncio
import copy
import hashlib
import json
import math
import os
import threading
import time
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("shadowd.cognition.self_improvement")

import re as _re

_SANITIZE_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 2000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


class OutcomeType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    PENDING = "pending"


class FeedbackSource(str, Enum):
    EXPLICIT_USER = "explicit_user"
    IMPLICIT_BEHAVIOR = "implicit_behavior"
    OUTCOME_COMPARISON = "outcome_comparison"
    SYSTEM_METRIC = "system_metric"


@dataclass
class Prediction:
    prediction_id: str = ""
    content: str = ""
    confidence: float = 0.5
    predicted_outcome: str = ""
    domain: Optional[str] = None
    model: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    actual_outcome: Optional[str] = None
    accuracy: Optional[float] = None
    outcome_type: OutcomeType = OutcomeType.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.prediction_id:
            self.prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
        if len(self.metadata) > 50:
            self.metadata = dict(list(self.metadata.items())[:50])

    def resolve(self, actual_outcome: str, outcome_type: OutcomeType = OutcomeType.NEUTRAL):
        self.actual_outcome = actual_outcome
        self.outcome_type = outcome_type
        self.resolved_at = datetime.now(timezone.utc).isoformat()

        if self.predicted_outcome and actual_outcome:
            similarity = self._text_similarity(self.predicted_outcome, actual_outcome)
            self.accuracy = similarity

    def _text_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return 0.0
        intersection = a_words & b_words
        union = a_words | b_words
        return len(intersection) / len(union)

    def to_dict(self) -> Dict:
        return {
            "prediction_id": self.prediction_id,
            "content": self.content[:100],
            "confidence": round(self.confidence, 3),
            "predicted_outcome": (self.predicted_outcome or "")[:100],
            "domain": self.domain,
            "accuracy": round(self.accuracy, 3) if self.accuracy is not None else None,
            "outcome_type": self.outcome_type.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class DecisionRecord:
    MAX_FEEDBACK = 100

    decision_id: str = ""
    action: str = ""
    reasoning: str = ""
    autonomy_level: str = "auto"
    predicted_outcome: str = ""
    actual_outcome: str = ""
    confidence: float = 0.5
    domain: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    outcome_type: OutcomeType = OutcomeType.PENDING
    feedback: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.decision_id:
            self.decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        if len(self.metadata) > 50:
            self.metadata = dict(list(self.metadata.items())[:50])

    def resolve(self, outcome: str, outcome_type: OutcomeType = OutcomeType.NEUTRAL):
        self.actual_outcome = outcome
        self.outcome_type = outcome_type
        self.resolved_at = datetime.now(timezone.utc).isoformat()

    def add_feedback(self, source: FeedbackSource, rating: float, comment: str = ""):
        comment = _sanitize_content(comment, 500)
        self.feedback.append({
            "source": source.value,
            "rating": round(max(0, min(1, rating)), 3),
            "comment": comment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.feedback) > self.MAX_FEEDBACK:
            self.feedback = self.feedback[-self.MAX_FEEDBACK:]

    def get_average_rating(self) -> float:
        if not self.feedback:
            return 0.5
        return sum(f["rating"] for f in self.feedback) / len(self.feedback)

    def to_dict(self) -> Dict:
        return {
            "decision_id": self.decision_id,
            "action": self.action[:100],
            "confidence": round(self.confidence, 3),
            "outcome_type": self.outcome_type.value,
            "domain": self.domain,
            "feedback_count": len(self.feedback),
            "avg_rating": round(self.get_average_rating(), 3),
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "metadata": copy.deepcopy(self.metadata),
            "feedback": copy.deepcopy(self.feedback),
        }


class AccuracyCalibrator:
    _MAX_BY_MODEL = 100
    _MAX_BY_DOMAIN = 100

    def __init__(self):
        self._predictions: List[Prediction] = []
        self._by_model: Dict[str, List[Prediction]] = defaultdict(list)
        self._by_domain: Dict[str, List[Prediction]] = defaultdict(list)
        self._by_confidence_bucket: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_prediction(self, prediction: Prediction):
        with self._lock:
            self._predictions.append(prediction)
            if prediction.model:
                self._by_model[prediction.model].append(prediction)
                if len(self._by_model) > self._MAX_BY_MODEL:
                    oldest_key = next(iter(self._by_model))
                    evicted = self._by_model.pop(oldest_key)
                    evicted_ids = {p.prediction_id for p in evicted}
                    self._predictions = [p for p in self._predictions if p.prediction_id not in evicted_ids]
                    for domain_key in list(self._by_domain):
                        self._by_domain[domain_key] = [p for p in self._by_domain[domain_key] if p.prediction_id not in evicted_ids]
                        if not self._by_domain[domain_key]:
                            del self._by_domain[domain_key]
            if prediction.domain:
                self._by_domain[prediction.domain].append(prediction)
                if len(self._by_domain) > self._MAX_BY_DOMAIN:
                    oldest_key = next(iter(self._by_domain))
                    evicted = self._by_domain.pop(oldest_key)
                    evicted_ids = {p.prediction_id for p in evicted}
                    self._predictions = [p for p in self._predictions if p.prediction_id not in evicted_ids]
            if len(self._predictions) > 5000:
                self._predictions = self._predictions[-5000:]

    def get_overall_accuracy(self) -> float:
        resolved = [p for p in self._predictions if p.accuracy is not None]
        if not resolved:
            return 0.5
        return sum(p.accuracy for p in resolved) / len(resolved)

    def get_accuracy_by_model(self, model: str) -> float:
        resolved = [p for p in self._by_model.get(model, []) if p.accuracy is not None]
        if not resolved:
            return 0.5
        return sum(p.accuracy for p in resolved) / len(resolved)

    def get_accuracy_by_domain(self, domain: str) -> float:
        resolved = [p for p in self._by_domain.get(domain, []) if p.accuracy is not None]
        if not resolved:
            return 0.5
        return sum(p.accuracy for p in resolved) / len(resolved)

    def get_confidence_calibration(self) -> Dict[str, float]:
        with self._lock:
            calibration = {}
            for p in self._predictions:
                if p.accuracy is not None:
                    bucket = f"{int(p.confidence * 10) / 10:.1f}"
                    if bucket not in calibration:
                        calibration[bucket] = []
                    calibration[bucket].append(p.accuracy)

            return {
                bucket: sum(accs) / len(accs)
                for bucket, accs in sorted(calibration.items())
            }

    def get_calibration_error(self) -> float:
        calibration = self.get_confidence_calibration()
        if not calibration:
            return 0.0
        errors = [abs(float(bucket) - accuracy) for bucket, accuracy in calibration.items()]
        return sum(errors) / len(errors) if errors else 0.0

    def get_stats(self) -> Dict:
        with self._lock:
            resolved = [p for p in self._predictions if p.accuracy is not None]
            return {
                "total_predictions": len(self._predictions),
                "resolved_predictions": len(resolved),
                "overall_accuracy": round(self._get_overall_accuracy_unlocked(), 3),
                "calibration_error": round(self._get_calibration_error_unlocked(), 3),
                "models": list(self._by_model.keys()),
                "domains": list(self._by_domain.keys()),
            }

    def _get_overall_accuracy_unlocked(self) -> float:
        resolved = [p for p in self._predictions if p.accuracy is not None]
        if not resolved:
            return 0.5
        return sum(p.accuracy for p in resolved) / len(resolved)

    def _get_calibration_error_unlocked(self) -> float:
        calibration = {}
        for p in self._predictions:
            if p.accuracy is not None:
                bucket = f"{int(p.confidence * 10) / 10:.1f}"
                if bucket not in calibration:
                    calibration[bucket] = []
                calibration[bucket].append(p.accuracy)
        if not calibration:
            return 0.0
        errors = [abs(float(bucket) - sum(accs) / len(accs)) for bucket, accs in calibration.items()]
        return sum(errors) / len(errors) if errors else 0.0

    def get_predictions_snapshot(self) -> List[Prediction]:
        with self._lock:
            return [copy.deepcopy(p) for p in self._predictions]


class StrategyEvolver:
    _MAX_STRATEGIES = 500

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self._strategy_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_strategy_outcome(self, strategy: str, outcome_score: float):
        with self._lock:
            self._strategy_performance[strategy].append(outcome_score)
            if len(self._strategy_performance[strategy]) > 100:
                self._strategy_performance[strategy] = self._strategy_performance[strategy][-100:]

            current_weight = self._strategy_weights[strategy]
            recent_avg = sum(self._strategy_performance[strategy][-10:]) / min(len(self._strategy_performance[strategy]), 10)
            new_weight = current_weight + self.learning_rate * (recent_avg - current_weight)
            self._strategy_weights[strategy] = max(0.1, min(2.0, new_weight))

            if len(self._strategy_weights) > self._MAX_STRATEGIES:
                oldest = next(iter(self._strategy_weights))
                del self._strategy_weights[oldest]
                self._strategy_performance.pop(oldest, None)

    def get_strategy_weight(self, strategy: str) -> float:
        with self._lock:
            return self._strategy_weights[strategy]

    def get_best_strategies(self, top_k: int = 5) -> List[Tuple[str, float]]:
        with self._lock:
            strategies = [(s, w) for s, w in self._strategy_weights.items()]
            strategies.sort(key=lambda x: x[1], reverse=True)
            return strategies[:top_k]

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "strategies_count": len(self._strategy_weights),
                "best_strategies": self.get_best_strategies_unlocked(5),
                "learning_rate": self.learning_rate,
            }

    def get_best_strategies_unlocked(self, top_k: int = 5) -> List[Tuple[str, float]]:
        strategies = [(s, w) for s, w in self._strategy_weights.items()]
        strategies.sort(key=lambda x: x[1], reverse=True)
        return strategies[:top_k]


class ConfidenceCalibrator:
    def __init__(self):
        self._confidence_history: List[Tuple[float, bool]] = []
        self._lock = threading.Lock()

    def record(self, confidence: float, was_correct: bool):
        with self._lock:
            self._confidence_history.append((confidence, was_correct))
            if len(self._confidence_history) > 5000:
                self._confidence_history = self._confidence_history[-5000:]

    def get_calibration_offset(self, stated_confidence: float) -> float:
        with self._lock:
            if not self._confidence_history:
                return 0.0

            nearby = [(c, correct) for c, correct in self._confidence_history
                      if abs(c - stated_confidence) < 0.1]
            if not nearby:
                return 0.0

            actual_accuracy = sum(1 for _, correct in nearby if correct) / len(nearby)
            return actual_accuracy - stated_confidence

    def calibrate(self, stated_confidence: float) -> float:
        offset = self.get_calibration_offset(stated_confidence)
        return max(0.0, min(1.0, stated_confidence + offset * 0.5))

    def get_stats(self) -> Dict:
        with self._lock:
            if not self._confidence_history:
                return {"total_records": 0, "avg_offset": 0.0}
            total = len(self._confidence_history)
            correct = sum(1 for _, c in self._confidence_history if c)
            return {
                "total_records": total,
                "accuracy": round(correct / total, 3) if total > 0 else 0.0,
                "avg_confidence": round(sum(c for c, _ in self._confidence_history) / total, 3) if total > 0 else 0.0,
            }


class SelfImprovementEngine:
    """
    The Self-Improvement Engine for Shadow AI.

    Records predictions, tracks outcomes, calibrates confidence,
    and evolves strategies based on real-world results.

    This is the "learning" part of the shadow - not just observing,
    but actively improving its own models and strategies.
    """

    _MAX_DECISIONS = 10000
    _MAX_OPEN_DECISIONS = 1000
    _MAX_OBSERVERS = 20

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / "digital-shadow" / "data" / "self_improvement"
        self.data_dir.mkdir(parents=True, mode=0o700, exist_ok=True)

        self.calibrator = AccuracyCalibrator()
        self.evolver = StrategyEvolver(learning_rate=0.1)
        self.confidence_calibrator = ConfidenceCalibrator()

        self._decisions: Dict[str, DecisionRecord] = {}
        self._open_decisions: Dict[str, DecisionRecord] = {}
        self._improvement_history: List[Dict] = []
        self._calibration_task: Optional[asyncio.Task] = None
        self._calibration_interval: float = 3600.0
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()

        self._observers: List[Callable] = []
        self._observers_lock = threading.Lock()
        self._stats = {
            "predictions_made": 0,
            "decisions_recorded": 0,
            "outcomes_resolved": 0,
            "feedback_received": 0,
            "improvement_cycles": 0,
            "brier_score": 0.0,
        }
        self._stats_lock = threading.Lock()

    def record_prediction(self, content: str, predicted_outcome: str,
                           confidence: float = 0.5, domain: Optional[str] = None,
                           model: str = "default") -> Prediction:
        content = _sanitize_content(content)
        predicted_outcome = _sanitize_content(predicted_outcome, 500)
        domain = _sanitize(domain, 99) if domain else None
        model = _sanitize(model, 100)
        prediction = Prediction(
            content=content,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            domain=domain,
            model=model,
        )
        self.calibrator.record_prediction(prediction)
        with self._lock:
            with self._stats_lock:
                self._stats["predictions_made"] += 1
        return prediction

    def resolve_prediction(self, prediction: Prediction, actual_outcome: str,
                            outcome_type: OutcomeType = OutcomeType.NEUTRAL) -> float:
        actual_outcome = _sanitize_content(actual_outcome, 500)
        prediction.resolve(actual_outcome, outcome_type)
        with self._lock:
            with self._stats_lock:
                self._stats["outcomes_resolved"] += 1
        self.confidence_calibrator.record(prediction.confidence, prediction.accuracy is not None and prediction.accuracy > 0.5)
        return prediction.accuracy or 0.0

    def record_decision(self, action: str, reasoning: str = "",
                          predicted_outcome: str = "", confidence: float = 0.5,
                          domain: Optional[str] = None,
                          autonomy_level: str = "auto") -> DecisionRecord:
        valid_levels = {"auto", "notify", "ask", "forbid"}
        if autonomy_level not in valid_levels:
            autonomy_level = "auto"
        action = _sanitize_content(action, 500)
        reasoning = _sanitize_content(reasoning, 2000)
        predicted_outcome = _sanitize_content(predicted_outcome, 500)
        domain = _sanitize(domain, 99) if domain else None
        decision = DecisionRecord(
            action=action,
            reasoning=reasoning,
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            domain=domain,
            autonomy_level=autonomy_level,
        )
        with self._lock:
            self._decisions[decision.decision_id] = decision
            if len(self._decisions) > self._MAX_DECISIONS:
                oldest = list(self._decisions.keys())[:len(self._decisions) - self._MAX_DECISIONS]
                for k in oldest:
                    self._decisions.pop(k, None)
                    self._open_decisions.pop(k, None)
            self._open_decisions[decision.decision_id] = decision
            if len(self._open_decisions) > self._MAX_OPEN_DECISIONS:
                oldest_open = list(self._open_decisions.keys())[:len(self._open_decisions) - self._MAX_OPEN_DECISIONS]
                for k in oldest_open:
                    self._open_decisions.pop(k, None)
            with self._stats_lock:
                self._stats["decisions_recorded"] += 1
        return decision

    def resolve_decision(self, decision_id: str, actual_outcome: str,
                          outcome_type: OutcomeType = OutcomeType.NEUTRAL) -> Optional[DecisionRecord]:
        actual_outcome = _sanitize_content(actual_outcome, 500)
        with self._lock:
            decision = self._decisions.get(decision_id)
            if not decision:
                return None

            decision.resolve(actual_outcome, outcome_type)
            self._open_decisions.pop(decision_id, None)

            strategy = f"{decision.autonomy_level}_{decision.domain or 'general'}"
            score = 1.0 if outcome_type in (OutcomeType.POSITIVE, OutcomeType.NEUTRAL) else 0.0
            self.evolver.record_strategy_outcome(strategy, score)

            with self._stats_lock:
                self._stats["outcomes_resolved"] += 1
        return decision

    def add_feedback(self, decision_id: str, source: FeedbackSource,
                      rating: float, comment: str = "") -> bool:
        with self._lock:
            decision = self._decisions.get(decision_id)
            if not decision:
                return False
            decision.add_feedback(source, rating, comment)
            with self._stats_lock:
                self._stats["feedback_received"] += 1

            if rating >= 0.7:
                self.evolver.record_strategy_outcome(decision.autonomy_level, rating)
            elif rating < 0.3:
                self.evolver.record_strategy_outcome(decision.autonomy_level, rating)

        return True

    async def run_improvement_cycle(self) -> Dict:
        with self._lock:
            with self._stats_lock:
                self._stats["improvement_cycles"] += 1
            open_count = len(self._open_decisions)

        calibration_error = self.calibrator.get_calibration_error()
        overall_accuracy = self.calibrator.get_overall_accuracy()
        best_strategies = self.evolver.get_best_strategies(5)

        improvements = []
        if calibration_error > 0.2:
            improvements.append({
                "area": "confidence_calibration",
                "issue": f"Calibration error is {calibration_error:.2f} (should be < 0.1)",
                "action": "Adjust confidence estimates based on historical accuracy",
            })

        if overall_accuracy < 0.5:
            improvements.append({
                "area": "prediction_accuracy",
                "issue": f"Overall accuracy is {overall_accuracy:.2f} (should be > 0.5)",
                "action": "Consider switching to a better prediction model or strategy",
            })

        if open_count > 10:
            improvements.append({
                "area": "outcome_tracking",
                "issue": f"{open_count} decisions pending outcome resolution",
                "action": "Resolve pending decisions to improve learning",
            })

        with self._lock:
            stats_snapshot = dict(self._stats)

        report = {
            "cycle_number": stats_snapshot["improvement_cycles"],
            "overall_accuracy": round(overall_accuracy, 3),
            "calibration_error": round(calibration_error, 3),
            "best_strategies": best_strategies,
            "improvements": improvements,
            "stats": stats_snapshot,
        }

        with self._history_lock:
            self._improvement_history.append(report)
            if len(self._improvement_history) > 100:
                self._improvement_history = self._improvement_history[-100:]

        with self._observers_lock:
            observers_snapshot = list(self._observers)
        for observer in observers_snapshot:
            try:
                observer(report)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                pass

        self._save_state()
        return report

    def calibrate_confidence(self, stated_confidence: float) -> float:
        return self.confidence_calibrator.calibrate(stated_confidence)

    def get_open_decisions(self) -> List[Dict]:
        with self._lock:
            return [copy.deepcopy(d.to_dict()) for d in self._open_decisions.values()]

    def get_recent_decisions(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            decisions = sorted(self._decisions.values(), key=lambda d: d.created_at, reverse=True)
            return [copy.deepcopy(d.to_dict()) for d in decisions[:limit]]

    def add_observer(self, observer: Callable) -> bool:
        with self._observers_lock:
            if len(self._observers) >= self._MAX_OBSERVERS:
                return False
            self._observers.append(observer)
        return True

    def get_improvement_history(self, limit: int = 10) -> List[Dict]:
        with self._history_lock:
            return [copy.deepcopy(h) for h in self._improvement_history[-limit:]]

    def get_brier_score(self) -> float:
        resolved = [p for p in self.calibrator._predictions if p.accuracy is not None and p.confidence is not None]
        if not resolved:
            return 0.0
        scores = []
        for p in resolved:
            outcome = 1.0 if p.accuracy > 0.5 else 0.0
            scores.append((p.confidence - outcome) ** 2)
        return sum(scores) / len(scores) if scores else 0.0

    async def start_auto_calibration(self):
        if self._calibration_task and not self._calibration_task.done():
            return
        self._calibration_task = asyncio.create_task(self._calibration_loop())

    async def stop_auto_calibration(self):
        if self._calibration_task and not self._calibration_task.done():
            self._calibration_task.cancel()
            try:
                await self._calibration_task
            except asyncio.CancelledError:
                pass

    async def _calibration_loop(self):
        while True:
            try:
                await asyncio.sleep(self._calibration_interval)
                with self._lock:
                    decisions_count = len(self._decisions)
                if len(self.calibrator._predictions) > 5 or decisions_count > 5:
                    report = await self.run_improvement_cycle()
                    logger.info("Auto-calibration: accuracy=%.3f", report.get('overall_accuracy', 0))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Auto-calibration error: %s", type(e).__name__)
                await asyncio.sleep(300)

    def _save_state(self):
        try:
            state_file = self.data_dir / "improvement_state.json"
            data = {
                "decisions_count": len(self._decisions),
                "open_decisions": len(self._open_decisions),
                "improvement_cycles": self._improvement_history[-1:] if self._improvement_history else [],
                "brier_score": self.get_brier_score(),
            }
            tmp = state_file.with_suffix('.json.tmp')
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(json.dumps(data, indent=2, default=str))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(str(tmp), str(state_file))
            except (OSError, TypeError, ValueError):
                try:
                    os.unlink(str(tmp))
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to save improvement state: %s", type(e).__name__)

    def get_stats(self) -> Dict:
        with self._stats_lock:
            self._stats["brier_score"] = round(self.get_brier_score(), 4)
            stats_copy = dict(self._stats)
        with self._lock:
            open_count = len(self._open_decisions)
            total_count = len(self._decisions)
        return {
            "stats": stats_copy,
            "calibrator_stats": self.calibrator.get_stats(),
            "evolver_stats": self.evolver.get_stats(),
            "confidence_calibrator_stats": self.confidence_calibrator.get_stats(),
            "open_decisions": open_count,
            "total_decisions": total_count,
        }