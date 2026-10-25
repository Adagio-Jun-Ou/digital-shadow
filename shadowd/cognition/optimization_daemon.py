"""
Cognitive Optimization Daemon — Autonomous parameter tuning from performance data.

Watches PerformanceProfiler, AnomalyCorrelationEngine, and HealthDashboard in a
background loop. When modules degrade, it proposes and tests parameter adjustments.
Uses evolutionary strategies to discover optimal configurations without human input.

Architecture:
- Observe: Pull metrics from profiler, anomalies from correlator, health from dashboard
- Analyze: Compute health decay, detect degradation patterns
- Optimize: Generate parameter tweaks via evolutionary perturbation
- Apply: Atomic config update with pre/post validation
- Verify: Monitor for N cycles to confirm improvement, auto-rollback on regression
"""

import copy
import logging
import math
import re
import threading
import time as time_module
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

def _sanitize(text: str, max_length: int = 100) -> str:
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]

def _sanitize_content(text: str, max_length: int = 5000) -> str:
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_RE.sub('', text).strip()[:max_length]

_MAX_HISTORY = 1000
_MAX_TUNINGS = 500
_MAX_PARAMS_PER_MODULE = 20
_VERIFICATION_CYCLES = 5
_DEFAULT_TICK_SECONDS = 60


class TuningStatus(Enum):
    IDLE = "idle"
    PROPOSING = "proposing"
    TESTING = "testing"
    ACCEPTED = "accepted"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class TunableParam:
    param_name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float = 0.01
    optimal_value: Optional[float] = None
    last_tuned_at: Optional[datetime] = None

    def __post_init__(self):
        self.param_name = _sanitize(self.param_name, 50)
        self.current_value = max(self.min_value, min(self.max_value, self.current_value))

    def perturb(self, sigma: float = 0.1) -> float:
        import random
        delta = random.gauss(0, sigma * self.step_size * (self.max_value - self.min_value))
        return max(self.min_value, min(self.max_value, self.current_value + delta))


@dataclass
class TuningAttempt:
    attempt_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    module_name: str = ""
    param_changes: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    status: str = "idle"
    baseline_health: float = 1.0
    post_health: float = 1.0
    cycles_verified: int = 0
    improvement: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: str = ""

    def __post_init__(self):
        self.module_name = _sanitize(self.module_name, 100)
        self.status = _sanitize(self.status, 20)
        self.error_message = _sanitize_content(self.error_message, 500)
        self.improvement = round(self.improvement, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "module_name": self.module_name,
            "status": self.status,
            "baseline_health": round(self.baseline_health, 3),
            "post_health": round(self.post_health, 3),
            "improvement": self.improvement,
            "cycles_verified": self.cycles_verified,
            "started_at": self.started_at.isoformat(),
        }


class CognitiveOptimizationDaemon:
    _MAX_HISTORY = _MAX_HISTORY
    _MAX_TUNINGS = _MAX_TUNINGS

    def __init__(self, shadow_os=None, tick_seconds: float = _DEFAULT_TICK_SECONDS):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._os = shadow_os

        self._tick_seconds = max(5.0, min(3600.0, tick_seconds))
        self._running = False
        self._params: Dict[str, Dict[str, TunableParam]] = defaultdict(dict)
        self._tuning_history: deque = deque(maxlen=_MAX_TUNINGS)
        self._active_tuning: Optional[TuningAttempt] = None
        self._health_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=_MAX_HISTORY)
        )

        self._stats: Dict[str, int] = {
            "ticks_run": 0,
            "degradations_detected": 0,
            "tunings_attempted": 0,
            "tunings_accepted": 0,
            "tunings_rolled_back": 0,
            "params_registered": 0,
        }

    def register_param(self, module_name: str, param_name: str, value: float,
                       min_val: float, max_val: float, step: float = 0.01) -> None:
        module_name = _sanitize(module_name, 100)
        param_name = _sanitize(param_name, 50)
        with self._lock:
            self._params[module_name][param_name] = TunableParam(
                param_name=param_name, current_value=value,
                min_value=min_val, max_value=max_val, step_size=step,
            )
        with self._stats_lock:
            self._stats["params_registered"] = sum(len(p) for p in self._params.values())

    def get_params(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if module_name:
                module_name = _sanitize(module_name, 100)
                params = self._params.get(module_name, {})
                return {
                    k: {"value": v.current_value, "optimal": v.optimal_value}
                    for k, v in params.items()
                }
            return {
                mod: {k: {"value": v.current_value, "optimal": v.optimal_value}
                      for k, v in params.items()}
                for mod, params in self._params.items()
            }

    def tick(self) -> Optional[Dict[str, Any]]:
        if not self._os:
            return None

        with self._stats_lock:
            self._stats["ticks_run"] += 1

        health_data = self._collect_health()
        self._update_health_history(health_data)

        if active.status == "testing":
            return self._verify_tuning()

        degradation = self._detect_degradation(health_data)
        if degradation:
            return self._attempt_tuning(degradation)

        return {"action": "no_degradation", "modules_healthy": len(health_data)}

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(_MAX_TUNINGS, limit))
        with self._lock:
            return [t.to_dict() for t in list(self._tuning_history)[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        stats["active_tuning"] = self._active_tuning is not None
        with self._lock:
            stats["total_params"] = sum(len(p) for p in self._params.values())
        return stats

    def _collect_health(self) -> Dict[str, float]:
        health = {}
        try:
            h = self._os.get_health()
            modules = h.get("modules", {})
            for mod, status in modules.items():
                if isinstance(status, dict):
                    health[mod] = 1.0 if status.get("healthy") else 0.5
                else:
                    health[mod] = 0.7
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("Health collection failed: %s", type(e).__name__)
        return health

    def _update_health_history(self, health: Dict[str, float]) -> None:
        with self._lock:
            for mod, score in health.items():
                self._health_history[mod].append({
                    "timestamp": datetime.now(timezone.utc),
                    "score": score,
                })

    def _detect_degradation(self, health: Dict[str, float]) -> Optional[str]:
        with self._lock:
            for mod, score in health.items():
                if score >= 0.7:
                    continue
                history = list(self._health_history[mod])
                if len(history) < 3:
                    continue
                recent = [h["score"] for h in history[-5:]]
                if len(recent) >= 2 and all(
                    recent[i] <= recent[i - 1] for i in range(1, len(recent))
                ):
                    with self._stats_lock:
                        self._stats["degradations_detected"] += 1
                    return mod
        return None

    def _attempt_tuning(self, module_name: str) -> Dict[str, Any]:
        with self._lock:
            params = self._params.get(module_name, {})
            if not params:
                return {"action": "no_tunable_params", "module": module_name}

            baseline = 0.5
            history = list(self._health_history[module_name])
            if history:
                baseline = history[-1]["score"]

            changes = {}
            for name, param in list(params.items())[:3]:
                new_val = param.perturb(0.2)
                changes[name] = (param.current_value, new_val)
                param.current_value = new_val

            attempt = TuningAttempt(
                module_name=module_name,
                param_changes=changes,
                status="testing",
                baseline_health=round(baseline, 3),
            )
            self._active_tuning = attempt
            self._tuning_history.append(attempt)

        with self._stats_lock:
            self._stats["tunings_attempted"] += 1

        return {"action": "tuning_started", "module": module_name,
                "attempt_id": attempt.attempt_id}

    def _verify_tuning(self) -> Dict[str, Any]:
        attempt = self._active_tuning
        if not attempt:
            return {"action": "no_active_tuning"}

        attempt.cycles_verified += 1
        current_health = 0.5
        history = list(self._health_history.get(attempt.module_name, []))
        if history:
            current_health = history[-1]["score"]

        if attempt.cycles_verified < _VERIFICATION_CYCLES:
            return {"action": "verifying", "module": attempt.module_name,
                    "cycles": attempt.cycles_verified}

        improvement = round(current_health - attempt.baseline_health, 4)
        attempt.post_health = round(current_health, 3)
        attempt.improvement = improvement
        attempt.completed_at = datetime.now(timezone.utc)

        if improvement >= 0.0:
            attempt.status = "accepted"
            with self._stats_lock:
                self._stats["tunings_accepted"] += 1
            result = {"action": "tuning_accepted", "improvement": improvement}
        else:
            attempt.status = "rolled_back"
            self._rollback_params(attempt)
            with self._stats_lock:
                self._stats["tunings_rolled_back"] += 1
            result = {"action": "tuning_rolled_back", "regression": improvement}

        self._active_tuning = None
        return result

    def _rollback_params(self, attempt: TuningAttempt) -> None:
        with self._lock:
            params = self._params.get(attempt.module_name, {})
            for name, (old_val, _new_val) in attempt.param_changes.items():
                if name in params:
                    params[name].current_value = old_val
