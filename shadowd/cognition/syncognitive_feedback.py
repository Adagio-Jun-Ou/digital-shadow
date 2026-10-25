#!/usr/bin/env python3
"""
SyncognitiveFeedback — System Observes Own Thinking.

Metacognitive feedback loops where the system monitors and adjusts its own
cognitive processes. Observes process execution, detects biases, corrects
confidence, and optimizes resource allocation through self-reflective loops.

Thread-safe, bounded, sanitized, deepcopy returns.
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
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.syncognitive_feedback")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_OBSERVATIONS = 5000
_MAX_LOOPS = 500
_MAX_PROFILES = 100
_MAX_PER_PROCESS = 200
_MAX_METADATA_KEYS = 20


class FeedbackType(str, Enum):
    SELF_CORRECTION = "self_correction"
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"
    BIAS_DETECTION = "bias_detection"
    PROCESS_OPTIMIZATION = "process_optimization"
    RESOURCE_REALLOCATION = "resource_reallocation"
    STRATEGY_SWITCH = "strategy_switch"


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
class CognitiveObservation:
    observation_id: str
    process_name: str
    observation_type: FeedbackType
    details: str
    confidence_before: float
    confidence_after: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.observation_id:
            self.observation_id = uuid.uuid4().hex[:12]
        self.observation_id = _sanitize(self.observation_id, 50)
        self.process_name = _sanitize(self.process_name, 100)
        self.details = _sanitize_content(self.details, 2000)
        self.confidence_before = _clamp(self.confidence_before)
        self.confidence_after = _clamp(self.confidence_after)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "observation_id": self.observation_id,
            "process_name": self.process_name,
            "observation_type": self.observation_type.value,
            "details": self.details,
            "confidence_before": round(self.confidence_before, 4),
            "confidence_after": round(self.confidence_after, 4),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class FeedbackLoop:
    loop_id: str
    source_process: str
    target_process: str
    feedback_type: FeedbackType
    strength: float
    evidence_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.loop_id:
            self.loop_id = uuid.uuid4().hex[:12]
        self.loop_id = _sanitize(self.loop_id, 50)
        self.source_process = _sanitize(self.source_process, 100)
        self.target_process = _sanitize(self.target_process, 100)
        self.strength = _clamp(self.strength)
        self.evidence_count = max(0, self.evidence_count)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "loop_id": self.loop_id,
            "source_process": self.source_process,
            "target_process": self.target_process,
            "feedback_type": self.feedback_type.value,
            "strength": round(self.strength, 4),
            "evidence_count": self.evidence_count,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class ProcessProfile:
    process_name: str
    avg_confidence: float
    avg_latency: float
    error_rate: float
    self_corrections: int = 0
    optimization_count: int = 0
    last_observed: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.process_name = _sanitize(self.process_name, 100)
        self.avg_confidence = _clamp(self.avg_confidence)
        self.avg_latency = _clamp(self.avg_latency, 0.0, 1.0)
        self.error_rate = _clamp(self.error_rate)
        self.self_corrections = max(0, self.self_corrections)
        self.optimization_count = max(0, self.optimization_count)
        self.last_observed = _sanitize_content(self.last_observed, 500)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "process_name": self.process_name,
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_latency": round(self.avg_latency, 4),
            "error_rate": round(self.error_rate, 4),
            "self_corrections": self.self_corrections,
            "optimization_count": self.optimization_count,
            "last_observed": self.last_observed,
            "metadata": copy.deepcopy(self.metadata),
        }


class SyncognitiveFeedback:
    """Metacognitive feedback loops for self-observation and self-correction.

    The system monitors its own cognitive processes, detects biases,
    adjusts confidence levels, and optimizes resource allocation through
    structured feedback loops.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._observations: Dict[str, CognitiveObservation] = {}
        self._loops: Dict[str, FeedbackLoop] = {}
        self._profiles: Dict[str, ProcessProfile] = {}
        self._observation_order: deque = deque(maxlen=_MAX_OBSERVATIONS)
        self._loop_order: deque = deque(maxlen=_MAX_LOOPS)
        self._process_obs: Dict[str, List[str]] = {}
        self._stats = {
            "observations_created": 0,
            "loops_created": 0,
            "self_corrections": 0,
            "bias_detections": 0,
            "confidence_adjustments": 0,
            "optimizations": 0,
        }

    def observe_process(self, process_name: str, details: str,
                        confidence: float,
                        observation_type: FeedbackType) -> CognitiveObservation:
        observation = CognitiveObservation(
            observation_id=uuid.uuid4().hex[:12],
            process_name=process_name,
            observation_type=observation_type,
            details=details,
            confidence_before=confidence,
            confidence_after=confidence,
        )
        with self._lock:
            if len(self._observations) >= _MAX_OBSERVATIONS:
                evicted_id = self._observation_order.popleft()
                self._observations.pop(evicted_id, None)
                for obs_list in self._process_obs.values():
                    try:
                        obs_list.remove(evicted_id)
                    except ValueError:
                        pass
                logger.warning("Evicted observation %s", evicted_id)
            self._observations[observation.observation_id] = observation
            self._observation_order.append(observation.observation_id)
            self._process_obs.setdefault(observation.process_name, []).append(
                observation.observation_id
            )
            if len(self._process_obs[observation.process_name]) > _MAX_PER_PROCESS:
                excess = len(self._process_obs[observation.process_name]) - _MAX_PER_PROCESS
                remove_ids = self._process_obs[observation.process_name][:excess]
                self._process_obs[observation.process_name] = (
                    self._process_obs[observation.process_name][excess:]
                )
                logger.warning("Evicted %d per-process overflow observations for %s", len(remove_ids), observation.process_name)
                for rid in remove_ids:
                    self._observations.pop(rid, None)
                    try:
                        self._observation_order.remove(rid)
                    except ValueError:
                        pass
            self._update_profile(observation)
            with self._stats_lock:
                self._stats["observations_created"] += 1
        return copy.deepcopy(observation)

    def create_feedback_loop(self, source: str, target: str,
                             feedback_type: FeedbackType,
                             strength: float) -> FeedbackLoop:
        loop = FeedbackLoop(
            loop_id=uuid.uuid4().hex[:12],
            source_process=source,
            target_process=target,
            feedback_type=feedback_type,
            strength=strength,
        )
        with self._lock:
            for existing in self._loops.values():
                if (existing.source_process == loop.source_process and
                        existing.target_process == loop.target_process and
                        existing.feedback_type == loop.feedback_type):
                    existing.evidence_count += 1
                    existing.strength = _clamp(
                        (existing.strength + loop.strength) / 2
                    )
                    return copy.deepcopy(existing)
            if len(self._loops) >= _MAX_LOOPS:
                evicted_id = self._loop_order.popleft()
                self._loops.pop(evicted_id, None)
                logger.warning("Evicted feedback loop %s", evicted_id)
            self._loops[loop.loop_id] = loop
            self._loop_order.append(loop.loop_id)
            with self._stats_lock:
                self._stats["loops_created"] += 1
        return copy.deepcopy(loop)

    def propagate_feedback(self, loop_id: str,
                           observation: CognitiveObservation) -> Dict:
        lid = _sanitize(loop_id, 50)
        with self._lock:
            loop = self._loops.get(lid)
            if not loop:
                return {"propagated": False, "reason": "loop_not_found"}
            adjustments: Dict[str, Any] = {
                "source_process": loop.source_process,
                "target_process": loop.target_process,
                "feedback_type": loop.feedback_type.value,
                "adjustments": {},
            }
            if loop.feedback_type == FeedbackType.CONFIDENCE_ADJUSTMENT:
                delta = (observation.confidence_after - observation.confidence_before) * loop.strength
                adjustments["adjustments"]["confidence_delta"] = round(delta, 4)
            elif loop.feedback_type == FeedbackType.RESOURCE_REALLOCATION:
                adjustments["adjustments"]["resource_shift"] = round(loop.strength * 0.1, 4)
            elif loop.feedback_type == FeedbackType.STRATEGY_SWITCH:
                adjustments["adjustments"]["strategy_override"] = loop.strength >= 0.6
            elif loop.feedback_type == FeedbackType.BIAS_DETECTION:
                adjustments["adjustments"]["bias_flag"] = True
                adjustments["adjustments"]["bias_strength"] = round(loop.strength, 4)
            elif loop.feedback_type == FeedbackType.SELF_CORRECTION:
                adjustments["adjustments"]["correction_applied"] = True
                adjustments["adjustments"]["correction_strength"] = round(loop.strength, 4)
            elif loop.feedback_type == FeedbackType.PROCESS_OPTIMIZATION:
                adjustments["adjustments"]["optimization_hint"] = round(loop.strength * 0.15, 4)
            loop.evidence_count += 1
            loop.strength = _clamp(min(1.0, loop.strength + 0.01))
            adjustments["evidence_count"] = loop.evidence_count
            adjustments["propagated"] = True
            return adjustments

    def correct_self(self, process_name: str, correction: str,
                     confidence_delta: float) -> CognitiveObservation:
        pname = _sanitize(process_name, 100)
        corr = _sanitize_content(correction, 2000)
        with self._lock:
            profile = self._profiles.get(pname)
            confidence_before = profile.avg_confidence if profile else 0.5
            confidence_after = _clamp(confidence_before + confidence_delta)
            observation = CognitiveObservation(
                observation_id=uuid.uuid4().hex[:12],
                process_name=pname,
                observation_type=FeedbackType.SELF_CORRECTION,
                details=corr,
                confidence_before=confidence_before,
                confidence_after=confidence_after,
            )
            if len(self._observations) >= _MAX_OBSERVATIONS:
                evicted_id = self._observation_order.popleft()
                self._observations.pop(evicted_id, None)
                logger.warning("Evicted observation %s", evicted_id)
            self._observations[observation.observation_id] = observation
            self._observation_order.append(observation.observation_id)
            self._process_obs.setdefault(pname, []).append(observation.observation_id)
            if profile:
                profile.self_corrections += 1
                profile.avg_confidence = confidence_after
                profile.last_observed = observation.created_at
            with self._stats_lock:
                self._stats["observations_created"] += 1
                self._stats["self_corrections"] += 1
        return copy.deepcopy(observation)

    def detect_bias(self, process_name: str) -> List[CognitiveObservation]:
        pname = _sanitize(process_name, 100)
        with self._lock:
            obs_ids = list(self._process_obs.get(pname, []))
            results = []
            for oid in obs_ids:
                obs = self._observations.get(oid)
                if obs and obs.observation_type == FeedbackType.BIAS_DETECTION:
                    results.append(obs)
            if not results:
                all_obs = [self._observations[oid] for oid in obs_ids
                           if oid in self._observations]
                confidence_vals = [o.confidence_after for o in all_obs]
                if len(confidence_vals) >= 3:
                    mean_conf = sum(confidence_vals) / len(confidence_vals)
                    variance = sum((c - mean_conf) ** 2 for c in confidence_vals) / len(confidence_vals)
                    if variance < 0.001:
                        bias_obs = CognitiveObservation(
                            observation_id=uuid.uuid4().hex[:12],
                            process_name=pname,
                            observation_type=FeedbackType.BIAS_DETECTION,
                            details="Low variance in confidence indicates potential confirmation bias",
                            confidence_before=mean_conf,
                            confidence_after=_clamp(mean_conf - 0.05),
                        )
                        results.append(bias_obs)
            return [copy.deepcopy(r) for r in results[:20]]

    def get_process_profile(self, process_name: str) -> Optional[ProcessProfile]:
        pname = _sanitize(process_name, 100)
        with self._lock:
            profile = self._profiles.get(pname)
            return copy.deepcopy(profile) if profile else None

    def _update_profile(self, observation: CognitiveObservation):
        pname = observation.process_name
        if pname not in self._profiles:
            if len(self._profiles) >= _MAX_PROFILES:
                oldest = next(iter(self._profiles))
                logger.warning("Evicted process profile %s", oldest)
                del self._profiles[oldest]
            self._profiles[pname] = ProcessProfile(
                process_name=pname,
                avg_confidence=observation.confidence_after,
                avg_latency=0.5,
                error_rate=0.0,
                last_observed=observation.created_at,
            )
        else:
            profile = self._profiles[pname]
            n = max(1, sum(1 for oid in self._process_obs.get(pname, [])
                          if oid in self._observations))
            profile.avg_confidence = _clamp(
                (profile.avg_confidence * (n - 1) + observation.confidence_after) / n
            )
            profile.last_observed = observation.created_at
            if observation.observation_type == FeedbackType.SELF_CORRECTION:
                profile.self_corrections += 1
            elif observation.observation_type == FeedbackType.PROCESS_OPTIMIZATION:
                profile.optimization_count += 1

    def get_observation(self, observation_id: str) -> Optional[CognitiveObservation]:
        oid = _sanitize(observation_id, 50)
        with self._lock:
            obs = self._observations.get(oid)
            return copy.deepcopy(obs) if obs else None

    def list_observations(self, process_name: str = "",
                          limit: int = 100) -> List[CognitiveObservation]:
        limit = max(1, min(500, limit))
        with self._lock:
            if process_name:
                pname = _sanitize(process_name, 100)
                obs_ids = self._process_obs.get(pname, [])[-limit:]
            else:
                obs_ids = list(self._observation_order)[-limit:]
            return [copy.deepcopy(self._observations[oid]) for oid in obs_ids
                    if oid in self._observations]

    def list_feedback_loops(self, limit: int = 100) -> List[FeedbackLoop]:
        limit = max(1, min(500, limit))
        with self._lock:
            loop_ids = list(self._loop_order)[:limit]
            return [copy.deepcopy(self._loops[lid]) for lid in loop_ids
                    if lid in self._loops]

    def get_stats(self) -> Dict:
        with self._lock:
            stats = dict(self._stats)
            stats["observation_count"] = len(self._observations)
            stats["loop_count"] = len(self._loops)
            stats["profile_count"] = len(self._profiles)
            return stats