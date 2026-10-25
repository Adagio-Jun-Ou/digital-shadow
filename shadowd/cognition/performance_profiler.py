"""
ShadowAI Performance Profiler — Module-level benchmarking and optimization tracking.

Tracks per-module metrics:
1. Execution time (mean, p95, p99)
2. Memory footprint
3. Call frequency / throughput
4. Error rates
5. Cache hit ratios

Provides optimization recommendations based on real data.
"""

import copy
import logging
import re
import threading
import time as time_module
import uuid
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 100) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 1000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_RE.sub('', text)
    return text.strip()[:max_length]


_MAX_SAMPLES_PER_MODULE = 500
_MAX_MODULES = 200
_MAX_RECOMMENDATIONS = 200


@dataclass
class TimingSample:
    module_name: str
    operation: str
    duration_ms: float
    memory_bytes: int = 0
    error_occurred: bool = False
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self):
        self.module_name = _sanitize(self.module_name, 100)
        self.operation = _sanitize(self.operation, 100)
        self.duration_ms = max(0.0, self.duration_ms)
        self.memory_bytes = max(0, self.memory_bytes)


@dataclass
class ModuleProfile:
    module_name: str
    total_calls: int = 0
    total_errors: int = 0
    total_cache_hits: int = 0
    durations_ms: List[float] = field(default_factory=list)
    mean_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    error_rate: float = 0.0
    cache_hit_ratio: float = 0.0
    calls_per_second: float = 0.0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "total_cache_hits": self.total_cache_hits,
            "mean_duration_ms": round(self.mean_duration_ms, 3),
            "p95_duration_ms": round(self.p95_duration_ms, 3),
            "p99_duration_ms": round(self.p99_duration_ms, 3),
            "error_rate": round(self.error_rate, 4),
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
            "calls_per_second": round(self.calls_per_second, 4),
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


@dataclass
class OptimizationRecommendation:
    recommendation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    module_name: str = ""
    priority: str = "medium"
    category: str = ""
    description: str = ""
    expected_improvement_pct: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied: bool = False

    def __post_init__(self):
        self.module_name = _sanitize(self.module_name, 100)
        self.priority = _sanitize(self.priority, 20)
        self.category = _sanitize(self.category, 50)
        self.description = _sanitize_content(self.description, 2000)
        self.expected_improvement_pct = max(0.0, min(100.0, self.expected_improvement_pct))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "module_name": self.module_name,
            "priority": self.priority,
            "category": self.category,
            "description": self.description[:200],
            "expected_improvement_pct": round(self.expected_improvement_pct, 1),
            "applied": self.applied,
        }


class PerformanceProfiler:
    _MAX_SAMPLES_PER_MODULE = _MAX_SAMPLES_PER_MODULE
    _MAX_MODULES = _MAX_MODULES
    _MAX_RECOMMENDATIONS = _MAX_RECOMMENDATIONS

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=_MAX_SAMPLES_PER_MODULE)
        )
        self._module_order: deque = deque(maxlen=_MAX_MODULES)
        self._recommendations: Dict[str, OptimizationRecommendation] = {}
        self._rec_order: deque = deque(maxlen=_MAX_RECOMMENDATIONS)

        self._stats: Dict[str, int] = {
            "samples_recorded": 0,
            "modules_profiled": 0,
            "recommendations_generated": 0,
        }

    def record(
        self,
        module_name: str,
        operation: str,
        duration_ms: float,
        memory_bytes: int = 0,
        error: bool = False,
        cache_hit: bool = False,
    ) -> str:
        sample = TimingSample(
            module_name=module_name,
            operation=operation,
            duration_ms=duration_ms,
            memory_bytes=memory_bytes,
            error_occurred=error,
            cache_hit=cache_hit,
        )

        with self._lock:
            self._samples[module_name].append(sample)
            if module_name not in self._module_order:
                self._module_order.append(module_name)

        with self._stats_lock:
            self._stats["samples_recorded"] += 1
            self._stats["modules_profiled"] = len(self._samples)

        return sample.sample_id

    def get_module_profile(self, module_name: str) -> Optional[ModuleProfile]:
        module_name = _sanitize(module_name, 100)

        with self._lock:
            samples = list(self._samples.get(module_name, []))
            if not samples:
                return None

        durations = sorted([s.duration_ms for s in samples])
        total = len(durations)
        errors = sum(1 for s in samples if s.error_occurred)
        hits = sum(1 for s in samples if s.cache_hit)

        mean_dur = sum(durations) / total
        p95_idx = int(total * 0.95)
        p99_idx = int(total * 0.99)

        first_seen = min(s.timestamp for s in samples)
        last_seen = max(s.timestamp for s in samples)
        span_seconds = max(0.001, (last_seen - first_seen).total_seconds())

        return ModuleProfile(
            module_name=module_name,
            total_calls=total,
            total_errors=errors,
            total_cache_hits=hits,
            durations_ms=durations,
            mean_duration_ms=round(mean_dur, 3),
            p95_duration_ms=round(durations[min(p95_idx, total - 1)], 3),
            p99_duration_ms=round(durations[min(p99_idx, total - 1)], 3),
            error_rate=round(errors / total, 4),
            cache_hit_ratio=round(hits / total, 4),
            calls_per_second=round(total / span_seconds, 4),
            first_seen=first_seen,
            last_seen=last_seen,
        )

    def get_all_profiles(self) -> List[Dict[str, Any]]:
        with self._lock:
            module_names = list(self._samples.keys())

        profiles = []
        for name in sorted(module_names):
            profile = self.get_module_profile(name)
            if profile:
                profiles.append(profile.to_dict())
        return profiles

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        with self._lock:
            module_names = list(self._samples.keys())

        recommendations = []
        for name in module_names:
            profile = self.get_module_profile(name)
            if profile is None:
                continue

            recs = self._analyze_for_recommendations(profile)
            recommendations.extend(recs)

        with self._lock:
            for rec in recommendations:
                self._recommendations[rec.recommendation_id] = rec
                self._rec_order.append(rec.recommendation_id)
            while len(self._recommendations) > self._MAX_RECOMMENDATIONS:
                old_id = self._rec_order.popleft()
                self._recommendations.pop(old_id, None)

        with self._stats_lock:
            self._stats["recommendations_generated"] += len(recommendations)

        return [r.to_dict() for r in recommendations]

    def _analyze_for_recommendations(self, profile: ModuleProfile) -> List[OptimizationRecommendation]:
        recs = []

        if profile.error_rate > 0.05:
            recs.append(OptimizationRecommendation(
                module_name=profile.module_name,
                priority="high",
                category="error_rate",
                description=f"Error rate at {round(profile.error_rate * 100, 1)}% — investigate root causes",
                expected_improvement_pct=50.0,
            ))

        if profile.p95_duration_ms > 100:
            recs.append(OptimizationRecommendation(
                module_name=profile.module_name,
                priority="high",
                category="latency",
                description=f"P95 latency at {profile.p95_duration_ms}ms — consider async or caching",
                expected_improvement_pct=40.0,
            ))

        if profile.cache_hit_ratio < 0.5 and profile.total_calls > 50:
            recs.append(OptimizationRecommendation(
                module_name=profile.module_name,
                priority="medium",
                category="caching",
                description=f"Cache hit ratio at {round(profile.cache_hit_ratio * 100, 1)}% — add result caching",
                expected_improvement_pct=30.0,
            ))

        if profile.calls_per_second > 10:
            recs.append(OptimizationRecommendation(
                module_name=profile.module_name,
                priority="medium",
                category="throughput",
                description=f"High throughput at {profile.calls_per_second} calls/s — consider batching",
                expected_improvement_pct=25.0,
            ))

        if profile.p95_duration_ms > 5 * profile.mean_duration_ms and profile.total_calls > 20:
            recs.append(OptimizationRecommendation(
                module_name=profile.module_name,
                priority="low",
                category="tail_latency",
                description=f"P95 ({profile.p95_duration_ms}ms) is {round(profile.p95_duration_ms / max(1, profile.mean_duration_ms), 1)}x the mean ({profile.mean_duration_ms}ms)",
                expected_improvement_pct=20.0,
            ))

        return recs

    def get_recommendations(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(_MAX_RECOMMENDATIONS, limit))
        with self._lock:
            recs = sorted(
                list(self._recommendations.values()),
                key=lambda r: r.created_at, reverse=True,
            )
            return [r.to_dict() for r in recs[:limit]]

    def get_slow_modules(self, threshold_ms: float = 50, top_n: int = 10) -> List[Dict[str, Any]]:
        profiles = self.get_all_profiles()
        profiles.sort(key=lambda p: p.get("p95_duration_ms", 0) or 0, reverse=True)
        slow = [p for p in profiles if (p.get("p95_duration_ms", 0) or 0) > threshold_ms]
        return slow[:top_n]

    def get_error_prone_modules(self, threshold: float = 0.01, top_n: int = 10) -> List[Dict[str, Any]]:
        profiles = self.get_all_profiles()
        profiles.sort(key=lambda p: p.get("error_rate", 0) or 0, reverse=True)
        buggy = [p for p in profiles if (p.get("error_rate", 0) or 0) > threshold]
        return buggy[:top_n]

    def get_cache_inefficient_modules(self, threshold: float = 0.5, top_n: int = 10) -> List[Dict[str, Any]]:
        profiles = self.get_all_profiles()
        profiles.sort(key=lambda p: p.get("cache_hit_ratio", 0) or 0)
        inefficient = [p for p in profiles if (p.get("cache_hit_ratio", 0) or 0) < threshold]
        return inefficient[:top_n]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_modules = len(self._samples)
            total_samples = sum(len(s) for s in self._samples.values())
            total_recs = len(self._recommendations)

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_modules"] = total_modules
        stats["total_samples"] = total_samples
        stats["total_recommendations"] = total_recs
        return stats
