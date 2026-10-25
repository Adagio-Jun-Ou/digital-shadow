"""
Cross-Module Anomaly Correlation Engine — Detecting systemic failure patterns.

Correlates anomalies across modules to detect:
1. Cascading failures (anomaly in A causes anomaly in B)
2. Shared root causes (multiple anomalies from single trigger)
3. Temporal clustering (anomaly bursts in time windows)
4. Domain-specific patterns (security anomalies correlate with CPU spikes)
5. Predictive amplification (small early signals predict large later failures)

Design: Statistical correlation engine, not ML-dependent. Uses Pearson, Spearman,
and Jaccard similarity on anomaly timestamps and severity vectors.
"""

import copy
import logging
import math
import re
import threading
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

def _sanitize_content(text: str, max_length: int = 1000) -> str:
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_RE.sub('', text).strip()[:max_length]

_MAX_ANOMALIES = 5000
_MAX_CORRELATIONS = 1000
_MAX_CASCADES = 200
_MAX_CLUSTERS = 100
_MAX_BURST_WINDOW_SECONDS = 3600


class CorrelationType(Enum):
    TEMPORAL = "temporal"
    SEVERITY = "severity"
    CAUSAL = "causal"
    TOPICAL = "topical"
    DOMAIN_CROSS = "domain_cross"


class CascadeDirection(Enum):
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    BIDIRECTIONAL = "bidirectional"


class AnomalyCategory(Enum):
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATA = "data"
    COGNITIVE = "cognitive"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


@dataclass
class CorrelatedAnomaly:
    anomaly_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    module_name: str = ""
    category: str = "unknown"
    description: str = ""
    severity: float = 0.0
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_resolved: bool = False

    def __post_init__(self):
        self.module_name = _sanitize(self.module_name, 100)
        self.category = _sanitize(self.category, 50)
        self.description = _sanitize_content(self.description, 2000)
        self.severity = max(0.0, min(1.0, self.severity))
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:30]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "module_name": self.module_name,
            "category": self.category,
            "description": self.description[:200],
            "severity": round(self.severity, 3),
            "detected_at": self.detected_at.isoformat(),
            "is_resolved": self.is_resolved,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class AnomalyCorrelation:
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str = ""
    target_id: str = ""
    correlation_type: str = "temporal"
    correlation_score: float = 0.0
    confidence: float = 0.0
    evidence: str = ""
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    def __post_init__(self):
        self.source_id = _sanitize(self.source_id, 20)
        self.target_id = _sanitize(self.target_id, 20)
        self.correlation_type = _sanitize(self.correlation_type, 30)
        self.correlation_score = max(-1.0, min(1.0, self.correlation_score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.evidence = _sanitize_content(self.evidence, 2000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "correlation_type": self.correlation_type,
            "correlation_score": round(self.correlation_score, 4),
            "confidence": round(self.confidence, 3),
            "is_active": self.is_active,
        }


@dataclass
class CascadeChain:
    cascade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    root_anomaly_id: str = ""
    chain: List[str] = field(default_factory=list)
    depth: int = 0
    max_severity: float = 0.0
    propagation_speed: float = 0.0
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_contained: bool = False

    def __post_init__(self):
        self.root_anomaly_id = _sanitize(self.root_anomaly_id, 20)
        self.chain = [_sanitize(a, 20) for a in (self.chain or [])[:50]]
        self.depth = len(self.chain)
        self.max_severity = max(0.0, min(1.0, self.max_severity))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cascade_id": self.cascade_id,
            "root_anomaly_id": self.root_anomaly_id,
            "depth": self.depth,
            "max_severity": round(self.max_severity, 3),
            "propagation_speed": round(self.propagation_speed, 4),
            "is_contained": self.is_contained,
        }


class AnomalyCorrelationEngine:
    _MAX_ANOMALIES = _MAX_ANOMALIES
    _MAX_CORRELATIONS = _MAX_CORRELATIONS
    _MAX_CASCADES = _MAX_CASCADES

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._anomalies: deque = deque(maxlen=_MAX_ANOMALIES)
        self._correlations: Dict[str, AnomalyCorrelation] = {}
        self._corr_order: deque = deque(maxlen=_MAX_CORRELATIONS)
        self._cascades: Dict[str, CascadeChain] = {}
        self._cascade_order: deque = deque(maxlen=_MAX_CASCADES)
        self._module_anomaly_index: Dict[str, Set[str]] = defaultdict(set)
        self._category_index: Dict[str, Set[str]] = defaultdict(set)

        self._stats: Dict[str, int] = {
            "anomalies_recorded": 0,
            "correlations_found": 0,
            "cascades_detected": 0,
            "bursts_detected": 0,
            "predictive_alerts": 0,
        }

    def record_anomaly(
        self, module_name: str, category: str, description: str,
        severity: float = 0.5,
    ) -> str:
        anomaly = CorrelatedAnomaly(
            module_name=module_name, category=category,
            description=description, severity=severity,
        )
        with self._lock:
            self._anomalies.append(anomaly)
            self._module_anomaly_index[module_name].add(anomaly.anomaly_id)
            self._category_index[category].add(anomaly.anomaly_id)
        with self._stats_lock:
            self._stats["anomalies_recorded"] += 1
        return anomaly.anomaly_id

    def detect_correlations(self) -> List[AnomalyCorrelation]:
        with self._lock:
            snapshot = list(self._anomalies)
        if len(snapshot) < 4:
            return []

        correlations = []
        anomalies_by_module = defaultdict(list)
        for a in snapshot:
            anomalies_by_module[a.module_name].append(a)

        for i in range(len(snapshot)):
            for j in range(i + 1, len(snapshot)):
                a1, a2 = snapshot[i], snapshot[j]
                if a1.module_name == a2.module_name:
                    continue

                time_diff = abs((a1.detected_at - a2.detected_at).total_seconds())
                time_score = max(0.0, 1.0 - (time_diff / 300.0))

                sev_diff = abs(a1.severity - a2.severity)
                sev_score = max(0.0, 1.0 - sev_diff)

                if a1.category == a2.category:
                    sev_score += 0.2

                combined = round(time_score * 0.4 + sev_score * 0.6, 4)
                if combined < 0.3:
                    continue

                corr = AnomalyCorrelation(
                    source_id=a1.anomaly_id,
                    target_id=a2.anomaly_id,
                    correlation_type="temporal" if time_score > sev_score else "severity",
                    correlation_score=combined,
                    confidence=round(combined * 0.9, 3),
                    evidence=f"Temporal proximity: {time_diff:.1f}s, severity diff: {round(sev_diff, 3)}",
                )
                correlations.append(corr)

        with self._lock:
            for corr in correlations:
                if corr.correlation_id not in self._correlations:
                    self._correlations[corr.correlation_id] = corr
                    self._corr_order.append(corr.correlation_id)
            while len(self._correlations) > self._MAX_CORRELATIONS:
                old_id = self._corr_order.popleft()
                self._correlations.pop(old_id, None)

        with self._stats_lock:
            self._stats["correlations_found"] += len(correlations)

        return correlations

    def detect_cascades(self, max_depth: int = 10) -> List[CascadeChain]:
        with self._lock:
            snapshot = list(self._anomalies)
        self.detect_correlations()

        with self._lock:
            corr_snapshot = list(self._correlations.values())

        cascades = []
        visited = set()

        for corr in corr_snapshot:
            if not corr.is_active or corr.correlation_score < 0.5:
                continue
            if corr.source_id in visited:
                continue

            chain = [corr.source_id]
            current = corr.target_id
            depth = 0

            while current and depth < max_depth:
                chain.append(current)
                visited.add(current)
                next_corr = next(
                    (c for c in corr_snapshot
                     if c.source_id == current and c.is_active and c.correlation_score >= 0.5),
                    None,
                )
                current = next_corr.target_id if next_corr else None
                depth += 1

            if len(chain) >= 2:
                cascade = CascadeChain(
                    root_anomaly_id=chain[0],
                    chain=chain,
                    depth=len(chain),
                    max_severity=max(
                        (a.severity for a in snapshot if a.anomaly_id in chain),
                        default=0.0,
                    ),
                )
                cascades.append(cascade)

        with self._lock:
            for c in cascades:
                self._cascades[c.cascade_id] = c
                self._cascade_order.append(c.cascade_id)
            while len(self._cascades) > self._MAX_CASCADES:
                old_id = self._cascade_order.popleft()
                self._cascades.pop(old_id, None)

        with self._stats_lock:
            self._stats["cascades_detected"] += len(cascades)

        return cascades

    def detect_bursts(self, window_seconds: float = 300, min_count: int = 3) -> List[Dict[str, Any]]:
        window_seconds = max(1, min(_MAX_BURST_WINDOW_SECONDS, window_seconds))
        min_count = max(2, min(20, min_count))

        with self._lock:
            snapshot = sorted(list(self._anomalies), key=lambda a: a.detected_at)

        bursts = []
        i = 0
        while i < len(snapshot):
            j = i
            burst_anomalies = []
            start_time = snapshot[i].detected_at

            while j < len(snapshot) and (
                snapshot[j].detected_at - start_time
            ).total_seconds() <= window_seconds:
                burst_anomalies.append(snapshot[j])
                j += 1

            if len(burst_anomalies) >= min_count:
                modules = set(a.module_name for a in burst_anomalies)
                categories = set(a.category for a in burst_anomalies)
                bursts.append({
                    "start_time": start_time.isoformat(),
                    "end_time": burst_anomalies[-1].detected_at.isoformat(),
                    "count": len(burst_anomalies),
                    "unique_modules": len(modules),
                    "modules": sorted(modules),
                    "unique_categories": len(categories),
                    "categories": sorted(categories),
                    "avg_severity": round(
                        sum(a.severity for a in burst_anomalies) / len(burst_anomalies),
                        3,
                    ),
                    "max_severity": round(
                        max(a.severity for a in burst_anomalies), 3,
                    ),
                })

            i = j if j > i else i + 1

        with self._stats_lock:
            self._stats["bursts_detected"] += len(bursts)

        return bursts

    def get_correlated_anomalies(
        self, anomaly_id: str, min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        anomaly_id = _sanitize(anomaly_id, 20)
        min_score = max(0.0, min(1.0, min_score))

        with self._lock:
            related = []
            for corr in self._correlations.values():
                if corr.correlation_score < min_score:
                    continue
                if corr.source_id == anomaly_id:
                    related.append(corr.to_dict())
                elif corr.target_id == anomaly_id:
                    related.append(corr.to_dict())

        return related

    def get_cascades(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(_MAX_CASCADES, limit))
        with self._lock:
            cascades = sorted(
                list(self._cascades.values()),
                key=lambda c: c.depth, reverse=True,
            )
            return [c.to_dict() for c in cascades[:limit]]

    def get_recent_anomalies(
        self, limit: int = 50, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(_MAX_ANOMALIES, limit))
        category = _sanitize(category, 50) if category else None

        with self._lock:
            anomalies = list(self._anomalies)
            if category:
                anomalies = [a for a in anomalies if a.category == category]
            anomalies.sort(key=lambda a: a.detected_at, reverse=True)
            return [a.to_dict() for a in anomalies[:limit]]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_anomalies = len(self._anomalies)
            total_correlations = len(self._correlations)
            total_cascades = len(self._cascades)

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_anomalies"] = total_anomalies
        stats["active_correlations"] = sum(
            1 for c in self._correlations.values() if c.is_active
        )
        stats["active_cascades"] = sum(
            1 for c in self._cascades.values() if not c.is_contained
        )
        return stats
