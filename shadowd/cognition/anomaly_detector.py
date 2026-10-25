#!/usr/bin/env python3
"""
AnomalyDetectionEngine - Real-time Anomaly Detection for ShadowOS

Detects anomalies across all cognitive subsystems using:
- Statistical anomaly detection (Z-score on observation streams)
- Behavioral deviation detection (compares against personality baseline)
- Temporal anomaly detection (unusual timing patterns)
- Cross-correlation anomaly detection (correlated deviations)

Inspired by: Isolation Forest concepts, Etsy's Skyline anomaly detection,
and the 12-Factor Agent principle of self-monitoring.
"""

import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import time
import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..atomic_io import fsync_dir

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()

def _sanitize_content(text: str, max_length: int = 10000) -> str:
    if text is None:
        return ""
    return _SANITIZE_CONTENT_RE.sub('', str(text)).strip()[:max_length]

logger = logging.getLogger("shadowd.cognition.anomaly_detector")


class AnomalyType(str, Enum):
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    CORRELATION = "correlation"
    THRESHOLD = "threshold"


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    anomaly_id: str = ""
    anomaly_type: AnomalyType = AnomalyType.STATISTICAL
    severity: AnomalySeverity = AnomalySeverity.LOW
    metric: str = ""
    value: float = 0.0
    expected_range: Tuple[float, float] = (0.0, 0.0)
    z_score: float = 0.0
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    acknowledged: bool = False

    def __post_init__(self):
        if not self.anomaly_id:
            self.anomaly_id = f"anom_{uuid.uuid4().hex[:12]}"
        self.metric = _sanitize(self.metric)[:100]
        self.description = _sanitize_content(self.description)[:10000]

    def to_dict(self) -> Dict:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "metric": self.metric,
            "value": round(self.value, 4),
            "expected_range": [round(self.expected_range[0], 4), round(self.expected_range[1], 4)],
            "z_score": round(self.z_score, 4),
            "description": self.description,
            "context": copy.deepcopy(self.context),
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
        }


class MetricStream:
    _MAX_HISTORY = 500

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._values: List[float] = []
        self._timestamps: List[float] = []

    def add(self, value: float):
        self._values.append(value)
        self._timestamps.append(time.monotonic())
        if len(self._values) > self._MAX_HISTORY:
            self._values = self._values[-self._MAX_HISTORY:]
            self._timestamps = self._timestamps[-self._MAX_HISTORY:]

    @property
    def count(self) -> int:
        return len(self._values)

    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def stddev(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean()
        variance = sum((x - m) ** 2 for x in self._values) / (len(self._values) - 1)
        return variance ** 0.5

    def z_score(self, value: float) -> float:
        sd = self.stddev()
        if sd == 0:
            return 0.0
        m = self.mean()
        return (value - m) / sd

    def expected_range(self, n_std: float = 2.0) -> Tuple[float, float]:
        m = self.mean()
        sd = self.stddev()
        return (m - n_std * sd, m + n_std * sd)

    def recent(self, n: int = 10) -> List[float]:
        return self._values[-n:]

    def clear(self):
        self._values.clear()
        self._timestamps.clear()


class AnomalyDetectionEngine:
    """
    Real-time anomaly detection engine for ShadowOS.

    Monitors metric streams from all subsystems and detects:
    - Statistical outliers (Z-score > threshold)
    - Behavioral deviations from personality baseline
    - Temporal anomalies (events at unusual times)
    - Correlated anomalies (multiple metrics diverging together)

    All detections produce Anomaly objects that can be:
    - Logged for audit
    - Fed into the cognitive graph for reactive reasoning
    - Used to trigger escalations
    - Included in daily briefings
    """

    _MAX_STREAMS = 200
    _MAX_ANOMALIES = 1000
    _MAX_BASELINES = 1000
    _ANOMALY_DEQUE_MAXLEN = 1000
    _Z_SCORE_THRESHOLD = 2.5
    _BEHAVIORAL_DEVIATION_THRESHOLD = 0.4
    _TEMPORAL_UNUSUAL_HOURS = (0, 5)

    _MAX_COLLECTION_PURPOSES = 500

    def __init__(self, z_threshold: float = 2.5, data_dir: Optional[str] = None):
        self._lock = threading.Lock()
        self._streams: Dict[str, MetricStream] = {}
        self._streams_metadata: Dict[str, Dict] = {}
        self._anomalies: deque = deque(maxlen=self._ANOMALY_DEQUE_MAXLEN)
        self._baselines: Dict[str, Dict] = {}
        self._z_threshold = z_threshold
        self._data_dir = data_dir
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = 60.0
        self._monitoring = False
        self._stats = {
            "observations_processed": 0,
            "anomalies_detected": 0,
            "behavioral_anomalies": 0,
            "temporal_anomalies": 0,
            "correlation_anomalies": 0,
        }
        self._stats_lock = threading.Lock()
        if data_dir:
            os.makedirs(data_dir, mode=0o700, exist_ok=True)
            self._load()

    def register_metric(self, name: str, description: str = "", unit: str = "") -> None:
        name = _sanitize(name)[:100]
        description = _sanitize_content(description)[:500]
        with self._lock:
            if name not in self._streams:
                if len(self._streams) >= self._MAX_STREAMS:
                    self._evict_oldest_stream()
                self._streams[name] = MetricStream()
                self._streams_metadata[name] = {
                    "description": description,
                    "unit": unit,
                    "registered_at": datetime.now(timezone.utc).isoformat(),
                }

    def observe(self, metric: str, value: float, context: Optional[Dict] = None) -> Optional[Anomaly]:
        """
        Observe a metric value and check for anomalies.

        Returns an Anomaly if detected, None otherwise.
        """
        metric = _sanitize(metric)[:100]
        with self._lock:
            if metric not in self._streams:
                if len(self._streams) >= self._MAX_STREAMS:
                    self._evict_oldest_stream()
                self._streams[metric] = MetricStream()

            stream = self._streams[metric]
            stream.add(value)
            if context:
                context = {
                    _sanitize(str(k))[:100]: _sanitize_content(str(v), 500) if isinstance(v, str) else v
                    for k, v in list(context.items())[:20]
                }
            with self._stats_lock:
                self._stats["observations_processed"] += 1

            if stream.count < 5:
                return None

            z = stream.z_score(value)
            expected = stream.expected_range(self._z_threshold)

            if abs(z) > self._z_threshold:
                severity = self._z_to_severity(abs(z))
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    metric=metric,
                    value=value,
                    expected_range=expected,
                    z_score=z,
                    description=f"Statistical anomaly: {metric}={value:.2f} (z={z:.2f}, expected {expected[0]:.2f}-{expected[1]:.2f})",
                    context=context or {},
                )
                self._record_anomaly(anomaly)
                return anomaly

            return None

    def observe_temporal(self, event_type: str, timestamp: Optional[datetime] = None) -> Optional[Anomaly]:
        """
        Check if an event occurred at an unusual time.

        Detects events during unusual hours (0-5 AM) or with
        unusual frequency patterns.
        """
        event_type = _sanitize(event_type)[:100]
        with self._lock:
            ts = timestamp or datetime.now(timezone.utc)
            hour = ts.hour
            stream_key = f"temporal_{event_type}"

            if stream_key not in self._streams:
                self._streams[stream_key] = MetricStream()
                self._streams[stream_key].add(float(hour))
                return None

            stream = self._streams[stream_key]
            stream.add(float(hour))

            if stream.count < 10:
                return None

            z = stream.z_score(float(hour))

            if self._TEMPORAL_UNUSUAL_HOURS[0] <= hour < self._TEMPORAL_UNUSUAL_HOURS[1]:
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=AnomalySeverity.MEDIUM,
                    metric=f"temporal_{event_type}",
                    value=float(hour),
                    expected_range=(6.0, 23.0),
                    z_score=z,
                    description=f"Temporal anomaly: {event_type} at unusual hour {hour}:00 (z={z:.2f})",
                    context={"event_type": event_type, "hour": hour},
                )
                self._record_anomaly(anomaly)
                with self._stats_lock:
                    self._stats["temporal_anomalies"] += 1
                return anomaly

            return None

    def observe_behavioral(
        self,
        user_id: str,
        metric: str,
        value: float,
        personality_baseline: Optional[Dict] = None,
    ) -> Optional[Anomaly]:
        """
        Detect behavioral deviation from personality baseline.

        Compares current observation against expected patterns derived
        from the personality engine's behavioral baseline.
        """
        with self._lock:
            baseline = personality_baseline or self._baselines.get(user_id, {})
            if not baseline:
                return None

            expected = baseline.get(metric, {})
            if not expected:
                return None

            expected_mean = expected.get("mean", value)
            expected_std = expected.get("std", 0.1)

            deviation = abs(value - expected_mean) / max(expected_std, 0.01)

            if deviation > self._BEHAVIORAL_DEVIATION_THRESHOLD * self._z_threshold:
                severity = self._z_to_severity(deviation)
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.BEHAVIORAL,
                    severity=severity,
                    metric=f"behavioral_{metric}",
                    value=value,
                    expected_range=(expected_mean - 2 * expected_std, expected_mean + 2 * expected_std),
                    z_score=deviation,
                    description=f"Behavioral anomaly: {metric}={value:.2f} deviates {deviation:.1f}σ from baseline",
                    context={"user_id": hashlib.sha256(user_id.encode()).hexdigest()[:16] if user_id else None, "baseline": copy.deepcopy(expected) if expected else {}},
                )
                self._record_anomaly(anomaly)
                with self._stats_lock:
                    self._stats["behavioral_anomalies"] += 1
                return anomaly

            return None

    def set_baseline(self, user_id: str, metrics: Dict[str, Dict]) -> None:
        """
        Set behavioral baseline for a user.
        Each metric baseline has 'mean' and 'std' keys.
        """
        with self._lock:
            self._baselines[user_id] = metrics
            if len(self._baselines) > self._MAX_BASELINES:
                oldest = next(iter(self._baselines))
                logger.warning("Evicting oldest baseline '%s' to stay under cap %s", oldest, self._MAX_BASELINES)
                del self._baselines[oldest]

    def detect_correlation_anomaly(
        self,
        metric_pairs: List[Tuple[str, str]],
        current_values: Dict[str, float],
    ) -> List[Anomaly]:
        """
        Detect correlated anomalies across multiple metrics.

        When multiple metrics deviate simultaneously, it suggests
        a systemic issue rather than individual outliers.
        
        Args:
            metric_pairs: Specific pairs of metrics to check for correlation.
                If empty, checks all metrics in current_values.
            current_values: Current metric values keyed by metric name.
        """
        with self._lock:
            anomalies = []
            co_deviations = 0

            metrics_to_check = set(current_values.keys())
            if metric_pairs:
                paired_metrics = set()
                for m1, m2 in metric_pairs:
                    paired_metrics.add(m1)
                    paired_metrics.add(m2)
                metrics_to_check = metrics_to_check & paired_metrics

            for metric in metrics_to_check:
                if metric not in self._streams:
                    continue
                value = current_values.get(metric)
                if value is None:
                    continue
                stream = self._streams[metric]
                if stream.count < 5:
                    continue
                z = stream.z_score(value)
                if abs(z) > self._z_threshold:
                    co_deviations += 1

            if co_deviations >= 3:
                severity = AnomalySeverity.HIGH if co_deviations >= 5 else AnomalySeverity.MEDIUM
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.CORRELATION,
                    severity=severity,
                    metric="correlation_group",
                    value=float(co_deviations),
                    expected_range=(0.0, 2.0),
                    z_score=float(co_deviations),
                    description=f"Correlation anomaly: {co_deviations} metrics deviating simultaneously",
                    context={
                        "co_deviations": co_deviations,
                        "metrics": list(current_values.keys()),
                    },
                )
                self._record_anomaly(anomaly)
                anomalies.append(anomaly)
                with self._stats_lock:
                    self._stats["correlation_anomalies"] += 1

            return anomalies

    def get_anomalies(
        self,
        severity: Optional[AnomalySeverity] = None,
        anomaly_type: Optional[AnomalyType] = None,
        limit: int = 50,
        unacknowledged_only: bool = False,
    ) -> List[Dict]:
        """Get anomalies, optionally filtered by type/severity."""
        with self._lock:
            results = list(self._anomalies)
            if severity:
                results = [a for a in results if a.severity == severity]
            if anomaly_type:
                results = [a for a in results if a.anomaly_type == anomaly_type]
            if unacknowledged_only:
                results = [a for a in results if not a.acknowledged]
            return [a.to_dict() for a in results[-limit:]]

    def acknowledge(self, anomaly_id: str) -> bool:
        """Mark an anomaly as acknowledged."""
        with self._lock:
            for a in self._anomalies:
                if a.anomaly_id == anomaly_id:
                    a.acknowledged = True
                    return True
            return False

    def get_stats(self) -> Dict:
        with self._stats_lock:
            total_observations = self._stats["observations_processed"]
            anomalies_detected = self._stats["anomalies_detected"]
            behavioral_anomalies = self._stats["behavioral_anomalies"]
            temporal_anomalies = self._stats["temporal_anomalies"]
            correlation_anomalies = self._stats["correlation_anomalies"]
        with self._lock:
            streams = len(self._streams)
            unacknowledged = sum(1 for a in self._anomalies if not a.acknowledged)
            baselines = len(self._baselines)
        return {
            "streams": streams,
            "total_observations": total_observations,
            "anomalies_detected": anomalies_detected,
            "behavioral_anomalies": behavioral_anomalies,
            "temporal_anomalies": temporal_anomalies,
            "correlation_anomalies": correlation_anomalies,
            "unacknowledged": unacknowledged,
            "baselines": baselines,
        }

    def get_stream_stats(self) -> Dict[str, Dict]:
        """Get statistics for all metric streams."""
        with self._lock:
            result = {}
            for name, stream in self._streams.items():
                if stream.count > 0:
                    result[name] = {
                        "count": stream.count,
                        "mean": round(stream.mean(), 4),
                        "stddev": round(stream.stddev(), 4),
                        "last_value": stream._values[-1] if stream._values else None,
                    }
            return result

    def _record_anomaly(self, anomaly: Anomaly) -> None:
        self._anomalies.append(anomaly)
        with self._stats_lock:
            self._stats["anomalies_detected"] += 1
        logger.info("Anomaly detected: %s", anomaly.description.replace(chr(10), ' ').replace(chr(13), ' '))

    def _z_to_severity(self, z: float) -> AnomalySeverity:
        if z > 4.0:
            return AnomalySeverity.CRITICAL
        elif z > 3.0:
            return AnomalySeverity.HIGH
        elif z > 2.0:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW

    def _evict_oldest_stream(self) -> None:
        if not self._streams:
            return
        oldest_key = next(iter(self._streams))
        logger.warning("Evicting oldest stream '%s' to stay under cap %s", oldest_key, self._MAX_STREAMS)
        del self._streams[oldest_key]
        self._streams_metadata.pop(oldest_key, None)

    def _save(self) -> None:
        """Persist baselines and anomalies to disk."""
        if not self._data_dir:
            return
        path = os.path.join(self._data_dir, "anomaly_state.json")
        try:
            with self._lock, self._stats_lock:
                data = {
                    "baselines": copy.deepcopy(self._baselines),
                    "anomalies": [a.to_dict() for a in self._anomalies],
                    "stats": dict(self._stats),
                }
            tmp = path + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                try:
                    f = os.fdopen(fd, 'w')
                except (OSError, ValueError):
                    # fdopen failed — close the raw descriptor or Windows
                    # keeps the temp file locked and unlink fails.
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    raise
                with f:
                    f.write(json.dumps(data, indent=2, default=str))
                    f.flush()
                    os.fsync(f.fileno())
                if self._data_dir:
                    fsync_dir(self._data_dir)
                os.replace(tmp, path)
            except (OSError, TypeError, ValueError):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save anomaly state: %s", type(e).__name__)

    def _load(self) -> None:
        """Load baselines and anomalies from disk."""
        if not self._data_dir:
            return
        path = os.path.join(self._data_dir, "anomaly_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                data = json.loads(f.read())
            self._baselines = data.get("baselines", {})
            for adata in data.get("anomalies", []):
                try:
                    self._anomalies.append(Anomaly(
                        anomaly_id=adata.get("anomaly_id", ""),
                        anomaly_type=AnomalyType(adata.get("anomaly_type", "statistical")),
                        severity=AnomalySeverity(adata.get("severity", "low")),
                        metric=adata.get("metric", ""),
                        value=adata.get("value", 0.0),
                        description=adata.get("description", ""),
                        context=adata.get("context", {}),
                    ))
                except (ValueError, KeyError):
                    pass
            loaded_stats = data.get("stats", {})
            for key in self._stats:
                if key in loaded_stats:
                    val = loaded_stats[key]
                    if isinstance(val, (int, float)):
                        self._stats[key] = int(val)
            logger.info("Loaded anomaly state: %s baselines, %s anomalies", len(self._baselines), len(self._anomalies))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning("Failed to load anomaly state: %s", type(e).__name__)

    async def start_monitoring(self) -> None:
        """Start periodic anomaly monitoring loop."""
        if self._monitoring_task and not self._monitoring_task.done():
            return
        self._monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Cancel the monitoring task."""
        self._monitoring = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self) -> None:
        """Periodically check all metric streams for anomalies."""
        while self._monitoring:
            try:
                await asyncio.sleep(self._monitoring_interval)
                with self._lock:
                    stream_snapshot = {name: stream.recent(5) for name, stream in self._streams.items() if stream.count >= 5}
                current_values = {}
                for name, vals in stream_snapshot.items():
                    if vals:
                        current_values[name] = vals[-1]
                if len(current_values) >= 3:
                    await asyncio.get_running_loop().run_in_executor(
                        None, self.detect_correlation_anomaly, [], current_values
                    )
            except asyncio.CancelledError:
                break
            except (OSError, TypeError, ValueError, RuntimeError, AttributeError) as e:
                logger.error("Monitoring loop error: %s", type(e).__name__)