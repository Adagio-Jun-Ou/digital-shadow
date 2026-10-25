#!/usr/bin/env python3
"""Sprint 20 tests - AnomalyDetectionEngine feature.

Tests:
- MetricStream: add, mean, stddev, z_score, expected_range
- AnomalyDetectionEngine: observe, temporal, behavioral, correlation
- Anomaly: creation, to_dict, acknowledgment
- API endpoint registration
- Cap enforcement
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetricStream:
    """MetricStream unit tests."""

    def test_add_and_count(self):
        from shadowd.cognition.anomaly_detector import MetricStream
        ms = MetricStream()
        assert ms.count == 0
        ms.add(1.0)
        ms.add(2.0)
        assert ms.count == 2

    def test_mean(self):
        from shadowd.cognition.anomaly_detector import MetricStream
        ms = MetricStream()
        ms.add(2.0)
        ms.add(4.0)
        ms.add(6.0)
        assert abs(ms.mean() - 4.0) < 0.001

    def test_stddev(self):
        from shadowd.cognition.anomaly_detector import MetricStream
        ms = MetricStream()
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            ms.add(float(v))
        assert abs(ms.stddev() - 2.0) < 0.5

    def test_z_score(self):
        from shadowd.cognition.anomaly_detector import MetricStream
        ms = MetricStream()
        for v in [10, 10, 10, 10, 10]:
            ms.add(float(v))
        assert ms.z_score(10.0) == 0.0
        ms.add(20.0)
        ms.add(20.0)
        assert ms.z_score(30.0) > 2.0

    def test_expected_range(self):
        from shadowd.cognition.anomaly_detector import MetricStream
        ms = MetricStream()
        for v in [10, 10, 10, 10, 10]:
            ms.add(float(v))
        low, high = ms.expected_range(n_std=2.0)
        assert low <= 10.0 <= high

    def test_history_cap(self):
        from shadowd.cognition.anomaly_detector import MetricStream
        ms = MetricStream()
        for i in range(600):
            ms.add(float(i))
        assert ms.count <= ms._MAX_HISTORY

    def test_empty_stats(self):
        from shadowd.cognition.anomaly_detector import MetricStream
        ms = MetricStream()
        assert ms.mean() == 0.0
        assert ms.stddev() == 0.0


class TestAnomaly:
    """Anomaly dataclass tests."""

    def test_auto_id(self):
        from shadowd.cognition.anomaly_detector import Anomaly
        a = Anomaly()
        assert a.anomaly_id.startswith("anom_")

    def test_to_dict(self):
        from shadowd.cognition.anomaly_detector import Anomaly, AnomalyType, AnomalySeverity
        a = Anomaly(
            anomaly_type=AnomalyType.STATISTICAL,
            severity=AnomalySeverity.HIGH,
            metric="test_metric",
            value=100.0,
            description="Test anomaly",
        )
        d = a.to_dict()
        assert d["anomaly_type"] == "statistical"
        assert d["severity"] == "high"
        assert d["metric"] == "test_metric"
        assert d["acknowledged"] is False

    def test_acknowledge(self):
        from shadowd.cognition.anomaly_detector import Anomaly
        a = Anomaly()
        assert not a.acknowledged
        a.acknowledged = True
        assert a.acknowledged


class TestAnomalyDetectionEngine:
    """AnomalyDetectionEngine core tests."""

    def test_register_metric(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.register_metric("test_metric", "Test description", "units")
        assert "test_metric" in engine._streams

    def test_observe_no_anomaly_with_few_points(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        result = engine.observe("test", 5.0)
        assert result is None

    def test_observe_detects_statistical_anomaly(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine, AnomalyType
        engine = AnomalyDetectionEngine(z_threshold=2.0)
        for v in [10.0] * 20:
            engine.observe("test_metric", v)
        anomaly = engine.observe("test_metric", 50.0)
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.STATISTICAL
        assert abs(anomaly.z_score) > 2.0

    def test_observe_normal_value_no_anomaly(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(z_threshold=2.5)
        import random
        random.seed(42)
        for _ in range(50):
            engine.observe("test_metric", random.gauss(10.0, 2.0))
        result = engine.observe("test_metric", 10.5)
        assert result is None

    def test_temporal_anomaly_unusual_hour(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine, AnomalyType
        engine = AnomalyDetectionEngine()
        for i in range(15):
            engine.observe_temporal("test_event")
        ts = datetime(2025, 1, 1, 3, 0, tzinfo=timezone.utc)
        anomaly = engine.observe_temporal("test_event", timestamp=ts)
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.TEMPORAL

    def test_temporal_no_anomaly_during_day(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        for i in range(15):
            engine.observe_temporal("test_event")
        ts = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)
        anomaly = engine.observe_temporal("test_event", timestamp=ts)
        assert anomaly is None

    def test_behavioral_anomaly(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine, AnomalyType
        engine = AnomalyDetectionEngine()
        engine.set_baseline("user1", {
            "typing_speed": {"mean": 60.0, "std": 5.0},
        })
        anomaly = engine.observe_behavioral("user1", "typing_speed", 150.0)
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.BEHAVIORAL

    def test_behavioral_no_anomaly_near_baseline(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.set_baseline("user1", {
            "typing_speed": {"mean": 60.0, "std": 5.0},
        })
        anomaly = engine.observe_behavioral("user1", "typing_speed", 62.0)
        assert anomaly is None

    def test_correlation_anomaly(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine, AnomalyType
        engine = AnomalyDetectionEngine()
        import random
        random.seed(42)
        for _ in range(30):
            base = random.gauss(10.0, 2.0)
            engine.observe("metric_a", base)
            engine.observe("metric_b", base + random.gauss(0, 0.5))
            engine.observe("metric_c", random.gauss(10.0, 2.0))
            engine.observe("metric_d", random.gauss(10.0, 2.0))
        anomalies = engine.detect_correlation_anomaly(
            [("metric_a", "metric_b"), ("metric_c", "metric_d")],
            {"metric_a": 50.0, "metric_b": 50.0, "metric_c": 50.0, "metric_d": 50.0},
        )
        assert len(anomalies) > 0
        assert anomalies[0].anomaly_type == AnomalyType.CORRELATION

    def test_acknowledge(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(z_threshold=1.5)
        for v in [10.0] * 20:
            engine.observe("test", v)
        anomaly = engine.observe("test", 100.0)
        assert anomaly is not None
        assert not anomaly.acknowledged
        result = engine.acknowledge(anomaly.anomaly_id)
        assert result is True
        assert anomaly.acknowledged

    def test_acknowledge_not_found(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        result = engine.acknowledge("nonexistent_id")
        assert result is False


class TestAnomalyDetectionCaps:
    """Test cap enforcement."""

    def test_max_anomalies(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        assert engine._MAX_ANOMALIES == 1000

    def test_max_streams(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        assert engine._MAX_STREAMS == 200

    def test_stream_eviction(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine._MAX_STREAMS = 5
        for i in range(10):
            engine.observe(f"metric_{i}", float(i))
        assert len(engine._streams) <= 5


class TestAnomalyDetectionStats:
    """Test stats reporting."""

    def test_stats_empty(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        stats = engine.get_stats()
        assert stats["anomalies_detected"] == 0
        assert stats["streams"] == 0

    def test_stats_after_observations(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        for v in [10.0] * 20:
            engine.observe("test", v)
        stats = engine.get_stats()
        assert stats["total_observations"] == 20
        assert stats["streams"] == 1

    def test_get_anomalies_filtered(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine, AnomalySeverity
        engine = AnomalyDetectionEngine(z_threshold=1.5)
        for v in [10.0] * 20:
            engine.observe("test", v)
        engine.observe("test", 100.0)
        anomalies = engine.get_anomalies(severity=AnomalySeverity.HIGH)
        assert isinstance(anomalies, list)

    def test_get_stream_stats(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        for v in [10.0, 20.0, 30.0]:
            engine.observe("test_metric", v)
        stats = engine.get_stream_stats()
        assert "test_metric" in stats
        assert stats["test_metric"]["count"] == 3


class TestAnomalySeverityMapping:
    """Test Z-score to severity mapping."""

    def test_z_to_severity(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine, AnomalySeverity
        engine = AnomalyDetectionEngine()
        assert engine._z_to_severity(1.0) == AnomalySeverity.LOW
        assert engine._z_to_severity(2.5) == AnomalySeverity.MEDIUM
        assert engine._z_to_severity(3.5) == AnomalySeverity.HIGH
        assert engine._z_to_severity(5.0) == AnomalySeverity.CRITICAL


class TestAnomalyExportedFromModule:
    """Test that anomaly_detector is properly exported from cognition."""

    def test_import_from_cognition(self):
        from shadowd.cognition import AnomalyDetectionEngine, Anomaly, AnomalyType, AnomalySeverity
        assert AnomalyDetectionEngine is not None
        assert Anomaly is not None
        assert AnomalyType.STATISTICAL is not None
        assert AnomalySeverity.HIGH is not None