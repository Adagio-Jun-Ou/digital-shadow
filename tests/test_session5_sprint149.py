#!/usr/bin/env python3
"""Sprint 149 — Cross-module integration tests for EvolutionGuide + OracleBridge."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestEvolutionGuideDeviceContextIntegration:
    """Test EvolutionGuide.ingest_device_context() with device data."""

    def setup_method(self):
        from shadowd.cognition.evolution_guide import EvolutionGuide
        self.eg = EvolutionGuide()

    def test_ingest_high_stress(self):
        ctx = {"health": {"stress_level": 0.85, "heart_rate": 75, "sleep_hours": 7}}
        self.eg.ingest_device_context(ctx)
        obs_list = list(self.eg._observations.values())
        stress_obs = [o for o in obs_list if "stress" in o.description.lower()]
        assert len(stress_obs) >= 1

    def test_ingest_high_heart_rate(self):
        ctx = {"health": {"stress_level": 0.3, "heart_rate": 110, "sleep_hours": 7}}
        self.eg.ingest_device_context(ctx)
        obs_list = list(self.eg._observations.values())
        hr_obs = [o for o in obs_list if "heart rate" in o.description.lower()]
        assert len(hr_obs) >= 1

    def test_ingest_sleep_deficit(self):
        ctx = {"health": {"stress_level": 0.2, "heart_rate": 65, "sleep_hours": 3.5}}
        self.eg.ingest_device_context(ctx)
        obs_list = list(self.eg._observations.values())
        sleep_obs = [o for o in obs_list if "sleep" in o.description.lower()]
        assert len(sleep_obs) >= 1

    def test_ingest_distracted(self):
        ctx = {"focus": {"focus_state": "distracted"}}
        self.eg.ingest_device_context(ctx)
        obs_list = list(self.eg._observations.values())
        focus_obs = [o for o in obs_list if "distracted" in o.description.lower()]
        assert len(focus_obs) >= 1

    def test_ingest_activity(self):
        ctx = {"activity": {"activity_type": "walking"}}
        self.eg.ingest_device_context(ctx)
        obs_list = list(self.eg._observations.values())
        act_obs = [o for o in obs_list if "walking" in o.description.lower()]
        assert len(act_obs) >= 1

    def test_ingest_resting_not_observed(self):
        ctx = {"activity": {"activity_type": "resting"}}
        self.eg.ingest_device_context(ctx)
        before = len(self.eg._observations)
        ctx2 = {"activity": {"activity_type": "unknown"}}
        self.eg.ingest_device_context(ctx2)
        assert len(self.eg._observations) == before

    def test_ingest_empty_context(self):
        self.eg.ingest_device_context({})
        assert len(self.eg._observations) == 0

    def test_ingest_with_to_dict_object(self):
        class FakeContext:
            def to_dict(self):
                return {"health": {"stress_level": 0.9}}
        self.eg.ingest_device_context(FakeContext())
        assert len(self.eg._observations) >= 1

    def test_ingest_exception_handling(self):
        self.eg.ingest_device_context(None)
        assert len(self.eg._observations) == 0


class TestEvolutionGuideBriefingIntegration:
    """Test EvolutionGuide.get_briefing_evolution_items()."""

    def setup_method(self):
        from shadowd.cognition.evolution_guide import EvolutionGuide
        self.eg = EvolutionGuide()

    def test_briefing_empty(self):
        items = self.eg.get_briefing_evolution_items()
        assert isinstance(items, list)
        assert len(items) == 0

    def test_briefing_with_observation(self):
        self.eg.add_observation(
            description="Consistently skipping breakfast before work",
            domain="health",
            observation_type="pattern",
            confidence=0.7,
        )
        items = self.eg.get_briefing_evolution_items(limit=10)
        assert isinstance(items, list)

    def test_briefing_with_suggestion(self):
        self.eg.generate_suggestions(domain="health", limit=3)
        items = self.eg.get_briefing_evolution_items(limit=5)
        assert isinstance(items, list)

    def test_briefing_limit(self):
        for i in range(5):
            self.eg.add_observation(
                description=f"Pattern {i}",
                domain="work",
                observation_type="behavior",
                confidence=0.5,
            )
        items = self.eg.get_briefing_evolution_items(limit=2)
        assert len(items) <= 2


class TestOracleBridgeTemporalIntegration:
    """Test OracleBridge.ingest_temporal_data()."""

    def setup_method(self):
        from shadowd.cognition.oracle_bridge import OracleBridge
        self.ob = OracleBridge()

    def test_ingest_temporal_patterns(self):
        class Pattern:
            pattern_type = "daily"
            frequency = 0.85
            confidence = 0.9
            trend = "increasing"

        class FakeTemporalReasoning:
            def detect_patterns(self):
                return [Pattern()]

        self.ob.ingest_temporal_data(FakeTemporalReasoning())
        stats = self.ob.get_stats()
        assert stats.get("temporal_ingestions", 0) >= 1

    def test_ingest_empty_temporal(self):
        class FakeTemporalReasoning:
            def detect_patterns(self):
                return []
        self.ob.ingest_temporal_data(FakeTemporalReasoning())
        stats = self.ob.get_stats()
        assert stats.get("temporal_ingestions", 0) == 0

    def test_ingest_no_detect_method(self):
        self.ob.ingest_temporal_data(None)
        stats = self.ob.get_stats()
        assert stats.get("temporal_ingestions", 0) == 0


class TestOracleBridgeAnomalyIntegration:
    """Test OracleBridge.ingest_anomalies()."""

    def setup_method(self):
        from shadowd.cognition.oracle_bridge import OracleBridge
        self.ob = OracleBridge()

    def test_ingest_anomaly_detector(self):
        class Anomaly:
            def to_dict(self):
                return {"severity": "high", "metric": "cpu_usage", "z_score": 3.5}

        class FakeAnomalyDetector:
            def get_recent_anomalies(self, limit=10):
                return [Anomaly()]

        self.ob.ingest_anomalies(FakeAnomalyDetector())
        stats = self.ob.get_stats()
        assert stats.get("anomaly_ingestions", 0) >= 1

    def test_ingest_empty_anomalies(self):
        class FakeAnomalyDetector:
            def get_recent_anomalies(self, limit=10):
                return []
        self.ob.ingest_anomalies(FakeAnomalyDetector())
        stats = self.ob.get_stats()
        assert stats.get("anomaly_ingestions", 0) == 0

    def test_ingest_no_method(self):
        self.ob.ingest_anomalies(None)
        stats = self.ob.get_stats()
        assert stats.get("anomaly_ingestions", 0) == 0


class TestShadowOSIntegration:
    """Test ShadowOS convenience methods for cross-module integration."""

    def setup_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        self.os = ShadowOS()

    def test_sync_evolution_from_device(self):
        result = self.os.sync_evolution_from_device()
        assert isinstance(result, list)

    def test_sync_oracle_from_cognition(self):
        result = self.os.sync_oracle_from_cognition()
        assert isinstance(result, list)

    def test_get_evolution_briefing_items(self):
        items = self.os.get_evolution_briefing_items(limit=5)
        assert isinstance(items, list)
        assert len(items) <= 5

    def test_sync_evolution_with_context(self):
        from shadowd.cognition.device_context import DeviceType
        self.os.device_context.register_device(
            device_id="test_dev_123", device_type=DeviceType.IPHONE, name="Test Phone"
        )
        self.os.device_context.ingest_iphone("test_dev_123", {
            "stress_level": 0.9,
            "heart_rate": 110,
            "sleep_hours": 3,
        })
        result = self.os.sync_evolution_from_device()
        assert "device_context" in result

    def test_sync_oracle_with_temporal(self):
        self.os.temporal_reasoning.record_event("test_event")
        result = self.os.sync_oracle_from_cognition()
        assert "temporal_reasoning" in result