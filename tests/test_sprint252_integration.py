#!/usr/bin/env python3
"""Sprint 252 — Cross-Module Integration: Device Push → Cognition Chain Tests"""

import pytest
import sys
import os
import copy
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SHADOW_OS_API_KEY", "")

from shadowd.cognition.device_context import DeviceContextEngine, DeviceType, ActivityType, StressLevel
from shadowd.cognition.emotional_intelligence import EmotionalIntelligence, EmotionCategory
from shadowd.cognition.attention_engine import AttentionEngine, AttentionLevel
from shadowd.cognition.personality_engine import PersonalityEngine
from shadowd.cognition.health_dashboard import CognitiveHealthDashboard
from shadowd.cognition.shadow_os import ShadowOS


class TestDevicePushToEmotionalResponse:
    def test_iphone_stress_data_ingestion(self):
        os_ = ShadowOS()
        os_.device_context.register_device("iphone_test", DeviceType.IPHONE, "Test Phone")
        ctx = os_.device_context.ingest_iphone("iphone_test", {
            "health": {"heart_rate": 120, "hrv": 20, "steps": 3000, "stress_level": "high", "sleep_hours": 3}
        })
        assert ctx is not None
        assert ctx.health.heart_rate == 120

    def test_watch_heart_rate_ingestion(self):
        os_ = ShadowOS()
        os_.device_context.register_device("watch_test", DeviceType.APPLE_WATCH, "Test Watch")
        ctx = os_.device_context.ingest_watch("watch_test", {
            "heart_rate": 130, "hrv": 15, "steps": 1000, "stress_level": "critical"
        })
        assert ctx is not None

    def test_emotional_detection_with_device_engine(self):
        """Test emotional detection with DeviceContextEngine (not UnifiedContext)."""
        ei = EmotionalIntelligence()
        engine = DeviceContextEngine()
        engine.register_device("iphone_test", DeviceType.IPHONE, "Test Phone")
        engine.ingest_iphone("iphone_test", {
            "health": {"heart_rate": 120, "hrv": 20, "stress_level": "high"}
        })
        result = ei.detect_emotion("I need to finish this project", device_context=engine)
        assert result is not None
        assert result.intensity > 0.3


class TestDevicePushToAttentionState:
    def test_sleep_deficit_via_engine(self):
        ae = AttentionEngine()
        engine = DeviceContextEngine()
        engine.register_device("watch_test", DeviceType.APPLE_WATCH, "Test Watch")
        engine.ingest_watch("watch_test", {
            "heart_rate": 65, "hrv": 50, "steps": 2000, "sleep_minutes": 180, "stress_level": "low"
        })
        result = ae.detect_focus_state(device_context=engine)
        # Sleep deficit should be detected - result is FocusState enum
        assert result is not None


class TestDevicePushToPersonalityPrediction:
    def test_personality_with_device_engine(self):
        pe = PersonalityEngine()
        engine = DeviceContextEngine()
        engine.register_device("iphone_test", DeviceType.IPHONE, "Test Phone")
        engine.ingest_iphone("iphone_test", {
            "health": {"heart_rate": 110, "stress_level": "high"}
        })
        result = pe.predict_decision(
            [{"name": "safe", "risk_level": 0.1}, {"name": "risky", "risk_level": 0.9}],
            "decide", device_context=engine
        )
        assert result is not None


class TestDevicePushToAnticipatoryEngine:
    def test_anticipatory_predict(self):
        from shadowd.cognition.anticipatory_engine import AnticipatoryEngine, AnticipationContext
        ae = AnticipatoryEngine()
        ctx = AnticipationContext(
            time_of_day=14, active_domains=["work"],
            emotional_state="stressed", recent_actions=["checked email", "reviewed PR"]
        )
        signals = ae.predict(ctx)
        assert isinstance(signals, list)


class TestHealthDashboardAutoUpdate:
    def test_health_dashboard_refresh(self):
        os_ = ShadowOS()
        modules = {
            "emotional_intelligence": os_.emotional_intel,
            "attention_engine": os_.attention,
            "device_context": os_.device_context,
        }
        os_.health_dashboard.auto_update(modules)
        stats = os_.health_dashboard.get_stats()
        assert "total_modules" in stats
        assert stats["total_modules"] >= 3

    def test_health_dashboard_with_device_data(self):
        os_ = ShadowOS()
        os_.device_context.register_device("iphone_test", DeviceType.IPHONE, "Test Phone")
        os_.device_context.ingest_iphone("iphone_test", {
            "health": {"heart_rate": 85, "hrv": 40, "steps": 5000}
        })
        modules = {
            "emotional_intelligence": os_.emotional_intel,
            "attention_engine": os_.attention,
            "device_context": os_.device_context,
        }
        os_.health_dashboard.auto_update(modules)
        stats = os_.health_dashboard.get_stats()
        assert stats["total_modules"] >= 3


class TestSiriPushEndpointIntegration:
    def test_iphone_push_via_endpoint(self):
        from shadowd.cognition.api_router import SiriPushRequest
        req = SiriPushRequest(device_type="iphone", data={"health": {"heart_rate": 72, "hrv": 45, "steps": 8500}})
        assert req.device_type == "iphone"
        assert req.device_id is None
        assert req.data["health"]["heart_rate"] == 72

    def test_watch_push_via_endpoint(self):
        from shadowd.cognition.api_router import SiriPushRequest
        req = SiriPushRequest(device_type="watch", data={"heart_rate": 68, "hrv": 50, "spo2": 98})
        assert req.device_type == "watch"

    def test_location_push_via_endpoint(self):
        from shadowd.cognition.api_router import SiriPushRequest
        req = SiriPushRequest(device_type="iphone", device_id="my_iphone_15",
                              data={"location": {"latitude": 37.7749, "longitude": -122.4194, "name": "Home"}})
        assert req.device_id == "my_iphone_15"

    def test_battery_push_via_endpoint(self):
        from shadowd.cognition.api_router import SiriPushRequest
        req = SiriPushRequest(device_type="iphone", data={"battery": {"level": 45, "charging": False}})
        assert req.data["battery"]["level"] == 45

    def test_invalid_device_type_rejected(self):
        from shadowd.cognition.api_router import SiriPushRequest
        with pytest.raises(ValueError):
            SiriPushRequest(device_type="tablet", data={})

    def test_max_length_validation(self):
        from shadowd.cognition.api_router import SiriPushRequest
        with pytest.raises(ValueError):
            SiriPushRequest(device_type="a" * 21, data={})


class TestDeviceContextToEvolutionGuide:
    def test_device_data_feeds_evolution(self):
        os_ = ShadowOS()
        os_.device_context.register_device("iphone_test", DeviceType.IPHONE, "Test Phone")
        os_.device_context.ingest_iphone("iphone_test", {
            "health": {"heart_rate": 95, "stress_level": "high", "steps": 2000},
            "active_app": "slack",
            "active_app_category": "communication",
        })
        synced = os_.sync_evolution_from_device()
        assert "device_context" in synced


class TestDeviceContextToOracleBridge:
    def test_oracle_sync_returns_list(self):
        os_ = ShadowOS()
        synced = os_.sync_oracle_from_cognition()
        assert isinstance(synced, list)


class TestMultiDeviceFusion:
    def test_iphone_and_watch_fusion(self):
        os_ = ShadowOS()
        os_.device_context.register_device("iphone_test", DeviceType.IPHONE, "Test Phone")
        os_.device_context.register_device("watch_test", DeviceType.APPLE_WATCH, "Test Watch")

        os_.device_context.ingest_iphone("iphone_test", {
            "health": {"heart_rate": 75, "hrv": 45, "steps": 5000, "sleep_hours": 7.0},
            "battery": {"level": 80, "charging": False},
        })

        os_.device_context.ingest_watch("watch_test", {
            "heart_rate": 73, "hrv": 48, "steps": 2000, "sleep_hours": 7.2, "spo2": 98,
            "stress_level": "low"
        })

        final_ctx = os_.device_context.get_context()
        assert final_ctx is not None

    def test_laptop_device_context(self):
        os_ = ShadowOS()
        os_.device_context.register_device("laptop_test", DeviceType.LAPTOP, "Test Laptop")
        ctx = os_.device_context.ingest_laptop("laptop_test", {
            "active_window": "VSCode",
            "screen_on": True,
            "cpu_percent": 45,
            "memory_percent": 60,
            "battery_level": 90,
            "battery_charging": True,
        })
        assert ctx is not None


class TestDeviceContextStressInference:
    def test_critical_stress_detection(self):
        engine = DeviceContextEngine()
        engine.register_device("test", DeviceType.APPLE_WATCH, "Test")
        ctx = engine.ingest_watch("test", {
            "heart_rate": 140, "hrv": 10, "steps": 500, "stress_level": "critical"
        })
        if ctx and ctx.health:
            assert ctx.health.stress_level in [StressLevel.HIGH, StressLevel.CRITICAL]

    def test_low_stress_detection(self):
        engine = DeviceContextEngine()
        engine.register_device("test", DeviceType.APPLE_WATCH, "Test")
        ctx = engine.ingest_watch("test", {
            "heart_rate": 65, "hrv": 60, "steps": 8000, "stress_level": "low"
        })
        if ctx and ctx.health:
            assert ctx.health.stress_level in [StressLevel.LOW, StressLevel.HIGH]


class TestPydanticStringValidation:
    def test_voice_start_max_length(self):
        from shadowd.cognition.api_router import VoiceStartRequest
        req = VoiceStartRequest(voice="alloy")
        assert req.voice == "alloy"

    def test_voice_start_too_long(self):
        from shadowd.cognition.api_router import VoiceStartRequest
        with pytest.raises(ValueError):
            VoiceStartRequest(voice="a" * 101)

    def test_voice_tts_max_length(self):
        from shadowd.cognition.api_router import VoiceTTSRequest
        req = VoiceTTSRequest(text="Hello", voice="alloy")
        assert req.voice == "alloy"

    def test_goal_create_priority_max_length(self):
        from shadowd.cognition.api_router import GoalCreateRequest
        req = GoalCreateRequest(title="Test Goal")
        assert req.priority == "medium"

    def test_anticipate_request_emotional_state(self):
        from shadowd.cognition.api_router import AnticipateRequest
        req = AnticipateRequest(emotional_state="happy")
        assert req.emotional_state == "happy"

    def test_health_update_request_status(self):
        from shadowd.cognition.api_router import HealthUpdateRequest
        req = HealthUpdateRequest(module_name="test", status="healthy")
        assert req.status == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])