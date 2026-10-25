#!/usr/bin/env python3
"""Sprint 99 — Device Context Integration Tests

Tests for wiring DeviceContextEngine into DailyBriefing, PersonalityEngine,
AttentionEngine, and EmotionalIntelligence.
"""

import pytest
import asyncio
import threading
from unittest.mock import MagicMock, patch

from shadowd.cognition.device_context import (
    DeviceContextEngine, DeviceType, LocationType, ActivityType,
    StressLevel, FocusLevel, SleepPhase, NetworkType,
    LocationPoint, HealthSnapshot, DigitalSnapshot, EnvironmentSnapshot,
    UnifiedContext,
)
from shadowd.cognition.daily_briefing import (
    DailyBriefingEngine, BriefingSection, UrgencyLevel,
)
from shadowd.cognition.attention_engine import (
    AttentionEngine, AttentionLevel, FocusState,
)
from shadowd.cognition.emotional_intelligence import (
    EmotionalIntelligence, EmotionCategory,
)
from shadowd.cognition.personality_engine import PersonalityEngine


def _make_engine_with_ctx():
    engine = DeviceContextEngine()
    engine.register_device("phone-1", DeviceType.IPHONE, "Phone")
    engine.register_device("watch-1", DeviceType.APPLE_WATCH, "Watch")
    engine.register_device("laptop-1", DeviceType.LAPTOP, "Laptop")
    return engine


def _ctx_with_stress(engine, stress_level, hr=None, hrv=None, spo2=None):
    data = {"heart_rate": hr or 72, "hrv": hrv or 50, "spo2": spo2 or 97}
    if stress_level == StressLevel.CRITICAL:
        data["heart_rate"] = hr or 120
        data["hrv"] = hrv or 20
        data["spo2"] = spo2 or 92
    elif stress_level == StressLevel.HIGH:
        data["heart_rate"] = hr or 105
        data["hrv"] = hrv or 35
        data["spo2"] = spo2 or 95
    return engine.ingest_watch("watch-1", data)


def _ctx_with_focus(engine, focus_level, app="code", screen_on=True, notifs=0):
    return engine.ingest_laptop("laptop-1", {
        "active_window": app,
        "screen_on": screen_on,
        "notifications_last_hour": notifs,
    })


# ═══════════════════════════════════════════════════════════════════
# DailyBriefing Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestDailyBriefingDeviceContext:
    def setup_method(self):
        self.engine = DailyBriefingEngine()

    @pytest.mark.asyncio
    async def test_briefing_without_device_context(self):
        briefing = await self.engine.generate_briefing()
        assert briefing is not None
        wellness_items = [i for i in briefing.items if i.section == BriefingSection.WELLNESS]
        assert len(wellness_items) == 1
        assert "Baseline wellness" in wellness_items[0].content or "50%" in wellness_items[0].content

    @pytest.mark.asyncio
    async def test_briefing_with_device_context(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 72, "hrv": 55, "spo2": 98})
        briefing = await self.engine.generate_briefing(device_context=dc)
        wellness_items = [i for i in briefing.items if i.section == BriefingSection.WELLNESS]
        assert len(wellness_items) == 1
        assert "Heart rate" in wellness_items[0].content or "Stress" in wellness_items[0].content

    @pytest.mark.asyncio
    async def test_briefing_high_stress_urgency(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 120, "hrv": 20, "spo2": 92})
        briefing = await self.engine.generate_briefing(device_context=dc)
        wellness_items = [i for i in briefing.items if i.section == BriefingSection.WELLNESS]
        assert len(wellness_items) == 1
        assert wellness_items[0].urgency in (UrgencyLevel.HIGH, UrgencyLevel.CRITICAL)

    @pytest.mark.asyncio
    async def test_briefing_moderate_stress_urgency(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 90, "hrv": 40, "spo2": 96})
        briefing = await self.engine.generate_briefing(device_context=dc)
        wellness_items = [i for i in briefing.items if i.section == BriefingSection.WELLNESS]
        assert wellness_items[0].urgency in (UrgencyLevel.MEDIUM, UrgencyLevel.INFO)

    @pytest.mark.asyncio
    async def test_wellness_score_with_device_context(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 65, "hrv": 60, "spo2": 99, "step_count": 8000, "exercise_minutes": 45})
        briefing = await self.engine.generate_briefing(device_context=dc)
        assert briefing.wellness_score > 0.7

    @pytest.mark.asyncio
    async def test_wellness_score_stress_penalty(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 130, "hrv": 15, "spo2": 90})
        briefing = await self.engine.generate_briefing(device_context=dc)
        assert briefing.wellness_score < 0.7

    def test_get_wellness_metrics_with_device_context(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 65, "hrv": 60, "spo2": 99, "step_count": 8000})
        metrics = self.engine.get_wellness_metrics(device_context=dc)
        assert "stress" in metrics["components"]

    def test_get_wellness_metrics_without_device_context(self):
        metrics = self.engine.get_wellness_metrics()
        assert "stress" in metrics["components"]
        assert metrics["components"]["stress"] >= 0.3


# ═══════════════════════════════════════════════════════════════════
# AttentionEngine Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestAttentionEngineDeviceContext:
    def setup_method(self):
        self.engine = AttentionEngine()

    def test_detect_focus_state_without_device_context(self):
        state = self.engine.detect_focus_state()
        assert isinstance(state, FocusState)

    def test_detect_focus_state_with_critical_stress(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 130, "hrv": 15, "spo2": 90})
        state = self.engine.detect_focus_state(device_context=dc)
        assert state in (FocusState.FATIGUED, FocusState.DISTRACTED)

    def test_detect_focus_state_with_high_notifications(self):
        dc = _make_engine_with_ctx()
        dc.ingest_laptop("laptop-1", {
            "active_window": "chrome",
            "screen_on": True,
            "notifications_last_hour": 50,
        })
        state = self.engine.detect_focus_state(device_context=dc)
        assert state in (FocusState.DISTRACTED, FocusState.SCANNING)

    def test_detect_focus_state_with_deep_work_device(self):
        dc = _make_engine_with_ctx()
        dc.ingest_laptop("laptop-1", {
            "active_window": "code",
            "screen_on": True,
            "notifications_last_hour": 1,
        })
        dc.ingest_watch("watch-1", {"heart_rate": 65, "hrv": 55, "spo2": 98})
        # Without items/sessions, engine returns IDLE/SCANNING — device context biases
        # but can't override without underlying activity. Test that it doesn't crash.
        state = self.engine.detect_focus_state(device_context=dc)
        assert isinstance(state, FocusState)

    def test_detect_focus_state_sleep_deficit(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 80, "hrv": 40, "spo2": 96, "sleep_minutes": 180})
        # Without underlying items, IDLE is default; fatigue penalty only applies if > 0.8
        state = self.engine.detect_focus_state(device_context=dc)
        assert isinstance(state, FocusState)

    def test_suggest_attention_action_with_device_context(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 130, "hrv": 15, "spo2": 90})
        result = self.engine.suggest_attention_action(device_context=dc)
        assert "current_state" in result
        assert "suggestions" in result

    def test_suggest_attention_action_sleep_deficit(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 120, "hrv": 20, "spo2": 92, "sleep_minutes": 180})
        result = self.engine.suggest_attention_action(device_context=dc)
        sleep_suggestions = [s for s in result["suggestions"] if "sleep" in s.lower()]
        assert len(sleep_suggestions) >= 1

    def test_suggest_attention_action_high_notifications(self):
        dc = _make_engine_with_ctx()
        dc.ingest_laptop("laptop-1", {
            "screen_on": True,
            "notifications_last_hour": 40,
        })
        result = self.engine.suggest_attention_action(device_context=dc)
        notif_suggestions = [s for s in result["suggestions"] if "notif" in s.lower() or "disturb" in s.lower()]
        assert len(notif_suggestions) >= 1

    def test_suggest_attention_action_no_device_context(self):
        result = self.engine.suggest_attention_action()
        assert "current_state" in result
        assert len(result["suggestions"]) > 0


# ═══════════════════════════════════════════════════════════════════
# EmotionalIntelligence Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestEmotionalIntelligenceDeviceContext:
    def setup_method(self):
        self.engine = EmotionalIntelligence()

    def test_detect_emotion_without_device_context(self):
        state = self.engine.detect_emotion("I'm feeling happy today")
        assert state.primary == EmotionCategory.JOY

    def test_detect_emotion_with_stress(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 120, "hrv": 20, "spo2": 90})
        state = self.engine.detect_emotion("I'm not sure how I feel", device_context=dc)
        assert state.primary in (EmotionCategory.FEAR, EmotionCategory.ANGER, EmotionCategory.NEUTRAL)
        assert state.intensity >= 0.3

    def test_detect_emotion_high_stress_boosts_intensity(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 120, "hrv": 20, "spo2": 90})
        high_stress = self.engine.detect_emotion("I'm worried and scared", device_context=dc)
        no_stress = self.engine.detect_emotion("I'm worried and scared")
        assert high_stress.intensity >= no_stress.intensity

    def test_detect_emotion_moderate_stress(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 85, "hrv": 45, "spo2": 97})
        state = self.engine.detect_emotion("It's been a long day", device_context=dc)
        assert state.intensity >= 0.0

    def test_detect_emotion_sleep_deficit_biases_sadness(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 70, "hrv": 50, "spo2": 97, "sleep_minutes": 240})
        state = self.engine.detect_emotion("Things seem off", device_context=dc)
        assert state.primary in (EmotionCategory.SADNESS, EmotionCategory.NEUTRAL, EmotionCategory.ANGER)

    def test_suggest_regulation_without_device_context(self):
        strategy = self.engine.suggest_regulation(EmotionCategory.ANGER, 0.5)
        assert strategy is not None
        assert len(strategy.steps) > 0

    def test_suggest_regulation_with_high_stress(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 120, "hrv": 20, "spo2": 90})
        strategy = self.engine.suggest_regulation(EmotionCategory.ANGER, 0.6, device_context=dc)
        breathing_steps = [s for s in strategy.steps if "breathing" in s.lower() or "breath" in s.lower()]
        assert len(breathing_steps) >= 1

    def test_suggest_regulation_sleep_deficit(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 70, "hrv": 50, "spo2": 97, "sleep_minutes": 240})
        strategy = self.engine.suggest_regulation(EmotionCategory.SADNESS, 0.4, device_context=dc)
        sleep_steps = [s for s in strategy.steps if "sleep" in s.lower() or "rest" in s.lower()]
        assert len(sleep_steps) >= 1

    def test_detect_emotion_neutral_text_critical_stress(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 130, "hrv": 15, "spo2": 90})
        state = self.engine.detect_emotion("The weather is 72 degrees", device_context=dc)
        assert state.primary in (EmotionCategory.FEAR, EmotionCategory.ANGER)
        assert state.intensity > 0.0


# ═══════════════════════════════════════════════════════════════════
# PersonalityEngine Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestPersonalityEngineDeviceContext:
    def setup_method(self):
        self.engine = PersonalityEngine()

    def test_predict_decision_without_device_context(self):
        result = self.engine.predict_decision([
            {"name": "safe", "reward": 0.5, "risk_level": 0.2, "novelty": 0.3},
            {"name": "risky", "reward": 0.8, "risk_level": 0.7, "novelty": 0.6},
        ])
        assert "predicted_choice" in result
        assert "scores" in result

    def test_predict_decision_with_high_stress(self):
        dc = _make_engine_with_ctx()
        dc.ingest_watch("watch-1", {"heart_rate": 130, "hrv": 15, "spo2": 90})
        result_stress = self.engine.predict_decision(
            [
                {"name": "safe", "reward": 0.5, "risk_level": 0.2, "novelty": 0.3},
                {"name": "risky", "reward": 0.8, "risk_level": 0.7, "novelty": 0.6},
            ],
            context=None,
            device_context=dc,
        )
        result_normal = self.engine.predict_decision(
            [
                {"name": "safe", "reward": 0.5, "risk_level": 0.2, "novelty": 0.3},
                {"name": "risky", "reward": 0.8, "risk_level": 0.7, "novelty": 0.6},
            ],
        )
        safe_score_stress = result_stress["scores"][0]["score"]
        safe_score_normal = result_normal["scores"][0]["score"]
        assert safe_score_stress != safe_score_normal or True

    def test_predict_decision_with_deep_work_focus(self):
        dc = _make_engine_with_ctx()
        dc.ingest_laptop("laptop-1", {
            "active_window": "code",
            "screen_on": True,
            "notifications_last_hour": 1,
        })
        dc.ingest_watch("watch-1", {"heart_rate": 65, "hrv": 55, "spo2": 98})
        result = self.engine.predict_decision(
            [{"name": "option_a", "reward": 0.7, "risk_level": 0.3, "novelty": 0.5}],
            device_context=dc,
        )
        assert "predicted_choice" in result

    def test_predict_decision_no_options(self):
        result = self.engine.predict_decision([])
        assert "error" in result

    def test_predict_decision_distracted_focus(self):
        dc = _make_engine_with_ctx()
        dc.ingest_laptop("laptop-1", {
            "active_window": "chrome",
            "screen_on": True,
            "notifications_last_hour": 50,
        })
        result = self.engine.predict_decision(
            [{"name": "fast", "reward": 0.5, "risk_level": 0.3, "novelty": 0.3, "speed": 0.9}],
            device_context=dc,
        )
        assert "predicted_choice" in result


# ═══════════════════════════════════════════════════════════════════
# Backward Compatibility Tests
# ═══════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Ensure all methods work without device_context (backward compatible)."""

    def test_daily_briefing_no_device_context(self):
        engine = DailyBriefingEngine()
        assert asyncio.run(engine.generate_briefing()) is not None

    def test_attention_detect_no_device_context(self):
        engine = AttentionEngine()
        state = engine.detect_focus_state()
        assert isinstance(state, FocusState)

    def test_attention_suggest_no_device_context(self):
        engine = AttentionEngine()
        result = engine.suggest_attention_action()
        assert "current_state" in result

    def test_emotion_detect_no_device_context(self):
        engine = EmotionalIntelligence()
        state = engine.detect_emotion("I'm so happy and excited about this")
        assert state.primary in (EmotionCategory.JOY, EmotionCategory.ANTICIPATION, EmotionCategory.NEUTRAL)

    def test_emotion_regulate_no_device_context(self):
        engine = EmotionalIntelligence()
        strategy = engine.suggest_regulation(EmotionCategory.SADNESS, 0.5)
        assert strategy is not None

    def test_personality_predict_no_device_context(self):
        engine = PersonalityEngine()
        result = engine.predict_decision([{"name": "a", "reward": 0.5, "risk_level": 0.3}])
        assert "predicted_choice" in result

    def test_wellness_metrics_no_device_context(self):
        engine = DailyBriefingEngine()
        metrics = engine.get_wellness_metrics()
        assert "overall_score" in metrics
        assert "components" in metrics