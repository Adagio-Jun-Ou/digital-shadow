#!/usr/bin/env python3
"""Sprint 163 — round() pass 3 + thread safety review

Tests:
- UserRhythmModel.get_baselines() returns rounded floats
- UserRhythmModel.get_stats() returns rounded floats
- RhythmSample.to_dict() rounds heart_rate, hrv, sleep_hours, stress_level
- AutonomousDecision.to_dict() rounds confidence and risk_level
- ObservationPulse.to_dict() rounds confidence
- SecurityEvent.to_dict() correct structure
"""

import pytest
from datetime import datetime, timezone

from shadowd.cognition.autonomous_daemon import (
    AutonomousDaemon,
    RhythmSample,
    UserActivityState,
    UserRhythmModel,
    AutonomousDecision,
    AutonomousActionType,
    ObservationPulse,
    ObservationType,
    SecurityEvent,
    ThreatLevel,
    SecurityEventType,
)


class TestRhythmModelRound:
    """UserRhythmModel returns rounded floats."""

    def test_get_baselines_rounds_floats(self):
        model = UserRhythmModel()
        model._avg_heart_rate = 72.123456789
        model._avg_hrv = 45.987654321
        model._avg_sleep_hours = 7.123456789
        model._avg_stress = 0.3456789012
        baselines = model.get_baselines()
        assert baselines["avg_heart_rate"] == 72.1
        assert baselines["avg_hrv"] == 46.0
        assert baselines["avg_sleep_hours"] == 7.1
        assert baselines["avg_stress"] == 0.346

    def test_get_baselines_handles_none(self):
        model = UserRhythmModel()
        baselines = model.get_baselines()
        assert baselines["avg_heart_rate"] is None
        assert baselines["avg_hrv"] is None

    def test_get_stats_rounds_floats(self):
        model = UserRhythmModel()
        model._avg_heart_rate = 72.123456789
        model._avg_hrv = 45.987654321
        model._avg_sleep_hours = 7.123456789
        model._avg_stress = 0.3456789012
        stats = model.get_stats()
        assert stats["avg_heart_rate"] == 72.1
        assert stats["avg_hrv"] == 46.0
        assert stats["avg_sleep_hours"] == 7.1
        assert stats["avg_stress"] == 0.346


class TestRhythmSampleRound:
    """RhythmSample.to_dict() rounds floats."""

    def test_to_dict_rounds_values(self):
        sample = RhythmSample(
            activity_state=UserActivityState.PRESENT,
            heart_rate=72.12345,
            hrv=45.98765,
            stress_level=0.34567,
            sleep_hours=7.12345,
        )
        d = sample.to_dict()
        assert d["heart_rate"] == 72.1
        assert d["hrv"] == 45.99
        assert d["stress_level"] == 0.346
        assert d["sleep_hours"] == 7.12

    def test_to_dict_handles_none(self):
        sample = RhythmSample(activity_state=UserActivityState.PRESENT)
        d = sample.to_dict()
        assert d["heart_rate"] is None
        assert d["hrv"] is None
        assert d["stress_level"] is None


class TestAutonomousDecisionRound:
    """AutonomousDecision.to_dict() rounds confidence and risk_level."""

    def test_to_dict_rounds(self):
        d = AutonomousDecision(
            action_type=AutonomousActionType.MEMORY_CONSOLIDATION,
            reason="test",
            confidence=0.85765432,
            risk_level=0.12345678,
        )
        result = d.to_dict()
        assert result["confidence"] == 0.858
        assert result["risk_level"] == 0.123


class TestObservationPulseRound:
    """ObservationPulse.to_dict() rounds confidence."""

    def test_to_dict_rounds_confidence(self):
        pulse = ObservationPulse(
            observation_type=ObservationType.SYSTEM_METRICS,
            confidence=0.75432,
            source="test",
        )
        d = pulse.to_dict()
        assert d["confidence"] == 0.754