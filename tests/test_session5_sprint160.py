#!/usr/bin/env python3
"""Sprint 160 — datetime.fromisoformat() Crash Guards

Tests:
- UserRhythmModel.ingest() handles malformed timestamps gracefully
- SecurityGuardian.record_auth_failure() handles malformed event timestamps gracefully
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from shadowd.cognition.autonomous_daemon import (
    AutonomousDaemon,
    RhythmSample,
    UserActivityState,
    UserRhythmModel,
    SecurityEvent,
    ThreatLevel,
    SecurityGuardian,
)


class TestRhythmModelMalformedTimestamp:
    """UserRhythmModel.ingest() does not crash on bad timestamps."""

    def test_malformed_timestamp_uses_current_hour(self):
        model = UserRhythmModel()
        sample = RhythmSample(
            timestamp="not-a-date",
            activity_state=UserActivityState.PRESENT,
        )
        model.ingest(sample)
        current_hour = datetime.now(timezone.utc).hour
        assert current_hour in model._hourly_activity

    def test_empty_timestamp_uses_current_hour(self):
        model = UserRhythmModel()
        sample = RhythmSample(
            timestamp="",
            activity_state=UserActivityState.IDLE,
        )
        model.ingest(sample)
        current_hour = datetime.now(timezone.utc).hour
        assert current_hour in model._hourly_activity

    def test_none_timestamp_uses_current_hour(self):
        model = UserRhythmModel()
        sample = RhythmSample(
            timestamp=None,
            activity_state=UserActivityState.AWAY,
            heart_rate=72.0,
        )
        model.ingest(sample)
        current_hour = datetime.now(timezone.utc).hour
        assert current_hour in model._hourly_activity
        assert model._avg_heart_rate == 72.0

    def test_valid_timestamp_parses_correctly(self):
        model = UserRhythmModel()
        sample = RhythmSample(
            timestamp="2025-01-15T08:30:00+00:00",
            activity_state=UserActivityState.PRESENT,
        )
        model.ingest(sample)
        assert 8 in model._hourly_activity

    def test_malformed_timestamp_does_not_crash_ingest(self):
        model = UserRhythmModel()
        for ts in ["garbage", "", "2025-13-45", None, "T00:00"]:
            sample = RhythmSample(
                timestamp=ts,
                activity_state=UserActivityState.PRESENT,
            )
            model.ingest(sample)
        assert model._sample_count == 5


class TestSecurityGuardianMalformedTimestamp:
    """record_auth_failure() handles malformed event timestamps gracefully."""

    def test_malformed_event_timestamp_skipped_in_count(self):
        sg = SecurityGuardian()
        bad_event = SecurityEvent(
            event_type="auth_failure",
            description="test",
            source="ssh",
            threat_level=ThreatLevel.LOW,
            timestamp="not-a-date",
        )
        sg._auth_failures = [bad_event]
        result = sg.record_auth_failure("localhost")
        assert result is not None

    def test_all_malformed_timestamps_no_crash(self):
        sg = SecurityGuardian()
        for ts in ["garbage", "", "2025-99-99", None]:
            bad_event = SecurityEvent(
                event_type="auth_failure",
                description="test",
                source="ssh",
                threat_level=ThreatLevel.LOW,
                timestamp=ts,
            )
            sg._auth_failures.append(bad_event)
        result = sg.record_auth_failure("localhost")
        assert result is not None
        assert result.threat_level == ThreatLevel.LOW

    def test_mixed_valid_invalid_timestamps_count_correctly(self):
        sg = SecurityGuardian()
        for _ in range(4):
            event = SecurityEvent(
                event_type="auth_failure",
                description="test",
                source="ssh",
                threat_level=ThreatLevel.LOW,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            sg._auth_failures.append(event)
        bad_event = SecurityEvent(
            event_type="auth_failure",
            description="test",
            source="ssh",
            threat_level=ThreatLevel.LOW,
            timestamp="garbage",
        )
        sg._auth_failures.append(bad_event)
        result = sg.record_auth_failure("localhost")
        assert sg._stats["auth_failures_total"] == 1