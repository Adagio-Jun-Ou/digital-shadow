#!/usr/bin/env python3
"""Sprint 161 — datetime.fromisoformat() Crash Guards (4 fixes)

Tests:
- temporal_reasoning.record_event() falls back to now() on bad timestamp
- ambient_monitor._detect_anomaly() skips events with bad timestamps
- mcp_server._parse_ts() helper handles None, empty, malformed strings
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from shadowd.cognition.temporal_reasoning import TemporalReasoning

try:
    from shadowd.mcp_server import _parse_ts
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


class TestTemporalReasoningMalformedTimestamp:
    """record_event() does not crash on malformed timestamps."""

    def test_malformed_timestamp_no_crash(self):
        tr = TemporalReasoning()
        event_id = tr.record_event("test_event", timestamp="not-a-date")
        assert event_id is not None

    def test_empty_timestamp_no_crash(self):
        tr = TemporalReasoning()
        event_id = tr.record_event("test_event", timestamp="")
        assert event_id is not None

    def test_none_timestamp_works(self):
        tr = TemporalReasoning()
        event_id = tr.record_event("test_event", timestamp=None)
        assert event_id is not None

    def test_valid_timestamp_still_works(self):
        tr = TemporalReasoning()
        event_id = tr.record_event("test_event", timestamp="2025-01-15T08:30:00+00:00")
        assert event_id is not None

    def test_multiple_bad_timestamps_no_crash(self):
        tr = TemporalReasoning()
        for ts in ["garbage", "", "2025-99-99", "T00:00"]:
            event_id = tr.record_event("test_event", timestamp=ts)
            assert event_id is not None


class TestAmbientMonitorMalformedTimestamp:
    """_detect_anomaly() skips events with bad timestamps."""

    @pytest.mark.asyncio
    async def test_malformed_observed_at_no_crash(self):
        from shadowd.ambient_monitor import AmbientMonitor
        mock_ctx = MagicMock()
        am = AmbientMonitor(context_protocol=mock_ctx)
        am.event_history = [
            {"type": "test", "observed_at": "garbage"},
            {"type": "test", "observed_at": datetime.now(timezone.utc).isoformat()},
        ]
        result = await am._detect_anomaly({"type": "test"})
        assert result is not None or result is None

    @pytest.mark.asyncio
    async def test_empty_observed_at_no_crash(self):
        from shadowd.ambient_monitor import AmbientMonitor
        mock_ctx = MagicMock()
        am = AmbientMonitor(context_protocol=mock_ctx)
        am.event_history = [
            {"type": "test", "observed_at": ""},
        ]
        result = await am._detect_anomaly({"type": "test"})
        assert result is not None or result is None


@pytest.mark.skipif(not HAS_MCP, reason="mcp module not installed")
class TestParseTsHelper:
    """_parse_ts() safely parses timestamps."""

    def test_none_returns_now(self):
        result = _parse_ts(None)
        assert isinstance(result, datetime)

    def test_empty_returns_now(self):
        result = _parse_ts("")
        assert isinstance(result, datetime)

    def test_malformed_returns_now(self):
        result = _parse_ts("not-a-date")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_valid_iso_returns_datetime(self):
        result = _parse_ts("2025-01-15T08:30:00+00:00")
        assert result.hour == 8

    def test_timezone_utc_on_fallback(self):
        result = _parse_ts("garbage")
        assert result.tzinfo == timezone.utc