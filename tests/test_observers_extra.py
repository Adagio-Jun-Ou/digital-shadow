"""Tests for the extra observers: process, browser, calendar.

We don't exercise the polling loops here — those are best-effort and
environment-dependent — but verify the pure helpers (ICS parser, history file
discovery) and that the observer functions exit cleanly when their backends
are disabled."""

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shadowd.observers_extra import (
    _parse_ics_events,
    _sanitize,
    _find_history_files,
    observe_processes,
    observe_browser_history,
    observe_calendar,
)


class TestSanitize:
    def test_strips_control_chars(self):
        assert _sanitize("hello\x00world") == "helloworld"

    def test_caps_length(self):
        assert len(_sanitize("a" * 1000, max_len=20)) == 20

    def test_handles_unicode(self):
        # Real text with multi-byte chars should pass through.
        assert _sanitize("héllo wörld 中文 😀") == "héllo wörld 中文 😀"


class TestIcsParser:
    def test_empty(self):
        assert _parse_ics_events("") == []

    def test_no_events(self):
        text = "BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR\n"
        assert _parse_ics_events(text) == []

    def test_single_event(self):
        text = (
            "BEGIN:VEVENT\n"
            "UID:abc-123\n"
            "SUMMARY:Team Standup\n"
            "DTSTART:20260514T140000Z\n"
            "LOCATION:Zoom\n"
            "END:VEVENT\n"
        )
        events = _parse_ics_events(text)
        assert len(events) == 1
        e = events[0]
        assert e["uid"] == "abc-123"
        assert e["summary"] == "Team Standup"
        assert e["location"] == "Zoom"
        assert e["dtstart"].startswith("2026-05-14T14:00:00")

    def test_multiple_events(self):
        text = (
            "BEGIN:VEVENT\nUID:1\nSUMMARY:A\nDTSTART:20260514T140000Z\nEND:VEVENT\n"
            "BEGIN:VEVENT\nUID:2\nSUMMARY:B\nDTSTART:20260515T150000Z\nEND:VEVENT\n"
        )
        events = _parse_ics_events(text)
        assert len(events) == 2
        assert events[0]["uid"] == "1"
        assert events[1]["uid"] == "2"

    def test_handles_dtstart_with_tzid_param(self):
        # DTSTART;TZID=America/New_York:20260514T100000
        text = (
            "BEGIN:VEVENT\nUID:x\nSUMMARY:Y\n"
            "DTSTART;TZID=America/New_York:20260514T100000\n"
            "END:VEVENT\n"
        )
        events = _parse_ics_events(text)
        assert len(events) == 1
        # We pass through the value; the actual TZ resolution is best-effort.

    def test_skips_malformed_event(self):
        text = "BEGIN:VEVENT\nUID:bad\n"  # no END:VEVENT
        events = _parse_ics_events(text)
        # Should not raise; the unterminated event is dropped.
        assert events == []


class TestHistoryFileDiscovery:
    def test_returns_list(self):
        # Just verify it doesn't crash; the actual list depends on the host.
        result = _find_history_files()
        assert isinstance(result, list)
        for p in result:
            assert isinstance(p, Path)
            assert p.exists()


class TestObserverDisableFlags:
    def test_processes_observer_short_circuits_when_disabled(self):
        shadow = MagicMock()
        with patch.dict(os.environ, {"SHADOW_OBSERVE_PROCESSES": "false"}):
            # Should return immediately without ever calling shadow.learn.
            asyncio.run(observe_processes(shadow))
        shadow.learn.assert_not_called()

    def test_browser_observer_short_circuits_when_disabled(self):
        shadow = MagicMock()
        with patch.dict(os.environ, {"SHADOW_OBSERVE_BROWSER": "false"}):
            asyncio.run(observe_browser_history(shadow))
        shadow.learn.assert_not_called()

    def test_calendar_observer_short_circuits_when_disabled(self):
        shadow = MagicMock()
        with patch.dict(os.environ, {"SHADOW_OBSERVE_CALENDAR": "false"}):
            asyncio.run(observe_calendar(shadow))
        shadow.learn.assert_not_called()
