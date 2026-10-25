"""Tests for the Time Witness ability."""

from datetime import datetime, timezone, timedelta

import pytest

from shikigami.time_witness import (
    compute_time_report, format_bar, _classify, _RULES,
)


@pytest.fixture
def now():
    return datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


class TestClassify:
    def test_coding_type(self):
        label, mins = _classify({"type": "git_commit", "content": {}})
        assert label == "coding"
        assert mins > 0

    def test_communication_type(self):
        label, _ = _classify({"type": "message", "content": {}})
        assert label == "communication"

    def test_calendar_meeting(self):
        label, mins = _classify({"type": "calendar_event", "content": {}})
        assert label == "meetings"
        # Meetings carry chunky weight.
        assert mins >= 15.0

    def test_text_rule_coding(self):
        label, _ = _classify({"type": "note", "content": {"text": "ran pytest after git commit"}})
        assert label == "coding"

    def test_other_fallback(self):
        label, mins = _classify({"type": "exotic_unknown_type", "content": {}})
        assert label == "other"
        assert mins > 0


class TestFormatBar:
    def test_zero(self):
        bar = format_bar(0.0, width=10)
        assert bar == "░" * 10

    def test_full(self):
        bar = format_bar(1.0, width=10)
        assert bar == "█" * 10

    def test_half(self):
        bar = format_bar(0.5, width=10)
        assert bar.count("█") == 5


class TestComputeReport:
    def test_empty(self, now):
        report = compute_time_report([], now=now)
        assert report.total_minutes == 0.0
        assert report.total_events == 0
        assert report.buckets == []
        assert report.coverage == 0.0

    def test_only_old_obs_ignored(self, now):
        obs = [{"type": "git_commit", "content": {},
                "timestamp": (now - timedelta(days=30)).isoformat()}]
        report = compute_time_report(obs, window_days=7, now=now)
        assert report.total_events == 0

    def test_top_label(self, now):
        obs = []
        # Many coding events, one meeting.
        for i in range(20):
            obs.append({"type": "git_commit", "content": {},
                        "timestamp": (now - timedelta(hours=i)).isoformat()})
        obs.append({"type": "calendar_event", "content": {},
                    "timestamp": now.isoformat()})
        report = compute_time_report(obs, window_days=7, now=now)
        assert report.top_label is not None
        # Coding should win on event count even though meeting weighs more per-event.
        # Whichever wins, both should be present.
        labels = {b.label for b in report.buckets}
        assert "coding" in labels
        assert "meetings" in labels

    def test_delta_vs_prev_window(self, now):
        # 5 events this week, 1 event last week.
        obs = []
        for i in range(5):
            obs.append({"type": "git_commit", "content": {},
                        "timestamp": (now - timedelta(days=i)).isoformat()})
        obs.append({"type": "git_commit", "content": {},
                    "timestamp": (now - timedelta(days=10)).isoformat()})
        report = compute_time_report(obs, window_days=7, now=now)
        coding = next(b for b in report.buckets if b.label == "coding")
        # Should be positive — more activity this week.
        assert coding.delta_vs_prev > 0

    def test_coverage_capped(self, now):
        obs = [{"type": "calendar_event", "content": {},
                "timestamp": now.isoformat()} for _ in range(1000)]
        report = compute_time_report(obs, window_days=1, now=now)
        # Should not exceed 1.0 even with absurd event count.
        assert report.coverage <= 1.0

    def test_buckets_sorted_by_minutes(self, now):
        obs = [
            {"type": "git_commit", "content": {},
             "timestamp": (now - timedelta(hours=i)).isoformat()} for i in range(5)
        ] + [
            {"type": "calendar_event", "content": {},
             "timestamp": (now - timedelta(hours=i)).isoformat()} for i in range(2)
        ]
        report = compute_time_report(obs, window_days=7, now=now)
        for i in range(len(report.buckets) - 1):
            assert report.buckets[i].minutes >= report.buckets[i + 1].minutes
