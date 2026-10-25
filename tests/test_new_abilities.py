"""Tests for the three new shikigami: Energy Flow, Focus Sight, Connection."""

from datetime import datetime, timezone, timedelta

import pytest

from shikigami.energy_curve import compute_energy_curve, render_curve_ascii, _level
from shikigami.focus_blocks import (
    FocusBlock, find_focus_blocks, current_focus_state, summarize_focus,
)
from shikigami.connection_strength import (
    Connection, score_connection, score_all_connections, trend_marker,
)


def _obs(otype, dt, content=None):
    return {"type": otype, "content": content or {}, "timestamp": dt.isoformat()}


@pytest.fixture
def now():
    return datetime(2026, 5, 13, 14, 0, 0, tzinfo=timezone.utc)


# ════════════════════════════════════════════════════════════════════
# Energy Flow
# ════════════════════════════════════════════════════════════════════

class TestEnergyLevel:
    @pytest.mark.parametrize("score,expected", [
        (0.0, "low"), (0.32, "low"),
        (0.34, "medium"), (0.65, "medium"),
        (0.67, "high"), (1.0, "high"),
    ])
    def test_thresholds(self, score, expected):
        assert _level(score) == expected


class TestEnergyCurve:
    def test_empty_observations(self, now):
        report = compute_energy_curve([], now=now)
        assert report.sample_size == 0
        assert len(report.hourly_curve) == 24

    def test_peak_at_busy_hour(self, now):
        # 10 events at hour 10, 1 event at each other hour.
        obs = []
        for _ in range(10):
            obs.append(_obs("x", now.replace(hour=10, minute=0)))
        for h in range(24):
            if h != 10:
                obs.append(_obs("x", now.replace(hour=h)))
        report = compute_energy_curve(obs, now=now)
        # Smoothing pulls the peak slightly but hour 10 should still dominate.
        assert report.peak_hour in (9, 10, 11)

    def test_forecast_length(self, now):
        report = compute_energy_curve([_obs("x", now)], now=now)
        assert len(report.forecast_6h) == 6
        for f in report.forecast_6h:
            assert 0 <= f["hour"] <= 23
            assert 0.0 <= f["score"] <= 1.0
            assert f["level"] in ("low", "medium", "high")

    def test_current_hour_present(self, now):
        report = compute_energy_curve([_obs("x", now)], now=now)
        assert report.current_hour == 14

    def test_ascii_render(self, now):
        report = compute_energy_curve([_obs("x", now)], now=now)
        chart = render_curve_ascii(report.hourly_curve, report.current_hour)
        # Marker for current hour should be present.
        assert "*" in chart


# ════════════════════════════════════════════════════════════════════
# Focus Sight
# ════════════════════════════════════════════════════════════════════

class TestFocusBlocks:
    def test_empty(self):
        assert find_focus_blocks([]) == []

    def test_single_event_not_a_block(self, now):
        assert find_focus_blocks([_obs("x", now)]) == []

    def test_detects_pure_block(self, now):
        # 6 events of "code" within 30 minutes, no gaps.
        obs = [_obs("code", now + timedelta(minutes=i * 5)) for i in range(6)]
        blocks = find_focus_blocks(obs, min_block_minutes=15)
        assert len(blocks) == 1
        b = blocks[0]
        assert b.dominant_type == "code"
        assert b.purity == 1.0
        assert b.duration_min >= 15

    def test_gap_splits_blocks(self, now):
        # First block: 4 events at start.
        obs = [_obs("code", now + timedelta(minutes=i * 5)) for i in range(4)]
        # Big gap (60 min), then second block.
        obs += [_obs("code", now + timedelta(minutes=80 + i * 5)) for i in range(4)]
        blocks = find_focus_blocks(obs, max_gap_minutes=10, min_block_minutes=10)
        assert len(blocks) == 2

    def test_low_purity_filtered(self, now):
        # Wildly mixed types — no single dominant.
        types = ["a", "b", "c", "d", "e", "f"]
        obs = [_obs(t, now + timedelta(minutes=i * 3)) for i, t in enumerate(types)]
        blocks = find_focus_blocks(obs, min_block_minutes=10, min_purity=0.5)
        assert blocks == []


class TestCurrentFocus:
    def test_idle_when_empty(self, now):
        s = current_focus_state([], now=now)
        assert s["state"] == "idle"

    def test_idle_when_only_old(self, now):
        old = _obs("code", now - timedelta(hours=2))
        s = current_focus_state([old], now=now, idle_minutes=15)
        assert s["state"] == "idle"

    def test_in_flow(self, now):
        obs = [_obs("code", now - timedelta(minutes=i)) for i in range(5)]
        s = current_focus_state(obs, now=now)
        assert s["state"] == "in_flow"
        assert s["dominant"] == "code"

    def test_scattered_when_types_switch(self, now):
        obs = [
            _obs("code", now - timedelta(minutes=4)),
            _obs("clipboard", now - timedelta(minutes=3)),
            _obs("note", now - timedelta(minutes=2)),
            _obs("file_change", now - timedelta(minutes=1)),
        ]
        s = current_focus_state(obs, now=now)
        assert s["state"] == "scattered"


class TestSummarize:
    def test_empty_summary(self):
        s = summarize_focus([])
        assert s["total_blocks"] == 0
        assert s["favorite_focus_type"] is None

    def test_summary_picks_weighted_favorite(self):
        # 1 long code block + 1 short writing block — code should dominate.
        b1 = FocusBlock("a", "b", 60.0, "code", 0.9, 12)
        b2 = FocusBlock("c", "d", 15.0, "writing", 0.9, 4)
        s = summarize_focus([b1, b2])
        assert s["total_blocks"] == 2
        assert s["favorite_focus_type"] == "code"
        assert s["longest_block_minutes"] == 60.0


# ════════════════════════════════════════════════════════════════════
# Connection
# ════════════════════════════════════════════════════════════════════

class TestConnection:
    def test_no_data(self, now):
        c = score_connection("Sarah", "friend", [], now=now)
        assert c.score < 0.3
        assert c.trend == "dormant"
        assert c.interactions_30d == 0

    def test_recent_interactions_score_high(self, now):
        obs = []
        for i in range(8):
            obs.append({
                "type": "message",
                "content": {"body": f"Sarah said hi great work"},
                "timestamp": (now - timedelta(days=i)).isoformat(),
            })
        c = score_connection("Sarah", "friend", obs, now=now)
        assert c.score > 0.5
        assert c.interactions_30d == 8

    def test_dormant_when_no_recent(self, now):
        old = {
            "type": "message",
            "content": {"body": "Sarah was here long ago"},
            "timestamp": (now - timedelta(days=120)).isoformat(),
        }
        c = score_connection("Sarah", "friend", [old], now=now)
        assert c.trend == "dormant"

    def test_word_boundary_match(self, now):
        # "Sarahara" should NOT match "Sarah".
        obs = [{
            "type": "message",
            "content": {"body": "Visited Sarahara desert"},
            "timestamp": now.isoformat(),
        }]
        c = score_connection("Sarah", "friend", obs, now=now)
        assert c.interactions_30d == 0

    def test_rising_trend(self, now):
        # 0 events in 60-30d window, 5 events in last 30d → rising.
        obs = []
        for i in range(5):
            obs.append({
                "type": "message",
                "content": {"body": "Sarah catch up"},
                "timestamp": (now - timedelta(days=i)).isoformat(),
            })
        c = score_connection("Sarah", "friend", obs, now=now)
        assert c.trend == "rising"

    def test_score_all_sorts_strongest_first(self, now):
        people = {
            "Sarah": {"relation": "friend"},
            "Bob": {"relation": "coworker"},
        }
        obs = []
        for i in range(5):
            obs.append({
                "type": "message",
                "content": {"body": "Sarah lunch"},
                "timestamp": (now - timedelta(days=i)).isoformat(),
            })
        # No interactions with Bob.
        scores = score_all_connections(people, obs, now=now)
        assert scores[0].name == "Sarah"
        assert scores[1].name == "Bob"

    def test_trend_marker(self):
        assert trend_marker("rising") == "↑"
        assert trend_marker("dormant") == "⌛"
        assert trend_marker("nonsense") == "•"
