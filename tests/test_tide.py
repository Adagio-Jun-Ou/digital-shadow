"""Tests for the Tide sentiment-trend ability."""

import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from shikigami import get_registry, ShikigamiType
from shikigami.tide import compute_tide, _iso_week_key, _linear_slope


NOW = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)


def _note(text: str, days_ago: float) -> dict:
    return {
        "type": "note",
        "content": {"text": text},
        "timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
    }


class TestLinearSlope:
    def test_slope_positive_for_rising(self):
        s = _linear_slope([0.0, 0.1, 0.2, 0.3, 0.4])
        assert s > 0

    def test_slope_negative_for_falling(self):
        s = _linear_slope([0.4, 0.3, 0.2, 0.1, 0.0])
        assert s < 0

    def test_slope_zero_for_flat(self):
        s = _linear_slope([0.5, 0.5, 0.5, 0.5])
        assert s == 0.0

    def test_single_value(self):
        assert _linear_slope([0.5]) == 0.0


class TestIsoWeekKey:
    def test_consistent_for_same_week(self):
        dt = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)  # Mon
        dt2 = datetime(2026, 5, 17, 23, 0, tzinfo=timezone.utc)  # Sun same wk
        assert _iso_week_key(dt)[0] == _iso_week_key(dt2)[0]


class TestComputeTide:
    def test_empty_returns_neutral(self):
        r = compute_tide([])
        assert r.total_samples == 0
        assert r.overall_label == "neutral"
        assert r.slope == "stable"
        assert r.weeks == []

    def test_positive_corpus_reads_positive(self):
        obs = [_note("amazing wonderful great love it", d) for d in range(14)]
        r = compute_tide(obs)
        assert r.overall_avg > 0.1
        assert r.overall_label in ("positive", "very_positive")

    def test_negative_corpus_reads_negative(self):
        obs = [_note("terrible awful hate it broken disaster", d) for d in range(14)]
        r = compute_tide(obs)
        assert r.overall_avg < -0.1
        assert r.overall_label in ("negative", "very_negative")

    def test_rising_trend_detected(self):
        # Earlier: negative; later: positive.
        obs = (
            [_note("terrible awful broken", d) for d in range(20, 14, -1)] +
            [_note("amazing wonderful great", d) for d in range(7, 0, -1)]
        )
        r = compute_tide(obs)
        assert r.slope == "rising"
        assert r.slope_value > 0

    def test_falling_trend_detected(self):
        obs = (
            [_note("amazing wonderful great", d) for d in range(20, 14, -1)] +
            [_note("terrible awful broken", d) for d in range(7, 0, -1)]
        )
        r = compute_tide(obs)
        assert r.slope == "falling"
        assert r.slope_value < 0

    def test_best_and_worst_weeks_distinct(self):
        obs = (
            [_note("amazing", d) for d in range(20, 14, -1)] +
            [_note("terrible", d) for d in range(7, 0, -1)]
        )
        r = compute_tide(obs)
        assert r.best_week is not None
        assert r.worst_week is not None
        assert r.best_week != r.worst_week

    def test_weekly_buckets_populated(self):
        obs = [_note("good", d) for d in range(14)]
        r = compute_tide(obs)
        for w in r.weeks:
            assert w.sample_count > 0
            assert -1.0 <= w.avg_sentiment <= 1.0
            assert w.iso_week.startswith(("2026-W", "2025-W"))

    def test_top_positive_words_captured(self):
        obs = [_note("amazing wonderful great", d) for d in range(7)]
        r = compute_tide(obs)
        flat = []
        for w in r.weeks:
            flat.extend(w.top_positive)
        assert any(word in flat for word in ("amazing", "wonderful", "great"))

    def test_ignores_non_text_observations(self):
        obs = [
            {"type": "wifi", "content": {"ssid": "Home"}, "timestamp": NOW.isoformat()},
            {"type": "file_change", "content": {"path": "/x"}, "timestamp": NOW.isoformat()},
            _note("amazing", 1),
        ]
        r = compute_tide(obs)
        assert r.total_samples == 1

    def test_ignores_empty_text(self):
        obs = [
            {"type": "note", "content": {"text": ""}, "timestamp": NOW.isoformat()},
            _note("good", 1),
        ]
        r = compute_tide(obs)
        assert r.total_samples == 1

    def test_streak_detection(self):
        # All positive samples in sequence.
        obs = [_note("amazing wonderful", d) for d in range(10, 0, -1)]
        r = compute_tide(obs)
        assert r.longest_positive_streak >= 5


class TestTideViaRegistry:
    def test_executes_via_registry(self):
        r = get_registry()
        obs = [_note("happy great wonderful", d) for d in range(10)]
        res = asyncio.run(r.execute(ShikigamiType.TIDE, "overview", {"observations": obs}))
        assert res.success
        assert "Tide" in res.output
        # Should mention sentiment direction.
        assert any(label in res.output for label in
                   ("positive", "negative", "neutral", "very_positive", "very_negative"))

    def test_data_is_json_serializable(self):
        import json
        r = get_registry()
        obs = [_note("good and motivated", d) for d in range(14)]
        res = asyncio.run(r.execute(ShikigamiType.TIDE, "overview", {"observations": obs}))
        json.dumps(res.data)

    def test_empty_observations_still_outputs(self):
        r = get_registry()
        res = asyncio.run(r.execute(ShikigamiType.TIDE, "overview", {"observations": []}))
        assert res.success
        assert len(res.output) >= 80
