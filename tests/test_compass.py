"""Tests for the Compass goal-alignment ability."""

import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from shikigami.compass import (
    Goal, CompassStore, score_goal, trend_marker,
    _compute_alignment, _compute_trend,
)


@pytest.fixture
def now():
    return datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return CompassStore(db_path=tmp_path / "goals.json")


class TestGoalDataclass:
    def test_matches_word_boundary(self):
        g = Goal(goal_id="g1", title="Learn Rust", keywords=["rust"],
                 created_at="2026-01-01T00:00:00+00:00")
        assert g.matches("Working on rust today")
        assert g.matches("rust is great")
        assert not g.matches("I trust him")  # 'trust' shouldn't match 'rust'

    def test_matches_empty(self):
        g = Goal(goal_id="g1", title="x", keywords=[],
                 created_at="2026-01-01T00:00:00+00:00")
        assert not g.matches("anything")

    def test_matches_case_insensitive(self):
        g = Goal(goal_id="g1", title="x", keywords=["python"],
                 created_at="2026-01-01T00:00:00+00:00")
        assert g.matches("Wrote some PYTHON today")


class TestStorePersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "goals.json"
        s = CompassStore(db_path=path)
        g = s.add("Ship the feature", ["ship", "feature"], target_date="2026-06-01T00:00:00+00:00")
        assert g.goal_id.startswith("goal_")

        s2 = CompassStore(db_path=path)
        assert g.goal_id in s2.goals
        assert s2.goals[g.goal_id].title == "Ship the feature"

    def test_remove(self, store):
        g = store.add("test goal", ["test"])
        assert store.remove(g.goal_id)
        assert g.goal_id not in store.goals
        assert not store.remove("nonexistent")

    def test_status_update(self, store):
        g = store.add("test goal", ["test"])
        assert store.set_status(g.goal_id, "done").status == "done"
        assert store.set_status("missing", "done") is None

    def test_corrupt_load_fails_safe(self, tmp_path):
        path = tmp_path / "goals.json"
        path.write_text("{not valid json")
        s = CompassStore(db_path=path)
        assert s.goals == {}


class TestComputeAlignment:
    def test_high_when_above_baseline(self):
        # mentions_30d=12 vs baseline=2 → score should clamp near max.
        assert _compute_alignment(12, 2.0, None) >= 0.9

    def test_low_when_below_baseline(self):
        assert _compute_alignment(0, 5.0, None) == 0.0

    def test_urgency_boost_near_target(self):
        # Same frequency but urgent target — boost up to max.
        baseline = _compute_alignment(6, 2.0, None)
        urgent = _compute_alignment(6, 2.0, days_to_target=5)
        assert urgent >= baseline


class TestComputeTrend:
    def test_dormant_when_zero(self):
        assert _compute_trend(0, 0, 30) == "dormant"

    def test_rising(self):
        # Expected in last 7 ~ 30 * 7/30 = 7; got 15 → rising.
        assert _compute_trend(15, 30, 30) == "rising"

    def test_declining(self):
        # Expected 7, got 2 → declining.
        assert _compute_trend(2, 30, 30) == "declining"

    def test_stable(self):
        assert _compute_trend(7, 30, 30) == "stable"


class TestScoreGoal:
    def test_no_observations(self, now):
        g = Goal(goal_id="g1", title="x", keywords=["python"],
                 created_at=(now - timedelta(days=5)).isoformat())
        p = score_goal(g, [], now=now)
        assert p.mentions_total == 0
        assert p.alignment == 0.0
        assert p.trend == "dormant"

    def test_mentions_after_goal_count(self, now):
        g = Goal(goal_id="g1", title="x", keywords=["python"],
                 created_at=(now - timedelta(days=14)).isoformat())
        obs = []
        for i in range(5):
            obs.append({
                "type": "note", "content": {"text": "python code"},
                "timestamp": (now - timedelta(days=i)).isoformat(),
            })
        # Before the goal — shouldn't count toward "mentions_total".
        obs.append({
            "type": "note", "content": {"text": "python pre-goal"},
            "timestamp": (now - timedelta(days=20)).isoformat(),
        })
        p = score_goal(g, obs, now=now)
        assert p.mentions_total == 5
        assert p.mentions_pre_goal == 1
        assert p.alignment > 0.0
        assert p.trend in ("rising", "stable")

    def test_target_date_overdue(self, now):
        g = Goal(goal_id="g1", title="x", keywords=["x"],
                 created_at=(now - timedelta(days=30)).isoformat(),
                 target_date=(now - timedelta(days=3)).isoformat())
        p = score_goal(g, [], now=now)
        assert p.days_to_target == -3


class TestTrendMarker:
    @pytest.mark.parametrize("trend,marker", [
        ("rising", "↑"), ("declining", "↓"),
        ("stable", "→"), ("dormant", "⌛"),
        ("nonsense", "•"),
    ])
    def test_markers(self, trend, marker):
        assert trend_marker(trend) == marker
