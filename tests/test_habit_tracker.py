"""Tests for the Habit Tracker (11th shikigami)."""

from datetime import datetime, timezone, timedelta

import pytest

from shikigami.habit_tracker import (
    Habit, detect_habits, find_broken_streaks,
    format_habit, _iso_week_id, _weeks_consecutive, _compute_streak,
)


def _obs(otype, dt):
    return {"type": otype, "content": {}, "timestamp": dt.isoformat()}


def _make_weekly_obs(otype, weeks, weekday=0, hour=9):
    """Build N weekly observations, one per ISO week, all at the same
    weekday + hour. Latest one is `now - 0 weeks`."""
    now = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)  # Wed
    obs = []
    for w in range(weeks):
        # Snap to the requested weekday in week-of (now - w).
        target = now - timedelta(weeks=w)
        days_to_subtract = (target.weekday() - weekday) % 7
        d = target - timedelta(days=days_to_subtract)
        d = d.replace(hour=hour, minute=0, second=0, microsecond=0)
        obs.append(_obs(otype, d))
    return obs, now


class TestStreakCalculation:
    def test_isoweek_id(self):
        a = datetime(2026, 5, 13, tzinfo=timezone.utc)
        b = datetime(2026, 5, 13, 23, 59, tzinfo=timezone.utc)
        # Same iso week.
        assert _iso_week_id(a) == _iso_week_id(b)

    def test_consecutive(self):
        assert _weeks_consecutive((2026, 10), (2026, 11))
        assert not _weeks_consecutive((2026, 10), (2026, 12))
        assert _weeks_consecutive((2026, 10), (2026, 10))

    def test_year_wraparound(self):
        assert _weeks_consecutive((2025, 52), (2026, 1))

    def test_compute_streak_empty(self):
        assert _compute_streak([], (2026, 20)) == (0, 0)

    def test_compute_streak_full_consecutive(self):
        weeks = [(2026, 18), (2026, 19), (2026, 20)]
        longest, current = _compute_streak(weeks, (2026, 20))
        assert longest == 3
        assert current == 3

    def test_compute_streak_with_gap(self):
        weeks = [(2026, 10), (2026, 11), (2026, 13), (2026, 14)]
        longest, current = _compute_streak(weeks, (2026, 14))
        assert longest == 2  # the 10-11 OR 13-14 streak
        assert current == 2

    def test_compute_streak_broken_at_current(self):
        # Last firing was 3 weeks before "now".
        weeks = [(2026, 15), (2026, 16)]
        longest, current = _compute_streak(weeks, (2026, 20))
        assert longest == 2
        assert current == 0  # broken


class TestDetectHabits:
    def test_empty(self):
        assert detect_habits([]) == []

    def test_below_min_frequency(self):
        obs, now = _make_weekly_obs("morning_run", weeks=2)
        habits = detect_habits(obs, min_frequency=3, now=now)
        assert habits == []

    def test_strong_weekly_habit(self):
        obs, now = _make_weekly_obs("morning_run", weeks=6, weekday=0, hour=8)
        habits = detect_habits(obs, min_frequency=3, now=now)
        assert len(habits) == 1
        h = habits[0]
        assert h.activity_type == "morning_run"
        assert h.bucket.startswith("Mon ")
        assert h.frequency == 6
        assert h.streak >= 5  # most recent firings should chain
        assert h.consistency == 1.0

    def test_inconsistent_activity_filtered(self):
        # 5 observations spread across 5 different days/hours — low consistency.
        now = datetime(2026, 5, 13, tzinfo=timezone.utc)
        obs = []
        for i in range(5):
            t = now - timedelta(days=i, hours=i)
            obs.append(_obs("scattered", t))
        habits = detect_habits(obs, min_consistency=0.5, now=now)
        # All in different buckets → max consistency 1/5 = 0.2 — filtered.
        assert habits == []

    def test_picks_strongest_bucket(self):
        # 6 firings: 4 on Monday 9am, 2 on Tuesday 10am.
        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        obs = []
        for w in range(4):
            d = (now - timedelta(weeks=w))
            d = d - timedelta(days=(d.weekday() - 0) % 7)
            obs.append(_obs("standup", d.replace(hour=9, minute=0, second=0, microsecond=0)))
        for w in range(2):
            d = (now - timedelta(weeks=w))
            d = d - timedelta(days=(d.weekday() - 1) % 7)
            obs.append(_obs("standup", d.replace(hour=10, minute=0, second=0, microsecond=0)))
        habits = detect_habits(obs, min_frequency=3, now=now)
        assert habits[0].bucket.startswith("Mon ")


class TestBrokenStreaks:
    def test_no_breaks_when_active(self):
        h = Habit(
            activity_type="run", bucket="Mon 09", frequency=10,
            consistency=1.0, avg_per_week=1.0, streak=4, longest_streak=4,
            last_seen="2026-05-13T09:00:00+00:00", days_since=0, days_observed=28,
        )
        assert find_broken_streaks([h]) == []

    def test_flags_broken_long_streak(self):
        h = Habit(
            activity_type="run", bucket="Mon 09", frequency=10,
            consistency=1.0, avg_per_week=1.0, streak=0, longest_streak=8,
            last_seen="2026-04-01T09:00:00+00:00", days_since=14, days_observed=80,
        )
        assert find_broken_streaks([h]) == [h]

    def test_does_not_flag_short_history(self):
        # Habit that only had a 2-week streak doesn't deserve a break alert.
        h = Habit(
            activity_type="run", bucket="Mon 09", frequency=2,
            consistency=1.0, avg_per_week=1.0, streak=0, longest_streak=2,
            last_seen="2026-04-01T09:00:00+00:00", days_since=14, days_observed=20,
        )
        assert find_broken_streaks([h]) == []


class TestFormatting:
    def test_format(self):
        h = Habit(
            activity_type="run", bucket="Mon 09", frequency=10,
            consistency=1.0, avg_per_week=1.0, streak=4, longest_streak=4,
            last_seen="2026-05-13T09:00:00+00:00", days_since=0, days_observed=28,
        )
        s = format_habit(h)
        assert "run" in s
        assert "Mon 09" in s
        assert "streak=4w" in s

    def test_format_broken(self):
        h = Habit(
            activity_type="run", bucket="Mon 09", frequency=10,
            consistency=1.0, avg_per_week=1.0, streak=0, longest_streak=8,
            last_seen="2026-04-01T09:00:00+00:00", days_since=14, days_observed=80,
        )
        assert "BROKEN" in format_habit(h)
