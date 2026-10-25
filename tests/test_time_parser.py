"""Tests for the natural-language time-window parser."""

from datetime import datetime, timezone, timedelta

import pytest

from shikigami.time_parser import parse_time_window, in_window, _to_int


class TestNumberWord:
    def test_digits(self):
        assert _to_int("3") == 3
        assert _to_int("42") == 42

    def test_words(self):
        assert _to_int("a") == 1
        assert _to_int("one") == 1
        assert _to_int("ten") == 10

    def test_unknown(self):
        assert _to_int("seventeenish") is None


@pytest.fixture
def now():
    # A predictable Wednesday 14:30 UTC.
    return datetime(2026, 5, 13, 14, 30, 0, tzinfo=timezone.utc)


class TestTimeWindow:
    def test_today(self, now):
        start, end = parse_time_window("what did I do today?", now=now)
        assert start.date() == now.date()
        assert end.date() == now.date() + timedelta(days=1)

    def test_yesterday(self, now):
        start, end = parse_time_window("yesterday's events", now=now)
        assert start.date() == (now - timedelta(days=1)).date()
        assert end == start + timedelta(days=1)

    def test_this_week(self, now):
        start, end = parse_time_window("this week", now=now)
        # Monday is weekday 0.
        assert start.weekday() == 0
        assert end == now

    def test_last_week(self, now):
        start, end = parse_time_window("show me last week", now=now)
        # Should be Monday-to-Monday of the previous week.
        assert start.weekday() == 0
        assert (end - start) == timedelta(days=7)

    def test_past_hour(self, now):
        start, end = parse_time_window("the past hour", now=now)
        assert end == now
        assert end - start == timedelta(hours=1)

    def test_last_3_hours(self, now):
        start, end = parse_time_window("last 3 hours", now=now)
        assert end - start == timedelta(hours=3)

    def test_last_24_hours(self, now):
        start, end = parse_time_window("in the last 24 hours", now=now)
        assert end - start == timedelta(hours=24)

    def test_word_number(self, now):
        result = parse_time_window("past two days", now=now)
        assert result is not None
        start, end = result
        assert end - start == timedelta(days=2)

    def test_this_month(self, now):
        start, end = parse_time_window("this month so far", now=now)
        assert start.day == 1
        assert end == now

    def test_last_month(self, now):
        start, end = parse_time_window("last month", now=now)
        assert start.day == 1
        # End should be the 1st of the current month.
        assert end.day == 1
        assert (end.year, end.month) > (start.year, start.month) or \
               (end.year == start.year + 1)

    def test_no_match(self, now):
        assert parse_time_window("random nonsense", now=now) is None
        assert parse_time_window("", now=now) is None


class TestInWindow:
    def test_inside(self):
        win = (datetime(2026, 5, 1, tzinfo=timezone.utc),
               datetime(2026, 5, 31, tzinfo=timezone.utc))
        assert in_window("2026-05-15T10:00:00+00:00", win)

    def test_outside(self):
        win = (datetime(2026, 5, 1, tzinfo=timezone.utc),
               datetime(2026, 5, 31, tzinfo=timezone.utc))
        assert not in_window("2026-06-15T10:00:00+00:00", win)

    def test_malformed(self):
        win = (datetime(2026, 5, 1, tzinfo=timezone.utc),
               datetime(2026, 5, 31, tzinfo=timezone.utc))
        assert not in_window("not a timestamp", win)
        assert not in_window(None, win)

    def test_naive_timestamp_assumed_utc(self):
        # ISO without tz info should be treated as UTC.
        win = (datetime(2026, 5, 1, tzinfo=timezone.utc),
               datetime(2026, 5, 31, tzinfo=timezone.utc))
        assert in_window("2026-05-15T10:00:00", win)
