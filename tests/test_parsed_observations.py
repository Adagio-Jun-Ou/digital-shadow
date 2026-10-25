"""Tests for the observation pre-parse cache."""

from datetime import datetime, timezone

import pytest

from shikigami.parsed_observations import ParsedObservations, _parse_ts


class TestParseTs:
    def test_iso_z_suffix(self):
        t = _parse_ts("2026-05-15T12:00:00Z")
        assert t is not None
        assert t.tzinfo is timezone.utc

    def test_iso_with_offset(self):
        t = _parse_ts("2026-05-15T12:00:00+00:00")
        assert t is not None

    def test_naive_becomes_utc(self):
        t = _parse_ts("2026-05-15T12:00:00")
        assert t is not None
        assert t.tzinfo is timezone.utc

    def test_passthrough_datetime(self):
        original = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
        assert _parse_ts(original) == original

    def test_naive_datetime_gets_tz(self):
        naive = datetime(2026, 5, 15, 12, 0)
        t = _parse_ts(naive)
        assert t is not None
        assert t.tzinfo is timezone.utc

    def test_invalid_returns_none(self):
        assert _parse_ts("not-a-date") is None

    def test_none_returns_none(self):
        assert _parse_ts(None) is None


class TestParsedObservations:
    def test_acts_like_list(self):
        raw = [
            {"type": "a", "timestamp": "2026-05-01T10:00:00Z"},
            {"type": "b", "timestamp": "2026-05-02T10:00:00Z"},
        ]
        p = ParsedObservations(raw)
        assert len(p) == 2
        assert p[0]["type"] == "a"
        assert list(p) == raw

    def test_filters_non_dict(self):
        p = ParsedObservations([{"type": "a"}, "garbage", None, 42, {"type": "b"}])
        assert len(p) == 2

    def test_timestamp_caches(self):
        obs = {"type": "x", "timestamp": "2026-05-01T10:00:00Z"}
        p = ParsedObservations([obs])
        a = p.timestamp_for(obs)
        b = p.timestamp_for(obs)
        # Cache returns the same object reference.
        assert a is b

    def test_timestamp_for_missing(self):
        obs = {"type": "x"}
        p = ParsedObservations([obs])
        assert p.timestamp_for(obs) is None

    def test_timestamp_for_invalid(self):
        obs = {"type": "x", "timestamp": "garbage"}
        p = ParsedObservations([obs])
        assert p.timestamp_for(obs) is None

    def test_precompute_runs(self):
        raw = [{"type": "x", "timestamp": "2026-05-01T10:00:00Z"} for _ in range(50)]
        p = ParsedObservations(raw)
        p.precompute_timestamps()
        # Cache should be populated.
        assert len(p._ts_cache) == 50

    def test_by_type(self):
        raw = [
            {"type": "note", "id": 1},
            {"type": "wifi", "id": 2},
            {"type": "note", "id": 3},
        ]
        p = ParsedObservations(raw)
        notes = p.by_type("note")
        assert len(notes) == 2
        assert {o["id"] for o in notes} == {1, 3}
        assert p.by_type("nonexistent") == []

    def test_types_property(self):
        p = ParsedObservations([{"type": "a"}, {"type": "b"}, {"type": "a"}])
        assert p.types == {"a", "b"}

    def test_compatible_with_isinstance_list_check(self):
        # ParsedObservations is NOT a list; abilities use isinstance(obs, list)
        # in some places, so make sure those paths still work via iteration.
        p = ParsedObservations([{"type": "a"}])
        # We can iterate it.
        items = [o for o in p]
        assert items[0]["type"] == "a"

    def test_raw_attribute(self):
        raw = [{"type": "a"}, {"type": "b"}]
        p = ParsedObservations(raw)
        assert p.raw == raw

    def test_iteration_order_preserved(self):
        raw = [{"type": f"t{i}", "i": i} for i in range(20)]
        p = ParsedObservations(raw)
        assert [o["i"] for o in p] == list(range(20))
