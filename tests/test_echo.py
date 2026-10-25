"""Tests for the Echo counterfactual ability."""

import asyncio
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

import pytest

from shikigami import get_registry, ShikigamiType
from shikigami.echo import compute_echo, _looks_like_decision, _overlap, _bag_of_words


# compute_echo ages observations against the real clock, so timestamps must be
# relative to "now" rather than a frozen date or the lookback window slowly
# expires every test that uses it.
NOW = datetime.now(timezone.utc)


def _msg(text: str, days_ago: float) -> dict:
    return {
        "type": "message",
        "content": {"body": text},
        "timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
    }


def _fake_tracker(decisions):
    return SimpleNamespace(decisions=decisions)


class TestLooksLikeDecision:
    def test_accepted(self):
        assert _looks_like_decision("accepted the offer")

    def test_rejected(self):
        assert _looks_like_decision("rejected the proposal")

    def test_git_commit(self):
        assert _looks_like_decision("git commit -m 'wip'")

    def test_chose(self):
        assert _looks_like_decision("chose option B")

    def test_negative_case(self):
        assert not _looks_like_decision("just a regular note")

    def test_empty(self):
        assert not _looks_like_decision("")


class TestOverlap:
    def test_identical_sets(self):
        a = b = {"one", "two", "three"}
        assert _overlap(a, b) == 1.0

    def test_disjoint(self):
        assert _overlap({"a"}, {"b"}) == 0.0

    def test_empty(self):
        assert _overlap(set(), {"a"}) == 0.0


class TestComputeEcho:
    def test_empty_history_no_counterfactuals(self):
        r = compute_echo([], decision_tracker=None)
        assert r.decisions_seen == 0
        assert r.counterfactuals == []
        assert r.top_regret is None

    def test_observation_decisions_seen(self):
        obs = [_msg("accepted the contract immediately", d) for d in (1, 2, 3)]
        r = compute_echo(obs, decision_tracker=None)
        # Each "accepted ..." counts as one decision observation.
        assert r.decisions_seen >= 3

    def test_counterfactual_generated_with_history(self):
        # Past: "accepted similar offer" with low satisfaction (2/10)
        # Past: "declined similar offer" with high satisfaction (8/10)
        # Recent: "accepted the contract" — alternative should show as relief opportunity
        past_iso = (NOW - timedelta(days=45)).isoformat()
        past_iso2 = (NOW - timedelta(days=60)).isoformat()
        past_iso3 = (NOW - timedelta(days=50)).isoformat()
        recent_obs = [_msg("accepted the contract immediately", 2)]
        tracker = _fake_tracker([
            {"timestamp": past_iso, "situation": "offer for contract work",
             "choice": "accepted", "satisfaction": 2},
            {"timestamp": past_iso2, "situation": "another contract offer",
             "choice": "accepted", "satisfaction": 3},
            {"timestamp": past_iso3, "situation": "similar contract offer",
             "choice": "declined", "satisfaction": 8},
        ])
        r = compute_echo(recent_obs, decision_tracker=tracker)
        assert len(r.counterfactuals) >= 1
        # Best alternative should have positive delta (better than chosen).
        assert r.top_regret is not None
        assert r.top_regret.delta > 0

    def test_only_looks_backward(self):
        # Future-dated decisions in history should not be alternatives.
        future_iso = (NOW + timedelta(days=10)).isoformat()
        past_iso = (NOW - timedelta(days=2)).isoformat()
        tracker = _fake_tracker([
            {"timestamp": future_iso, "situation": "x", "choice": "B", "satisfaction": 9},
            {"timestamp": past_iso, "situation": "x decision", "choice": "A", "satisfaction": 5},
        ])
        recent = [_msg("accepted decision x", 1)]
        r = compute_echo(recent, decision_tracker=tracker)
        # The future B should be filtered out.
        for c in r.counterfactuals:
            assert c.alt_choice != "B"

    def test_requires_satisfaction_data(self):
        # History without satisfaction shouldn't generate counterfactuals.
        tracker = _fake_tracker([
            {"timestamp": (NOW - timedelta(days=5)).isoformat(),
             "situation": "contract offer", "choice": "accepted", "satisfaction": None},
            {"timestamp": (NOW - timedelta(days=6)).isoformat(),
             "situation": "contract offer", "choice": "declined", "satisfaction": None},
        ])
        r = compute_echo([_msg("accepted contract", 1)], decision_tracker=tracker)
        assert r.counterfactuals == []

    def test_regret_pressure_ordering(self):
        past = (NOW - timedelta(days=20)).isoformat()
        tracker = _fake_tracker([
            {"timestamp": past, "situation": "big decision", "choice": "A", "satisfaction": 3},
            {"timestamp": past, "situation": "big decision", "choice": "A", "satisfaction": 4},
            {"timestamp": past, "situation": "big decision", "choice": "B", "satisfaction": 9},
            {"timestamp": past, "situation": "big decision", "choice": "B", "satisfaction": 8},
            {"timestamp": past, "situation": "big decision", "choice": "C", "satisfaction": 6},
        ])
        recent = [_msg("accepted big decision", 1)]
        r = compute_echo(recent, decision_tracker=tracker)
        # Sorted by regret_pressure descending.
        if len(r.counterfactuals) >= 2:
            assert r.counterfactuals[0].regret_pressure >= r.counterfactuals[1].regret_pressure


class TestEchoViaRegistry:
    def test_executes_via_registry(self):
        r = get_registry()
        res = asyncio.run(r.execute(ShikigamiType.ECHO, "what if", {"observations": []}))
        assert res.success
        assert "Echo" in res.output
        assert len(res.output) >= 80

    def test_data_is_json_serializable(self):
        import json
        r = get_registry()
        res = asyncio.run(r.execute(ShikigamiType.ECHO, "what if", {"observations": []}))
        json.dumps(res.data)
