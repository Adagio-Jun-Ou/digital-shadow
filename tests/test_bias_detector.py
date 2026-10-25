"""Tests for the cognitive bias detector."""

from datetime import datetime, timezone, timedelta

import pytest

from shikigami.bias_detector import (
    BiasSignal, detect_biases,
    _detect_confirmation_bias, _detect_recency_bias,
    _detect_sunk_cost, _detect_bandwagon,
)


@pytest.fixture
def now():
    return datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


def _obs(text, ts, otype="note"):
    return {"type": otype, "content": {"text": text}, "timestamp": ts.isoformat()}


class TestConfirmationBias:
    def test_no_data(self):
        assert _detect_confirmation_bias([]) is None

    def test_detects_one_sided_praise(self, now):
        obs = []
        for i in range(6):
            obs.append(_obs("the new framework is great python", now - timedelta(days=i)))
        sig = _detect_confirmation_bias(obs)
        assert sig is not None
        assert sig.name == "confirmation"
        assert sig.strength > 0.3

    def test_balanced_view_not_flagged(self, now):
        obs = [
            _obs("the new framework is great python", now),
            _obs("the new framework is terrible python sometimes", now - timedelta(days=1)),
            _obs("python is great today", now - timedelta(days=2)),
        ]
        sig = _detect_confirmation_bias(obs)
        # Critical mention exists, so this should NOT flag.
        assert sig is None


class TestRecencyBias:
    def test_no_data(self, now):
        assert _detect_recency_bias([], now=now) is None

    def test_recent_dominant_flag(self, now):
        # 5 emotional events in last 2 days, 1 in the prior 12.
        obs = []
        for i in range(5):
            obs.append(_obs("today was a disaster, just terrible", now - timedelta(hours=i * 6)))
        obs.append(_obs("good day overall, amazing", now - timedelta(days=8)))
        sig = _detect_recency_bias(obs, now=now)
        assert sig is not None
        assert sig.name == "recency"

    def test_emotionally_balanced_not_flagged(self, now):
        # Even spread of emotional events across the 14-day window.
        obs = []
        for d in range(14):
            obs.append(_obs("amazing day", now - timedelta(days=d)))
        sig = _detect_recency_bias(obs, now=now)
        # Recent share is roughly 2/14 = 0.14 → below 0.4 threshold.
        assert sig is None


class TestSunkCost:
    def test_no_decisions(self):
        assert _detect_sunk_cost([]) is None

    def test_repeated_low_satisfaction_flagged(self, now):
        # All three classify to the same action category ("accept").
        decisions = [
            {"choice": "accepted the offer immediately", "satisfaction": 3},
            {"choice": "accepted the terms", "satisfaction": 2},
            {"choice": "accepted without reviewing", "satisfaction": 4},
        ]
        sig = _detect_sunk_cost(decisions)
        assert sig is not None
        assert sig.name == "sunk_cost"

    def test_satisfactory_outcomes_not_flagged(self):
        decisions = [
            {"choice": "negotiated firmly", "satisfaction": 9},
            {"choice": "negotiated for more", "satisfaction": 8},
            {"choice": "counter-proposed", "satisfaction": 9},
        ]
        # Average ≥ 7 — no sunk cost.
        assert _detect_sunk_cost(decisions) is None


class TestBandwagon:
    def test_no_people_no_signal(self, now):
        obs = [_obs("just my own text here random words", now)]
        assert _detect_bandwagon(obs, tracked_people=None, now=now) is None

    def test_mirrors_tracked_person(self, now):
        people = {"Alice": {"relation": "friend"}}
        obs = []
        # Alice uses 'kubernetes' a lot.
        for i in range(4):
            obs.append({
                "type": "message",
                "content": {"text": "kubernetes scaling deployment kubernetes", "from": "alice"},
                "timestamp": (now - timedelta(days=i)).isoformat(),
            })
        # User starts using it too.
        for i in range(4):
            obs.append(_obs("starting kubernetes migration today kubernetes", now - timedelta(days=i)))
        sig = _detect_bandwagon(obs, tracked_people=people, now=now)
        assert sig is not None
        assert sig.name == "bandwagon"


class TestDetectBiases:
    def test_empty_returns_empty(self):
        assert detect_biases([]) == []

    def test_threshold_filters_weak(self, now):
        # Weak confirmation: 2 favorable mentions, below the 4+ threshold.
        obs = [_obs("great python today", now - timedelta(days=i)) for i in range(2)]
        signals = detect_biases(obs)
        # Either empty or contains no confirmation bias.
        assert all(s.name != "confirmation" for s in signals)

    def test_signals_sorted_by_strength(self, now):
        # Build observations triggering multiple biases.
        obs = []
        for i in range(8):
            obs.append(_obs("the new tool is great python", now - timedelta(days=i)))
        # And add recent emotion-heavy events for recency bias.
        for i in range(5):
            obs.append(_obs("terrible today, just a disaster", now - timedelta(hours=i * 4)))
        decisions = [
            {"choice": "accepted on the spot", "satisfaction": 2},
            {"choice": "accepted on the spot", "satisfaction": 3},
            {"choice": "accepted on the spot", "satisfaction": 4},
        ]
        signals = detect_biases(obs, decisions=decisions, now=now)
        assert len(signals) >= 2
        # Sorted strongest first.
        for i in range(len(signals) - 1):
            assert signals[i].strength >= signals[i + 1].strength
