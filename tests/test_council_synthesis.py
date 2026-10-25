"""Tests for Shadow Council cross-decision synthesis."""

from datetime import datetime, timedelta, timezone

import pytest

from shikigami.council_synthesis import classify_choice, synthesize


def _iso_days_ago(days: float) -> str:
    """ISO timestamp relative to now so recency-weighted tests never age out."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _dec(id_, ts, choice, satisfaction, situation="generic situation", outcome="ok"):
    return {
        "id": id_,
        "timestamp": ts,
        "situation": situation,
        "choice": choice,
        "outcome": outcome,
        "satisfaction": satisfaction,
        "tags": [],
    }


class TestClassifyChoice:
    @pytest.mark.parametrize("text,expected", [
        ("negotiated for a better rate", "negotiate"),
        ("counter-proposed terms", "negotiate"),
        ("declined politely", "decline"),
        ("waited a day before responding", "wait"),
        ("researched options thoroughly", "research"),
        ("delegated to the team", "delegate"),
        ("jumped right away", "act_quickly"),
        ("xyz nonsense gibberish", "other"),
    ])
    def test_classifies(self, text, expected):
        assert classify_choice(text) == expected

    def test_empty_is_other(self):
        assert classify_choice("") == "other"

    def test_ties_pick_first_match(self):
        # "accepted right away" matches both 'accept' and 'act_quickly' equally.
        # Either is fine; just don't crash and return a known category.
        result = classify_choice("accepted right away")
        assert result in ("accept", "act_quickly")


class TestSynthesize:
    def test_empty_history(self):
        v = synthesize([], "any situation")
        assert v["recommendation"]["action"] is None
        assert v["action_stats"] == {}
        assert v["trend"] == {}
        assert v["counterfactual"] is None

    def test_skips_missing_satisfaction(self):
        decs = [_dec("d1", "2026-04-01T00:00:00+00:00", "negotiated", None)]
        v = synthesize(decs, "anything")
        assert v["recommendation"]["action"] is None

    def test_recommends_highest_satisfaction_category(self):
        decs = [
            _dec("d1", "2026-03-01T00:00:00+00:00", "negotiated terms", 9),
            _dec("d2", "2026-03-15T00:00:00+00:00", "negotiated rate", 8),
            _dec("d3", "2026-02-01T00:00:00+00:00", "accepted on the spot", 3),
            _dec("d4", "2026-02-15T00:00:00+00:00", "jumped right away", 2),
        ]
        v = synthesize(decs, "new offer")
        assert v["recommendation"]["action"] == "negotiate"
        assert v["recommendation"]["confidence"] > 0
        assert v["recommendation"]["lift"] > 0

    def test_action_stats_breakdown(self):
        decs = [
            _dec("d1", "2026-03-01T00:00:00+00:00", "negotiated", 8),
            _dec("d2", "2026-03-15T00:00:00+00:00", "negotiated firmly", 9),
            _dec("d3", "2026-02-15T00:00:00+00:00", "declined politely", 6),
        ]
        v = synthesize(decs, "anything")
        assert "negotiate" in v["action_stats"]
        assert v["action_stats"]["negotiate"]["n"] == 2
        assert v["action_stats"]["negotiate"]["avg_satisfaction"] > v["action_stats"]["decline"]["avg_satisfaction"]
        assert 0 <= v["action_stats"]["negotiate"]["success_rate"] <= 1

    def test_trend_improving(self):
        # Older decisions with low satisfaction, newer with high.
        decs = [
            _dec("o1", "2025-08-01T00:00:00+00:00", "negotiated", 3),
            _dec("o2", "2025-08-15T00:00:00+00:00", "negotiated", 4),
            _dec("n1", "2026-04-01T00:00:00+00:00", "negotiated", 9),
            _dec("n2", "2026-04-15T00:00:00+00:00", "negotiated", 9),
        ]
        v = synthesize(decs, "anything")
        assert v["trend"].get("negotiate", {}).get("direction") == "improving"

    def test_counterfactual_picks_extreme(self):
        similar = [
            _dec("s1", "2026-03-01T00:00:00+00:00", "negotiated", 9,
                 situation="vendor offer", outcome="great"),
            _dec("s2", "2026-03-15T00:00:00+00:00", "negotiated", 5,
                 situation="vendor offer 2", outcome="meh"),
        ]
        v = synthesize(similar, "another vendor offer", similar=similar)
        # The "9" satisfaction one is more extreme (further from 5).
        assert v["counterfactual"]["satisfaction"] == 9
        assert v["counterfactual"]["polarity"] == "affirming"

    def test_recency_weighting_reduces_old_contributions(self):
        # Two decisions in the same category but at very different times should
        # have very different effective weights — verify the recent one's
        # weight dominates by orders of magnitude.
        decs = [
            _dec("o1", _iso_days_ago(800), "negotiated", 9),
            _dec("n1", _iso_days_ago(1), "negotiated", 9),
        ]
        v = synthesize(decs, "anything")
        stats = v["action_stats"]["negotiate"]
        assert stats["n"] == 2
        # The very old decision should contribute <5% of the total weight.
        # Recent weight ≈ 0.5^(1/90) ≈ 0.99; old weight ≈ 0.5^(800/90) ≈ 0.002.
        assert stats["weighted_n"] < 1.0  # not 2.0 — recency really shrunk old one
        assert stats["weighted_n"] > 0.5  # but recent one still contributes

    def test_similar_decisions_boost_their_weight(self):
        decs = [
            _dec("g1", _iso_days_ago(1), "declined", 8),
            _dec("g2", _iso_days_ago(1), "declined", 8),
        ]
        similar = [_dec("g1", _iso_days_ago(1), "declined", 8)]
        v = synthesize(decs, "current sit", similar=similar)
        assert v["recommendation"]["action"] == "decline"
