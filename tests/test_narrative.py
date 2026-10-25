"""Tests for cross-ability narrative weaving."""

import pytest

from shikigami.narrative import weave, _signals_from, _health_score, _detect_themes


class TestSignalsFrom:
    def test_tide_signals(self):
        s = _signals_from("tide", {"slope": "falling", "overall_label": "negative",
                                    "longest_negative_streak": 7})
        assert any("falling" in x for x in s)
        assert any("negative streak" in x for x in s)

    def test_lantern_dormant(self):
        s = _signals_from("lantern", {"active": [], "dormant": ["a", "b", "c"]})
        assert any("dormant" in x for x in s)
        assert any("no active" in x for x in s)

    def test_compass_alignment(self):
        s = _signals_from("compass", {"alignment_score": 0.2, "trend": "declining"})
        assert any("0.20" in x for x in s)

    def test_mirror_rising(self):
        s = _signals_from("mirror", {"rising": ["anxiety"], "avg_self_sentiment": -0.3})
        assert any("anxiety" in x for x in s)
        assert any("harsh self-talk" in x for x in s)

    def test_echo_top_regret(self):
        s = _signals_from("echo", {
            "counterfactual_count": 1,
            "top_regret": {"alt_choice": "declined the offer", "delta": 3.5},
        })
        assert any("regret" in x for x in s)

    def test_unknown_ability_no_signals(self):
        assert _signals_from("nonexistent", {}) == []

    def test_handles_empty_data(self):
        assert _signals_from("tide", {}) == []

    def test_handles_non_dict_data(self):
        assert _signals_from("tide", None) == []
        assert _signals_from("tide", "garbage") == []


class TestHealthScore:
    def test_neutral_baseline(self):
        # All defaults: should land near 0.5.
        h = _health_score({})
        assert 0.4 <= h <= 0.6

    def test_positive_signals_raise_score(self):
        positive = {
            "tide": {"overall_avg": 0.5, "slope": "rising"},
            "mirror": {"avg_self_sentiment": 0.4},
            "compass": {"alignment_score": 0.9},
            "lantern": {"active": ["proj_a"]},
            "focus_sight": {"fragmentation_score": 0.3},
        }
        assert _health_score(positive) > 0.6

    def test_negative_signals_lower_score(self):
        negative = {
            "tide": {"overall_avg": -0.5, "slope": "falling"},
            "mirror": {"avg_self_sentiment": -0.4},
            "compass": {"alignment_score": 0.1},
            "focus_sight": {"fragmentation_score": 0.9},
        }
        assert _health_score(negative) < 0.4

    def test_clamped_to_0_1(self):
        # Extreme inputs.
        extreme = {
            "tide": {"overall_avg": -5.0, "slope": "falling"},
            "mirror": {"avg_self_sentiment": -5.0},
            "compass": {"alignment_score": -1.0},
            "focus_sight": {"fragmentation_score": 5.0},
        }
        h = _health_score(extreme)
        assert 0.0 <= h <= 1.0


class TestDetectThemes:
    def test_finds_recurring(self):
        signals = ["[tide] mood falling", "[mirror] mood declining", "[other] mood is heavy"]
        themes = _detect_themes(signals)
        assert "mood" in themes

    def test_short_tokens_ignored(self):
        signals = ["[a] is to be", "[b] is to be"]
        # "is", "to", "be" should be ignored (< 4 chars).
        themes = _detect_themes(signals)
        # Nothing >= 4 chars repeats meaningfully.
        for t in themes:
            assert len(t) >= 4


class TestWeave:
    def test_empty_output(self):
        n = weave({})
        assert "No usable signals" in n.summary
        assert n.signals == []
        assert n.recommendations == []

    def test_full_pipeline(self):
        data = {
            "tide": {"overall_avg": -0.3, "slope": "falling",
                     "overall_label": "negative", "longest_negative_streak": 6},
            "mirror": {"avg_self_sentiment": -0.25, "rising": ["anxiety"]},
            "lantern": {"active": [], "dormant": ["x", "y", "z", "w"]},
            "compass": {"alignment_score": 0.25, "trend": "declining"},
        }
        n = weave(data)
        assert n.signals  # non-empty
        # Two recommendations expected: reflective break + audit dormant.
        titles = {r.title for r in n.recommendations}
        assert "Take a reflective break" in titles
        assert "Audit dormant projects" in titles
        # Recommendations sorted by priority ascending.
        prios = [r.priority for r in n.recommendations]
        assert prios == sorted(prios)
        # Health gauge should be on the low side given the signals.
        assert n.health_score < 0.5

    def test_focus_energy_rec(self):
        n = weave({
            "focus_sight": {"fragmentation_score": 0.7},
            "energy_flow": {"peak_hour": 9, "trough_hour": 14},
        })
        assert any("deep-work block" in r.title for r in n.recommendations)

    def test_burnout_signal_fires(self):
        """Burnout rec requires ≥3 of: falling mood, anxiety/doubt rising,
        low alignment, mostly-dormant projects, harsh self-talk."""
        n = weave({
            "tide": {"slope": "falling", "overall_avg": -0.3, "overall_label": "negative"},
            "mirror": {"rising": ["anxiety", "doubt"], "avg_self_sentiment": -0.3},
            "compass": {"alignment_score": 0.2},
            "lantern": {"active": [], "dormant": ["a", "b", "c", "d"]},
        })
        burnout = [r for r in n.recommendations if "Burnout" in r.title]
        assert burnout, "burnout recommendation should fire"
        # It must be priority 1 (most urgent).
        assert burnout[0].priority == 1

    def test_burnout_does_not_fire_on_mild_signals(self):
        # Only 1 signal — should NOT trigger burnout.
        n = weave({
            "tide": {"slope": "falling", "overall_avg": -0.1, "overall_label": "negative"},
        })
        assert not any("Burnout" in r.title for r in n.recommendations)

    def test_contradiction_mood_vs_alignment(self):
        """tide rising + compass declining → mood-alignment contradiction."""
        n = weave({
            "tide": {"slope": "rising", "overall_avg": 0.3, "overall_label": "positive"},
            "compass": {"alignment_score": 0.3, "trend": "declining"},
        })
        # Should pick up the dissonance.
        descs = " ".join(c.description for c in n.contradictions)
        assert "drifting" in descs.lower() or "alignment" in descs.lower()

    def test_contradiction_falling_mood_rising_alignment(self):
        n = weave({
            "tide": {"slope": "falling", "overall_avg": -0.3, "overall_label": "negative"},
            "compass": {"alignment_score": 0.8, "trend": "rising"},
        })
        assert any("costing you energy" in c.description for c in n.contradictions)

    def test_contradiction_kind_selftalk_negative_mood(self):
        n = weave({
            "tide": {"overall_label": "very_negative", "overall_avg": -0.6, "slope": "stable"},
            "mirror": {"avg_self_sentiment": 0.4},
        })
        assert any("dissonance" in c.description for c in n.contradictions)

    def test_no_contradictions_when_aligned(self):
        n = weave({
            "tide": {"slope": "rising", "overall_avg": 0.3, "overall_label": "positive"},
            "compass": {"alignment_score": 0.7, "trend": "rising"},
        })
        assert n.contradictions == []

    def test_recommendations_priority_ordered(self):
        """Sanity: when burnout fires, it should be first in the list (lowest
        priority number)."""
        n = weave({
            "tide": {"slope": "falling", "overall_avg": -0.5, "overall_label": "very_negative"},
            "mirror": {"rising": ["anxiety", "doubt"], "avg_self_sentiment": -0.4},
            "compass": {"alignment_score": 0.15},
            "lantern": {"active": [], "dormant": ["x", "y", "z", "w"]},
        })
        if n.recommendations:
            assert n.recommendations[0].priority == min(r.priority for r in n.recommendations)
