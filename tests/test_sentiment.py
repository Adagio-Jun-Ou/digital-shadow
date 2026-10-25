"""Tests for the lexicon sentiment + manipulation detector used by Social Radar."""

import pytest

from shikigami.sentiment import (
    analyze_sentiment,
    detect_manipulation,
    analyze_messages,
    _label_for,
)


class TestSentimentBasics:
    def test_neutral_empty(self):
        r = analyze_sentiment("")
        assert r.score == 0.0
        assert r.label == "neutral"

    def test_neutral_no_lexicon_hit(self):
        r = analyze_sentiment("the table has four legs")
        assert r.label == "neutral"

    def test_positive(self):
        r = analyze_sentiment("I love this project, it's wonderful")
        assert r.score > 0
        assert r.label in ("positive", "very_positive")
        assert "love" in r.positive_terms or "wonderful" in r.positive_terms

    def test_negative(self):
        r = analyze_sentiment("this is terrible and I am frustrated")
        assert r.score < 0
        assert r.label in ("negative", "very_negative")

    def test_intensifier_amplifies(self):
        weak = analyze_sentiment("I like this")
        strong = analyze_sentiment("I really really love this")
        assert strong.score > weak.score

    def test_negation_inverts(self):
        positive = analyze_sentiment("I love this")
        negated = analyze_sentiment("I do not love this")
        # Negation should drop the score (won't fully invert, but must move).
        assert negated.score < positive.score

    def test_exclamation_amplifies(self):
        plain = analyze_sentiment("terrible")
        loud = analyze_sentiment("terrible!!!!!")
        assert abs(loud.score) > abs(plain.score)


class TestSentimentLabels:
    @pytest.mark.parametrize("score,expected", [
        (0.0, "neutral"),
        (0.1, "neutral"),
        (0.2, "positive"),
        (0.6, "very_positive"),
        (-0.2, "negative"),
        (-0.6, "very_negative"),
    ])
    def test_label_thresholds(self, score, expected):
        assert _label_for(score) == expected


class TestManipulationDetection:
    def test_empty_returns_zero(self):
        r = detect_manipulation("")
        assert r.score == 0.0
        assert r.tactics == {}

    def test_no_manipulation_when_benign(self):
        r = detect_manipulation("hey, just checking in, how are you?")
        assert r.score == 0.0

    def test_urgency(self):
        r = detect_manipulation("you must do this ASAP")
        assert r.score > 0
        assert "urgency" in r.tactics or "pressure" in r.tactics

    def test_guilt(self):
        r = detect_manipulation("after all I've done for you, you owe me")
        assert r.score > 0
        assert "guilt" in r.tactics

    def test_isolation(self):
        r = detect_manipulation("don't tell anyone, just between us")
        assert r.score > 0
        assert "isolation" in r.tactics

    def test_multiple_patterns_saturate_not_explode(self):
        # Stuff every tactic into one message — score must stay within [0,1].
        msg = (
            "You must do this right now or else! After all I've done for you, "
            "you owe me. Either you do it or you let me down. Last chance, "
            "don't tell anyone. Everyone agrees this is right."
        )
        r = detect_manipulation(msg)
        assert 0.0 < r.score <= 1.0


class TestMessageAggregation:
    def test_empty_aggregate(self):
        agg = analyze_messages([])
        assert agg["count"] == 0
        assert agg["avg_sentiment"] == 0.0

    def test_drops_non_strings(self):
        agg = analyze_messages([None, 42, "real message"])
        assert agg["count"] == 1

    def test_aggregates_sentiment(self):
        agg = analyze_messages([
            "this is amazing!",
            "really wonderful",
            "I love it",
        ])
        assert agg["count"] == 3
        assert agg["avg_sentiment"] > 0
        assert agg["sentiment_label"] in ("positive", "very_positive")

    def test_detects_declining_trend(self):
        # First half positive, second half negative → declining.
        msgs = [
            "wonderful great amazing",
            "love this fantastic",
            "good okay",
            "terrible disappointing awful",
            "hate this furious",
            "broken upset frustrated",
        ]
        agg = analyze_messages(msgs)
        assert agg["sentiment_trend"] == "declining"

    def test_detects_improving_trend(self):
        msgs = [
            "terrible disappointing awful",
            "hate this furious",
            "broken upset frustrated",
            "wonderful great amazing",
            "love this fantastic",
            "good okay",
        ]
        agg = analyze_messages(msgs)
        assert agg["sentiment_trend"] == "improving"

    def test_example_surfaces_worst_offender(self):
        msgs = [
            "hi how are you",
            "all good here",
            "DON'T TELL ANYONE, you must do this RIGHT NOW or else!",
        ]
        agg = analyze_messages(msgs)
        assert agg["examples"], "should surface the high-manipulation message"
        assert "RIGHT NOW" in agg["examples"][0]["message"] or "DON'T TELL" in agg["examples"][0]["message"]
