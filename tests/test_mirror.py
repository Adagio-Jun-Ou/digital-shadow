"""Tests for the Mirror self-reflection ability."""

from datetime import datetime, timezone, timedelta

import pytest

from shikigami.mirror import (
    reflect, _matches_theme, _FIRST_PERSON, _text_of,
)


@pytest.fixture
def now():
    return datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


def _obs(otype, text, ts):
    return {"type": otype, "content": {"text": text}, "timestamp": ts.isoformat()}


class TestFirstPersonDetection:
    @pytest.mark.parametrize("text,expected", [
        ("I feel great", True),
        ("i'm anxious", True),
        ("my life is fine", True),
        ("the cat is on the mat", False),
        ("Their feelings matter", False),
        ("I've been thinking", True),
    ])
    def test_detection(self, text, expected):
        assert bool(_FIRST_PERSON.search(text)) == expected


class TestThemeMatching:
    def test_anxiety(self):
        themes = _matches_theme("I'm anxious about the presentation")
        assert "anxiety" in themes

    def test_growth(self):
        themes = _matches_theme("I'm learning to be better at this")
        assert "growth" in themes

    def test_burnout(self):
        themes = _matches_theme("I'm so exhausted, totally burnt out")
        assert "burnout" in themes

    def test_doubt(self):
        themes = _matches_theme("I feel like an imposter sometimes")
        assert "doubt" in themes

    def test_no_match(self):
        assert _matches_theme("the cat sat on the mat") == []


class TestTextExtraction:
    def test_clipboard(self):
        o = {"type": "clipboard", "content": {"text": "I am calm"}}
        assert _text_of(o) == "I am calm"

    def test_journal(self):
        o = {"type": "journal", "content": {"body": "Reflecting on the day"}}
        assert _text_of(o) == "Reflecting on the day"

    def test_unrelated_type_ignored(self):
        o = {"type": "terminal_command", "content": {"text": "I ran a command"}}
        assert _text_of(o) == ""


class TestReflect:
    def test_empty(self, now):
        report = reflect([], now=now)
        assert report.total_sentences == 0
        assert report.self_talk_sentences == 0
        assert report.theme_counts == {}

    def test_no_self_talk(self, now):
        obs = [_obs("note", "The cat is on the mat. The dog is hungry.", now)]
        report = reflect(obs, now=now)
        assert report.total_sentences == 2
        assert report.self_talk_sentences == 0

    def test_detects_self_talk_with_themes(self, now):
        obs = [
            _obs("note", "I'm feeling anxious about the launch.", now - timedelta(days=2)),
            _obs("note", "I am learning a lot today.", now - timedelta(days=1)),
            _obs("journal", "I'm grateful for my team.", now),
        ]
        report = reflect(obs, now=now)
        assert report.self_talk_sentences == 3
        assert report.self_talk_ratio == 1.0
        # Themes should include at least one of: anxiety, growth, gratitude.
        assert any(t in report.theme_counts for t in ("anxiety", "growth", "gratitude"))

    def test_self_sentiment(self, now):
        # Predominantly negative self-talk.
        obs = [
            _obs("note", "I feel terrible today, exhausted, stuck.", now),
            _obs("note", "I'm completely burnt out and frustrated.", now),
        ]
        report = reflect(obs, now=now)
        assert report.avg_self_sentiment < 0

    def test_rising_theme_detection(self, now):
        # 0 anxiety in week-2, 3 in last week.
        obs = []
        for i in range(3):
            obs.append(_obs("note", "I'm anxious about everything",
                            now - timedelta(days=i)))
        # Old entry without anxiety.
        obs.append(_obs("note", "I had a good day overall",
                        now - timedelta(days=10)))
        report = reflect(obs, now=now)
        assert "anxiety" in report.rising_themes

    def test_sample_hits_have_excerpts(self, now):
        obs = [_obs("note", "I'm anxious about the demo on Friday.", now)]
        report = reflect(obs, now=now)
        assert len(report.sample_hits) >= 1
        assert "anxious" in report.sample_hits[0].excerpt.lower()
