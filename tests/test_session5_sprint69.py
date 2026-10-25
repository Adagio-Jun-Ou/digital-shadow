#!/usr/bin/env python3
"""Tests for CognitiveReflection module and API endpoints."""

import pytest
import copy
import threading
from unittest.mock import patch
from datetime import datetime, timezone

from shadowd.cognition.cognitive_reflection import (
    CognitiveReflection, ReflectionType,
    ReflectionEntry, LearningInsight, ReflectionPattern,
    _sanitize, _LESSON_PATTERNS,
)


# ═══════════════════════════════════════════════════════════════════
# TestReflectionEntry
# ═══════════════════════════════════════════════════════════════════

class TestReflectionEntry:
    def test_reflection_entry_to_dict(self):
        entry = ReflectionEntry(
            entry_id="ref_test123",
            topic="Daily standup",
            description="Reflected on morning standup",
            reflection_type=ReflectionType.DAILY,
            what_went_well=["Good communication"],
            what_could_improve=["Shorter updates"],
            lessons_learned=["Be concise"],
            confidence_before=0.6,
            confidence_after=0.8,
            domain="work",
            linked_decision_ids=["dec_abc"],
            sentiment=0.3,
            created_at="2025-01-01T00:00:00+00:00",
            metadata={"key": "value"},
        )
        d = entry.to_dict()
        assert d["entry_id"] == "ref_test123"
        assert d["reflection_type"] == "daily"
        assert d["confidence_before"] == 0.6
        assert d["confidence_after"] == 0.8
        assert d["topic"] == "Daily standup"

    def test_reflection_type_enum(self):
        assert ReflectionType.DAILY.value == "daily"
        assert ReflectionType.WEEKLY.value == "weekly"
        assert ReflectionType.MILESTONE.value == "milestone"
        assert ReflectionType.POST_DECISION.value == "post_decision"
        assert ReflectionType.RETROSPECTIVE.value == "retrospective"
        assert ReflectionType.LESSON_LEARNED.value == "lesson_learned"

    def test_reflection_type_from_string(self):
        assert ReflectionType("daily") == ReflectionType.DAILY
        assert ReflectionType("lesson_learned") == ReflectionType.LESSON_LEARNED

    def test_reflection_entry_default_values(self):
        entry = ReflectionEntry(
            entry_id="ref_abc",
            topic="test",
            description="",
            reflection_type=ReflectionType.DAILY,
            what_went_well=[],
            what_could_improve=[],
            lessons_learned=[],
            confidence_before=0.5,
            confidence_after=0.5,
            domain="",
            linked_decision_ids=[],
            sentiment=0.0,
            created_at="2025-01-01T00:00:00",
            metadata={},
        )
        assert entry.what_went_well == []
        assert entry.sentiment == 0.0


class TestLearningInsight:
    def test_insight_to_dict(self):
        insight = LearningInsight(
            insight_id="learn_test123",
            title="Be concise",
            description="Learned to be more concise",
            category="general",
            source_reflection_ids=["ref_abc"],
            frequency=2,
            confidence=0.7,
            applicable_domains=["work"],
            created_at="2025-01-01T00:00:00+00:00",
        )
        d = insight.to_dict()
        assert d["insight_id"] == "learn_test123"
        assert d["frequency"] == 2
        assert d["category"] == "general"


class TestReflectionPattern:
    def test_pattern_to_dict(self):
        pattern = ReflectionPattern(
            pattern_id="rpat_test123",
            pattern_type="recurring_mistake",
            description="Negative trend",
            frequency=5,
            first_seen="2025-01-01T00:00:00",
            last_seen="2025-01-10T00:00:00",
        )
        d = pattern.to_dict()
        assert d["pattern_id"] == "rpat_test123"
        assert d["pattern_type"] == "recurring_mistake"
        assert d["frequency"] == 5


# ═══════════════════════════════════════════════════════════════════
# TestSanitization
# ═══════════════════════════════════════════════════════════════════

class TestSanitization:
    def test_sanitize_basic(self):
        assert _sanitize("hello world", 200) == "hello world"

    def test_sanitize_truncation(self):
        long_text = "a" * 300
        assert len(_sanitize(long_text, 200)) == 200

    def test_sanitize_control_chars(self):
        text = "hello\x00world\x01test"
        assert "\x00" not in _sanitize(text, 200)
        assert "\x01" not in _sanitize(text, 200)

    def test_sanitize_non_string(self):
        assert _sanitize(123, 200) == "123"

    def test_sanitize_custom_max_length(self):
        text = "hello world"
        assert _sanitize(text, 5) == "hello"

    def test_sanitize_strips_whitespace(self):
        assert _sanitize("  hello  ", 200) == "hello"


# ═══════════════════════════════════════════════════════════════════
# TestCognitiveReflectionCore
# ═══════════════════════════════════════════════════════════════════

class TestCognitiveReflectionCore:
    def setup_method(self):
        self.cr = CognitiveReflection()

    def test_add_reflection_basic(self):
        entry = self.cr.add_reflection(topic="Sprint review", domain="engineering")
        assert entry.entry_id.startswith("ref_")
        assert entry.topic == "Sprint review"
        assert entry.domain == "engineering"
        assert entry.reflection_type == ReflectionType.DAILY
        assert entry.confidence_before == 0.5
        assert entry.confidence_after == 0.5
        assert entry.sentiment == 0.0

    def test_add_reflection_with_all_fields(self):
        entry = self.cr.add_reflection(
            topic="Post-mortem",
            description="Reflecting on project delivery",
            reflection_type="retrospective",
            what_went_well=["Good teamwork", "On-time delivery"],
            what_could_improve=["Better estimates", "More testing"],
            lessons_learned=["Should have estimated earlier", "Need more QA"],
            confidence_before=0.4,
            confidence_after=0.7,
            domain="project",
            linked_decision_ids=["dec_001"],
            sentiment=0.5,
            metadata={"sprint": "42"},
        )
        assert entry.reflection_type == ReflectionType.RETROSPECTIVE
        assert len(entry.what_went_well) == 2
        assert len(entry.what_could_improve) == 2
        assert len(entry.lessons_learned) == 2
        assert entry.confidence_before == 0.4
        assert entry.confidence_after == 0.7
        assert entry.domain == "project"
        assert entry.linked_decision_ids == ["dec_001"]
        assert entry.sentiment == 0.5

    def test_add_reflection_invalid_type_defaults(self):
        entry = self.cr.add_reflection(topic="test", reflection_type="invalid_type")
        assert entry.reflection_type == ReflectionType.DAILY

    def test_add_reflection_confidence_clamp(self):
        entry = self.cr.add_reflection(topic="test", confidence_before=1.5, confidence_after=-0.3)
        assert entry.confidence_before == 1.0
        assert entry.confidence_after == 0.0

    def test_add_reflection_sentiment_clamp(self):
        entry = self.cr.add_reflection(topic="test", sentiment=2.0)
        assert entry.sentiment == 1.0
        entry2 = self.cr.add_reflection(topic="test2", sentiment=-2.0)
        assert entry2.sentiment == -1.0

    def test_add_reflection_sanitizes_topic(self):
        entry = self.cr.add_reflection(topic="hello\x00world" + "a" * 300)
        assert "\x00" not in entry.topic
        assert len(entry.topic) <= 200

    def test_add_reflection_sanitizes_description(self):
        entry = self.cr.add_reflection(topic="test", description="test\x01desc" + "b" * 3000)
        assert "\x01" not in entry.description
        assert len(entry.description) <= 2000

    def test_add_reflection_sanitizes_domain(self):
        entry = self.cr.add_reflection(topic="test", domain="a" * 300)
        assert len(entry.domain) <= 100

    def test_add_reflection_limits_items(self):
        items = [f"item_{i}" for i in range(20)]
        entry = self.cr.add_reflection(
            topic="test",
            what_went_well=items,
            what_could_improve=items,
            lessons_learned=items,
        )
        assert len(entry.what_went_well) <= 10
        assert len(entry.what_could_improve) <= 10
        assert len(entry.lessons_learned) <= 10

    def test_add_reflection_limits_linked_decisions(self):
        decisions = [f"dec_{i}" for i in range(10)]
        entry = self.cr.add_reflection(topic="test", linked_decision_ids=decisions)
        assert len(entry.linked_decision_ids) <= 5

    def test_add_reflection_limits_metadata(self):
        meta = {f"key_{i}": f"val_{i}" for i in range(30)}
        entry = self.cr.add_reflection(topic="test", metadata=meta)
        assert len(entry.metadata) <= 20

    def test_add_reflection_metadata_values_truncated(self):
        meta = {"long_key": "x" * 600}
        entry = self.cr.add_reflection(topic="test", metadata=meta)
        assert len(entry.metadata["long_key"]) <= 500

    def test_add_reflection_metadata_non_string_values(self):
        meta = {"count": 42, "flag": True, "ratio": 3.14}
        entry = self.cr.add_reflection(topic="test", metadata=meta)
        assert entry.metadata["count"] == 42
        assert entry.metadata["flag"] is True
        assert entry.metadata["ratio"] == 3.14

    def test_add_reflection_item_sanitization(self):
        entry = self.cr.add_reflection(
            topic="test",
            what_went_well=["good\x00thing" + "a" * 300],
        )
        assert "\x00" not in entry.what_went_well[0]
        assert len(entry.what_went_well[0]) <= 200

    def test_get_reflection(self):
        entry = self.cr.add_reflection(topic="test", domain="dev")
        retrieved = self.cr.get_reflection(entry.entry_id)
        assert retrieved is not None
        assert retrieved.entry_id == entry.entry_id
        assert retrieved.topic == "test"

    def test_get_reflection_not_found(self):
        assert self.cr.get_reflection("nonexistent") is None

    def test_get_reflection_returns_deepcopy(self):
        entry = self.cr.add_reflection(topic="test")
        retrieved = self.cr.get_reflection(entry.entry_id)
        retrieved.topic = "modified"
        original = self.cr.get_reflection(entry.entry_id)
        assert original.topic == "test"

    def test_list_reflections(self):
        self.cr.add_reflection(topic="first", domain="a")
        self.cr.add_reflection(topic="second", domain="b")
        entries = self.cr.list_reflections()
        assert len(entries) == 2

    def test_list_reflections_filter_by_type(self):
        self.cr.add_reflection(topic="daily", reflection_type="daily")
        self.cr.add_reflection(topic="weekly", reflection_type="weekly")
        entries = self.cr.list_reflections(reflection_type="daily")
        assert len(entries) == 1
        assert entries[0].reflection_type == ReflectionType.DAILY

    def test_list_reflections_filter_by_domain(self):
        self.cr.add_reflection(topic="dev", domain="dev")
        self.cr.add_reflection(topic="ops", domain="ops")
        entries = self.cr.list_reflections(domain="dev")
        assert len(entries) == 1

    def test_list_reflections_limit(self):
        for i in range(10):
            self.cr.add_reflection(topic=f"test_{i}")
        entries = self.cr.list_reflections(limit=5)
        assert len(entries) == 5

    def test_list_reflections_invalid_type(self):
        self.cr.add_reflection(topic="test")
        entries = self.cr.list_reflections(reflection_type="invalid")
        assert len(entries) == 1

    def test_list_reflections_returns_deepcopy(self):
        self.cr.add_reflection(topic="test")
        entries = self.cr.list_reflections()
        entries[0].topic = "modified"
        originals = self.cr.list_reflections()
        assert originals[0].topic == "test"


# ═══════════════════════════════════════════════════════════════════
# TestExtractInsights
# ═══════════════════════════════════════════════════════════════════

class TestExtractInsights:
    def setup_method(self):
        self.cr = CognitiveReflection()

    def test_extract_insights_basic(self):
        entry = self.cr.add_reflection(
            topic="Sprint review",
            what_went_well=["Good teamwork"],
            what_could_improve=["Should have estimated better"],
            lessons_learned=["Need more testing"],
            domain="engineering",
        )
        insights = self.cr.extract_insights(entry.entry_id)
        assert len(insights) == 3

    def test_extract_insights_not_found(self):
        result = self.cr.extract_insights("nonexistent")
        assert result == []

    def test_extract_insights_empty_lists(self):
        entry = self.cr.add_reflection(topic="Empty reflection")
        insights = self.cr.extract_insights(entry.entry_id)
        assert len(insights) == 0

    def test_extract_insights_skips_empty_strings(self):
        entry = self.cr.add_reflection(
            topic="test",
            what_went_well=["", "  ", "valid"],
        )
        insights = self.cr.extract_insights(entry.entry_id)
        assert len(insights) == 1

    def test_extract_insights_classifies_errors(self):
        entry = self.cr.add_reflection(
            topic="test",
            lessons_learned=["I made a mistake in the design"],
        )
        insights = self.cr.extract_insights(entry.entry_id)
        assert any(i.category == "error_pattern" for i in insights)

    def test_extract_insights_classifies_discovery(self):
        entry = self.cr.add_reflection(
            topic="test",
            what_went_well=["I learned a new approach"],
        )
        insights = self.cr.extract_insights(entry.entry_id)
        assert any(i.category == "discovery" for i in insights)

    def test_extract_insights_classifies_forward_looking(self):
        entry = self.cr.add_reflection(
            topic="test",
            lessons_learned=["In the future we will automate"],
        )
        insights = self.cr.extract_insights(entry.entry_id)
        assert any(i.category == "forward_looking" for i in insights)

    def test_extract_insights_classifies_comparative(self):
        entry = self.cr.add_reflection(
            topic="test",
            what_could_improve=["This was worse than expected"],
        )
        insights = self.cr.extract_insights(entry.entry_id)
        assert any(i.category == "comparative" for i in insights)

    def test_extract_insights_missed_opportunity(self):
        entry = self.cr.add_reflection(
            topic="test",
            lessons_learned=["We should have started earlier"],
        )
        insights = self.cr.extract_insights(entry.entry_id)
        assert any(i.category == "missed_opportunity" for i in insights)

    def test_extract_insights_general_category(self):
        entry = self.cr.add_reflection(
            topic="test",
            what_went_well=["Team was solid"],
        )
        insights = self.cr.extract_insights(entry.entry_id)
        assert any(i.category == "general" for i in insights)

    def test_extract_insights_merges_duplicates(self):
        entry1 = self.cr.add_reflection(
            topic="test1",
            what_went_well=["Good communication"],
        )
        self.cr.extract_insights(entry1.entry_id)
        entry2 = self.cr.add_reflection(
            topic="test2",
            what_went_well=["Good communication"],
        )
        insights2 = self.cr.extract_insights(entry2.entry_id)
        all_insights = self.cr.get_learning_insights()
        matching = [i for i in all_insights if i.title == "Good communication"]
        assert len(matching) == 1
        assert matching[0].frequency == 2

    def test_extract_insights_captures_domain(self):
        entry = self.cr.add_reflection(
            topic="test",
            what_went_well=["something"],
            domain="engineering",
        )
        insights = self.cr.extract_insights(entry.entry_id)
        assert insights[0].applicable_domains == ["engineering"]

    def test_extract_insights_capped_source_ids(self):
        entry = self.cr.add_reflection(topic="test", what_went_well=["a"])
        insights = self.cr.extract_insights(entry.entry_id)
        assert len(insights[0].source_reflection_ids) <= 5


# ═══════════════════════════════════════════════════════════════════
# TestDetectPatterns
# ═══════════════════════════════════════════════════════════════════

class TestDetectPatterns:
    def setup_method(self):
        self.cr = CognitiveReflection()

    def test_detect_patterns_negative_sentiment(self):
        for i in range(3):
            self.cr.add_reflection(
                topic=f"fail_{i}",
                domain="bugs",
                sentiment=-0.5,
            )
        patterns = self.cr.detect_patterns(min_frequency=2)
        assert any(p.pattern_type == "recurring_mistake" for p in patterns)

    def test_detect_patterns_positive_sentiment(self):
        for i in range(3):
            self.cr.add_reflection(
                topic=f"win_{i}",
                domain="releases",
                sentiment=0.7,
            )
        patterns = self.cr.detect_patterns(min_frequency=2)
        assert any(p.pattern_type == "strength_pattern" for p in patterns)

    def test_detect_patterns_improvement_items(self):
        for i in range(3):
            self.cr.add_reflection(
                topic=f"test_{i}",
                what_could_improve=["Need better testing"],
            )
        patterns = self.cr.detect_patterns(min_frequency=2)
        assert any(p.pattern_type == "improvement_trend" for p in patterns)

    def test_detect_patterns_strength_items(self):
        for i in range(3):
            self.cr.add_reflection(
                topic=f"test_{i}",
                what_went_well=["Great teamwork"],
            )
        patterns = self.cr.detect_patterns(min_frequency=2)
        assert any(p.pattern_type == "strength_pattern" for p in patterns)

    def test_detect_patterns_empty(self):
        patterns = self.cr.detect_patterns()
        assert patterns == []

    def test_detect_patterns_min_frequency(self):
        self.cr.add_reflection(topic="test", domain="dev", sentiment=-0.5)
        patterns = self.cr.detect_patterns(min_frequency=3)
        assert len(patterns) == 0

    def test_detect_patterns_pattern_fields(self):
        for i in range(3):
            self.cr.add_reflection(topic=f"test_{i}", what_went_well=["Great teamwork"])
        patterns = self.cr.detect_patterns(min_frequency=2)
        assert len(patterns) > 0
        p = patterns[0]
        assert p.pattern_id.startswith("rpat_")
        assert p.pattern_type in ("recurring_mistake", "strength_pattern", "improvement_trend", "avoidance_pattern")
        assert p.frequency >= 2
        assert p.first_seen is not None
        assert p.last_seen is not None

    def test_detect_patterns_stored(self):
        for i in range(3):
            self.cr.add_reflection(topic=f"test_{i}", sentiment=0.8, domain="perf")
        patterns = self.cr.detect_patterns(min_frequency=2)
        stats = self.cr.get_stats()
        assert stats["patterns_detected"] > 0


# ═══════════════════════════════════════════════════════════════════
# TestConfidenceTrend
# ═══════════════════════════════════════════════════════════════════

class TestConfidenceTrend:
    def setup_method(self):
        self.cr = CognitiveReflection()

    def test_confidence_trend_basic(self):
        self.cr.add_reflection(
            topic="test1", confidence_before=0.3, confidence_after=0.7
        )
        self.cr.add_reflection(
            topic="test2", confidence_before=0.4, confidence_after=0.8
        )
        trend = self.cr.get_confidence_trend()
        assert len(trend["trend"]) == 2
        assert trend["average_delta"] > 0
        assert trend["direction"] == "learning"

    def test_confidence_trend_overconfidence(self):
        self.cr.add_reflection(
            topic="test1", confidence_before=0.9, confidence_after=0.3
        )
        self.cr.add_reflection(
            topic="test2", confidence_before=0.8, confidence_after=0.2
        )
        trend = self.cr.get_confidence_trend()
        assert trend["direction"] == "overconfidence"

    def test_confidence_trend_neutral(self):
        self.cr.add_reflection(
            topic="test1", confidence_before=0.5, confidence_after=0.52
        )
        trend = self.cr.get_confidence_trend()
        assert trend["direction"] == "neutral"

    def test_confidence_trend_domain_filter(self):
        self.cr.add_reflection(topic="test1", domain="dev", confidence_before=0.3, confidence_after=0.8)
        self.cr.add_reflection(topic="test2", domain="ops", confidence_before=0.7, confidence_after=0.3)
        trend = self.cr.get_confidence_trend(domain="dev")
        assert len(trend["trend"]) == 1
        assert trend["direction"] == "learning"

    def test_confidence_trend_limit(self):
        for i in range(30):
            self.cr.add_reflection(topic=f"test_{i}", confidence_before=0.3, confidence_after=0.7)
        trend = self.cr.get_confidence_trend(limit=10)
        assert len(trend["trend"]) == 10

    def test_confidence_trend_empty(self):
        trend = self.cr.get_confidence_trend()
        assert trend["trend"] == []
        assert trend["average_delta"] == 0.0
        assert trend["direction"] == "neutral"

    def test_confidence_trend_delta_calculation(self):
        self.cr.add_reflection(topic="test", confidence_before=0.4, confidence_after=0.7)
        trend = self.cr.get_confidence_trend()
        assert trend["trend"][0]["delta"] == 0.3


# ═══════════════════════════════════════════════════════════════════
# TestGetLearningInsights
# ═══════════════════════════════════════════════════════════════════

class TestGetLearningInsights:
    def setup_method(self):
        self.cr = CognitiveReflection()

    def test_get_learning_insights_basic(self):
        entry = self.cr.add_reflection(topic="test", what_went_well=["Good planning"])
        self.cr.extract_insights(entry.entry_id)
        insights = self.cr.get_learning_insights()
        assert len(insights) >= 1

    def test_get_learning_insights_category_filter(self):
        entry = self.cr.add_reflection(
            topic="test",
            what_went_well=["I learned a new technique"],
        )
        self.cr.extract_insights(entry.entry_id)
        all_insights = self.cr.get_learning_insights()
        categories = {i.category for i in all_insights}
        if "discovery" in categories:
            filtered = self.cr.get_learning_insights(category="discovery")
            assert all(i.category == "discovery" for i in filtered)

    def test_get_learning_insights_limit(self):
        entry = self.cr.add_reflection(
            topic="test",
            what_went_well=[f"Lesson {i}" for i in range(15)],
        )
        self.cr.extract_insights(entry.entry_id)
        insights = self.cr.get_learning_insights(limit=5)
        assert len(insights) <= 5

    def test_get_learning_insights_sorted_by_frequency(self):
        entry1 = self.cr.add_reflection(topic="t1", what_went_well=["Good code"])
        self.cr.extract_insights(entry1.entry_id)
        entry2 = self.cr.add_reflection(topic="t2", what_went_well=["Good code"])
        self.cr.extract_insights(entry2.entry_id)
        entry3 = self.cr.add_reflection(topic="t3", what_went_well=["Unique insight"])
        self.cr.extract_insights(entry3.entry_id)
        insights = self.cr.get_learning_insights()
        if len(insights) > 1:
            assert insights[0].frequency >= insights[-1].frequency


# ═══════════════════════════════════════════════════════════════════
# TestGetStats
# ═══════════════════════════════════════════════════════════════════

class TestGetStats:
    def setup_method(self):
        self.cr = CognitiveReflection()

    def test_get_stats_initial(self):
        stats = self.cr.get_stats()
        assert stats["reflections_added"] == 0
        assert stats["insights_extracted"] == 0
        assert stats["patterns_detected"] == 0
        assert stats["current_reflections"] == 0
        assert stats["current_insights"] == 0
        assert stats["current_patterns"] == 0

    def test_get_stats_after_add(self):
        self.cr.add_reflection(topic="test")
        stats = self.cr.get_stats()
        assert stats["reflections_added"] == 1
        assert stats["current_reflections"] == 1

    def test_get_stats_after_insights(self):
        entry = self.cr.add_reflection(topic="test", what_went_well=["good"])
        self.cr.extract_insights(entry.entry_id)
        stats = self.cr.get_stats()
        assert stats["insights_extracted"] == 1
        assert stats["current_insights"] >= 1

    def test_get_stats_after_patterns(self):
        for i in range(3):
            self.cr.add_reflection(topic=f"t{i}", sentiment=0.8, domain="perf")
        self.cr.detect_patterns(min_frequency=2)
        stats = self.cr.get_stats()
        assert stats["patterns_detected"] > 0


# ═══════════════════════════════════════════════════════════════════
# TestBoundsAndEdgeCases
# ═══════════════════════════════════════════════════════════════════

class TestBoundsAndEdgeCases:
    def setup_method(self):
        self.cr = CognitiveReflection()

    def test_eviction_at_cap(self):
        cr = CognitiveReflection()
        cr._MAX_REFLECTIONS = 5
        for i in range(7):
            cr.add_reflection(topic=f"test_{i}")
        stats = cr.get_stats()
        assert stats["current_reflections"] <= 5

    def test_eviction_at_insight_cap(self):
        entry = self.cr.add_reflection(topic="test", what_went_well=[f"lesson_{i}" for i in range(15)])
        cr_small = CognitiveReflection()
        cr_small._MAX_INSIGHTS = 5
        e = cr_small.add_reflection(topic="test", what_went_well=[f"x_{i}" for i in range(8)])
        cr_small.extract_insights(e.entry_id)
        assert cr_small.get_stats()["current_insights"] <= 5

    def test_eviction_at_pattern_cap(self):
        cr_small = CognitiveReflection()
        cr_small._MAX_PATTERNS = 3
        for i in range(6):
            cr_small.add_reflection(topic=f"t_{i}", sentiment=0.9, domain=f"d_{i % 3}")
        cr_small.detect_patterns(min_frequency=2)
        assert cr_small.get_stats()["current_patterns"] <= 3

    def test_thread_safety(self):
        cr = CognitiveReflection()
        errors = []

        def add_reflection(i):
            try:
                cr.add_reflection(topic=f"thread_test_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_reflection, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = cr.get_stats()
        assert stats["reflections_added"] == 20

    def test_empty_what_went_well(self):
        entry = self.cr.add_reflection(topic="test", what_went_well=None)
        assert entry.what_went_well == []

    def test_empty_what_could_improve(self):
        entry = self.cr.add_reflection(topic="test", what_could_improve=None)
        assert entry.what_could_improve == []

    def test_empty_lessons_learned(self):
        entry = self.cr.add_reflection(topic="test", lessons_learned=None)
        assert entry.lessons_learned == []

    def test_empty_linked_decisions(self):
        entry = self.cr.add_reflection(topic="test", linked_decision_ids=None)
        assert entry.linked_decision_ids == []

    def test_empty_metadata(self):
        entry = self.cr.add_reflection(topic="test", metadata=None)
        assert entry.metadata == {}

    def test_reflection_type_all_values(self):
        for rt in ["daily", "weekly", "milestone", "post_decision", "retrospective", "lesson_learned"]:
            entry = self.cr.add_reflection(topic="test", reflection_type=rt)
            assert entry.reflection_type.value == rt

    def test_list_reflections_returns_latest_first(self):
        self.cr.add_reflection(topic="first")
        self.cr.add_reflection(topic="second")
        entries = self.cr.list_reflections()
        assert entries[0].topic == "second"
        assert entries[1].topic == "first"

    def test_confidence_trend_single_entry(self):
        self.cr.add_reflection(topic="test", confidence_before=0.3, confidence_after=0.7)
        trend = self.cr.get_confidence_trend()
        assert len(trend["trend"]) == 1
        assert trend["trend"][0]["delta"] == 0.4

    def test_extract_insights_from_nonexistent_reflection(self):
        result = self.cr.extract_insights("nonexistent_id")
        assert result == []

    def test_detect_patterns_avoidance_pattern_via_improvement(self):
        for i in range(4):
            self.cr.add_reflection(
                topic=f"test_{i}",
                what_could_improve=["Need better documentation"],
            )
        patterns = self.cr.detect_patterns(min_frequency=2)
        improvement_patterns = [p for p in patterns if p.pattern_type == "improvement_trend"]
        assert len(improvement_patterns) > 0

    def test_detect_patterns_stores_patterns(self):
        for i in range(3):
            self.cr.add_reflection(
                topic=f"test_{i}",
                what_went_well=["Solid code quality"],
                sentiment=0.8,
                domain="engineering",
            )
        patterns = self.cr.detect_patterns(min_frequency=2)
        assert len(patterns) > 0
        assert self.cr.get_stats()["current_patterns"] > 0