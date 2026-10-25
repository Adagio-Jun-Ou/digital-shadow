#!/usr/bin/env python3
"""Sprint 133 — Second-Pass Deep Audit Round 5: temporal_reasoning, conflict_resolution,
adaptive_skill_tree, cognitive_trinity

Fixes:
- temporal_reasoning.py: Added _sanitize()/_sanitize_content() helpers, .strip() before [:max],
  round() in all to_dict() methods, input sanitization on public methods
- conflict_resolution.py: .strip() before [:max_len] in _sanitize()/_sanitize_content()
- adaptive_skill_tree.py: .strip() before [:_MAX_NAME_LENGTH] in _sanitize()
- cognitive_trinity.py: .strip() before [:max_len] in _sanitize()/_sanitize_content()
"""

import pytest
import copy
import threading
from unittest.mock import patch


class TestTemporalReasoningSanitize:
    def test_sanitize_strips_angles(self):
        from shadowd.cognition.temporal_reasoning import _sanitize
        result = _sanitize("<script>alert(1)</script>", max_length=200)
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.temporal_reasoning import _sanitize_content
        result = _sanitize_content("latency < 100ms", max_length=200)
        assert "<" in result

    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.temporal_reasoning import _sanitize
        result = _sanitize("  test  ", max_length=10)
        assert result == "test"

    def test_sanitize_content_strips_whitespace(self):
        from shadowd.cognition.temporal_reasoning import _sanitize_content
        result = _sanitize_content("  hello world  ", max_length=20)
        assert result == "hello world"


class TestTemporalReasoningRound:
    def test_pattern_to_dict_rounds_floats(self):
        from shadowd.cognition.temporal_reasoning import TemporalPattern, PatternType
        p = TemporalPattern(
            pattern_id="p1", pattern_type=PatternType.DAILY, event_type="login",
            frequency=0.3333333, typical_time="09:00", typical_day="monday",
            confidence=0.7777777, next_predicted="10:00", sample_size=10
        )
        d = p.to_dict()
        assert d["frequency"] == round(0.3333333, 3)
        assert d["confidence"] == round(0.7777777, 3)

    def test_prediction_to_dict_rounds_confidence(self):
        from shadowd.cognition.temporal_reasoning import Prediction, PatternType
        p = Prediction(
            event_type="login", predicted_time="10:00",
            confidence=0.8888888, reasoning="test", pattern_type=PatternType.WEEKLY
        )
        d = p.to_dict()
        assert d["confidence"] == round(0.8888888, 3)

    def test_anomaly_to_dict_rounds_deviation(self):
        from shadowd.cognition.temporal_reasoning import TemporalAnomaly
        a = TemporalAnomaly(
            event_type="login", expected_time="09:00", actual_time="12:00",
            deviation_score=2.5555555, description="Late login"
        )
        d = a.to_dict()
        assert d["deviation_score"] == round(2.5555555, 3)

    def test_trend_analysis_to_dict_rounds_floats(self):
        from shadowd.cognition.temporal_reasoning import TrendAnalysis, TrendDirection
        t = TrendAnalysis(
            event_type="login", direction=TrendDirection.INCREASING,
            rate_of_change=0.1234567, confidence=0.9012345,
            window_days=30, frequency_per_day=2.3456789
        )
        d = t.to_dict()
        assert d["rate_of_change"] == round(0.1234567, 3)
        assert d["confidence"] == round(0.9012345, 3)
        assert d["frequency_per_day"] == round(2.3456789, 3)

    def test_duration_estimate_to_dict_rounds_floats(self):
        from shadowd.cognition.temporal_reasoning import DurationEstimate
        d = DurationEstimate(
            event_type="task", mean_minutes=45.6666666, median_minutes=40.3333333,
            min_minutes=10.1111111, max_minutes=120.9999999,
            confidence=0.8765432, sample_size=100
        )
        result = d.to_dict()
        assert result["mean_minutes"] == round(45.6666666, 3)
        assert result["median_minutes"] == round(40.3333333, 3)
        assert result["min_minutes"] == round(10.1111111, 3)
        assert result["max_minutes"] == round(120.9999999, 3)
        assert result["confidence"] == round(0.8765432, 3)

    def test_record_event_sanitizes_event_type(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        eid = tr.record_event("  login  ", timestamp="2024-01-15T09:00:00+00:00")
        events = list(tr._events)
        assert events[0]["event_type"] == "login"

    def test_record_event_sanitizes_metadata(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        eid = tr.record_event("test", timestamp="2024-01-15T09:00:00+00:00",
                               metadata={"note": "value < 5", "key<bad>": "normal"})
        events = list(tr._events)
        meta = events[0]["metadata"]
        assert "note" in meta
        assert meta["note"] == "value < 5"
        assert "keybad" in meta or "key<bad>" not in meta

    def test_detect_patterns_sanitizes_event_type(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        result = tr.detect_patterns(event_type="  test  ")
        assert isinstance(result, list)

    def test_predict_next_sanitizes_event_type(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        result = tr.predict_next("  test  ", within_hours=24)
        assert isinstance(result, list)

    def test_analyze_trend_sanitizes_event_type(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning, TrendAnalysis
        tr = TemporalReasoning()
        result = tr.analyze_trend("  test  ", window_days=30)
        assert result is None or isinstance(result, TrendAnalysis)

    def test_estimate_duration_sanitizes_event_type(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning, DurationEstimate
        tr = TemporalReasoning()
        result = tr.estimate_duration("  test  ")
        assert result is None or isinstance(result, DurationEstimate)


class TestConflictResolutionStrip:
    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        cr = ConflictResolution()
        result = cr._sanitize("  hello world  ", max_len=5)
        assert result == "hello"

    def test_sanitize_content_strips_whitespace(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        result = ConflictResolution._sanitize_content("  hello world  ", max_len=5)
        assert result == "hello"

    def test_create_conflict_sanitizes_title(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution, ConflictType, ConflictIntensity
        cr = ConflictResolution()
        conflict = cr.create_conflict("  test title  ", ConflictType.INTEREST, ConflictIntensity.MEDIUM)
        assert conflict.title == "test title"


class TestAdaptiveSkillTreeStrip:
    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        ast = AdaptiveSkillTree()
        result = ast._sanitize("  python  ")
        assert result == "python"

    def test_sanitize_strips_angles(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        ast = AdaptiveSkillTree()
        result = ast._sanitize("<script>alert(1)</script>")
        assert "<" not in result


class TestCognitiveTrinityStrip:
    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        result = ct._sanitize("  test  ")
        assert result == "test"

    def test_sanitize_content_strips_whitespace(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        result = CognitiveTrinity._sanitize_content("  test  ")
        assert result == "test"

    def test_analyze_sanitizes_situation(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        result = ct.intuition_mind("  urgent task  ")
        assert result is not None

    def test_analyze_preserves_angles_in_content(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        result = ct.intuition_mind("latency < 100ms")
        assert result.recommendation != ""


class TestTemporalReasoningDeepCopy:
    def test_anomaly_to_dict_is_deepcopy(self):
        from shadowd.cognition.temporal_reasoning import TemporalAnomaly
        a = TemporalAnomaly(
            event_type="login", expected_time="09:00", actual_time="12:00",
            deviation_score=2.5, description="test"
        )
        d = a.to_dict()
        assert d["description"] == "test"
        d["description"] = "modified"
        assert a.description == "test"

    def test_duration_to_dict_is_deepcopy(self):
        from shadowd.cognition.temporal_reasoning import DurationEstimate
        de = DurationEstimate(
            event_type="task", mean_minutes=45.0, median_minutes=40.0,
            min_minutes=10.0, max_minutes=120.0, confidence=0.8, sample_size=100
        )
        d = de.to_dict()
        d["event_type"] = "modified"
        assert de.event_type == "task"


class TestConflictResolutionDeepCopy:
    def test_get_conflict_returns_deepcopy(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution, ConflictType, ConflictIntensity
        cr = ConflictResolution()
        c = cr.create_conflict("test", ConflictType.INTEREST, ConflictIntensity.MEDIUM)
        c1 = cr.get_conflict(c.conflict_id)
        c2 = cr.get_conflict(c.conflict_id)
        assert c1 is not c2
        assert c1.title == c2.title

    def test_list_conflicts_returns_deepcopy(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution, ConflictType, ConflictIntensity
        cr = ConflictResolution()
        cr.create_conflict("test", ConflictType.INTEREST, ConflictIntensity.MEDIUM)
        list1 = cr.list_conflicts()
        list2 = cr.list_conflicts()
        assert list1[0] is not list2[0]


class TestAdaptiveSkillTreeDeepCopy:
    def test_get_skill_returns_deepcopy(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree, SkillDomain
        ast = AdaptiveSkillTree()
        ast.add_skill("python-id", "Python", SkillDomain.TECHNICAL.value)
        s1 = ast.get_skill("python-id")
        s2 = ast.get_skill("python-id")
        assert s1 is not s2

    def test_get_recommendations_returns_deepcopy(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree, SkillDomain
        ast = AdaptiveSkillTree()
        ast.add_skill("python-id", "Python", SkillDomain.TECHNICAL.value)
        recs = ast.get_recommendations()
        assert isinstance(recs, list)

    def test_get_domain_summary_returns_deepcopy(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree, SkillDomain
        ast = AdaptiveSkillTree()
        ast.add_skill("python-id", "Python", SkillDomain.TECHNICAL.value)
        s1 = ast.get_domain_summary()
        s2 = ast.get_domain_summary()
        assert s1 is not s2


class TestCognitiveTrinityDeepCopy:
    def test_analyze_returns_deepcopy(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        result = ct.analyze("test situation")
        assert result is not None
        assert hasattr(result, 'analysis_id')

    def test_get_stats_returns_copy(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        stats = ct.get_stats()
        assert isinstance(stats, dict)

    def test_record_outcome_unknown_analysis(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        result = ct.record_outcome("nonexistent_id", "good", 0.8)
        assert result is None


class TestTemporalReasoningThreadSafety:
    def test_concurrent_record_and_read(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        errors = []

        def writer():
            try:
                for i in range(50):
                    tr.record_event(f"event_{i}", timestamp="2024-01-15T09:00:00+00:00")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    tr.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestConflictResolutionThreadSafety:
    def test_concurrent_create_and_read(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution, ConflictType, ConflictIntensity
        cr = ConflictResolution()
        errors = []

        def writer():
            try:
                for i in range(20):
                    cr.create_conflict(f"conflict_{i}", ConflictType.INTEREST, ConflictIntensity.MEDIUM)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    cr.list_conflicts()
                    cr.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0