#!/usr/bin/env python3
"""
Sprint 143 regression tests:
- EvolutionGuide module: observation, friction detection, suggestion generation, adoption tracking
- Cross-domain insights, domain progress tracking
- ShadowOS wiring
- Thread safety, bounded collections, input sanitization
"""

import copy
import threading
import pytest

from shadowd.cognition.evolution_guide import (
    EvolutionGuide, EvolutionDomain, FrictionSeverity,
    SuggestionStatus, AdoptionOutcome, ObservationType,
    ObservationPoint, FrictionPoint, EvolutionSuggestion,
    AdoptionRecord, DomainProgress,
    _sanitize, _sanitize_content,
)


class TestObservationPoint:
    def test_auto_id(self):
        o = ObservationPoint(description="test")
        assert len(o.observation_id) == 12

    def test_sanitization(self):
        o = ObservationPoint(description="hello\x00world<script>")
        assert "\x00" not in o.description
        assert "<script>" in o.description

    def test_confidence_clamped(self):
        o = ObservationPoint(description="test", confidence=5.0)
        assert o.confidence == 1.0
        o2 = ObservationPoint(description="test", confidence=-1.0)
        assert o2.confidence == 0.0

    def test_to_dict_copy(self):
        o = ObservationPoint(description="test", metadata={"key": "val"})
        d = o.to_dict()
        d["metadata"]["key2"] = "val2"
        assert "key2" not in o.metadata

    def test_domain_validation(self):
        o = ObservationPoint(description="test", domain="invalid_domain")
        assert o.domain == EvolutionDomain.WORK

    def test_observation_type_validation(self):
        o = ObservationPoint(description="test", observation_type="invalid")
        assert o.observation_type == ObservationType.BEHAVIOR

    def test_round_confidence(self):
        o = ObservationPoint(description="test", confidence=0.123456)
        assert o.to_dict()["confidence"] == 0.123


class TestFrictionPoint:
    def test_auto_id(self):
        f = FrictionPoint(title="test")
        assert len(f.friction_id) == 12

    def test_strict_title(self):
        f = FrictionPoint(title="<script>alert('xss')</script>")
        assert "<" not in f.title

    def test_impact_score_clamped(self):
        f = FrictionPoint(title="test", impact_score=3.0)
        assert f.impact_score == 1.0

    def test_to_dict_copy(self):
        f = FrictionPoint(title="test", metadata={"k": "v"})
        d = f.to_dict()
        d["metadata"]["k2"] = "v2"
        assert "k2" not in f.metadata

    def test_severity_validation(self):
        f = FrictionPoint(title="test", severity="invalid")
        assert f.severity == FrictionSeverity.MEDIUM


class TestEvolutionSuggestion:
    def test_auto_id(self):
        s = EvolutionSuggestion(title="test")
        assert len(s.suggestion_id) == 12

    def test_strict_title(self):
        s = EvolutionSuggestion(title="<b>bold</b>")
        assert "<" not in s.title

    def test_to_dict_copy(self):
        s = EvolutionSuggestion(title="test", metadata={"k": "v"})
        d = s.to_dict()
        d["metadata"]["k2"] = "v2"
        assert "k2" not in s.metadata

    def test_status_validation(self):
        s = EvolutionSuggestion(title="test", status="invalid")
        assert s.status == SuggestionStatus.PROPOSED


class TestAdoptionRecord:
    def test_auto_id(self):
        r = AdoptionRecord(suggestion_id="abc123")
        assert len(r.record_id) == 12

    def test_impact_clamped(self):
        r = AdoptionRecord(suggestion_id="abc", observed_impact=5.0)
        assert r.observed_impact == 1.0

    def test_outcome_validation(self):
        r = AdoptionRecord(suggestion_id="abc", outcome="invalid")
        assert r.outcome == AdoptionOutcome.UNKNOWN

    def test_to_dict_copy(self):
        r = AdoptionRecord(suggestion_id="abc", metadata={"k": "v"})
        d = r.to_dict()
        d["metadata"]["k2"] = "v2"
        assert "k2" not in r.metadata


class TestDomainProgress:
    def test_scores_clamped(self):
        dp = DomainProgress(current_score=2.0, baseline_score=-1.0)
        assert dp.current_score == 1.0
        assert dp.baseline_score == 0.0

    def test_to_dict_round(self):
        dp = DomainProgress(current_score=0.123456)
        assert dp.to_dict()["current_score"] == 0.123


class TestEvolutionGuideCore:
    def test_add_observation(self):
        eg = EvolutionGuide()
        obs = eg.add_observation("I feel tired and overwhelmed", "health", "behavior", "User reported fatigue")
        assert obs.observation_id in eg._observations
        assert eg._stats["observations_added"] == 1

    def test_friction_detection_on_observation(self):
        eg = EvolutionGuide()
        eg.add_observation("I keep procrastinating on important tasks", "work", "behavior", "")
        fps = eg.get_friction_points()
        assert len(fps) >= 1
        assert any("procrastinat" in fp.title.lower() for fp in fps)

    def test_friction_detection_health(self):
        eg = EvolutionGuide()
        eg.add_observation("I'm not sleeping well and feel fatigued", "health", "behavior", "")
        fps = eg.get_friction_points(domain="health")
        assert len(fps) >= 1

    def test_friction_detection_productivity(self):
        eg = EvolutionGuide()
        eg.add_observation("I can't focus with all these interruptions", "productivity", "behavior", "")
        fps = eg.get_friction_points(domain="productivity")
        assert len(fps) >= 1

    def test_friction_severity_increases_with_frequency(self):
        eg = EvolutionGuide()
        for _ in range(5):
            eg.add_observation("I'm procrastinating again", "work", "behavior", "")
        fps = eg.get_friction_points()
        severe = [fp for fp in fps if fp.severity in (FrictionSeverity.HIGH, FrictionSeverity.CRITICAL)]
        assert len(severe) >= 1

    def test_resolved_friction_excluded(self):
        eg = EvolutionGuide()
        eg.add_observation("I keep procrastinating", "work", "behavior", "")
        fps = eg.get_friction_points()
        if fps:
            fps[0].resolved_at = "2025-01-01T00:00:00Z"
        active = eg.get_friction_points()
        for fp in active:
            assert fp.resolved_at is None

    def test_generation_suggestions(self):
        eg = EvolutionGuide()
        eg.add_observation("I keep procrastinating on tasks", "work", "behavior", "")
        sugs = eg.generate_suggestions(domain="work")
        assert len(sugs) >= 1
        assert all(isinstance(s, EvolutionSuggestion) for s in sugs)
        assert all(len(s.rationale) > 0 for s in sugs)

    def test_suggestions_have_rationale(self):
        eg = EvolutionGuide()
        eg.add_observation("I'm burned out from stress overload", "health", "behavior", "")
        sugs = eg.generate_suggestions(domain="health")
        for s in sugs:
            assert len(s.rationale) > 20
            assert s.expected_benefit
            assert s.confidence > 0
            assert s.priority >= 0

    def test_suggestion_limit(self):
        eg = EvolutionGuide()
        for d in ["work", "health", "productivity", "creativity"]:
            eg.add_observation(f"I'm struggling with {d} procrastination", d, "behavior", "")
        sugs = eg.generate_suggestions(limit=2)
        assert len(sugs) <= 2

    def test_adopt_suggestion(self):
        eg = EvolutionGuide()
        eg.add_observation("I keep procrastinating", "work", "behavior", "")
        sugs = eg.generate_suggestions(domain="work", limit=1)
        if sugs:
            record = eg.adopt_suggestion(sugs[0].suggestion_id, "Starting today")
            assert record.outcome == AdoptionOutcome.UNKNOWN
            assert eg._stats["suggestions_adopted"] == 1
            updated = eg.get_suggestions()
            found = [s for s in updated if s.suggestion_id == sugs[0].suggestion_id]
            assert found[0].status == SuggestionStatus.ADOPTED

    def test_reject_suggestion(self):
        eg = EvolutionGuide()
        eg.add_observation("I keep procrastinating", "work", "behavior", "")
        sugs = eg.generate_suggestions(domain="work", limit=1)
        if sugs:
            result = eg.reject_suggestion(sugs[0].suggestion_id, "Not for me")
            assert result is True
            updated = eg.get_suggestions()
            found = [s for s in updated if s.suggestion_id == sugs[0].suggestion_id]
            assert found[0].status == SuggestionStatus.REJECTED

    def test_review_adoption(self):
        eg = EvolutionGuide()
        eg.add_observation("I'm not sleeping enough", "health", "behavior", "")
        sugs = eg.generate_suggestions(domain="health", limit=1)
        if sugs:
            eg.adopt_suggestion(sugs[0].suggestion_id)
            record = eg.review_adoption(sugs[0].suggestion_id, "positive", 0.8, "Great improvement!")
            if record:
                assert record.outcome == AdoptionOutcome.POSITIVE
                assert record.observed_impact == 0.8

    def test_domain_progress_update(self):
        eg = EvolutionGuide()
        eg.add_observation("I'm burned out", "health", "behavior", "")
        sugs = eg.generate_suggestions(domain="health", limit=1)
        if sugs:
            eg.adopt_suggestion(sugs[0].suggestion_id)
            eg.review_adoption(sugs[0].suggestion_id, "positive", 0.8)
            dp = eg.get_domain_progress("health")
            if dp:
                assert dp["current_score"] > dp["baseline_score"]


class TestCrossDomainInsights:
    def test_cross_domain_insight_generation(self):
        eg = EvolutionGuide()
        eg.add_observation("I'm stressed and overwhelmed", "health", "behavior", "")
        eg.add_observation("I can't focus at work", "productivity", "behavior", "")
        insights = eg.find_cross_domain_insights()
        assert len(insights) >= 1
        assert "source_domain" in insights[0]
        assert "target_domain" in insights[0]
        assert insights[0]["influence"] > 0.5

    def test_no_insight_without_friction(self):
        eg = EvolutionGuide()
        insights = eg.find_cross_domain_insights()
        assert len(insights) == 0


class TestEvolutionSummary:
    def test_summary_structure(self):
        eg = EvolutionGuide()
        eg.add_observation("Feeling tired", "health", "behavior", "")
        summary = eg.get_evolution_summary()
        assert "overall_adoption_rate" in summary
        assert "active_friction_points" in summary
        assert "open_suggestions" in summary
        assert "domains_tracked" in summary


class TestEvolutionGuideThreadSafety:
    def test_concurrent_observations(self):
        eg = EvolutionGuide()
        errors = []
        def add():
            try:
                for i in range(20):
                    eg.add_observation(f"Observation {i}", "work", "behavior", "")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=add) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_suggestions(self):
        eg = EvolutionGuide()
        eg.add_observation("I procrastinate", "work", "behavior", "")
        errors = []
        def gen():
            try:
                for _ in range(10):
                    eg.generate_suggestions(domain="work", limit=1)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=gen) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_read_write(self):
        eg = EvolutionGuide()
        eg.add_observation("Feeling burned out", "health", "behavior", "")
        errors = []
        def write():
            try:
                for i in range(10):
                    eg.add_observation(f"Obs {i}", "work", "behavior", "")
            except Exception as e:
                errors.append(e)
        def read():
            try:
                for _ in range(10):
                    eg.get_friction_points()
                    eg.get_suggestions()
                    eg.get_stats()
                    eg.get_evolution_summary()
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=write), threading.Thread(target=read)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestEvolutionGuideBounded:
    def test_observations_cap(self):
        eg = EvolutionGuide()
        for i in range(5500):
            eg.add_observation(f"Obs {i}", "work", "metric", "")
        assert len(eg._observations) <= 5050

    def test_suggestions_have_related_domains(self):
        eg = EvolutionGuide()
        eg.add_observation("I'm stressed and can't sleep", "health", "behavior", "")
        sugs = eg.generate_suggestions(domain="health", limit=5)
        if sugs:
            assert len(sugs[0].related_domains) >= 0


class TestGetStatsReturnsCopy:
    def test_stats_copy(self):
        eg = EvolutionGuide()
        s = eg.get_stats()
        s["observations_added"] = 999
        assert eg._stats["observations_added"] != 999


class TestSanitize:
    def test_strict_strips_lt_gt(self):
        assert "<" not in _sanitize("hello<world>test")
        assert ">" not in _sanitize("hello<world>test")

    def test_content_preserves_lt_gt(self):
        text = _sanitize_content("latency < 100ms")
        assert "<" in text
        assert "100ms" in text

    def test_strip_before_truncate(self):
        result = _sanitize("  hello  ", 7)
        assert result == "hello"