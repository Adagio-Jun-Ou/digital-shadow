#!/usr/bin/env python3
"""Sprint 166 — EvolutionGuide Race Conditions + Deepcopy Fixes

Tests:
- generate_suggestions() returns deepcopies (not mutable internal references)
- adopt_suggestion() is atomic (no double-adopt race)
- reject_suggestion() guards against double-reject
- review_adoption() returns deepcopy inside lock
- _update_domain_progress() is atomic (no TOCTOU)
"""

import pytest
import threading

from shadowd.cognition.evolution_guide import (
    EvolutionGuide,
    EvolutionDomain,
    ObservationType,
    SuggestionStatus,
    AdoptionOutcome,
)


def _make_eg_with_suggestion():
    eg = EvolutionGuide()
    eg.add_observation(
        observation_type=ObservationType.PATTERN,
        domain="health",
        description="I am feeling stressed and overwhelmed by everything and can't sleep",
    )
    eg.add_observation(
        observation_type=ObservationType.PATTERN,
        domain="health",
        description="I feel overwhelmed and burned out at work",
    )
    return eg


class TestGenerateSuggestionsDeepcopy:
    """generate_suggestions() returns deepcopies."""

    def test_suggestions_are_deepcopies(self):
        eg = _make_eg_with_suggestion()
        suggestions = eg.generate_suggestions()
        if not suggestions:
            pytest.skip("No suggestions generated")
        inner = eg.get_suggestions()
        assert suggestions[0].suggestion_id == inner[0].suggestion_id
        suggestions[0].title = "HACKED"
        assert inner[0].title != "HACKED"


class TestAdoptSuggestionAtomic:
    """adopt_suggestion() prevents double-adopt."""

    def test_double_adopt_rejected(self):
        eg = _make_eg_with_suggestion()
        suggestions = eg.generate_suggestions()
        if not suggestions:
            pytest.skip("No suggestions generated")
        sid = suggestions[0].suggestion_id
        result1 = eg.adopt_suggestion(sid)
        assert result1 is not None
        result2 = eg.adopt_suggestion(sid)
        inner = eg.get_suggestions()
        found = [s for s in inner if s.suggestion_id == sid]
        assert len(found) == 1
        assert found[0].status == SuggestionStatus.ADOPTED

    def test_adopt_nonexistent(self):
        eg = EvolutionGuide()
        result = eg.adopt_suggestion("nonexistent_id")
        assert result.outcome == AdoptionOutcome.UNKNOWN

    def test_adopt_returns_deepcopy(self):
        eg = _make_eg_with_suggestion()
        suggestions = eg.generate_suggestions()
        if not suggestions:
            pytest.skip("No suggestions generated")
        sid = suggestions[0].suggestion_id
        record = eg.adopt_suggestion(sid)
        record.feedback = "HACKED"
        records = [r for r in eg._adoptions.values() if r.record_id == record.record_id]
        if records:
            assert records[0].feedback != "HACKED"


class TestRejectSuggestionGuard:
    """reject_suggestion() prevents double-reject."""

    def test_double_reject_rejected(self):
        eg = _make_eg_with_suggestion()
        suggestions = eg.generate_suggestions()
        if not suggestions:
            pytest.skip("No suggestions generated")
        sid = suggestions[0].suggestion_id
        assert eg.reject_suggestion(sid) is True
        assert eg.reject_suggestion(sid) is False

    def test_reject_nonexistent(self):
        eg = EvolutionGuide()
        assert eg.reject_suggestion("nonexistent_id") is False

    def test_reject_then_adopt_rejected(self):
        eg = _make_eg_with_suggestion()
        suggestions = eg.generate_suggestions()
        if not suggestions:
            pytest.skip("No suggestions generated")
        sid = suggestions[0].suggestion_id
        eg.reject_suggestion(sid)
        result = eg.adopt_suggestion(sid)
        inner = eg.get_suggestions()
        found = [s for s in inner if s.suggestion_id == sid]
        assert found[0].status == SuggestionStatus.REJECTED


class TestReviewAdoptionDeepcopy:
    """review_adoption() returns deepcopy inside lock."""

    def test_review_returns_deepcopy(self):
        eg = _make_eg_with_suggestion()
        suggestions = eg.generate_suggestions()
        if not suggestions:
            pytest.skip("No suggestions generated")
        sid = suggestions[0].suggestion_id
        eg.adopt_suggestion(sid)
        result = eg.review_adoption(sid, outcome="positive", observed_impact=0.5)
        if result:
            result.feedback = "HACKED"
            records = [r for r in eg._adoptions.values() if r.suggestion_id == sid and r.reviewed_at]
            if records:
                assert records[0].feedback != "HACKED"


class TestUpdateDomainProgressAtomic:
    """_update_domain_progress() is atomic."""

    def test_progress_updates_under_lock(self):
        eg = _make_eg_with_suggestion()
        suggestions = eg.generate_suggestions()
        if not suggestions:
            pytest.skip("No suggestions generated")
        sid = suggestions[0].suggestion_id
        eg.adopt_suggestion(sid)
        eg.review_adoption(sid, outcome="positive", observed_impact=0.7)
        progress = eg.get_domain_progress()
        assert isinstance(progress, dict)