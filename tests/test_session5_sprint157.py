#!/usr/bin/env python3
"""Sprint 157 — DailyBriefing EvolutionGuide Integration Tests

Tests:
- generate_briefing() accepts and passes evolution_guide to _add_insights()
- BriefingItem generation from friction points and suggestions
- Urgency mapping from EvolutionGuide severity
- Scheduled briefing uses self.evolution_guide
- _add_insights narrow except
- DeviceContext + EvolutionGuide integration in briefing
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from shadowd.cognition.daily_briefing import (
    DailyBriefingEngine,
    DailyBriefing,
    BriefingItem,
    BriefingSection,
    UrgencyLevel,
)
from shadowd.cognition.evolution_guide import (
    EvolutionGuide, FrictionSeverity, FrictionPoint, SuggestionStatus,
)


class TestBriefingEvolutionGuideParameter:
    """generate_briefing() accepts evolution_guide and passes it through."""

    @pytest.fixture
    def engine(self):
        return DailyBriefingEngine()

    @pytest.mark.asyncio
    async def test_generate_briefing_with_evolution_guide(self, engine):
        eg = EvolutionGuide()
        for _ in range(5):
            eg.add_observation("procrastination and putting off tasks", domain="work")
        result = await engine.generate_briefing(evolution_guide=eg)
        friction_items = [i for i in result.items if "Friction" in i.title]
        assert len(friction_items) >= 1

    @pytest.mark.asyncio
    async def test_generate_briefing_without_evolution_guide(self, engine):
        result = await engine.generate_briefing()
        friction_items = [i for i in result.items if "Friction" in i.title]
        assert len(friction_items) == 0

    @pytest.mark.asyncio
    async def test_generate_briefing_with_none_evolution_guide(self, engine):
        result = await engine.generate_briefing(evolution_guide=None)
        friction_items = [i for i in result.items if "Friction" in i.title]
        assert len(friction_items) == 0

    @pytest.mark.asyncio
    async def test_engine_stores_evolution_guide(self):
        eg = EvolutionGuide()
        engine = DailyBriefingEngine(evolution_guide=eg)
        assert engine.evolution_guide is eg

    @pytest.mark.asyncio
    async def test_engine_default_evolution_guide_is_none(self):
        engine = DailyBriefingEngine()
        assert engine.evolution_guide is None


class TestBriefingInsightsFromEvolutionGuide:
    """_add_insights generates BriefingItems from EvolutionGuide data."""

    @pytest.fixture
    def engine(self):
        return DailyBriefingEngine()

    def _make_high_severity_eg(self):
        """Create an EvolutionGuide with enough observations to trigger severity."""
        eg = EvolutionGuide()
        for _ in range(8):
            eg.add_observation("sleep deprivation and insomnia patterns", domain="health")
        return eg

    @pytest.mark.asyncio
    async def test_friction_points_appear_in_briefing(self, engine):
        eg = EvolutionGuide()
        for _ in range(5):
            eg.add_observation("procrastination and putting off tasks", domain="work")
        result = await engine.generate_briefing(evolution_guide=eg)
        friction_items = [i for i in result.items
                         if i.section == BriefingSection.INSIGHTS and "Friction" in i.title]
        assert len(friction_items) >= 1

    @pytest.mark.asyncio
    async def test_suggestions_appear_in_briefing(self, engine):
        eg = self._make_high_severity_eg()
        eg.generate_suggestions(domain="health")
        result = await engine.generate_briefing(evolution_guide=eg)
        suggestion_items = [i for i in result.items
                           if i.section == BriefingSection.INSIGHTS
                           and "Evolution suggestion" in i.title]
        assert len(suggestion_items) >= 1

    @pytest.mark.asyncio
    async def test_suggestions_are_low_urgency(self, engine):
        eg = self._make_high_severity_eg()
        eg.generate_suggestions(domain="health")
        result = await engine.generate_briefing(evolution_guide=eg)
        suggestion_items = [i for i in result.items
                           if "Evolution suggestion" in i.title]
        for item in suggestion_items:
            assert item.urgency == UrgencyLevel.LOW

    @pytest.mark.asyncio
    async def test_suggestion_uses_rationale_as_content(self, engine):
        eg = self._make_high_severity_eg()
        eg.generate_suggestions(domain="health")
        result = await engine.generate_briefing(evolution_guide=eg)
        suggestion_items = [i for i in result.items if "Evolution suggestion" in i.title]
        if suggestion_items:
            assert any(len(i.content) > 20 for i in suggestion_items)

    @pytest.mark.asyncio
    async def test_insights_section_always_has_pattern_insight(self, engine):
        result = await engine.generate_briefing()
        pattern_items = [i for i in result.items
                        if i.section == BriefingSection.INSIGHTS
                        and "Pattern insight" in i.title]
        assert len(pattern_items) >= 1


class TestBriefingInsightsUrgencyMapping:
    """Verify urgency level mapping from EvolutionGuide severity."""

    @pytest.mark.asyncio
    async def test_critical_severity_maps_to_high(self):
        engine = DailyBriefingEngine()
        eg = EvolutionGuide()
        # Force a friction point with critical severity
        obs = eg.add_observation("stress overload and burnout", domain="health")
        fp = eg._create_friction_from_observation(obs, "stress_overload", 0.8)
        fp.severity = FrictionSeverity.CRITICAL
        # Get briefing items directly and verify mapping
        items = eg.get_briefing_evolution_items(limit=5)
        critical_items = [i for i in items if i.get("severity") == "critical"]
        assert len(critical_items) >= 1
        assert critical_items[0]["urgency"] == "high"

    @pytest.mark.asyncio
    async def test_low_severity_maps_to_medium_urgency_in_briefing_items(self):
        eg = EvolutionGuide()
        obs = eg.add_observation("routine work", domain="work")
        fp = eg._create_friction_from_observation(obs, "test_low", 0.1)
        fp.severity = FrictionSeverity.LOW
        items = eg.get_briefing_evolution_items(limit=5)
        low_items = [i for i in items if i.get("severity") == "low"]
        assert len(low_items) >= 1
        assert low_items[0]["urgency"] == "medium"

    @pytest.mark.asyncio
    async def test_high_severity_maps_to_high_urgency_in_briefing_items(self):
        eg = EvolutionGuide()
        obs = eg.add_observation("meeting overload", domain="productivity")
        fp = eg._create_friction_from_observation(obs, "meeting_overload", 0.7)
        fp.severity = FrictionSeverity.HIGH
        items = eg.get_briefing_evolution_items(limit=5)
        high_items = [i for i in items if i.get("severity") == "high"]
        assert len(high_items) >= 1
        assert high_items[0]["urgency"] == "high"

    @pytest.mark.asyncio
    async def test_critical_friction_creates_high_urgency_briefing_item(self):
        engine = DailyBriefingEngine()
        eg = EvolutionGuide()
        obs = eg.add_observation("stress overload and burnout", domain="health")
        fp = eg._create_friction_from_observation(obs, "stress_overload", 0.8)
        fp.severity = FrictionSeverity.CRITICAL
        result = await engine.generate_briefing(evolution_guide=eg)
        high_items = [i for i in result.items
                     if "Friction" in i.title and i.urgency == UrgencyLevel.HIGH]
        assert len(high_items) >= 1

    @pytest.mark.asyncio
    async def test_low_friction_creates_low_urgency_briefing_item(self):
        engine = DailyBriefingEngine()
        eg = EvolutionGuide()
        obs = eg.add_observation("routine work", domain="work")
        fp = eg._create_friction_from_observation(obs, "test_low", 0.1)
        fp.severity = FrictionSeverity.LOW
        result = await engine.generate_briefing(evolution_guide=eg)
        low_items = [i for i in result.items
                    if "Friction" in i.title and i.urgency == UrgencyLevel.LOW]
        assert len(low_items) >= 1


class TestScheduledBriefingUsesEvolutionGuide:
    """Scheduled briefing loop passes self.evolution_guide."""

    def test_evolution_guide_stored_in_engine(self):
        eg = EvolutionGuide()
        engine = DailyBriefingEngine(evolution_guide=eg)
        assert engine.evolution_guide is eg

    def test_evolution_guide_can_be_set_after_init(self):
        engine = DailyBriefingEngine()
        eg = EvolutionGuide()
        engine.evolution_guide = eg
        assert engine.evolution_guide is eg


class TestAddInsightsNarrowExcept:
    """_add_insights catches only specific exceptions."""

    @pytest.mark.asyncio
    async def test_add_insights_handles_broken_evolution_guide(self):
        engine = DailyBriefingEngine()
        broken_eg = MagicMock()
        broken_eg.get_briefing_evolution_items.side_effect = ValueError("test error")
        result = await engine.generate_briefing(evolution_guide=broken_eg)
        assert result is not None
        assert isinstance(result, DailyBriefing)


class TestBriefingFrictionDomainInItem:
    """Friction items have domain set correctly."""

    @pytest.mark.asyncio
    async def test_friction_item_has_domain(self):
        engine = DailyBriefingEngine()
        eg = EvolutionGuide()
        for _ in range(5):
            eg.add_observation("procrastination and putting off tasks", domain="work")
        result = await engine.generate_briefing(evolution_guide=eg)
        friction_items = [i for i in result.items if "Friction" in i.title]
        assert any(i.domain == "work" for i in friction_items)

    @pytest.mark.asyncio
    async def test_health_friction_has_health_domain(self):
        engine = DailyBriefingEngine()
        eg = EvolutionGuide()
        for _ in range(5):
            eg.add_observation("sleep deprivation and insomnia patterns", domain="health")
        result = await engine.generate_briefing(evolution_guide=eg)
        friction_items = [i for i in result.items if "Friction" in i.title]
        assert any(i.domain == "health" for i in friction_items)


class TestBriefingEvolutionGuideWithDeviceContext:
    """EvolutionGuide + DeviceContext both feed into briefing."""

    @pytest.mark.asyncio
    async def test_both_evolution_and_device_context(self):
        engine = DailyBriefingEngine()
        eg = EvolutionGuide()
        for _ in range(5):
            eg.add_observation("procrastination and context switching", domain="work")

        from shadowd.cognition.device_context import DeviceContextEngine
        dce = DeviceContextEngine()

        result = await engine.generate_briefing(
            device_context=dce,
            evolution_guide=eg,
        )
        assert isinstance(result, DailyBriefing)
        friction_items = [i for i in result.items if "Friction" in i.title]
        assert len(friction_items) >= 1

    @pytest.mark.asyncio
    async def test_device_context_wellness_with_evolution(self):
        engine = DailyBriefingEngine()
        eg = EvolutionGuide()
        for _ in range(5):
            eg.add_observation("stress overload and burnout", domain="health")

        from shadowd.cognition.device_context import DeviceContextEngine, DeviceType
        dce = DeviceContextEngine()
        dce.register_device("iphone-test", DeviceType.IPHONE, name="Test iPhone")

        result = await engine.generate_briefing(
            device_context=dce,
            evolution_guide=eg,
        )
        wellness_items = [i for i in result.items if i.section == BriefingSection.WELLNESS]
        assert len(wellness_items) >= 1


class TestEvolutionGuideIngestDeviceContext:
    """EvolutionGuide.ingest_device_context() creates observations from device data."""

    def test_ingest_stress_creates_health_observation(self):
        eg = EvolutionGuide()
        ctx = {
            "health": {"stress_level": 0.8, "heart_rate": 85, "sleep_hours": 7},
            "activity": {"activity_type": "working"},
            "focus": {"focus_state": "active"},
        }
        eg.ingest_device_context(ctx)
        stats = eg.get_stats()
        assert stats["observations_added"] >= 1

    def test_ingest_high_heart_rate_creates_observation(self):
        eg = EvolutionGuide()
        ctx = {
            "health": {"stress_level": 0.3, "heart_rate": 110, "sleep_hours": 7},
            "activity": {},
            "focus": {},
        }
        eg.ingest_device_context(ctx)
        stats = eg.get_stats()
        assert stats["observations_added"] >= 1

    def test_ingest_sleep_deficit_creates_observation(self):
        eg = EvolutionGuide()
        ctx = {
            "health": {"stress_level": 0.3, "heart_rate": 70, "sleep_hours": 4},
            "activity": {},
            "focus": {},
        }
        eg.ingest_device_context(ctx)
        stats = eg.get_stats()
        assert stats["observations_added"] >= 1

    def test_ingest_distracted_creates_productivity_observation(self):
        eg = EvolutionGuide()
        ctx = {
            "health": {"stress_level": 0.3, "heart_rate": 70, "sleep_hours": 7},
            "activity": {},
            "focus": {"focus_state": "distracted"},
        }
        eg.ingest_device_context(ctx)
        stats = eg.get_stats()
        assert stats["observations_added"] >= 1

    def test_ingest_normal_context_no_observations(self):
        eg = EvolutionGuide()
        ctx = {
            "health": {"stress_level": 0.2, "heart_rate": 65, "sleep_hours": 8},
            "activity": {},
            "focus": {"focus_state": "active"},
        }
        eg.ingest_device_context(ctx)
        stats = eg.get_stats()
        assert stats["observations_added"] == 0

    def test_ingest_with_object_context(self):
        eg = EvolutionGuide()
        mock_ctx = MagicMock()
        mock_ctx.to_dict.return_value = {
            "health": {"stress_level": 0.9, "heart_rate": 105, "sleep_hours": 4},
            "activity": {"activity_type": "exercising"},
            "focus": {"focus_state": "distracted"},
        }
        eg.ingest_device_context(mock_ctx)
        stats = eg.get_stats()
        assert stats["observations_added"] >= 3

    def test_ingest_invalid_context_no_crash(self):
        eg = EvolutionGuide()
        eg.ingest_device_context(None)
        eg.ingest_device_context("invalid")
        eg.ingest_device_context(42)
        stats = eg.get_stats()
        assert stats["observations_added"] == 0


class TestSeverityUrgencyMappingBugFix:
    """Test that the >= 3 bug in get_briefing_evolution_items is fixed."""

    def test_critical_severity_urgency_is_high(self):
        eg = EvolutionGuide()
        obs = eg.add_observation("test", domain="work")
        fp = eg._create_friction_from_observation(obs, "test", 0.9)
        fp.severity = FrictionSeverity.CRITICAL
        items = eg.get_briefing_evolution_items(limit=10)
        critical = [i for i in items if i.get("severity") == "critical"]
        assert len(critical) >= 1
        assert critical[0]["urgency"] == "high"

    def test_high_severity_urgency_is_high(self):
        eg = EvolutionGuide()
        obs = eg.add_observation("test", domain="work")
        fp = eg._create_friction_from_observation(obs, "test", 0.7)
        fp.severity = FrictionSeverity.HIGH
        items = eg.get_briefing_evolution_items(limit=10)
        high = [i for i in items if i.get("severity") == "high"]
        assert len(high) >= 1
        assert high[0]["urgency"] == "high"

    def test_medium_severity_urgency_is_medium(self):
        eg = EvolutionGuide()
        obs = eg.add_observation("test", domain="work")
        fp = eg._create_friction_from_observation(obs, "test", 0.5)
        fp.severity = FrictionSeverity.MEDIUM
        items = eg.get_briefing_evolution_items(limit=10)
        medium = [i for i in items if i.get("severity") == "medium"]
        assert len(medium) >= 1
        assert medium[0]["urgency"] == "medium"

    def test_low_severity_urgency_is_medium(self):
        eg = EvolutionGuide()
        obs = eg.add_observation("test", domain="work")
        fp = eg._create_friction_from_observation(obs, "test", 0.1)
        fp.severity = FrictionSeverity.LOW
        items = eg.get_briefing_evolution_items(limit=10)
        low = [i for i in items if i.get("severity") == "low"]
        assert len(low) >= 1
        assert low[0]["urgency"] == "medium"