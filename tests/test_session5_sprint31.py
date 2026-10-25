#!/usr/bin/env python3
"""Regression tests for Session 5 Sprint 31 features.

Feature 1: DarkMemory search_with_boost + get_domain_summary
Feature 2: DailyBriefingEngine get_wellness_metrics
Feature 3: DifferentialPrivacy + CamouflageEngine.aggregate_with_dp
Feature 4: BlockStore search_blocks + diff_blocks
"""

import asyncio
import math
import pytest
from collections import Counter


# =============================================================================
# Feature 1: DarkMemory search_with_boost + get_domain_summary
# =============================================================================

class TestDarkMemorySearchWithBoost:
    @pytest.mark.asyncio
    async def test_search_with_boost_ranks_domain_higher(self, tmp_path):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType
        mem = DarkMemory(data_dir=str(tmp_path))
        try:
            await mem.store("Python programming tips", atom_type=AtomType.OBSERVATION, domain="tech", importance=0.7)
            await mem.store("Morning exercise routine", atom_type=AtomType.OBSERVATION, domain="health", importance=0.6)
            await mem.store("Machine learning with Python", atom_type=AtomType.DECISION, domain="tech", importance=0.8)
            await mem.store("Healthy dinner recipes", atom_type=AtomType.INSIGHT, domain="health", importance=0.5)
            results = await mem.search_with_boost("Python", domain_boost="tech")
            tech_results = [a for a in results if a.domain == "tech"]
            assert len(tech_results) > 0
        finally:
            mem.close()

    @pytest.mark.asyncio
    async def test_search_with_boost_no_boost_returns_results(self, tmp_path):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType
        mem = DarkMemory(data_dir=str(tmp_path))
        try:
            await mem.store("Python programming tips", atom_type=AtomType.OBSERVATION, domain="tech", importance=0.7)
            normal = await mem.search("Python")
            boosted = await mem.search_with_boost("Python", domain_boost=None)
            normal_ids = {a.atom_id for a in normal}
            boosted_ids = {a.atom_id for a in boosted}
            assert normal_ids == boosted_ids
        finally:
            mem.close()

    @pytest.mark.asyncio
    async def test_search_with_boost_respects_atom_type(self, tmp_path):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType
        mem = DarkMemory(data_dir=str(tmp_path))
        try:
            await mem.store("Python programming tips", atom_type=AtomType.OBSERVATION, domain="tech", importance=0.7)
            await mem.store("Machine learning with Python", atom_type=AtomType.DECISION, domain="tech", importance=0.8)
            results = await mem.search_with_boost("Python", atom_type=AtomType.DECISION, domain_boost="tech")
            for r in results:
                assert r.atom_type == AtomType.DECISION
        finally:
            mem.close()


class TestDarkMemoryGetDomainSummary:
    @pytest.mark.asyncio
    async def test_domain_summary_counts(self, tmp_path):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType
        mem = DarkMemory(data_dir=str(tmp_path))
        await mem.store("a", domain="tech")
        await mem.store("b", domain="tech")
        await mem.store("c", domain="health")
        summary = mem.get_domain_summary()
        assert summary["tech"] == 2
        assert summary["health"] == 1
        mem.close()

    @pytest.mark.asyncio
    async def test_domain_summary_unclassified(self, tmp_path):
        from shadowd.cognition.dark_memory import DarkMemory
        mem = DarkMemory(data_dir=str(tmp_path))
        await mem.store("no domain atom")
        summary = mem.get_domain_summary()
        assert "unclassified" in summary
        assert summary["unclassified"] == 1
        mem.close()


# =============================================================================
# Feature 2: DailyBriefingEngine get_wellness_metrics
# =============================================================================

class TestDailyBriefingWellnessMetrics:
    @pytest.fixture
    def engine(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        return DailyBriefingEngine()

    @pytest.mark.asyncio
    async def test_wellness_metrics_default_before_briefing(self, engine):
        result = engine.get_wellness_metrics()
        assert result["overall_score"] == 0.7
        assert result["trend"] == "stable"
        assert len(result["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_wellness_metrics_after_briefing(self, engine):
        await engine.generate_briefing()
        result = engine.get_wellness_metrics()
        assert "overall_score" in result
        assert "components" in result
        assert "sleep" in result["components"]
        assert "activity" in result["components"]
        assert "social" in result["components"]
        assert "stress" in result["components"]
        assert result["trend"] in ("improving", "stable", "declining")

    @pytest.mark.asyncio
    async def test_wellness_metrics_in_briefing_metadata(self, engine):
        briefing = await engine.generate_briefing()
        assert "wellness_metrics" in briefing.metadata
        wm = briefing.metadata["wellness_metrics"]
        assert "overall_score" in wm
        assert "components" in wm


class TestDailyBriefingWellnessTrend:
    @pytest.mark.asyncio
    async def test_trend_declining_with_urgent_items(self, tmp_path):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine, BriefingItem, BriefingSection, UrgencyLevel, DailyBriefing
        engine = DailyBriefingEngine()
        calm = DailyBriefing(items=[
            BriefingItem(section=BriefingSection.TODAY, title="t1", content="c1", urgency=UrgencyLevel.INFO)
        ])
        calm.wellness_score = 0.8
        engine._briefing_history.append(calm)
        engine._wellness_history.append(0.8)

        urgent = DailyBriefing(items=[
            BriefingItem(section=BriefingSection.TODAY, title="t1", content="c1", urgency=UrgencyLevel.CRITICAL),
            BriefingItem(section=BriefingSection.TODAY, title="t2", content="c2", urgency=UrgencyLevel.HIGH),
        ])
        urgent.wellness_score = 0.4
        engine._briefing_history.append(urgent)

        result = engine.get_wellness_metrics()
        assert result["components"]["stress"] > 0.4


# =============================================================================
# Feature 3: DifferentialPrivacy + CamouflageEngine.aggregate_with_dp
# =============================================================================

class TestDifferentialPrivacy:
    def test_add_noise_returns_different_value(self):
        from shadowd.cognition.stealth_layer import DifferentialPrivacy
        dp = DifferentialPrivacy(epsilon=1.0)
        noisy = dp.add_noise(100.0, sensitivity=1.0)
        assert isinstance(noisy, float)
        assert abs(noisy - 100.0) < 50

    def test_add_noise_int_no_negative(self):
        from shadowd.cognition.stealth_layer import DifferentialPrivacy
        dp = DifferentialPrivacy(epsilon=10.0)
        for _ in range(100):
            result = dp.add_noise_int(5, sensitivity=1)
            assert result >= 0

    def test_add_noise_counts_preserves_keys(self):
        from shadowd.cognition.stealth_layer import DifferentialPrivacy
        dp = DifferentialPrivacy(epsilon=5.0)
        counts = {"apple": 10, "banana": 20, "cherry": 0}
        noisy = dp.add_noise_counts(counts)
        assert set(noisy.keys()) == set(counts.keys())
        assert all(v >= 0 for v in noisy.values())

    def test_add_noise_counts_no_negative(self):
        from shadowd.cognition.stealth_layer import DifferentialPrivacy
        dp = DifferentialPrivacy(epsilon=1.0)
        counts = {"a": 0, "b": 1, "c": 2}
        for _ in range(50):
            noisy = dp.add_noise_counts(counts)
            assert all(v >= 0 for v in noisy.values())

    def test_explain_privacy(self):
        from shadowd.cognition.stealth_layer import DifferentialPrivacy
        dp = DifferentialPrivacy(epsilon=1.0)
        explanation = dp.explain_privacy()
        assert "ε=1.0" in explanation
        assert "63.2%" in explanation


class TestCamouflageEngineAggregatWithDp:
    def test_aggregate_with_dp_mixed_types(self):
        from shadowd.cognition.stealth_layer import CamouflageEngine
        ce = CamouflageEngine()
        data = {"count": 100, "rate": 0.85, "name": "test"}
        result = ce.aggregate_with_dp(data, epsilon=10.0)
        assert isinstance(result["count"], int)
        assert isinstance(result["rate"], float)
        assert result["name"] == "test"

    def test_aggregate_with_dp_no_negative_ints(self):
        from shadowd.cognition.stealth_layer import CamouflageEngine
        ce = CamouflageEngine()
        data = {"small": 0, "medium": 5}
        result = ce.aggregate_with_dp(data, epsilon=5.0)
        assert result["small"] >= 0
        assert result["medium"] >= 0


# =============================================================================
# Feature 4: BlockStore search_blocks + diff_blocks
# =============================================================================

class TestBlockStoreSearchBlocks:
    @pytest.fixture
    def store(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore()
        bs.initialize_default_blocks()
        return bs

    def test_search_blocks_finds_match(self, store):
        store.set_block("knowledge", "Python is a great programming language")
        results = store.search_blocks("python")
        assert len(results) == 1
        assert results[0]["label"] == "knowledge"
        assert "Python" in results[0]["value"]

    def test_search_blocks_case_insensitive(self, store):
        store.set_block("knowledge", "PYTHON programming")
        results = store.search_blocks("python")
        assert len(results) == 1

    def test_search_blocks_no_match_returns_empty(self, store):
        results = store.search_blocks("quantum_physics_xyz")
        assert len(results) == 0

    def test_search_blocks_truncates_long_value(self, store):
        long_val = "hello world " + "x" * 500
        store.set_block("knowledge", long_val)
        results = store.search_blocks("hello")
        assert len(results) > 0
        assert len(results[0]["value"]) <= 200

    def test_search_blocks_returns_last_modified(self, store):
        store.set_block("knowledge", "test content")
        results = store.search_blocks("test")
        assert results[0]["last_modified"] is not None


class TestBlockStoreDiffBlocks:
    @pytest.fixture
    def store(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore()
        bs.initialize_default_blocks()
        return bs

    def test_diff_blocks_added_lines(self, store):
        old = "line1\nline2"
        new = "line1\nline2\nline3"
        diff = store.diff_blocks("knowledge", old, new)
        assert "line3" in diff["added"]
        assert len(diff["removed"]) == 0

    def test_diff_blocks_removed_lines(self, store):
        old = "line1\nline2\nline3"
        new = "line1"
        diff = store.diff_blocks("knowledge", old, new)
        assert "line2" in diff["removed"]
        assert "line3" in diff["removed"]

    def test_diff_blocks_unchanged_lines(self, store):
        old = "line1\nline2"
        new = "line1\nline3"
        diff = store.diff_blocks("knowledge", old, new)
        assert "line1" in diff["unchanged"]

    def test_diff_blocks_summary(self, store):
        old = "a\nb"
        new = "a\nc"
        diff = store.diff_blocks("knowledge", old, new)
        assert "+1" in diff["summary"]
        assert "-1" in diff["summary"]

    def test_diff_blocks_empty_values(self, store):
        diff = store.diff_blocks("knowledge", "", "")
        assert diff["added"] == []
        assert diff["removed"] == []
        assert diff["summary"] == "+0/-0/~0 lines"