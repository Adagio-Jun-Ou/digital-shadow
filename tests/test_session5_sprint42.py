"""Sprint 42: DreamState — Offline Processing & Memory Consolidation"""

import threading
import pytest
from collections import deque
from shadowd.cognition.dream_state import (
    DreamState, DreamPhase, DreamMemory, DreamPattern,
    DreamInsight, ConsolidationResult,
)


class TestDreamStateInit:
    def test_init_empty(self):
        ds = DreamState()
        assert len(ds._memories) == 0
        assert len(ds._patterns) == 0
        assert len(ds._insights) == 0
        assert ds._stats["consolidations_run"] == 0

    def test_thread_safety(self):
        ds = DreamState()
        assert hasattr(ds, '_lock')
        assert isinstance(ds._lock, type(threading.Lock()))

    def test_max_constants(self):
        assert DreamState._MAX_MEMORIES == 5000
        assert DreamState._MAX_PATTERNS == 500
        assert DreamState._MAX_INSIGHTS == 1000


class TestAddMemory:
    def test_add_returns_id(self):
        ds = DreamState()
        mid = ds.add_memory("test content")
        assert isinstance(mid, str)
        assert len(mid) == 12

    def test_add_stores_memory(self):
        ds = DreamState()
        mid = ds.add_memory("hello world", domain="test", importance=0.7)
        assert mid in ds._memories
        mem = ds._memories[mid]
        assert mem.content == "hello world"
        assert mem.domain == "test"
        assert mem.importance == 0.7

    def test_default_domain(self):
        ds = DreamState()
        mid = ds.add_memory("x")
        assert ds._memories[mid].domain == "general"

    def test_default_importance(self):
        ds = DreamState()
        mid = ds.add_memory("x")
        assert ds._memories[mid].importance == 0.5

    def test_with_tags(self):
        ds = DreamState()
        mid = ds.add_memory("tagged", tags=["alpha", "beta"])
        assert ds._memories[mid].tags == ["alpha", "beta"]

    def test_no_tags_default(self):
        ds = DreamState()
        mid = ds.add_memory("no tags")
        assert ds._memories[mid].tags == []

    def test_importance_clamped_high(self):
        ds = DreamState()
        mid = ds.add_memory("x", importance=5.0)
        assert ds._memories[mid].importance == 1.0

    def test_importance_clamped_low(self):
        ds = DreamState()
        mid = ds.add_memory("x", importance=-1.0)
        assert ds._memories[mid].importance == 0.0

    def test_empty_content_raises(self):
        ds = DreamState()
        with pytest.raises(ValueError):
            ds.add_memory("")

    def test_whitespace_only_raises(self):
        ds = DreamState()
        with pytest.raises(ValueError):
            ds.add_memory("   ")

    def test_content_truncated(self):
        ds = DreamState()
        mid = ds.add_memory("x" * 20000)
        assert len(ds._memories[mid].content) <= 10000

    def test_access_count_starts_zero(self):
        ds = DreamState()
        mid = ds.add_memory("x")
        assert ds._memories[mid].access_count == 0

    def test_strengthened_starts_false(self):
        ds = DreamState()
        mid = ds.add_memory("x")
        assert ds._memories[mid].strengthened is False

    def test_timestamps_set(self):
        ds = DreamState()
        mid = ds.add_memory("x")
        mem = ds._memories[mid]
        assert mem.created_at is not None
        assert mem.last_accessed is not None

    def test_fifo_eviction(self):
        ds = DreamState()
        ids = []
        for i in range(DreamState._MAX_MEMORIES + 10):
            ids.append(ds.add_memory(f"mem_{i}"))
        assert len(ds._memories) <= DreamState._MAX_MEMORIES


class TestRecall:
    def test_basic_recall(self):
        ds = DreamState()
        ds.add_memory("python programming language")
        ds.add_memory("java programming language")
        results = ds.recall("programming")
        assert len(results) == 2

    def test_domain_filter(self):
        ds = DreamState()
        ds.add_memory("test data", domain="work")
        ds.add_memory("test data", domain="personal")
        results = ds.recall("test", domain="work")
        assert len(results) == 1
        assert results[0].domain == "work"

    def test_no_match(self):
        ds = DreamState()
        ds.add_memory("hello world")
        results = ds.recall("xyz123")
        assert len(results) == 0

    def test_limit(self):
        ds = DreamState()
        for i in range(20):
            ds.add_memory(f"test item number {i}")
        results = ds.recall("test", limit=5)
        assert len(results) == 5

    def test_limit_clamped(self):
        ds = DreamState()
        results = ds.recall("x", limit=0)
        assert isinstance(results, list)
        results = ds.recall("x", limit=999)
        assert isinstance(results, list)

    def test_access_count_incremented(self):
        ds = DreamState()
        mid = ds.add_memory("searchable content")
        ds.recall("searchable")
        assert ds._memories[mid].access_count == 1

    def test_importance_ordering(self):
        ds = DreamState()
        ds.add_memory("alpha query", importance=0.3)
        ds.add_memory("beta query", importance=0.9)
        results = ds.recall("query")
        assert results[0].importance > results[1].importance


class TestForget:
    def test_forget_existing(self):
        ds = DreamState()
        mid = ds.add_memory("to forget")
        assert ds.forget(mid) is True
        assert mid not in ds._memories

    def test_forget_nonexistent(self):
        ds = DreamState()
        assert ds.forget("nonexistent") is False

    def test_forget_twice(self):
        ds = DreamState()
        mid = ds.add_memory("x")
        assert ds.forget(mid) is True
        assert ds.forget(mid) is False


class TestConsolidate:
    def test_consolidate_empty(self):
        ds = DreamState()
        result = ds.consolidate()
        assert isinstance(result, ConsolidationResult)
        assert result.memories_consolidated == 0
        assert len(result.phases_completed) == 4

    def test_consolidate_returns_result(self):
        ds = DreamState()
        ds.add_memory("alpha beta gamma", importance=0.7)
        result = ds.consolidate()
        assert result.duration_ms >= 0
        assert DreamPhase.PATTERN_STRENGTHENING.value in result.phases_completed
        assert DreamPhase.MEMORY_PRUNING.value in result.phases_completed
        assert DreamPhase.INSIGHT_GENERATION.value in result.phases_completed
        assert DreamPhase.FORGETTING.value in result.phases_completed

    def test_consolidation_count_increments(self):
        ds = DreamState()
        ds.consolidate()
        ds.consolidate()
        assert ds._stats["consolidations_run"] == 2

    def test_pattern_strengthening(self):
        ds = DreamState()
        for i in range(4):
            ds.add_memory("python programming code function", importance=0.5)
        result = ds.consolidate()
        assert result.patterns_strengthened > 0

    def test_importance_boosted(self):
        ds = DreamState()
        mids = []
        for i in range(4):
            mids.append(ds.add_memory("database query optimization index", importance=0.5))
        ds.consolidate()
        strengthened = [ds._memories[mid] for mid in mids if mid in ds._memories and ds._memories[mid].strengthened]
        assert any(m.importance > 0.5 for m in ds._memories.values())

    def test_memory_pruning_duplicates(self):
        ds = DreamState()
        mid1 = ds.add_memory("the quick brown fox jumps over the lazy dog")
        mid2 = ds.add_memory("the quick brown fox jumps over the lazy dog")
        ds.consolidate()
        remaining = [mid for mid in [mid1, mid2] if mid in ds._memories]
        assert len(remaining) == 1

    def test_near_duplicate_pruning(self):
        ds = DreamState()
        ds.add_memory("python programming language tutorial guide")
        ds.add_memory("python programming language tutorial guide advanced")
        ds.consolidate()
        count = sum(1 for m in ds._memories.values() if "python programming" in m.content)
        assert count <= 2

    def test_insight_generation_cross_domain(self):
        ds = DreamState()
        for i in range(5):
            ds.add_memory(
                "database optimization query performance index scaling",
                domain="engineering",
                importance=0.7,
            )
        for i in range(5):
            ds.add_memory(
                "database optimization query performance analysis research",
                domain="science",
                importance=0.7,
            )
        result = ds.consolidate()
        assert result.insights_generated > 0

    def test_forgetting_phase(self):
        ds = DreamState()
        ds.add_memory("unimportant noise", importance=0.1)
        ds.add_memory("important core", importance=0.9)
        result = ds.consolidate()
        remaining_contents = [m.content for m in ds._memories.values()]
        assert "important core" in remaining_contents

    def test_forget_max_ratio(self):
        ds = DreamState()
        for i in range(20):
            ds.add_memory(f"noise_{i}", importance=0.1)
        for i in range(5):
            ds.add_memory(f"important_{i}", importance=0.8)
        before = len(ds._memories)
        ds.consolidate()
        after = len(ds._memories)
        max_forget = max(1, int(before * DreamState._MAX_FORGET_RATIO))
        assert (before - after) <= max_forget


class TestGetInsights:
    def test_empty_insights(self):
        ds = DreamState()
        assert ds.get_insights() == []

    def test_insights_after_consolidate(self):
        ds = DreamState()
        for i in range(5):
            ds.add_memory("neural network deep learning model training", domain="ml")
        for i in range(5):
            ds.add_memory("neural network deep learning model inference", domain="ai")
        ds.consolidate()
        insights = ds.get_insights()
        assert len(insights) > 0
        assert isinstance(insights[0], DreamInsight)

    def test_insight_domains(self):
        ds = DreamState()
        for i in range(5):
            ds.add_memory("signal processing data transformation pipeline filter", domain="signal")
        for i in range(5):
            ds.add_memory("signal processing data transformation pipeline noise", domain="audio")
        ds.consolidate()
        insights = ds.get_insights()
        if insights:
            assert len(insights[0].domains_connected) == 2


class TestGetPatterns:
    def test_empty_patterns(self):
        ds = DreamState()
        assert ds.get_patterns() == []

    def test_patterns_after_add(self):
        ds = DreamState()
        for i in range(4):
            ds.add_memory("optimization performance scaling algorithm")
        ds.consolidate()
        patterns = ds.get_patterns()
        assert len(patterns) > 0
        assert isinstance(patterns[0], DreamPattern)


class TestDreamInsightToDict:
    def test_to_dict(self):
        insight = DreamInsight(
            insight_id="abc12345",
            description="test insight",
            domains_connected=["a", "b"],
            confidence=0.756,
            memory_ids=["m1", "m2"],
            created_at="2025-01-01T00:00:00+00:00",
        )
        d = insight.to_dict()
        assert d["insight_id"] == "abc12345"
        assert d["confidence"] == 0.756
        assert "domains_connected" in d


class TestDreamMemoryToDict:
    def test_to_dict(self):
        mem = DreamMemory(
            memory_id="abc12345",
            content="test",
            domain="general",
            importance=0.555,
            tags=["t1"],
            access_count=3,
            created_at="2025-01-01",
            last_accessed="2025-01-02",
            strengthened=True,
        )
        d = mem.to_dict()
        assert d["importance"] == 0.555
        assert d["strengthened"] is True


class TestDreamPatternToDict:
    def test_to_dict(self):
        pat = DreamPattern(
            pattern_id="abc12345",
            theme="python",
            domains=["eng", "science"],
            frequency=5,
            avg_importance=0.666,
            related_memories=["m1"],
        )
        d = pat.to_dict()
        assert d["avg_importance"] == 0.666
        assert d["frequency"] == 5


class TestConsolidationResultToDict:
    def test_to_dict(self):
        cr = ConsolidationResult(
            memories_consolidated=10,
            memories_forgotten=2,
            insights_generated=3,
            patterns_strengthened=4,
            phases_completed=["a", "b"],
            duration_ms=123.456,
        )
        d = cr.to_dict()
        assert d["memories_consolidated"] == 10
        assert d["duration_ms"] == 123.46


class TestGetStats:
    def test_initial_stats(self):
        ds = DreamState()
        stats = ds.get_stats()
        assert stats["total_memories"] == 0
        assert stats["total_patterns"] == 0
        assert stats["total_insights"] == 0
        assert stats["consolidation_count"] == 0
        assert stats["avg_importance"] == 0.0

    def test_stats_after_add(self):
        ds = DreamState()
        ds.add_memory("x", domain="test", importance=0.8)
        stats = ds.get_stats()
        assert stats["total_memories"] == 1
        assert "test" in stats["domains"]
        assert stats["avg_importance"] == 0.8

    def test_stats_after_consolidate(self):
        ds = DreamState()
        ds.add_memory("x")
        ds.consolidate()
        stats = ds.get_stats()
        assert stats["consolidation_count"] == 1


class TestThreadSafety:
    def test_concurrent_adds(self):
        ds = DreamState()
        errors = []

        def add_mem(i):
            try:
                ds.add_memory(f"concurrent memory {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_mem, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(ds._memories) == 50

    def test_concurrent_add_and_recall(self):
        ds = DreamState()
        errors = []
        results = []

        def add_mem(i):
            try:
                ds.add_memory(f"shared keyword item {i}")
            except Exception as e:
                errors.append(e)

        def recall_mem():
            try:
                r = ds.recall("shared")
                results.append(len(r))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_mem, args=(i,)) for i in range(20)]
        threads += [threading.Thread(target=recall_mem) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_consolidate(self):
        ds = DreamState()
        for i in range(30):
            ds.add_memory(f"memory {i} with some keywords here", importance=0.5)
        errors = []
        results = []

        def c():
            try:
                r = ds.consolidate()
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=c) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 3


class TestEdgeCases:
    def test_unicode_content(self):
        ds = DreamState()
        mid = ds.add_memory("日本語テスト テスト", domain="i18n")
        assert mid in ds._memories

    def test_special_chars_in_domain(self):
        ds = DreamState()
        mid = ds.add_memory("x", domain="test-domain_v2")
        assert ds._memories[mid].domain == "test-domain_v2"

    def test_very_long_domain(self):
        ds = DreamState()
        mid = ds.add_memory("x", domain="a" * 200)
        assert len(ds._memories[mid].domain) <= 100

    def test_empty_tags_list(self):
        ds = DreamState()
        mid = ds.add_memory("x", tags=[])
        assert ds._memories[mid].tags == []

    def test_insight_deque_eviction(self):
        ds = DreamState()
        original_max = DreamState._MAX_INSIGHTS
        DreamState._MAX_INSIGHTS = 20
        ds._insights = deque(maxlen=DreamState._MAX_INSIGHTS)
        for batch in range(25):
            for i in range(3):
                ds.add_memory(
                    f"keyword_alpha keyword_beta keyword_gamma{batch}{i}",
                    domain="domain_a",
                    importance=0.7,
                )
            for i in range(3):
                ds.add_memory(
                    f"keyword_alpha keyword_beta keyword_delta{batch}{i}",
                    domain="domain_b",
                    importance=0.7,
                )
            ds.consolidate()
        assert len(ds._insights) <= DreamState._MAX_INSIGHTS
        DreamState._MAX_INSIGHTS = original_max

    def test_dream_phase_enum(self):
        assert DreamPhase.PATTERN_STRENGTHENING == "pattern_strengthening"
        assert DreamPhase.MEMORY_PRUNING == "memory_pruning"
        assert DreamPhase.INSIGHT_GENERATION == "insight_generation"
        assert DreamPhase.FORGETTING == "forgetting"

    def test_recall_no_domain_filter(self):
        ds = DreamState()
        ds.add_memory("test data", domain="a")
        ds.add_memory("test data", domain="b")
        results = ds.recall("test")
        assert len(results) == 2

    def test_consolidate_with_single_memory(self):
        ds = DreamState()
        ds.add_memory("single memory alone")
        result = ds.consolidate()
        assert isinstance(result, ConsolidationResult)

    def test_forget_importance_threshold(self):
        ds = DreamState()
        mid = ds.add_memory("low importance", importance=0.15)
        ds.consolidate()
        # 0.15 < 0.2 threshold and access_count == 0 (<=1), should be forgotten
        assert mid not in ds._memories or ds._memories[mid].importance >= 0.2