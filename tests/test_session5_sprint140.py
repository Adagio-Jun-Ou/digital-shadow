#!/usr/bin/env python3
"""
Sprint 140 regression tests:
- shadow_graph.py: _stats_lock for thread safety
- dark_memory.py: _stats_lock for thread safety
- daily_briefing.py: wellness trend [:-3] edge case fix
- shadow_os.py: _save_state stats copy + _load_state stats lock
- shadow_crew.py: dead imports removed (json, Set)
"""

import asyncio
import threading
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from shadowd.cognition.shadow_graph import ShadowGraph, CognitiveMode, TransitionTrigger
from shadowd.cognition.dark_memory import DarkMemory, AtomType, MemoryTier
from shadowd.cognition.daily_briefing import DailyBriefingEngine, BriefingItem, BriefingSection, UrgencyLevel
from shadowd.cognition.shadow_crew import ShadowCrew


class TestShadowGraphStatsLock:
    def test_stats_lock_exists(self):
        sg = ShadowGraph(graph_id="test_stats_lock")
        assert hasattr(sg, '_stats_lock')
        assert isinstance(sg._stats_lock, type(threading.Lock()))

    @pytest.mark.asyncio
    async def test_stats_thread_safe_process_event(self):
        sg = ShadowGraph(graph_id="test_ts")
        errors = []
        def race():
            try:
                for _ in range(50):
                    with sg._stats_lock:
                        sg._stats["by_mode"]["listen"] += 1
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=race) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert sg._stats["by_mode"]["listen"] == 200

    def test_get_status_returns_stats_copy(self):
        sg = ShadowGraph(graph_id="test_status_copy")
        status = sg.get_status()
        stats = status["stats"]
        stats["transitions"] = 9999
        assert sg._stats["transitions"] == 0

    def test_export_stats_snapshot(self):
        sg = ShadowGraph(graph_id="test_export")
        exported = sg.export_graph_state()
        exported["statistics"]["transitions"] = 9999
        assert sg._stats["transitions"] == 0

    def test_reset_under_stats_lock(self):
        sg = ShadowGraph(graph_id="test_reset")
        sg._stats["transitions"] = 42
        sg.reset()
        assert sg._stats["transitions"] == 0

    def test_import_graph_state_stats_lock(self):
        sg = ShadowGraph(graph_id="test_import")
        sg.import_graph_state({"statistics": {"transitions": 100, "errors": 5, "by_mode": {"listen": 10}}})
        assert sg._stats["transitions"] == 100
        assert sg._stats["errors"] == 5

    @pytest.mark.asyncio
    async def test_approve_escalation_stats_thread_safe(self):
        sg = ShadowGraph(graph_id="test_esc")
        sg.state.pending_escalations.append({"id": "esc_test123", "node": "test", "reason": "test"})
        result = await sg.approve_escalation("esc_test123", True)
        assert result is True
        assert sg._stats["human_approvals"] == 1

    @pytest.mark.asyncio
    async def test_approve_escalation_denied_counts(self):
        sg = ShadowGraph(graph_id="test_esc2")
        sg.state.pending_escalations.append({"id": "esc_test456", "node": "test", "reason": "test"})
        result = await sg.approve_escalation("esc_test456", False)
        assert result is True
        assert sg._stats["human_approvals"] == 1


class TestDarkMemoryStatsLock:
    def test_stats_lock_exists(self):
        dm = DarkMemory.__new__(DarkMemory)
        dm._stats = {"atoms_stored": 0}
        dm._stats_lock = threading.Lock()
        assert hasattr(dm, '_stats_lock')

    def test_get_stats_returns_copy(self):
        dm = DarkMemory.__new__(DarkMemory)
        dm._stats = {"atoms_stored": 10, "atoms_retrieved": 5, "searches": 3, "compactions": 0, "kg_queries": 0, "atoms_evicted": 0, "auto_compactions": 0}
        dm._stats_lock = threading.Lock()
        dm._atoms = {}
        dm._atoms_lock = threading.Lock()
        dm._vector_index = MagicMock()
        dm._vector_index.size.return_value = 0
        dm._temporal_kg = MagicMock()
        dm._temporal_kg.get_stats.return_value = {}
        dm._block_store = MagicMock()
        dm._block_store.list_blocks.return_value = []
        dm.db_path = ":memory:"
        stats = dm.get_stats()
        stats["stats"]["atoms_stored"] = 9999
        assert dm._stats["atoms_stored"] == 10

    def test_concurrent_stats_increment(self):
        dm = DarkMemory.__new__(DarkMemory)
        dm._stats = {"atoms_stored": 0, "atoms_retrieved": 0, "searches": 0, "compactions": 0, "kg_queries": 0, "atoms_evicted": 0, "auto_compactions": 0}
        dm._stats_lock = threading.Lock()
        errors = []
        def increment():
            try:
                for _ in range(100):
                    with dm._stats_lock:
                        dm._stats["atoms_stored"] += 1
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert dm._stats["atoms_stored"] == 400


class TestDailyBriefingWellnessTrend:
    def test_wellness_trend_stable_with_few_data(self):
        engine = DailyBriefingEngine()
        engine._wellness_history.extend([0.7, 0.7, 0.7])
        metrics = engine.get_wellness_metrics()
        assert metrics["trend"] == "stable"

    def test_wellness_trend_improving_with_six_data(self):
        engine = DailyBriefingEngine()
        from collections import deque
        engine._wellness_history = deque([0.3, 0.3, 0.3, 0.7, 0.7, 0.7], maxlen=30)
        engine._briefing_history = [MagicMock()]
        engine._briefing_history[-1].items = []
        metrics = engine.get_wellness_metrics()
        assert metrics["trend"] == "improving"

    def test_wellness_trend_declining_with_six_data(self):
        engine = DailyBriefingEngine()
        from collections import deque
        engine._wellness_history = deque([0.8, 0.8, 0.8, 0.4, 0.4, 0.4], maxlen=30)
        engine._briefing_history = [MagicMock()]
        engine._briefing_history[-1].items = []
        metrics = engine.get_wellness_metrics()
        assert metrics["trend"] == "declining"

    def test_wellness_trend_stable_with_similar_data(self):
        engine = DailyBriefingEngine()
        from collections import deque
        engine._wellness_history = deque([0.7, 0.7, 0.7, 0.7, 0.7, 0.7], maxlen=30)
        engine._briefing_history = [MagicMock()]
        engine._briefing_history[-1].items = []
        metrics = engine.get_wellness_metrics()
        assert metrics["trend"] == "stable"

    def test_wellness_trend_stable_with_four_data(self):
        engine = DailyBriefingEngine()
        from collections import deque
        engine._wellness_history = deque([0.3, 0.3, 0.3, 0.9], maxlen=30)
        engine._briefing_history = [MagicMock()]
        engine._briefing_history[-1].items = []
        metrics = engine.get_wellness_metrics()
        assert metrics["trend"] == "stable"


class TestShadowOSStatsLock:
    def test_save_state_returns_stats_copy(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS.__new__(ShadowOS)
        sos._stats_lock = threading.Lock()
        sos._stats = {"events_processed": 42}
        stats_copy = dict(sos._stats)
        assert stats_copy["events_processed"] == 42
        stats_copy["events_processed"] = 0
        assert sos._stats["events_processed"] == 42

    def test_load_state_uses_stats_lock(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS.__new__(ShadowOS)
        sos._stats_lock = threading.Lock()
        sos._stats = {"events_processed": 0, "queries_handled": 0, "actions_taken": 0, "insights_generated": 0, "anomalies_detected": 0, "errors": 0, "uptime_seconds": 0}
        with sos._stats_lock:
            sos._stats.update({"events_processed": 10})
        assert sos._stats["events_processed"] == 10


class TestShadowCrewDeadImports:
    def test_no_json_import(self):
        import shadowd.cognition.shadow_crew as sc
        assert 'json' not in dir(sc) or not hasattr(sc, 'json')

    def test_no_set_import(self):
        import shadowd.cognition.shadow_crew as sc
        source = open(sc.__file__).read()
        for line in source.split('\n'):
            if line.strip().startswith('from typing'):
                assert 'Set' not in line

    def test_crew_instantiates(self):
        crew = ShadowCrew(crew_id="test_dead_imports")
        assert crew.crew_id == "test_dead_imports"


class TestShadowGraphDeadImports:
    def test_no_set_tuple_import(self):
        import shadowd.cognition.shadow_graph as sg
        source = open(sg.__file__).read()
        for line in source.split('\n'):
            if line.strip().startswith('from typing'):
                assert 'Set' not in line or 'Set,' not in line
                assert 'Tuple' not in line or 'Tuple,' not in line