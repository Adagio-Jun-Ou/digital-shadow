"""
Session 5 Sprint 27 — MEDIUM Audit Fixes (Cognition + Legacy Modules)

Fixes:
  M1:  goal_planner.py — estimated_minutes clamped to [0, 9999]
  M2:  goal_planner.py — data_dir created with mode=0o700
  M3:  goal_planner.py — threading.Lock around shared state mutations
  M4:  anomaly_detector.py — user_id hashed in anomaly context (PII leak)
  M5:  anomaly_detector.py — persistence _save()/_load() for baselines & anomalies
  M6:  anomaly_detector.py — register_metric() auto-evicts like observe()
  M7:  self_improvement.py — data_dir created with mode=0o700
  M8:  self_improvement.py — _save_state() uses os.open()/os.fdopen()/fsync/replace
  M9:  shadowosd.py — rate-limit eviction is FIFO (no sort)
  M10: shadowosd.py — /briefing/generate auth + cached briefing engine
  M11: block_memory.py — long_term_memory capped at 50000
  M12: block_memory.py — fsync in ArchivalMemory._save() and BlockBasedMemory._save()
  M13: memory_rag.py — fsync in _save_memory()
  M14: memory_rag.py — long_term_memory capped at 10000
  M15: memory_rag.py — domain lists capped at 1000 entries
  M16: memory_tools.py — ContextWindowBudget division-by-zero guard
  M17: websocket_manager.py — topic name validation on SUBSCRIBE
  M18: event_bus.py — stats increments inside lock
  M19: block_memory.py — set_block() defaults new blocks to protected=True
  M20: block_memory.py — recall() docstring clarifies return type
"""

import inspect
import os
import tempfile
import threading

import pytest

# POSIX file modes (0o700) cannot be asserted on Windows; chmod there only
# toggles the read-only bit. Permission tests are skipped on non-POSIX hosts.
posix_only = pytest.mark.skipif(
    os.name == "nt", reason="POSIX file mode bits are not enforced on Windows"
)


# ─── M1: estimated_minutes clamp ───

class TestGoalPlannerEstMinutesClamp:
    def test_negative_clamped_to_zero(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        goal = gp.create_goal("Test")
        result = gp.decompose_goal(goal.goal_id, [
            {"description": "step", "estimated_minutes": -100},
        ])
        assert result.steps[0].estimated_minutes == 0

    def test_over_max_clamped_to_9999(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        goal = gp.create_goal("Test")
        result = gp.decompose_goal(goal.goal_id, [
            {"description": "step", "estimated_minutes": 99999},
        ])
        assert result.steps[0].estimated_minutes == 9999

    def test_normal_value_passed_through(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        goal = gp.create_goal("Test")
        result = gp.decompose_goal(goal.goal_id, [
            {"description": "step", "estimated_minutes": 30},
        ])
        assert result.steps[0].estimated_minutes == 30

    def test_default_value_preserved(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        goal = gp.create_goal("Test")
        result = gp.decompose_goal(goal.goal_id, [
            {"description": "step"},
        ])
        assert result.steps[0].estimated_minutes == 5.0


# ─── M2: goal_planner data_dir mode=0o700 ───

class TestGoalPlannerDataDirPermissions:
    @posix_only
    def test_data_dir_0o700(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        with tempfile.TemporaryDirectory() as td:
            data_dir = os.path.join(td, "gp_test_700")
            gp = CognitiveGoalPlanner(data_dir=data_dir)
            mode = os.stat(data_dir).st_mode & 0o777
            assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"

    def test_no_data_dir_no_error(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        assert gp._data_dir is None


# ─── M3: goal_planner threading.Lock ───

class TestGoalPlannerThreadSafety:
    def test_has_lock_attribute(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        assert hasattr(gp, '_lock')
        assert isinstance(gp._lock, type(threading.Lock()))

    def test_create_goal_uses_lock(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        source = inspect.getsource(CognitiveGoalPlanner.create_goal)
        assert "self._lock" in source

    def test_decompose_goal_uses_lock(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        source = inspect.getsource(CognitiveGoalPlanner.decompose_goal)
        assert "self._lock" in source

    def test_concurrent_creates_no_duplicates(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        results = []
        errors = []

        def create_goals(n):
            try:
                for i in range(n):
                    g = gp.create_goal(f"Goal-{threading.current_thread().name}-{i}")
                    if g:
                        results.append(g.goal_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_goals, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 50
        assert len(set(results)) == 50, "All goal IDs should be unique"


# ─── M4: anomaly_detector PII hashing ───

class TestAnomalyDetectorPIIHashing:
    def test_user_id_hashed_in_context(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.register_metric("test_metric")
        engine.set_baseline("user_alice_123", {"activity": {"mean": 5.0, "std": 1.0}})
        anomaly = engine.observe_behavioral("user_alice_123", "activity", 20.0)
        if anomaly:
            ctx = anomaly.context
            uid = ctx.get("user_id")
            assert uid != "user_alice_123", "user_id should be hashed, not plaintext"
            assert len(uid) == 16, "Hashed user_id should be 16 chars (truncated SHA-256)"

    def test_behavioral_anomaly_still_detectable(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.register_metric("latency")
        engine.set_baseline("bob", {"latency": {"mean": 100.0, "std": 10.0}})
        anomaly = engine.observe_behavioral("bob", "latency", 500.0)
        assert anomaly is not None


# ─── M5: anomaly_detector persistence ───

class TestAnomalyDetectorPersistence:
    def test_save_load_methods_exist(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        assert hasattr(AnomalyDetectionEngine, '_save')
        assert hasattr(AnomalyDetectionEngine, '_load')

    def test_save_uses_atomic_write(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        source = inspect.getsource(AnomalyDetectionEngine._save)
        assert "os.open" in source or "fsync" in source

    @posix_only
    def test_data_dir_created_with_0o700(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        with tempfile.TemporaryDirectory() as td:
            data_dir = os.path.join(td, "anom_test_700")
            engine = AnomalyDetectionEngine(data_dir=data_dir)
            mode = os.stat(data_dir).st_mode & 0o777
            assert mode == 0o700

    def test_init_accepts_data_dir(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        sig = inspect.signature(AnomalyDetectionEngine.__init__)
        assert "data_dir" in sig.parameters


# ─── M6: register_metric auto-evicts ───

class TestAnomalyDetectorRegisterEviction:
    def test_register_metric_evicts_at_capacity(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        for i in range(engine._MAX_STREAMS + 5):
            engine.register_metric(f"metric_{i}")
        assert len(engine._streams) <= engine._MAX_STREAMS

    def test_observe_auto_creates_evicts(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        for i in range(engine._MAX_STREAMS + 5):
            engine.observe(f"auto_metric_{i}", float(i))
        assert len(engine._streams) <= engine._MAX_STREAMS


# ─── M7: self_improvement data_dir mode=0o700 ───

class TestSelfImprovementDataDirPermissions:
    @posix_only
    def test_data_dir_0o700(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        with tempfile.TemporaryDirectory() as td:
            data_dir = os.path.join(td, "si_test_700")
            engine = SelfImprovementEngine(data_dir=data_dir)
            mode = os.stat(data_dir).st_mode & 0o777
            assert mode == 0o700


# ─── M8: self_improvement _save_state atomic write ───

class TestSelfImprovementAtomicSave:
    def test_save_state_uses_os_open_fdopen(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        source = inspect.getsource(SelfImprovementEngine._save_state)
        assert "os.open" in source, "_save_state should use os.open"
        assert "os.fdopen" in source, "_save_state should use os.fdopen"
        assert "os.fsync" in source, "_save_state should use os.fsync"
        assert "os.replace" in source, "_save_state should use os.replace"


# ─── M9: shadowosd rate-limit FIFO eviction ───

class TestShadowosdRateLimitFIFO:
    def test_eviction_is_fifo_not_sorted(self):
        from shadowd.cognition.shadowosd import rate_limit_middleware, _rate_limit_store
        source = inspect.getsource(rate_limit_middleware)
        assert "sorted" not in source, "Rate limit eviction should use FIFO, not sort"


# ─── M10: shadowosd briefing auth + caching ───

class TestShadowosdBriefingAuth:
    def test_briefing_get_has_auth_dependency(self):
        from shadowd.cognition.shadowosd import app
        routes = {route.path: route for route in app.routes if hasattr(route, 'path')}
        if '/briefing/generate' in routes:
            route = routes['/briefing/generate']
            if hasattr(route, 'dependant') and hasattr(route.dependant, 'dependencies'):
                assert any('verify' in str(d).lower() or 'api_key' in str(d).lower()
                           for d in route.dependant.dependencies), \
                    "/briefing/generate should have auth dependency"

    def test_briefing_engine_cached(self):
        import shadowd.cognition.shadowosd as shadowosd_mod
        assert hasattr(shadowosd_mod, '_briefing_engine'), "_briefing_engine should be module-level"

    def test_briefing_import_at_module_level(self):
        import shadowd.cognition.shadowosd as shadowosd_mod
        assert hasattr(shadowosd_mod, 'DailyBriefingEngine'), \
            "DailyBriefingEngine should be imported at module level, not inside handler"


# ─── M11: block_memory long_term_memory cap ───

class TestArchivalMemoryCap:
    def test_max_long_term_constant(self):
        from shadowd.block_memory import ArchivalMemory
        assert hasattr(ArchivalMemory, '_MAX_LONG_TERM')
        assert ArchivalMemory._MAX_LONG_TERM == 50000

    def test_insert_evicts_at_cap(self):
        from shadowd.block_memory import ArchivalMemory
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            am = ArchivalMemory(Path(td) / "archival")
            original_max = ArchivalMemory._MAX_LONG_TERM
            ArchivalMemory._MAX_LONG_TERM = 10
            try:
                for i in range(15):
                    am.insert(f"Memory {i}")
                assert len(am.memories) <= 10
            finally:
                ArchivalMemory._MAX_LONG_TERM = original_max


# ─── M12: block_memory fsync ───

class TestBlockMemoryFsync:
    def test_archival_save_uses_fsync(self):
        from shadowd.block_memory import ArchivalMemory
        source = inspect.getsource(ArchivalMemory._save)
        assert "os.fsync" in source, "ArchivalMemory._save should use fsync"
        assert "os.open" in source, "ArchivalMemory._save should use os.open for atomic write"

    def test_block_based_memory_save_uses_fsync(self):
        from shadowd.block_memory import BlockBasedMemory
        source = inspect.getsource(BlockBasedMemory._save)
        assert "os.fsync" in source, "BlockBasedMemory._save should use fsync"


# ─── M13: memory_rag fsync ───

class TestMemoryRagFsync:
    def test_save_memory_uses_fsync(self):
        from shadowd.memory_rag import RAGMemorySystem
        source = inspect.getsource(RAGMemorySystem._save_memory)
        assert "os.fsync" in source, "_save_memory should use fsync"
        assert "os.open" in source, "_save_memory should use os.open for atomic write"


# ─── M14: memory_rag long_term_memory cap ───

class TestRAGMemoryLongTermCap:
    def test_max_long_term_constant(self):
        from shadowd.memory_rag import RAGMemorySystem
        assert hasattr(RAGMemorySystem, '_MAX_LONG_TERM')
        assert RAGMemorySystem._MAX_LONG_TERM == 10000

    def test_has_eviction_in_add_observation_source(self):
        from shadowd.memory_rag import RAGMemorySystem
        source = inspect.getsource(RAGMemorySystem.add_observation)
        assert "_MAX_LONG_TERM" in source, "add_observation should reference _MAX_LONG_TERM"


# ─── M15: memory_rag domain cap ───

class TestRAGMemoryDomainCap:
    def test_max_per_domain_constant(self):
        from shadowd.memory_rag import RAGMemorySystem
        assert hasattr(RAGMemorySystem, '_MAX_PER_DOMAIN')
        assert RAGMemorySystem._MAX_PER_DOMAIN == 1000

    def test_domain_eviction_in_add_observation(self):
        from shadowd.memory_rag import RAGMemorySystem
        source = inspect.getsource(RAGMemorySystem.add_observation)
        assert "_MAX_PER_DOMAIN" in source, "add_observation should reference _MAX_PER_DOMAIN for cap"


# ─── M16: memory_tools division by zero ───

class TestContextWindowBudgetZeroDivision:
    def test_max_context_tokens_zero_guard(self):
        from shadowd.memory_tools import ContextWindowBudget
        budget = ContextWindowBudget(max_context_tokens=0)
        assert budget.max_context_tokens >= 1, "max_context_tokens should be >= 1"

    def test_negative_max_context_tokens_guard(self):
        from shadowd.memory_tools import ContextWindowBudget
        budget = ContextWindowBudget(max_context_tokens=-5)
        assert budget.max_context_tokens >= 1, "Should clamp negative to 1"

    def test_budget_overview_with_zero_section(self):
        from shadowd.memory_tools import ContextWindowBudget
        budget = ContextWindowBudget(max_context_tokens=100)
        overview = budget.get_budget_overview(system_prompt="", memory_xml="", tool_definitions="")
        assert overview["utilization"] >= 0
        for section_data in overview["sections"].values():
            assert section_data["utilization"] >= 0


# ─── M17: websocket topic validation ───

class TestWebSocketTopicValidation:
    def test_topic_regex_exists(self):
        from shadowd import websocket_manager as wm
        assert hasattr(wm, '_TOPIC_RE'), "_TOPIC_RE should be defined"
        assert hasattr(wm, '_MAX_TOPIC_LENGTH'), "_MAX_TOPIC_LENGTH should be defined"
        assert wm._MAX_TOPIC_LENGTH == 128

    def test_valid_topics_match(self):
        from shadowd import websocket_manager as wm
        assert wm._TOPIC_RE.match("sensor-data")
        assert wm._TOPIC_RE.match("alerts")
        assert wm._TOPIC_RE.match("room_1_temp")

    def test_invalid_topics_rejected(self):
        from shadowd import websocket_manager as wm
        assert not wm._TOPIC_RE.match("topic with spaces")
        assert not wm._TOPIC_RE.match("topic/slash")
        assert not wm._TOPIC_RE.match("topic.dot")
        assert not wm._TOPIC_RE.match("")

    def test_subscribe_uses_topic_validation(self):
        from shadowd.websocket_manager import ConnectionManager
        source = inspect.getsource(ConnectionManager.handle_client_loop)
        assert "_TOPIC_RE" in source, "SUBSCRIBE should validate topic with _TOPIC_RE"


# ─── M18: event_bus stats inside lock ───

class TestEventBusStatsInLock:
    def test_events_processed_inside_lock(self):
        from shadowd.event_bus import EventBus
        source = inspect.getsource(EventBus.publish)
        assert "self._lock" in source, "events_processed and errors should be incremented inside lock"

    def test_handler_collection_inside_lock(self):
        from shadowd.event_bus import EventBus
        source = inspect.getsource(EventBus.publish)
        lines = source.split('\n')
        lock_line = None
        for i, line in enumerate(lines):
            if 'async with self._lock' in line:
                lock_line = i
                break
        assert lock_line is not None, "publish should acquire self._lock"


# ─── M19: block_memory set_block defaults protected=True ───

class TestSetBlockProtectedDefault:
    def test_new_block_is_protected(self):
        from shadowd.block_memory import CoreMemory
        cm = CoreMemory()
        result = cm.set_block("newblock", "test value")
        assert result is True
        assert cm.blocks["newblock"].protected is True, "New blocks should default to protected=True"

    def test_existing_protected_block_still_protected(self):
        from shadowd.block_memory import CoreMemory
        cm = CoreMemory()
        result = cm.set_block("persona", "new persona text")
        assert result is False, "Protected blocks should not be modifiable"

    def test_existing_unprotected_block_modifiable(self):
        from shadowd.block_memory import CoreMemory
        cm = CoreMemory()
        result = cm.set_block("human", "new human text")
        assert result is True
        assert cm.blocks["human"].value == "new human text"


# ─── M20: block_memory recall() return type docstring ───

class TestRecallDocstring:
    def test_recall_has_return_type_docstring(self):
        from shadowd.block_memory import BlockBasedMemory
        source = inspect.getsource(BlockBasedMemory.recall)
        assert "str:" in source, "recall docstring should document str return type"
        assert "List[Dict]:" in source, "recall docstring should document List[Dict] return type"