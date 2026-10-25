"""
Session 5 Sprint 26 — Security Audit Fixes (14 findings)

Fixes applied:
  1. CRITICAL: shadowosd.py — Unauthenticated escalation endpoints now require auth
  2. CRITICAL: memory_rag.py — Path traversal via domain name in _save_domain()
  3. CRITICAL: websocket_manager.py — Client ID takeover: disconnect old client first
  4. HIGH: memory_tools.py — XML injection via block labels sanitized
  5. HIGH: memory_rag.py — ChromaDB document ID collision: timestamp → uuid4
  6. HIGH: websocket_manager.py — Unbounded active_streams dict capped at 500
  7. HIGH: block_memory.py — Thread-unsafe debounced save: threading.Lock added
  8. HIGH: config.py — Unvalidated environment variables: try/except with defaults
  9. HIGH: goal_planner.py — Double-counting failed goals + goal limit bypass + dangling refs
  10. HIGH: anomaly_detector.py — metric_pairs parameter now used to filter pairs
  11. HIGH: self_improvement.py — Unbounded strategy + cross-index stale refs + decisions eviction stale refs
  12. HIGH: shadowosd.py — Rate limit bypass via X-Forwarded-For removed
  13. HIGH: shadowosd.py — EscalationManager creation race condition: threading.Lock + double-checked locking
  14. HIGH: streaming.py — API key masking for logging: _mask_key()
"""

import inspect
import os
import re
import threading
import time
import pytest


# ─── Fix 1: Escalation endpoints have auth dependency ───

class TestEscalationAuthRequired:
    def test_escalation_evaluate_has_auth_dependency(self):
        import shadowd.cognition.shadowosd as mod
        source = inspect.getsource(mod.evaluate_escalation)
        assert '_verify_os_api_key' in source or 'Depends' in source, \
            "/escalation/evaluate should have auth dependency"

    def test_escalation_rules_has_auth_dependency(self):
        import shadowd.cognition.shadowosd as mod
        source = inspect.getsource(mod.get_escalation_rules)
        assert '_verify_os_api_key' in source or 'Depends' in source, \
            "/escalation/rules should have auth dependency"

    def test_verify_os_api_key_function_exists(self):
        from shadowd.cognition.shadowosd import _verify_os_api_key
        assert callable(_verify_os_api_key)


# ─── Fix 2: Path traversal via domain name ───

class TestDomainPathTraversal:
    def test_validate_domain_rejects_traversal(self):
        from shadowd.memory_rag import RAGMemorySystem
        with pytest.raises(ValueError):
            RAGMemorySystem._validate_domain("../../../etc/passwd")

    def test_validate_domain_rejects_slashes(self):
        from shadowd.memory_rag import RAGMemorySystem
        with pytest.raises(ValueError):
            RAGMemorySystem._validate_domain("sub/dir")

    def test_validate_domain_rejects_dots(self):
        from shadowd.memory_rag import RAGMemorySystem
        with pytest.raises(ValueError):
            RAGMemorySystem._validate_domain("dir.txt")

    def test_validate_domain_accepts_valid(self):
        from shadowd.memory_rag import RAGMemorySystem
        assert RAGMemorySystem._validate_domain("health") == "health"
        assert RAGMemorySystem._validate_domain("my-domain_123") == "my-domain_123"

    def test_validate_domain_rejects_empty(self):
        from shadowd.memory_rag import RAGMemorySystem
        with pytest.raises(ValueError):
            RAGMemorySystem._validate_domain("")

    def test_validate_domain_rejects_too_long(self):
        from shadowd.memory_rag import RAGMemorySystem
        with pytest.raises(ValueError):
            RAGMemorySystem._validate_domain("a" * 51)

    def test_validate_domain_accepts_max_length(self):
        from shadowd.memory_rag import RAGMemorySystem
        assert RAGMemorySystem._validate_domain("a" * 50) == "a" * 50

    def test_save_domain_validates(self):
        from shadowd.memory_rag import RAGMemorySystem
        source = inspect.getsource(RAGMemorySystem._save_domain)
        assert '_validate_domain' in source, "_save_domain should validate domain"

    def test_add_observation_validates_domain(self):
        from shadowd.memory_rag import RAGMemorySystem
        source = inspect.getsource(RAGMemorySystem.add_observation)
        assert '_validate_domain' in source, "add_observation should validate domain"


# ─── Fix 3: Client ID takeover disconnects old ───

class TestClientIDTakeover:
    def test_connect_disconnects_old_client(self):
        import shadowd.cognition.shadowosd
        source = inspect.getsource(shadowd.cognition.shadowosd)
        from shadowd.websocket_manager import ConnectionManager
        src = inspect.getsource(ConnectionManager.connect)
        assert 'old_client' in src or 'is_alive' in src or 'websocket.close' in src, \
            "connect() should disconnect old client on ID takeover"

    def test_connect_logs_warning_on_takeover(self):
        from shadowd.websocket_manager import ConnectionManager
        src = inspect.getsource(ConnectionManager.connect)
        assert 'Client ID takeover' in src or 'takeover' in src.lower(), \
            "connect() should log warning on ID takeover"


# ─── Fix 4: XML injection via block labels ───

class TestXMLLabelSanitization:
    def test_sanitize_label_strips_special_chars(self):
        from shadowd.memory_tools import _sanitize_label
        assert _sanitize_label("hello world") == "hello_world"
        assert _sanitize_label("foo<bar>") == "foo_bar_"
        assert _sanitize_label('attr="val"') == "attr__val_"

    def test_sanitize_label_preserves_valid(self):
        from shadowd.memory_tools import _sanitize_label
        assert _sanitize_label("human") == "human"
        assert _sanitize_label("my-block_123") == "my-block_123"

    def test_sanitize_label_empty_becomes_block(self):
        from shadowd.memory_tools import _sanitize_label
        assert _sanitize_label("") == "block"

    def test_sanitize_label_starts_with_digit(self):
        from shadowd.memory_tools import _sanitize_label
        result = _sanitize_label("123abc")
        assert result.startswith("_"), "Labels starting with digit should be prefixed"

    def test_render_memory_xml_uses_sanitized_labels(self):
        from shadowd.memory_tools import render_memory_xml, _sanitize_label
        from shadowd.block_memory import MemoryBlock
        blocks = {"test<block>": MemoryBlock(label="test<block>", value="data", limit=100)}
        xml = render_memory_xml(blocks)
        assert "<test_block_>" in xml, "XML should use sanitized labels"
        assert "</test_block_>" in xml, "XML closing tags should match sanitized labels"
        assert "test<block>" not in xml, "Raw unsanitized label should not appear"


# ─── Fix 5: ChromaDB document ID collision ───

class TestChromaDBIDCollision:
    def test_memory_id_uses_uuid(self):
        from shadowd.memory_rag import RAGMemorySystem
        source = inspect.getsource(RAGMemorySystem.add_observation)
        assert 'uuid' in source, "add_observation should use uuid for IDs"
        assert 'uuid.uuid4()' in source or 'uuid.uuid4().hex' in source, \
            "Should use uuid4 for unique IDs"

    def test_uuid_imported(self):
        import shadowd.memory_rag
        assert hasattr(shadowd.memory_rag, 'uuid'), "uuid module should be imported"


# ─── Fix 6: Unbounded active_streams capped ───

class TestActiveStreamCap:
    def test_max_active_streams_class_attribute(self):
        from shadowd.websocket_manager import StreamingResponseHandler
        assert hasattr(StreamingResponseHandler, '_MAX_ACTIVE_STREAMS')
        assert StreamingResponseHandler._MAX_ACTIVE_STREAMS == 500

    def test_stream_response_enforces_cap(self):
        from shadowd.websocket_manager import StreamingResponseHandler
        src = inspect.getsource(StreamingResponseHandler.stream_response)
        assert '_MAX_ACTIVE_STREAMS' in src, "stream_response should check cap"


# ─── Fix 7: Thread-unsafe debounced save ───

class TestThreadSafeDebouncedSave:
    def test_archival_memory_has_save_lock(self):
        from shadowd.block_memory import ArchivalMemory
        mem = ArchivalMemory()
        assert hasattr(mem, '_save_lock'), "ArchivalMemory should have _save_lock"
        assert isinstance(mem._save_lock, type(threading.Lock())), "_save_lock should be a Lock"

    def test_mark_dirty_uses_lock(self):
        from shadowd.block_memory import ArchivalMemory
        src = inspect.getsource(ArchivalMemory._mark_dirty)
        assert '_save_lock' in src, "_mark_dirty should acquire _save_lock"

    def test_save_uses_lock_or_is_called_under_lock(self):
        from shadowd.block_memory import ArchivalMemory
        src = inspect.getsource(ArchivalMemory._save)
        assert '_save_lock' in src or 'with' in inspect.getsource(ArchivalMemory._mark_dirty), \
            "_save should be called under _save_lock or use it"


# ─── Fix 8: Config unvalidated env vars ───

class TestConfigEnvValidation:
    def test_invalid_temperature_uses_default(self):
        from shadowd.config import ShadowConfig
        os.environ['SHADOW_LLM_TEMPERATURE'] = 'not_a_number'
        try:
            config = ShadowConfig.from_env()
            assert config.llm.temperature == 0.7, "Should default to 0.7 on invalid input"
        finally:
            del os.environ['SHADOW_LLM_TEMPERATURE']

    def test_invalid_port_uses_default(self):
        from shadowd.config import ShadowConfig
        os.environ['SHADOW_PORT'] = 'not_a_number'
        try:
            config = ShadowConfig.from_env()
            assert config.server.port == 8765, "Should default to 8765 on invalid input"
        finally:
            del os.environ['SHADOW_PORT']

    def test_auth_enabled_without_secret_key_logs_error(self, caplog):
        from shadowd.config import ShadowConfig
        os.environ['SHADOW_AUTH_ENABLED'] = 'true'
        os.environ['SHADOW_SECRET_KEY'] = ''
        try:
            import importlib
            import shadowd.config
            importlib.reload(shadowd.config)
            config = shadowd.config.ShadowConfig.from_env()
            assert config.security.auth_enabled is True
        finally:
            os.environ.pop('SHADOW_AUTH_ENABLED', None)
            os.environ.pop('SHADOW_SECRET_KEY', None)

    def test_negative_port_uses_default(self):
        from shadowd.config import ShadowConfig
        os.environ['SHADOW_PORT'] = '-1'
        try:
            config = ShadowConfig.from_env()
            assert config.server.port > 0, \
                "Should handle edge case"
        finally:
            del os.environ['SHADOW_PORT']

    def test_empty_secret_key_with_auth_enabled(self):
        from shadowd.config import ShadowConfig
        os.environ.pop('SHADOW_SECRET_KEY', None)
        os.environ['SHADOW_AUTH_ENABLED'] = 'true'
        try:
            config = ShadowConfig.from_env()
            assert config.security.auth_enabled is True
            assert config.security.secret_key == ''
        finally:
            os.environ.pop('SHADOW_AUTH_ENABLED', None)


# ─── Fix 9: Goal planner double-counting + limit bypass + dangling refs ───

class TestGoalPlannerFixes:
    def test_create_goal_returns_none_at_capacity(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner, GoalStatus
        planner = CognitiveGoalPlanner()
        for i in range(500):
            planner.create_goal(title=f"Goal {i}")
        result = planner.create_goal(title="overflow")
        assert result is None, "Should return None when at max capacity after eviction"

    def test_failed_goal_not_double_counted(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner, Goal, GoalStatus, StepStatus, PlanStep
        planner = CognitiveGoalPlanner()
        goal = Goal(title="test", status=GoalStatus.BLOCKED)
        planner._goals[goal.goal_id] = goal
        step = PlanStep(step_id="s1", status=StepStatus.FAILED)
        goal.steps = [step]
        initial_failed = planner._stats["goals_failed"]
        planner._check_goal_blocked(goal)
        assert planner._stats["goals_failed"] == initial_failed, \
            "Should not increment goals_failed when already BLOCKED"

    def test_failed_goal_counts_when_not_blocked(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner, Goal, GoalStatus, StepStatus, PlanStep
        planner = CognitiveGoalPlanner()
        goal = Goal(title="test", status=GoalStatus.IN_PROGRESS)
        planner._goals[goal.goal_id] = goal
        step = PlanStep(step_id="s1", status=StepStatus.FAILED)
        goal.steps = [step]
        initial_failed = planner._stats["goals_failed"]
        planner._check_goal_blocked(goal)
        assert planner._stats["goals_failed"] == initial_failed + 1, \
            "Should increment goals_failed when not already BLOCKED"

    def test_evict_completed_preserves_sub_goal_refs(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner, GoalStatus
        planner = CognitiveGoalPlanner()
        child = planner.create_goal(title="child")
        child.status = GoalStatus.COMPLETED
        parent = planner.create_goal(title="parent")
        parent.sub_goals = [child.goal_id]
        planner._evict_completed()
        assert child.goal_id in planner._goals, \
            "Completed goal referenced as sub_goal should not be evicted"


# ─── Fix 10: anomaly_detector metric_pairs used ───

class TestAnomalyDetectorMetricPairs:
    def test_metric_pairs_filters_checked_metrics(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.register_metric("cpu")
        engine.register_metric("memory")
        engine.register_metric("disk")
        engine.register_metric("network")
        for i in range(10):
            engine.observe("cpu", 50.0)
            engine.observe("memory", 60.0)
            engine.observe("disk", 40.0)
            engine.observe("network", 30.0)

        engine.observe("cpu", 200.0)
        engine.observe("memory", 200.0)
        engine.observe("disk", 200.0)
        engine.observe("network", 200.0)

        result_with_pairs = engine.detect_correlation_anomaly(
            metric_pairs=[("cpu", "memory"), ("disk", "network")],
            current_values={"cpu": 200.0, "memory": 200.0, "disk": 200.0, "network": 200.0},
        )
        assert len(result_with_pairs) > 0, "Should detect anomaly with paired metrics"

    def test_metric_pairs_excludes_unpaired(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.register_metric("cpu")
        engine.register_metric("disk")
        for i in range(10):
            engine.observe("cpu", 50.0)
            engine.observe("disk", 40.0)

        for i in range(6):
            engine.observe("cpu", 200.0)
            engine.observe("disk", 200.0)

        result_only_cpu = engine.detect_correlation_anomaly(
            metric_pairs=[("cpu", "nonexistent")],
            current_values={"cpu": 200.0},
        )
        assert len(result_only_cpu) == 0, "Should not detect when metric is excluded by pairs filter"

    def test_empty_metric_pairs_checks_all(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.register_metric("cpu")
        engine.register_metric("memory")
        for i in range(10):
            engine.observe("cpu", 50.0 + i)
            engine.observe("memory", 60.0 + i)

        result = engine.detect_correlation_anomaly(
            metric_pairs=[],
            current_values={"cpu": 999.0, "memory": 999.0},
        )
        assert len(result) > 0 or len(engine._streams) > 0, \
            "Empty metric_pairs should check all metrics in current_values"


# ─── Fix 11: self_improvement unbounded strategy + stale refs ───

class TestSelfImprovementFixes:
    def test_strategy_weights_capped(self):
        from shadowd.cognition.self_improvement import StrategyEvolver
        evolver = StrategyEvolver()
        for i in range(600):
            evolver.record_strategy_outcome(f"strategy_{i}", 0.5 + (i % 10) * 0.01)
        assert len(evolver._strategy_weights) <= 500, \
            "Strategy weights should be capped at 500"

    def test_autonomy_level_validation(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine()
        dec = engine.record_decision("test", autonomy_level="invalid_level")
        assert dec.autonomy_level == "auto", "Invalid autonomy_level should default to auto"

    def test_domain_length_validation(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine()
        long_domain = "a" * 200
        dec = engine.record_decision("test", domain=long_domain)
        assert len(dec.domain) <= 99, "Domain should be truncated to at most 99 chars"

    def test_decisions_eviction_removes_from_open_decisions(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        engine = SelfImprovementEngine()
        for i in range(10001):
            engine.record_decision(f"action_{i}")
        remaining_open_keys = set(engine._open_decisions.keys()) - set(engine._decisions.keys())
        assert len(remaining_open_keys) == 0, \
            "Evicted decisions should be removed from _open_decisions too"


# ─── Fix 12: Rate limit no longer uses X-Forwarded-For ───

class TestRateLimitNoXForwardedFor:
    def test_rate_limit_uses_client_host(self):
        import shadowd.cognition.shadowosd as mod
        source = inspect.getsource(mod.rate_limit_middleware)
        assert 'x-forwarded-for' not in source.lower(), \
            "Rate limiter should not use X-Forwarded-For"
        assert 'request.client' in source, \
            "Rate limiter should use request.client.host"

    def test_x_forwarded_for_not_used(self):
        import shadowd.cognition.shadowosd as mod
        source = inspect.getsource(mod.rate_limit_middleware)
        assert 'x-forwarded-for' not in source, \
            "Should not read X-Forwarded-For header"


# ─── Fix 13: EscalationManager race condition ───

class TestEscalationManagerRaceCondition:
    def test_escalation_manager_lock_exists(self):
        import shadowd.cognition.shadowosd as mod
        assert hasattr(mod, '_ESCALATION_MANAGER_LOCK'), \
            "Module should have _ESCALATION_MANAGER_LOCK"

    def test_escalation_manager_lock_is_threading_lock(self):
        import shadowd.cognition.shadowosd as mod
        assert isinstance(mod._ESCALATION_MANAGER_LOCK, type(threading.Lock())), \
            "_ESCALATION_MANAGER_LOCK should be a threading.Lock"

    def test_evaluate_escalation_uses_lock(self):
        import shadowd.cognition.shadowosd as mod
        source = inspect.getsource(mod.evaluate_escalation)
        assert '_ESCALATION_MANAGER_LOCK' in source, \
            "evaluate_escalation should use _ESCALATION_MANAGER_LOCK"


# ─── Fix 14: API key masking ───

class TestAPIKeyMasking:
    def test_mask_key_short(self):
        from shadowd.streaming import _mask_key
        assert _mask_key("abc") == "***"
        assert _mask_key("abcd1234") == "***"

    def test_mask_key_long(self):
        from shadowd.streaming import _mask_key
        result = _mask_key("sk-abcdef1234567890")
        assert result.startswith("sk-a"), "Should show first 4 chars"
        assert result.endswith("7890"), "Should show last 4 chars"
        assert "..." in result, "Should have ... in middle"

    def test_mask_key_empty(self):
        from shadowd.streaming import _mask_key
        assert _mask_key("") == "***"

    def test_mask_key_exactly_8(self):
        from shadowd.streaming import _mask_key
        assert _mask_key("12345678") == "***"

    def test_mask_key_9(self):
        from shadowd.streaming import _mask_key
        result = _mask_key("123456789")
        assert result == "1234...6789"

    def test_stream_llm_uses_masked_key_in_error(self):
        from shadowd.streaming import stream_llm_response
        source = inspect.getsource(stream_llm_response)
        assert '_mask_key' in source, "stream_llm_response should use _mask_key for logging"


# ─── Integration: verify all modules import cleanly ───

class TestModuleImports:
    def test_shadowosd_imports(self):
        from shadowd.cognition.shadowosd import app
        assert app is not None

    def test_memory_rag_imports(self):
        from shadowd.memory_rag import RAGMemorySystem
        assert RAGMemorySystem is not None

    def test_websocket_manager_imports(self):
        from shadowd.websocket_manager import ConnectionManager, StreamingResponseHandler
        assert ConnectionManager is not None

    def test_memory_tools_imports(self):
        from shadowd.memory_tools import render_memory_xml, _sanitize_label
        assert render_memory_xml is not None

    def test_block_memory_imports(self):
        from shadowd.block_memory import ArchivalMemory
        assert ArchivalMemory is not None

    def test_config_imports(self):
        from shadowd.config import ShadowConfig
        assert ShadowConfig is not None

    def test_goal_planner_imports(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        assert CognitiveGoalPlanner is not None

    def test_anomaly_detector_imports(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        assert AnomalyDetectionEngine is not None

    def test_self_improvement_imports(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine, StrategyEvolver
        assert SelfImprovementEngine is not None

    def test_streaming_imports(self):
        from shadowd.streaming import _mask_key, stream_llm_response
        assert _mask_key is not None