"""
Session 5 Sprint 28 — LOW Severity Audit Fixes (Cognition + Legacy Modules)

Fixes:
  L1: goal_planner.py — _load() re-validates dependencies with len(str(d)) < 100 filter
  L2: anomaly_detector.py — _anomalies uses collections.deque(maxlen=1000) for O(1) eviction
  L3: shadowosd.py — Module-level singleton comment documenting pattern and thread-safety
  L4: memory_rag.py — _classify_memory_type() uses word-boundary-aware keywords
  L5: event_bus.py — get_event_bus() uses threading.Lock for thread-safe singleton
  L6: block_memory.py — ArchivalMemory.search() access_count side effect documented
  L7: block_memory.py — WorkingMemory.add_message() trims before appending (no temp overflow)
  L8: config.py — from_yaml() type-checks numeric fields before overriding defaults
"""

import inspect
import os
import tempfile
import threading

import pytest


# ─── L1: goal_planner._load() re-validates dependencies ───

class TestGoalPlannerLoadDependenciesFilter:
    def test_load_filters_overlong_dependencies(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner, PlanStep, StepStatus
        gp = CognitiveGoalPlanner()
        goal = gp.create_goal("Test")
        result = gp.decompose_goal(goal.goal_id, [
            {"description": "step1"},
            {"description": "step2", "dependencies": ["x" * 100]},
            {"description": "step3", "dependencies": ["valid_dep"]},
        ])
        step2 = result.steps[1]
        assert len(step2.dependencies) == 0, f"Overlong dep should be filtered, got {step2.dependencies}"
        step3 = result.steps[2]
        assert step3.dependencies == ["valid_dep"]

    def test_load_revalidates_dependencies_from_disk(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        with tempfile.TemporaryDirectory() as td:
            gp1 = CognitiveGoalPlanner(data_dir=td)
            goal = gp1.create_goal("Persist test")
            gp1.decompose_goal(goal.goal_id, [
                {"description": "step with evil dep", "dependencies": ["a" * 120]},
            ])
            gp2 = CognitiveGoalPlanner(data_dir=td)
            loaded = gp2.get_goal(goal.goal_id)
            assert loaded is not None
            loaded_deps = loaded["steps"][0]["dependencies"]
            assert len(loaded_deps) == 0, f"Overlong deps should be filtered on load, got {loaded_deps}"

    def test_load_handles_non_list_dependencies(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        with tempfile.TemporaryDirectory() as td:
            import json
            gp = CognitiveGoalPlanner(data_dir=td)
            goal = gp.create_goal("Non-list deps")
            goal_id = goal.goal_id
            path = os.path.join(td, "goals.json")
            with open(path, 'r') as f:
                data = json.loads(f.read())
            data[goal_id]["steps"].append({
                "step_id": "test_step",
                "description": "bad deps",
                "dependencies": "not_a_list",
            })
            with open(path, 'w') as f:
                f.write(json.dumps(data))
            gp2 = CognitiveGoalPlanner(data_dir=td)
            loaded = gp2.get_goal(goal_id)
            assert loaded is not None
            bad_step = [s for s in loaded["steps"] if s.get("step_id") == "test_step"]
            assert len(bad_step) == 1
            assert bad_step[0]["dependencies"] == []


# ─── L2: anomaly_detector deque for O(1) eviction ───

class TestAnomalyDetectorDeque:
    def test_anomalies_is_deque(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        from collections import deque
        engine = AnomalyDetectionEngine()
        assert isinstance(engine._anomalies, deque), f"Expected deque, got {type(engine._anomalies)}"

    def test_deque_maxlen_is_1000(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        assert engine._anomalies.maxlen == 1000

    def test_deque_auto_eviction(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine, Anomaly, AnomalyType, AnomalySeverity
        engine = AnomalyDetectionEngine()
        for i in range(1050):
            engine._anomalies.append(Anomaly(
                anomaly_type=AnomalyType.STATISTICAL,
                severity=AnomalySeverity.LOW,
                metric=f"test_{i}",
            ))
        assert len(engine._anomalies) <= 1000

    def test_record_anomaly_no_manual_truncation(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        source = inspect.getsource(AnomalyDetectionEngine._record_anomaly)
        assert "_anomalies[-self._MAX_ANOMALIES:]" not in source, \
            "_record_anomaly should use deque auto-eviction, not manual slicing"

    def test_get_anomalies_works_with_deque(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.register_metric("cpu")
        for i in range(10):
            engine.observe("cpu", 100.0 + i)
        anomaly = engine.observe("cpu", 9999.0)
        assert anomaly is not None
        results = engine.get_anomalies()
        assert len(results) >= 1


# ─── L3: shadowosd singleton pattern comment ───

class TestShadowosdSingletonComment:
    def test_module_level_singleton_has_comment(self):
        import shadowd.cognition.shadowosd as mod
        source = inspect.getsource(mod)
        assert "singleton" in source.lower(), "Module should document singleton pattern"
        assert "_ESCALATION_MANAGER_LOCK" in source, "Should reference lock for thread-safety"


# ─── L4: memory_rag word-boundary keywords ───

class TestMemoryRagClassifyMemoryType:
    def test_will_with_space_is_commitment(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        assert mem._classify_memory_type("I will do it") == "commitment"

    def test_standalone_will_not_commitment(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        result = mem._classify_memory_type("You have strong willpower")
        assert result != "commitment", f"'will' inside 'willpower' should not match commitment, got {result}"

    def test_found_with_space_is_learning(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        assert mem._classify_memory_type("I found the bug") == "learning"

    def test_found_without_space_not_learning(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        result = mem._classify_memory_type("The newfound approach")
        assert result != "learning", f"'found' inside 'newfound' should not match learning, got {result}"

    def test_call_with_space_commitment(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        assert mem._classify_memory_type("I will call you") == "commitment"

    def test_call_inside_word_not_commitment(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        result = mem._classify_memory_type("This is a callback function")
        assert result != "commitment", f"'call' inside 'callback' should not match commitment, got {result}"

    def test_should_with_space_is_insight(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        assert mem._classify_memory_type("I should review this") == "insight"

    def test_should_without_space_not_insight(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        result = mem._classify_memory_type("This is shouldering the burden")
        assert result != "insight", f"'should' inside 'shouldering' should not match insight, got {result}"

    def test_later_with_space_commitment(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        assert mem._classify_memory_type("I'll do it later ") == "commitment"

    def test_plain_observation(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        assert mem._classify_memory_type("The sky is blue") == "observation"

    def test_tomorrow_with_space(self):
        from shadowd.memory_rag import RAGMemorySystem
        mem = RAGMemorySystem.__new__(RAGMemorySystem)
        assert mem._classify_memory_type("I'll call you tomorrow") == "commitment"


# ─── L5: event_bus thread-safe singleton ───

class TestEventBusThreadSafeSingleton:
    def test_get_event_bus_returns_same_instance(self):
        from shadowd.event_bus import get_event_bus, EventBus
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2, "get_event_bus should return the same singleton instance"

    def test_module_has_lock(self):
        import shadowd.event_bus as mod
        assert hasattr(mod, '_event_bus_lock'), "Module should have _event_bus_lock"
        assert isinstance(mod._event_bus_lock, type(threading.Lock()))

    def test_get_event_bus_uses_lock(self):
        from shadowd.event_bus import get_event_bus
        source = inspect.getsource(get_event_bus)
        assert "_event_bus_lock" in source, "get_event_bus should acquire _event_bus_lock"

    def test_concurrent_get_event_bus(self):
        import shadowd.event_bus as mod
        from shadowd.event_bus import get_event_bus, EventBus
        results = []
        lock = threading.Lock()
        buses = []

        def get_bus():
            bus = get_event_bus()
            with lock:
                buses.append(bus)

        threads = [threading.Thread(target=get_bus) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(buses) == 10
        assert all(b is buses[0] for b in buses), "All threads should get the same singleton"


# ─── L6: ArchivalMemory.search() side effect documentation ───

class TestArchivalSearchSideEffectDoc:
    def test_search_docstring_documents_access_count_mutation(self):
        from shadowd.block_memory import ArchivalMemory
        source = inspect.getsource(ArchivalMemory.search)
        assert "access_count" in source, "search() docstring should document access_count side effect"
        assert "side effect" in source.lower() or "mutates" in source.lower(), \
            "search() docstring should document the mutation as a side effect"

    def test_search_increments_access_count(self):
        from shadowd.block_memory import ArchivalMemory
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            am = ArchivalMemory(Path(td) / "archival")
            am.insert("python test content", {"source": "test"})
            am.insert("java test content", {"source": "test"})
            results = am.search("python")
            assert len(results) >= 1
            assert results[0]["access_count"] == 1


# ─── L7: WorkingMemory.add_message() trims before append ───

class TestWorkingMemoryTrimBeforeAppend:
    def test_no_temporary_overflow(self):
        from shadowd.block_memory import WorkingMemory
        wm = WorkingMemory(max_messages=3)
        wm.add_message("user", "msg1")
        wm.add_message("user", "msg2")
        wm.add_message("user", "msg3")
        assert len(wm.messages) == 3
        wm.add_message("user", "msg4")
        assert len(wm.messages) == 3, f"Should stay at max_messages=3, got {len(wm.messages)}"
        assert wm.messages[-1]["content"] == "msg4"
        assert wm.messages[0]["content"] == "msg2"

    def test_add_message_preserves_order_at_capacity(self):
        from shadowd.block_memory import WorkingMemory
        wm = WorkingMemory(max_messages=2)
        wm.add_message("user", "a")
        wm.add_message("user", "b")
        wm.add_message("user", "c")
        assert [m["content"] for m in wm.messages] == ["b", "c"]

    def test_add_message_never_exceeds_max(self):
        from shadowd.block_memory import WorkingMemory
        wm = WorkingMemory(max_messages=5)
        for i in range(20):
            wm.add_message("user", f"msg{i}")
            assert len(wm.messages) <= 5, f"After adding msg {i}, len={len(wm.messages)} exceeds max"


# ─── L8: config from_yaml type validation ───

class TestConfigYamlTypeValidation:
    def test_valid_temperature_float(self):
        from shadowd.config import ShadowConfig
        from pathlib import Path
        import tempfile
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.TemporaryDirectory() as td:
            config_file = Path(td) / "test_config.yaml"
            config_file.write_text(
                "llm:\n  temperature: 0.3\n  max_tokens: 4096\nserver:\n  port: 9090\n"
            )
            config = ShadowConfig.from_yaml(config_file)
            assert config.llm.temperature == 0.3
            assert config.llm.max_tokens == 4096
            assert config.server.port == 9090

    def test_invalid_temperature_string_ignored(self):
        from shadowd.config import ShadowConfig
        from pathlib import Path
        import tempfile
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.TemporaryDirectory() as td:
            config_file = Path(td) / "test_config.yaml"
            config_file.write_text("llm:\n  temperature: \"not_a_number\"\n")
            config = ShadowConfig.from_yaml(config_file)
            assert config.llm.temperature == 0.7, "String temperature should be ignored, default kept"

    def test_invalid_port_string_ignored(self):
        from shadowd.config import ShadowConfig
        from pathlib import Path
        import tempfile
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.TemporaryDirectory() as td:
            config_file = Path(td) / "test_config.yaml"
            config_file.write_text("server:\n  port: \"not_a_port\"\n")
            config = ShadowConfig.from_yaml(config_file)
            assert config.server.port == 8765, "String port should be ignored, default kept"

    def test_invalid_max_tokens_float_ignored(self):
        from shadowd.config import ShadowConfig
        from pathlib import Path
        import tempfile
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.TemporaryDirectory() as td:
            config_file = Path(td) / "test_config.yaml"
            config_file.write_text("llm:\n  max_tokens: 3.14\n")
            config = ShadowConfig.from_yaml(config_file)
            assert config.llm.max_tokens == 2048, "Float max_tokens should be ignored, default kept"

    def test_valid_int_as_temperature(self):
        from shadowd.config import ShadowConfig
        from pathlib import Path
        import tempfile
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.TemporaryDirectory() as td:
            config_file = Path(td) / "test_config.yaml"
            config_file.write_text("llm:\n  temperature: 1\n")
            config = ShadowConfig.from_yaml(config_file)
            assert config.llm.temperature == 1.0

    def test_invalid_core_limit_string_ignored(self):
        from shadowd.config import ShadowConfig
        from pathlib import Path
        import tempfile
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.TemporaryDirectory() as td:
            config_file = Path(td) / "test_config.yaml"
            config_file.write_text("memory:\n  core_limit: \"big\"\n")
            config = ShadowConfig.from_yaml(config_file)
            assert config.memory.core_limit == 2000

    def test_model_string_still_works(self):
        from shadowd.config import ShadowConfig
        from pathlib import Path
        import tempfile
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        with tempfile.TemporaryDirectory() as td:
            config_file = Path(td) / "test_config.yaml"
            config_file.write_text("llm:\n  model: llama3\n")
            config = ShadowConfig.from_yaml(config_file)
            assert config.llm.model == "llama3"

    def test_from_yaml_has_expect_type_helper(self):
        from shadowd.config import ShadowConfig
        source = inspect.getsource(ShadowConfig.from_yaml)
        assert "_expect_type" in source, "from_yaml should have _expect_type helper for type validation"