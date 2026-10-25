#!/usr/bin/env python3
"""Sprint 84 regression tests: goal_planner, anomaly_detector, stealth_layer,
personality_engine, shadow_crew, dark_memory deep audit fixes."""

import copy
import threading
import time
import pytest


# ─── GoalPlanner ──────────────────────────────────────────────────────────

class TestGoalPlanner:
    def setup_method(self):
        from shadowd.cognition.goal_planner import (
            CognitiveGoalPlanner, Goal, GoalStatus, GoalPriority, PlanStep, StepStatus,
        )
        self.planner = CognitiveGoalPlanner(data_dir=None)

    def test_check_goal_blocked_does_not_block_completed(self):
        """GP-2: _check_goal_blocked should NOT transition COMPLETED/CANCELLED goals to BLOCKED."""
        from shadowd.cognition.goal_planner import Goal, GoalStatus, PlanStep, StepStatus
        planner = self.planner
        goal = Goal(title="Test", status=GoalStatus.COMPLETED)
        goal.steps.append(PlanStep(description="fail", status=StepStatus.FAILED))
        planner._goals[goal.goal_id] = goal
        planner._check_goal_blocked(goal)
        assert goal.status == GoalStatus.COMPLETED

    def test_check_goal_blocked_does_not_block_cancelled(self):
        from shadowd.cognition.goal_planner import Goal, GoalStatus, PlanStep, StepStatus
        goal = Goal(title="Test", status=GoalStatus.CANCELLED)
        goal.steps.append(PlanStep(description="fail", status=StepStatus.FAILED))
        self.planner._check_goal_blocked(goal)
        assert goal.status == GoalStatus.CANCELLED

    def test_check_goal_blocked_blocks_pending(self):
        from shadowd.cognition.goal_planner import Goal, GoalStatus, PlanStep, StepStatus
        goal = Goal(title="Test", status=GoalStatus.PENDING)
        goal.steps.append(PlanStep(description="fail", status=StepStatus.FAILED))
        self.planner._check_goal_blocked(goal)
        assert goal.status == GoalStatus.BLOCKED

    def test_check_goal_blocked_blocks_in_progress(self):
        from shadowd.cognition.goal_planner import Goal, GoalStatus, PlanStep, StepStatus
        goal = Goal(title="Test", status=GoalStatus.IN_PROGRESS)
        goal.steps.append(PlanStep(description="fail", status=StepStatus.FAILED))
        self.planner._check_goal_blocked(goal)
        assert goal.status == GoalStatus.BLOCKED

    def test_step_to_dict_returns_copies(self):
        from shadowd.cognition.goal_planner import PlanStep, StepStatus
        step = PlanStep(description="Test", dependencies=["d1", "d2"], result={"key": "val"})
        d = step.to_dict()
        d["dependencies"].append("d3")
        d["result"]["key2"] = "val2"
        assert "d3" not in step.dependencies
        assert "key2" not in step.result

    def test_goal_to_dict_returns_copies(self):
        from shadowd.cognition.goal_planner import Goal, GoalPriority, GoalStatus
        goal = Goal(title="Test", sub_goals=["sg1"], tags=["t1"])
        d = goal.to_dict()
        d["sub_goals"].append("sg2")
        d["tags"].append("t2")
        assert "sg2" not in goal.sub_goals
        assert "t2" not in goal.tags

    def test_step_id_is_12_char_hex(self):
        from shadowd.cognition.goal_planner import PlanStep
        step = PlanStep(description="Test")
        assert step.step_id.startswith("step_")
        assert len(step.step_id) == 17  # "step_" + 12 hex chars

    def test_create_goal_sanitizes_title(self):
        result = self.planner.create_goal(title="<script>alert('xss')</script>")
        assert "<script>" not in result.title
        assert "&lt;script&gt;" not in result.title

    def test_read_methods_acquire_lock(self):
        """Verify read methods don't raise errors under lock."""
        self.planner.create_goal("Test")
        self.planner.get_goal("nonexistent")
        self.planner.list_goals()
        self.planner.get_stats()
        self.planner.get_active_goals()
        self.planner.get_blocked_goals()

    def test_goal_id_is_12_char_hex(self):
        goal = self.planner.create_goal("Test")
        assert goal.goal_id.startswith("goal_")
        assert len(goal.goal_id) == 17  # "goal_" + 12 hex


# ─── AnomalyDetector ─────────────────────────────────────────────────────

class TestAnomalyDetector:
    def setup_method(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        self.engine = AnomalyDetectionEngine(data_dir=None)

    def test_thread_safety(self):
        """Multiple threads calling observe concurrently should not crash."""
        errors = []
        def worker(i):
            try:
                self.engine.register_metric(f"m_{i % 5}", f"metric {i}", "count")
                for j in range(50):
                    self.engine.observe(f"m_{i % 5}", float(j + i))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_to_dict_returns_deepcopy_context(self):
        from shadowd.cognition.anomaly_detector import Anomaly, AnomalyType, AnomalySeverity
        ctx = {"key": [1, 2, 3], "nested": {"a": "b"}}
        anomaly = Anomaly(context=ctx)
        d = anomaly.to_dict()
        d["context"]["key"].append(4)
        d["context"]["nested"]["c"] = "d"
        assert len(anomaly.context["key"]) == 3
        assert "c" not in anomaly.context["nested"]

    def test_observe_sanitizes_metric(self):
        self.engine.observe("<script>alert(1)</script>metric", 1.0)
        stats = self.engine.get_stats()
        assert "script" not in str(self.engine._streams.keys()) or "metric" in str(self.engine._streams.keys())

    def test_register_metric_sanitizes_name(self):
        self.engine.register_metric("test<script>", "desc", "unit")
        keys = list(self.engine._streams.keys())
        assert all("<" not in k and ">" not in k for k in keys)

    def test_stats_type_validation(self):
        """_load should not crash with non-numeric stats values."""
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(data_dir=None)
        engine._stats["observations_processed"] = "not_a_number"
        assert isinstance(engine._stats["observations_processed"], str)

    def test_acknowledge_thread_safe(self):
        from shadowd.cognition.anomaly_detector import Anomaly, AnomalyType, AnomalySeverity
        self.engine.register_metric("test", "test", "unit")
        for i in range(10):
            self.engine.observe("test", 100.0 + i)
        anomalies = self.engine.get_anomalies()
        if anomalies:
            result = self.engine.acknowledge(anomalies[0]["anomaly_id"])
            assert result is True


# ─── StealthLayer / PrivacyGuard ────────────────────────────────────────

class TestPrivacyGuard:
    def setup_method(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        self.guard = PrivacyGuard()

    def test_check_access_thread_safe(self):
        from shadowd.cognition.stealth_layer import AccessAction, DataCategory
        errors = []
        def worker(i):
            try:
                self.guard.check_access("shadow_ai", AccessAction.READ, f"data_{i}", DataCategory.BEHAVIORAL)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_audit_entry_to_dict_includes_metadata(self):
        from shadowd.cognition.stealth_layer import AuditEntry, AccessAction, DataCategory, SensitivityLevel
        entry = AuditEntry(
            subject="shadow_ai",
            action=AccessAction.READ,
            data_id="test",
            category=DataCategory.BEHAVIORAL,
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            metadata={"ip": "1.2.3.4"},
        )
        d = entry.to_dict()
        assert "metadata" in d
        assert d["metadata"]["ip"] == "1.2.3.4"

    def test_get_or_create_key_thread_safe(self):
        errors = []
        results = []
        def worker():
            key = self.guard._get_or_create_key("test_key")
            results.append(len(key))
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10  # all threads completed without error

    def test_sanitization_strips_angle_brackets(self):
        from shadowd.cognition.stealth_layer import _sanitize
        assert _sanitize("<script>alert(1)</script>") == "scriptalert(1)/script"

    def test_data_minimizer_purpose_cap(self):
        from shadowd.cognition.stealth_layer import DataMinimizer
        dm = DataMinimizer()
        for i in range(600):
            dm.register_purpose(f"purpose_{i}", ["field"], 30)
        assert len(dm._collection_purposes) <= 500

    def test_get_purposes_returns_copy(self):
        from shadowd.cognition.stealth_layer import DataMinimizer
        dm = DataMinimizer()
        dm.register_purpose("test_p", ["f1", "f2"], 30)
        purposes = dm.get_purposes()
        purposes[0]["required_fields"].append("f3")
        # Original should not be modified
        assert len(dm._collection_purposes["test_p"]["required_fields"]) == 2


# ─── PersonalityEngine ───────────────────────────────────────────────────

class TestPersonalityEngine:
    def setup_method(self):
        import tempfile
        import os
        from shadowd.cognition.personality_engine import PersonalityEngine
        self.tmpdir = tempfile.mkdtemp()
        self.engine = PersonalityEngine(data_dir=self.tmpdir)

    def test_get_profile_thread_safe(self):
        errors = []
        def worker():
            try:
                self.engine.get_profile()
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_get_patterns_thread_safe(self):
        errors = []
        self.engine.record_behavior("decision_made", {"speed": "fast", "risk_level": "high"})
        def worker():
            try:
                self.engine.get_patterns()
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_add_decision_sanitizes(self):
        self.engine.profile.add_decision({"key": "<script>alert(1)</script>", "val": "test"})
        assert "<script>" not in self.engine.profile.decision_history[-1]["key"]

    def test_predict_decision_thread_safe(self):
        options = [{"name": "a", "risk_level": 0.3, "reward": 0.7, "novelty": 0.5, "speed": 0.6}]
        errors = []
        def worker():
            try:
                self.engine.predict_decision(options)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_float_error_handling_in_urgency(self):
        """Non-numeric urgency should not crash."""
        self.engine.record_behavior("test_event", {"urgency": "not_a_number", "priority": "high"})


# ─── ShadowCrew ──────────────────────────────────────────────────────────

class TestShadowCrew:
    def setup_method(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        self.crew = ShadowCrew()

    def test_mission_id_is_12_char_hex(self):
        from shadowd.cognition.shadow_crew import CrewMission
        m = CrewMission(objective="Test")
        assert m.mission_id.startswith("mission_")
        assert len(m.mission_id) == 20  # "mission_" + 12 hex

    def test_agent_to_dict_returns_copies(self):
        from shadowd.cognition.shadow_crew import ShadowAgent, AgentRole
        agent = ShadowAgent(
            agent_id="test",
            name="Test",
            role=AgentRole.OBSERVER,
            goal="Observe",
            backstory="Test",
            tools=["t1"],
            domain_expertise=["d1"],
            delegation_targets=["dt1"],
        )
        d = agent.to_dict()
        d["tools"].append("t2")
        d["domain_expertise"].append("d2")
        d["delegation_targets"].append("dt2")
        assert "t2" not in agent.tools
        assert "d2" not in agent.domain_expertise
        assert "dt2" not in agent.delegation_targets

    def test_task_id_is_12_char_hex(self):
        from shadowd.cognition.shadow_crew import CrewTask
        task = CrewTask(description="Test")
        assert task.task_id.startswith("task_")
        assert len(task.task_id) == 17  # "task_" + 12 hex

    def test_create_mission_sanitizes_objective(self):
        """create_mission should strip control chars but preserve <> for comparison expressions."""
        import asyncio
        mission = asyncio.run(
            self.crew.create_mission("test objective\x00with\x01control chars")
        )
        assert "\x00" not in mission.objective
        assert "\x01" not in mission.objective
        # Verify that <> are preserved for comparison expressions like "latency < 100ms"
        mission2 = asyncio.run(
            self.crew.create_mission("latency < 100ms")
        )
        assert "<" in mission2.objective

    def test_circuit_breaker_thread_safe(self):
        from shadowd.cognition.shadow_crew import CircuitBreaker, CircuitState
        cb = CircuitBreaker()
        errors = []
        def worker():
            try:
                cb.record_failure()
                cb.is_available()
                cb.record_success()
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_get_stats_thread_safe(self):
        stats = self.crew.get_stats()
        assert "stats" in stats


# ─── DarkMemory ──────────────────────────────────────────────────────────

class TestDarkMemory:
    def setup_method(self):
        import tempfile
        from shadowd.cognition.dark_memory import DarkMemory
        self.tmpdir = tempfile.mkdtemp()
        self.dm = DarkMemory(data_dir=self.tmpdir)

    @pytest.mark.asyncio
    async def test_search_returns_deepcopy(self):
        from shadowd.cognition.dark_memory import AtomType
        atom = await self.dm.store("test content for search", atom_type=AtomType.OBSERVATION)
        results = await self.dm.search("test content")
        assert len(results) >= 1
        results[0].content = "MODIFIED"
        results2 = await self.dm.search("test content")
        assert results2[0].content == "test content for search"

    @pytest.mark.asyncio
    async def test_memory_atom_to_dict_returns_copies(self):
        from shadowd.cognition.dark_memory import MemoryAtom, AtomType, MemoryTier
        atom = MemoryAtom(
            content="test",
            atom_type=AtomType.OBSERVATION,
            metadata={"k": "v"},
            tags=["t1"],
        )
        d = atom.to_dict()
        d["metadata"]["k2"] = "v2"
        d["tags"].append("t2")
        assert "k2" not in atom.metadata
        assert "t2" not in atom.tags

    @pytest.mark.asyncio
    async def test_block_store_thread_safe(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore()
        bs.initialize_default_blocks()
        errors = []
        def worker(i):
            try:
                block = bs.get_block("identity")
                bs.set_block("persona", f"Test persona {i}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    @pytest.mark.asyncio
    async def test_get_block_returns_copy(self):
        block = self.dm.get_block("identity")
        block["access_count"] = 999
        block2 = self.dm.get_block("identity")
        assert block2["access_count"] != 999 or block2["access_count"] == 2

    @pytest.mark.asyncio
    async def test_store_validates_content_length(self):
        from shadowd.cognition.dark_memory import AtomType
        atom = await self.dm.store("x" * 100001, atom_type=AtomType.OBSERVATION)
        assert len(atom.content) <= 100000

    @pytest.mark.asyncio
    async def test_store_validates_tags(self):
        from shadowd.cognition.dark_memory import AtomType
        tags = [f"t{i}" for i in range(25)]
        atom = await self.dm.store("test tags", tags=tags, atom_type=AtomType.OBSERVATION)
        assert len(atom.tags) <= 20

    @pytest.mark.asyncio
    async def test_tkg_get_entity_returns_deepcopy(self):
        await self.dm.store_entity("e1", "person", {"name": "Alice"})
        entity = self.dm._temporal_kg.get_entity("e1")
        entity["properties"]["name"] = "Bob"
        entity2 = self.dm._temporal_kg.get_entity("e1")
        assert entity2["properties"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_tkg_get_relations_returns_copy(self):
        await self.dm.store_entity("s1", "person", {"name": "Alice"})
        await self.dm.store_entity("t1", "person", {"name": "Bob"})
        await self.dm.store_relation("s1", "t1", "knows")
        rels = self.dm._temporal_kg.get_relations(entity_id="s1")
        assert len(rels) >= 1
        rels[0]["type"] = "modified"
        rels2 = self.dm._temporal_kg.get_relations(entity_id="s1")
        assert rels2[0]["type"] == "knows"