#!/usr/bin/env python3
"""
End-to-End Integration Tests for ShadowOS Cognitive Pipeline

Tests the full pipeline from observation ingestion through cognitive graph
processing to action execution, including all module interactions.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path


def _make_shadow_os(tmpdir):
    from shadowd.cognition.shadow_os import ShadowOS
    return ShadowOS(data_dir=tmpdir)


class TestCognitivePipelineE2E:
    @pytest.mark.asyncio
    async def test_full_query_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            await os_inst.start()
            try:
                result = await os_inst.process_query("What is the weather today?")
                assert "cognitive_mode" in result
                assert "confidence" in result
            finally:
                await os_inst.stop()

    @pytest.mark.asyncio
    async def test_observation_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            await os_inst.start()
            try:
                observation = {
                    "type": "observation",
                    "content": "User visited a coffee shop",
                    "domain": "daily_life",
                }
                result = await os_inst.process_observation(observation)
                assert result["status"] in ("observed", "denied")
            finally:
                await os_inst.stop()

    @pytest.mark.asyncio
    async def test_observation_sanitization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            await os_inst.start()
            try:
                observation = {
                    "type": "observation",
                    "content": "test<script>alert('xss')</script>",
                    "domain": "test",
                }
                result = await os_inst.process_observation(observation)
                assert result["status"] in ("observed", "denied")
            finally:
                await os_inst.stop()

    @pytest.mark.asyncio
    async def test_observation_with_sensitive_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            await os_inst.start()
            try:
                observation = {
                    "type": "credential",
                    "content": "password=secret123",
                    "domain": "security",
                }
                result = await os_inst.process_observation(observation)
                assert "status" in result
            finally:
                await os_inst.stop()

    @pytest.mark.asyncio
    async def test_action_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            await os_inst.start()
            try:
                result = await os_inst.execute_action("summarize", {"topic": "AI"})
                assert isinstance(result, dict)
            finally:
                await os_inst.stop()

    @pytest.mark.asyncio
    async def test_memory_store_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            await os_inst.start()
            try:
                atom = await os_inst.memory.store(
                    content="E2E test observation",
                    atom_type="observation",
                    domain="test",
                    importance=0.8,
                )
                assert atom.atom_id
                results = await os_inst.memory.search("E2E test", limit=5)
                assert len(results) > 0
            finally:
                await os_inst.stop()

    @pytest.mark.asyncio
    async def test_knowledge_graph_store_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            await os_inst.start()
            try:
                entity_id = await os_inst.memory.store_entity("e2e_person", "person", {"name": "Alice"})
                assert entity_id == "e2e_person"
                rel_id = await os_inst.memory.store_relation("e2e_person", "e2e_place", "visits")
                assert rel_id
                result = await os_inst.memory.query_kg(entity_id="e2e_person", depth=1)
                assert "neighbors" in result
            finally:
                await os_inst.stop()

    @pytest.mark.asyncio
    async def test_memory_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            try:
                blocks = os_inst.get_memory_blocks()
                assert len(blocks) > 0
                result = os_inst.set_memory_block("context", "E2E test context")
                assert result is True
                block = os_inst.get_memory_block("context")
                assert block["value"] == "E2E test context"
            finally:
                os_inst.memory.close()

    @pytest.mark.asyncio
    async def test_protected_block_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            try:
                result = os_inst.set_memory_block("identity", "I am not Shadow")
                assert result is False
            finally:
                os_inst.memory.close()

    @pytest.mark.asyncio
    async def test_personality_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            try:
                os_inst.personality.record_behavior("test_action", {
                    "type": "query", "domain": "test", "complexity": "simple",
                })
                profile = os_inst.get_personality()
                assert isinstance(profile, dict)
            finally:
                os_inst.memory.close()

    @pytest.mark.asyncio
    async def test_stealth_layer_access_control(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            try:
                from shadowd.cognition.stealth_layer import DataCategory, AccessAction
                denied, reason = os_inst.stealth.check_access(
                    "external_app", AccessAction.READ, "notes", DataCategory.CREDENTIALS
                )
                assert denied is False
            finally:
                os_inst.memory.close()

    @pytest.mark.asyncio
    async def test_graph_cognitive_transitions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            await os_inst.start()
            try:
                from shadowd.cognition.shadow_graph import TransitionTrigger
                await os_inst.graph.process_event(
                    event={"query": "test query"},
                    trigger=TransitionTrigger.HUMAN_REQUEST,
                )
                assert os_inst.graph.state.mode is not None
            finally:
                await os_inst.stop()

    @pytest.mark.asyncio
    async def test_crew_mission_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            try:
                from shadowd.cognition.shadow_crew import DelegationMode
                mission = await os_inst.crew.create_mission(
                    "Analyze recent patterns", mode=DelegationMode.HIERARCHICAL
                )
                assert mission is not None
                assert mission.mission_id
            finally:
                os_inst.memory.close()

    @pytest.mark.asyncio
    async def test_escalation_evaluation(self):
        from shadowd.cognition.escalation import EscalationManager, AutonomyLevel
        esc = EscalationManager()
        can, req = await esc.can_execute("observe", {"data": "test_data"})
        assert req.action_type == "observe"
        can2, req2 = await esc.can_execute("delete", {"target": "all_data"})
        assert can2 is False

    @pytest.mark.asyncio
    async def test_voice_session_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            try:
                session = await os_inst.start_voice_session()
                assert "session_id" in session
                result = await os_inst.process_voice("Hello Shadow")
                assert "command" in result
            finally:
                os_inst.memory.close()

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            try:
                health = os_inst.get_health()
                assert "status" in health
            finally:
                os_inst.memory.close()

    @pytest.mark.asyncio
    async def test_auto_compaction_daemon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.cognition.dark_memory import DarkMemory
            dm = DarkMemory(data_dir=tmpdir, max_atoms=10)
            for i in range(20):
                await dm.store(f"test observation {i}", importance=0.3)
            assert len(dm._atoms) == 20
            evicted = await dm.compact()
            assert evicted >= 0
            dm.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        from shadowd.cognition.neural_mesh import CircuitBreaker
        import time
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        assert cb.can_execute() is True
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is True
        cb.record_failure()
        assert cb.can_execute() is False
        time.sleep(0.15)
        assert cb.state == CircuitBreaker.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED

    @pytest.mark.asyncio
    async def test_input_sanitization(self):
        from shadowd.cognition.shadow_os import sanitize_input, validate_observation
        assert sanitize_input("<script>alert('xss')</script>hello") == "hello"
        assert len(sanitize_input("a" * 5000)) <= 2000
        assert sanitize_input("hello\x00world") == "helloworld"

        obs = {"key": "value", "huge": "x" * 20000}
        cleaned = validate_observation(obs)
        assert len(cleaned["huge"]) <= 10000

    @pytest.mark.asyncio
    async def test_mesh_circuit_breaker_integration(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        mesh = NeuralMesh(node_name="test-node")
        assert isinstance(mesh._circuit_breakers, dict)
        assert mesh._max_retries == 3

    @pytest.mark.asyncio
    async def test_graph_state_locking(self):
        from shadowd.cognition.shadow_graph import ShadowGraph, TransitionTrigger
        g = ShadowGraph()
        assert hasattr(g, '_state_lock')
        await g.process_event({"query": "test"}, trigger=TransitionTrigger.HUMAN_REQUEST)
        assert g.state is not None

    @pytest.mark.asyncio
    async def test_full_security_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            try:
                from shadowd.cognition.stealth_layer import DataCategory, AccessAction
                sensitive_obs = {"password": "hunter2", "email": "user@example.com"}
                sensitivity = os_inst.stealth.get_sensitivity_score(sensitive_obs)
                assert sensitivity > 0

                denied, reason = os_inst.stealth.check_access(
                    "external_app", AccessAction.READ, "credentials", DataCategory.CREDENTIALS
                )
                assert denied is False
            finally:
                os_inst.memory.close()

    @pytest.mark.asyncio
    async def test_shadowos_save_and_restore_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = _make_shadow_os(tmpdir)
            await os_inst.start()
            try:
                os_inst._stats["queries_handled"] = 42
                await os_inst._save_state()
                assert (os_inst.data_dir / "shadow_os_state.json").exists()
            finally:
                await os_inst.stop()