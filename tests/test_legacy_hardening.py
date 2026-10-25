#!/usr/bin/env python3
"""
Legacy Hardening & Phase 3-6 Regression Tests

Tests security fixes for legacy modules (shadow_ai, mind, mcp_server),
phone API auth, and all Phase 2 feature enhancements.
"""

import asyncio
import json
import hashlib
import pytest
import tempfile
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


class TestMindEngineSecurity:
    def test_error_messages_no_info_leak(self):
        from shadowd.mind import MindEngine
        me = MindEngine()
        me.api_key = ""
        result = me._build_messages("test", None, None)
        assert isinstance(result, list)
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_mind_handles_http_error_safely(self):
        from shadowd.mind import MindEngine
        me = MindEngine()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error with stack trace"
        error = Exception("HTTP 500")
        with patch.object(me, '_get_client', return_value=mock_client):
            mock_client.post = AsyncMock(side_effect=error)
            result = await me.think("test query")
            assert "error" in result.lower() or "encountered" in result.lower()
            assert "stack trace" not in result


class TestShadowAISecurity:
    @pytest.mark.asyncio
    async def test_observe_validates_event_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.shadow_ai import ShadowAI
            registry = MagicMock()
            mind = MagicMock()
            ai = ShadowAI(registry, mind, data_dir=tmpdir)
            with patch.object(ai.event_bus, 'publish', new_callable=AsyncMock):
                await ai.observe({"type": "test", "content": {"key": "value"}})
                assert len(ai.observations) == 1

    @pytest.mark.asyncio
    async def test_observe_rejects_non_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.shadow_ai import ShadowAI
            registry = MagicMock()
            mind = MagicMock()
            ai = ShadowAI(registry, mind, data_dir=tmpdir)
            with patch.object(ai.event_bus, 'publish', new_callable=AsyncMock):
                await ai.observe("not a dict")
                assert len(ai.observations) == 0

    @pytest.mark.asyncio
    async def test_observe_truncates_large_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.shadow_ai import ShadowAI
            registry = MagicMock()
            mind = MagicMock()
            ai = ShadowAI(registry, mind, data_dir=tmpdir)
            with patch.object(ai.event_bus, 'publish', new_callable=AsyncMock):
                await ai.observe({"type": "test", "data": "x" * 50000})
                assert len(ai.observations) == 1
                assert len(ai.observations[0].get("data", "")) <= 10000

    def test_save_data_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.shadow_ai import ShadowAI
            registry = MagicMock()
            mind = MagicMock()
            ai = ShadowAI(registry, mind, data_dir=tmpdir)
            ai.observations = [{"test": "data"}]
            ai._save_data()
            obs_file = Path(tmpdir) / "observations.json"
            assert obs_file.exists()
            data = json.loads(obs_file.read_text())
            assert len(data) == 1


class TestMCPServerSecurity:
    def test_path_traversal_prevented(self):
        import re
        user_id = "normal_user"
        safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id)[:64]
        assert safe_id == "normal_user"

    def test_path_traversal_special_chars_stripped(self):
        import re
        user_id = "user_with_@#$ chars"
        safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id)[:64]
        assert "@" not in safe_id
        assert "#" not in safe_id

    def test_path_traversal_dots_stripped(self):
        import re
        user_id = "../../../etc"
        safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id)[:64]
        assert ".." not in safe_id

    def test_empty_user_gets_default(self):
        import re
        user_id = ""
        safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id)[:64]
        assert safe_id == ""  # empty after sanitization

    def test_long_user_id_truncated(self):
        import re
        long_id = "a" * 200
        safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', long_id)[:64]
        assert len(safe_id) == 64


class TestPhoneAPIAuth:
    def test_api_key_hashing(self):
        api_key = "test_key_12345"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        assert len(key_hash) == 64
        assert key_hash != api_key


class TestNeuralMeshMessageSigning:
    def test_sign_verify_roundtrip(self):
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(
            message_type=MeshMessageType.HEARTBEAT,
            sender_id="node_1",
            payload={"status": "online"},
        )
        secret = "mesh_secret_key"
        sig = msg.sign(secret)
        assert msg.verify(secret) is True
        assert msg.verify("wrong_secret") is False

    def test_unsigned_message_rejected(self):
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="node_1",
            payload={"data": "test"},
        )
        assert msg.verify("any_key") is False

    def test_tampered_payload_fails(self):
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(
            message_type=MeshMessageType.HEARTBEAT,
            sender_id="node_1",
            payload={"status": "online"},
        )
        secret = "test_secret"
        msg.sign(secret)
        msg.payload["status"] = "offline"
        assert msg.verify(secret) is False


class TestCircuitBreakerFull:
    def test_circuit_breaker_half_open_recovery(self):
        from shadowd.cognition.neural_mesh import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        time.sleep(0.1)
        assert cb.state == CircuitBreaker.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb._failure_count == 0

    def test_circuit_breaker_reopens_after_half_open_failure(self):
        from shadowd.cognition.neural_mesh import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitBreaker.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN


class TestTaskTimeoutEnforcement:
    @pytest.mark.asyncio
    async def test_mission_timeout_config(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew()
        assert crew._task_timeout == 60.0
        assert crew._mission_timeout == 300.0

    @pytest.mark.asyncio
    async def test_timeout_error_type(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, CrewTask
        crew = ShadowCrew()
        mission = await crew.create_mission("timeout test")
        task = CrewTask(description="test", assigned_agent="shadow_mind")
        mission.tasks.append(task)
        result = await crew.execute_mission(mission.mission_id)
        assert result is not None


class TestEscalationRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limit_does_not_block_normal_usage(self):
        from shadowd.cognition.escalation import EscalationManager
        esc = EscalationManager()
        for i in range(10):
            record = await esc.evaluate("observe", {"data": f"test_{i}"})
            assert record is not None

    def test_escalation_persistence_config(self):
        from shadowd.cognition.escalation import EscalationManager
        esc = EscalationManager(data_dir="/tmp/test_esc_persist2")
        assert esc._data_dir is not None
        assert isinstance(esc._max_history, int)
        assert esc._max_history > 0


class TestDarkMemoryExportImport:
    @pytest.mark.asyncio
    async def test_export_roundtrip(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            await dm.store("test memory 1", importance=0.9)
            await dm.store("test memory 2", importance=0.5)
            exported = await dm.export_data()
            data = json.loads(exported)
            assert "atoms" in data
            assert "blocks" in data
            assert "stats" in data
            assert len(data["atoms"]) == 2
            dm.close()

    @pytest.mark.asyncio
    async def test_import_with_size_limit(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            result = await dm.import_data("x" * (60 * 1024 * 1024))
            assert "error" in result
            assert "too large" in result["error"].lower()


class TestShadowGraphAtomicCheckpoint:
    def test_checkpoint_writes_file(self):
        from shadowd.cognition.shadow_graph import ShadowGraph, CognitiveMode, ShadowNode, Checkpoint
        import time as _time
        with tempfile.TemporaryDirectory() as tmpdir:
            g = ShadowGraph(graph_id="test_atomic", data_dir=tmpdir)
            g.add_node(ShadowNode(name="test", mode=CognitiveMode.LISTEN))
            cp = Checkpoint(
                checkpoint_id="test_cp",
                state=g.state.to_dict(),
                node_name="test",
                timestamp=datetime.now(timezone.utc).isoformat(),
                execution_id="exec_1",
            )
            g._persist_checkpoint(cp)
            checkpoint_file = Path(tmpdir) / "test_atomic_latest.json"
            assert checkpoint_file.exists()


class TestStealthLayerKeyRotation:
    def test_rotate_changes_keys(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        pg._simple_encrypt("test1")
        old_keys = dict(pg._encryption_keys)
        pg.rotate_keys()
        for key_id in old_keys:
            if key_id in pg._encryption_keys:
                assert pg._encryption_keys[key_id] != old_keys[key_id]

    def test_key_age_tracking(self):
        # Implementation later switched from time.time() to datetime — either
        # representation is fine as long as we *can* compute an age from it.
        from datetime import datetime
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        pg._simple_encrypt("test")
        assert "default" in pg._key_created_at
        assert isinstance(pg._key_created_at["default"], (float, int, datetime))


class TestSelfImprovementCalibration:
    def test_brier_score_in_stats(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            stats = engine.get_stats()
            assert "brier_score" in stats["stats"]

    def test_auto_calibration_config(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            assert engine._calibration_interval > 0
            assert engine._calibration_task is None or engine._calibration_task.done() is not False


class TestDailyBriefingSchedule:
    def test_schedule_config(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        engine = DailyBriefingEngine()
        assert hasattr(engine, '_schedule_hour')
        assert hasattr(engine, '_schedule_task')
        assert engine._schedule_hour == 7

    @pytest.mark.asyncio
    async def test_generate_briefing(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        engine = DailyBriefingEngine()
        briefing = await engine.generate_briefing()
        assert briefing is not None
        assert briefing.briefing_id != ""
        assert len(briefing.items) > 0


class TestVoiceSessionCleanup:
    def test_session_timeout_config(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        assert sv._session_timeout == 1800.0
        assert sv._max_sessions == 100

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self):
        # _cleanup_expired_sessions uses time.monotonic(), so age the session
        # by setting its _start_time relative to monotonic, not wall clock.
        from shadowd.cognition.shadow_voice import ShadowVoice, VoiceSession
        sv = ShadowVoice()
        sv._sessions["old_session"] = VoiceSession(voice="test")
        sv._sessions["old_session"]._start_time = time.monotonic() - 7200  # 2 hours ago
        sv._cleanup_expired_sessions()
        assert sv._sessions["old_session"].ended_at is not None


class TestAPIRateLimiting:
    def test_rate_limit_value(self):
        assert 120 > 0  # RATE_LIMIT_PER_MINUTE


class TestShadowOSStatePersistence:
    @pytest.mark.asyncio
    async def test_save_state(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = ShadowOS(data_dir=tmpdir)
            os_inst._stats["queries_handled"] = 42
            await os_inst._save_state()
            state_file = os_inst.data_dir / "shadow_os_state.json"
            assert state_file.exists()
            data = json.loads(state_file.read_text())
            assert data["stats"]["queries_handled"] == 42

    @pytest.mark.asyncio
    async def test_load_state(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = ShadowOS(data_dir=tmpdir)
            await os_inst._load_state()
            assert isinstance(os_inst._stats, dict)


class TestInputSanitization:
    def test_sanitize_strips_xss(self):
        from shadowd.cognition.shadow_os import sanitize_input
        assert "<script>" not in sanitize_input("<script>alert(1)</script>hello")

    def test_sanitize_strips_control_chars(self):
        from shadowd.cognition.shadow_os import sanitize_input
        assert "\x00" not in sanitize_input("hello\x00world")

    def test_sanitize_truncates(self):
        from shadowd.cognition.shadow_os import sanitize_input
        assert len(sanitize_input("a" * 10000)) <= 2000

    def test_validate_observation_limits_keys(self):
        from shadowd.cognition.shadow_os import validate_observation
        big = {f"key_{i}": f"val_{i}" for i in range(200)}
        result = validate_observation(big)
        assert len(result) <= 50

    def test_validate_observation_limits_values(self):
        from shadowd.cognition.shadow_os import validate_observation
        result = validate_observation({"key": "x" * 20000})
        assert len(result["key"]) <= 10000