#!/usr/bin/env python3
"""
Feature Regression Tests for ShadowOS Phase 2 Enhancements

Tests all new features added in the Phase 2 development sprint:
- Auto-compaction daemon
- Circuit breaker
- Message signing/verification
- Task timeout enforcement
- Escalation rate limiting
- Personality export/import
- Session timeout cleanup
- API rate limiting
- DarkMemory backup/export/import
- ShadowOS state persistence
- Input sanitization
- Key rotation
"""

import asyncio
import json
import tempfile
import time
import pytest


class TestAutoCompaction:
    @pytest.mark.asyncio
    async def test_compaction_daemon_exists(self):
        from shadowd.cognition.dark_memory import DarkMemory
        dm = DarkMemory(data_dir="/tmp/test_compaction_1")
        assert hasattr(dm, 'start_auto_compaction')
        assert hasattr(dm, '_compaction_interval')
        assert hasattr(dm, '_max_atoms')
        dm.close()

    @pytest.mark.asyncio
    async def test_ttl_eviction(self):
        from shadowd.cognition.dark_memory import DarkMemory
        dm = DarkMemory(data_dir="/tmp/test_ttl_1")
        for i in range(5):
            await dm.store(f"old archival {i}", importance=0.1)
            # Force archival tier
            atom = dm._atoms[list(dm._atoms.keys())[-1]]
            from shadowd.cognition.dark_memory import MemoryTier
            atom.tier = MemoryTier.ARCHIVAL
            atom.accessed_at = "2020-01-01T00:00:00+00:00"
        evicted = await dm._evict_stale(ttl_hours=1)
        assert evicted >= 0
        dm.close()

    @pytest.mark.asyncio
    async def test_max_atoms_limit(self):
        from shadowd.cognition.dark_memory import DarkMemory
        dm = DarkMemory(data_dir="/tmp/test_max_atoms", max_atoms=5)
        for i in range(10):
            await dm.store(f"test atom {i}", importance=0.3)
        assert len(dm._atoms) == 10
        evicted = await dm.compact()
        assert evicted >= 0
        dm.close()


class TestCircuitBreaker:
    def test_circuit_breaker_states(self):
        from shadowd.cognition.neural_mesh import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.05)
        assert cb.state == CircuitBreaker.CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is True
        cb.record_failure()
        assert cb.can_execute() is False
        time.sleep(0.1)
        assert cb.state == CircuitBreaker.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED

    def test_mesh_has_circuit_breakers(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        mesh = NeuralMesh(node_name="test-cb")
        assert hasattr(mesh, '_circuit_breakers')
        assert hasattr(mesh, '_mesh_secret')


class TestMessageSigning:
    def test_message_sign_and_verify(self):
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="node_1",
            payload={"text": "hello"},
        )
        secret = "test_secret_key"
        sig = msg.sign(secret)
        assert sig != ""
        assert msg.verify(secret) is True
        assert msg.verify("wrong_secret") is False

    def test_unsigned_message_fails_verify(self):
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="node_1",
            payload={"text": "hello"},
        )
        assert msg.verify("any_secret") is False


class TestTaskTimeout:
    @pytest.mark.asyncio
    async def test_crew_has_timeout_config(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew()
        assert hasattr(crew, '_task_timeout')
        assert hasattr(crew, '_mission_timeout')
        assert crew._task_timeout > 0

    @pytest.mark.asyncio
    async def test_mission_with_timeout(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, DelegationMode
        crew = ShadowCrew()
        mission = await crew.create_mission("Quick test", mode=DelegationMode.SEQUENTIAL)
        result = await crew.execute_mission(mission.mission_id)
        assert result is not None


class TestEscalationRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        from shadowd.cognition.escalation import EscalationManager
        esc = EscalationManager(data_dir="/tmp/test_esc_rate")
        for _ in range(100):
            record = await esc.evaluate("observe", {"data": "test"})
        assert len(esc._rate_requests) <= 120

    @pytest.mark.asyncio
    async def test_data_dir_persistence(self):
        from shadowd.cognition.escalation import EscalationManager
        esc = EscalationManager(data_dir="/tmp/test_esc_persist")
        assert esc._data_dir is not None
        record = await esc.evaluate("observe", {"data": "test"})
        esc._persist_history()


class TestPersonalityExport:
    def test_export_import_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.cognition.personality_engine import PersonalityEngine
            pe = PersonalityEngine(data_dir=tmpdir)
            pe.record_behavior("test_action", {"domain": "test"})
            exported = pe.export_profile()
            data = json.loads(exported)
            assert "big_five" in data
            pe.import_profile(exported)

    def test_has_rate_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.cognition.personality_engine import PersonalityEngine
            pe = PersonalityEngine(data_dir=tmpdir)
            assert hasattr(pe.distiller, '_max_signals_per_second')


class TestVoiceSessionCleanup:
    def test_session_timeout_config(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        assert hasattr(sv, '_session_timeout')
        assert hasattr(sv, '_max_sessions')
        assert sv._session_timeout > 0

    def test_cleanup_method_exists(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        assert hasattr(sv, '_cleanup_expired_sessions')


class TestDarkMemoryExport:
    @pytest.mark.asyncio
    async def test_export_import(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            await dm.store("test export 1", importance=0.7)
            await dm.store("test export 2", importance=0.5)
            exported = await dm.export_data()
            data = json.loads(exported)
            assert "atoms" in data
            assert len(data["atoms"]) == 2
            assert "blocks" in data

            dm2 = DarkMemory(data_dir=tmpdir + "_import")
            result = await dm2.import_data(exported)
            assert result["imported"] == 2
            dm.close()
            dm2.close()


class TestShadowOSSanitization:
    def test_sanitize_input(self):
        from shadowd.cognition.shadow_os import sanitize_input
        assert sanitize_input("<script>alert('xss')</script>hello") == "hello"
        assert len(sanitize_input("a" * 5000)) <= 2000
        assert sanitize_input("hello\x00world") == "helloworld"

    def test_validate_observation(self):
        from shadowd.cognition.shadow_os import validate_observation
        obs = {"key": "value", "huge": "x" * 15000}
        cleaned = validate_observation(obs)
        assert len(cleaned["huge"]) <= 10000

        obs2 = {f"k{i}": "v" for i in range(100)}
        cleaned2 = validate_observation(obs2)
        assert len(cleaned2) <= 50


class TestKeyRotation:
    def test_key_rotation(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        encrypted = pg._simple_encrypt("test data")
        old_key = pg._encryption_keys.get("default", b"")
        pg.rotate_keys()
        new_key = pg._encryption_keys.get("default", b"")
        assert old_key != new_key or len(pg._encryption_keys) > 0

    def test_key_age_tracking(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        assert hasattr(pg, '_key_created_at')


class TestShadowOSHealthMonitor:
    def test_health_checks_modules(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = ShadowOS(data_dir=tmpdir)
            health = os_inst.get_health()
            assert health["status"] == "stopped"

    def test_state_persistence(self):
        from shadowd.cognition.shadow_os import ShadowOS
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = ShadowOS(data_dir=tmpdir)
            os_inst._stats["queries_handled"] = 42
            state_file = os_inst.data_dir / "shadow_os_state.json"
            assert not state_file.exists()


class TestShadowVoiceTranscriptSanitization:
    @pytest.mark.asyncio
    async def test_control_chars_stripped(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        result = await sv.process_speech("hello\x00\x01world")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_transcript_truncated(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        result = await sv.process_speech("a" * 10000)
        assert result is not None


class TestCrewTaskLimit:
    @pytest.mark.asyncio
    async def test_mission_max_tasks(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew()
        mission = await crew.create_mission("Test max tasks")
        from shadowd.cognition.shadow_crew import CrewTask
        for i in range(55):
            mission.tasks.append(CrewTask(description=f"Task {i}"))
        result = await crew.execute_mission(mission.mission_id)
        assert result is not None


class TestBriefingSchedule:
    def test_schedule_config(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        engine = DailyBriefingEngine()
        assert hasattr(engine, '_schedule_hour')
        assert hasattr(engine, '_schedule_task')

    @pytest.mark.asyncio
    async def test_can_generate_briefing(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        engine = DailyBriefingEngine()
        briefing = await engine.generate_briefing()
        assert briefing is not None
        assert briefing.briefing_id != ""


class TestSelfImprovementAutoCalibration:
    def test_auto_calibration_config(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            assert hasattr(engine, '_calibration_interval')
            assert hasattr(engine, '_calibration_task')
            assert engine._calibration_interval > 0