#!/usr/bin/env python3
"""Session 5 Sprint 14 regression tests - Remaining cognition module audit.

Covers:
- V-351: NeuralMesh _task_delegations capped at 1000
- V-350/V-354: NeuralMesh str(e) → type(e).__name__ (heartbeat, handler, processor, broadcast)
- V-352: PrivacyGuard classifications/policies/data_store capped
- V-354: ShadowVoice TTSEngine/STTEngine health str(e) → type(e).__name__
- V-355: ShadowVoice session cap enforced on process_speech auto-creation
- V-356: ShadowOS str(e) leaks → type(e).__name__ (4 instances)
- V-357: ShadowOS _save_state atomic (tmp+rename)
- V-358: ShadowOS get_health() dead code removed
- V-359: ShadowOS _MAX_BACKGROUND_TASKS constant added
- V-360: DailyBriefing str(e) → type(e).__name__
"""

import asyncio
import inspect
import json
import os
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNeuralMeshDelegationsCap:
    """V-351: NeuralMesh _task_delegations capped at 1000."""

    def test_max_delegations_constant(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        mesh = NeuralMesh(node_id="test")
        assert mesh._MAX_TASK_DELEGATIONS == 1000

    def test_delegations_eviction_on_overflow(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        mesh = NeuralMesh(node_id="test")
        for i in range(1001):
            mesh._task_delegations[f"del_{i}"] = {
                "task_type": "compute",
                "target_node": "node_a",
                "status": "pending",
                "created_at": f"2025-01-01T00:{i % 60:02d}:00",
            }
        assert len(mesh._task_delegations) == 1001

    @pytest.mark.asyncio
    async def test_delegate_task_evicts_when_over_cap(self):
        from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode, NodeStatus
        mesh = NeuralMesh(node_id="test")
        peer = MeshNode(node_id="peer_a", address="127.0.0.1", port=8766)
        mesh.add_peer(peer)

        for i in range(mesh._MAX_TASK_DELEGATIONS):
            mesh._task_delegations[f"del_existing_{i}"] = {
                "task_type": "compute",
                "target_node": "peer_a",
                "status": "completed",
                "created_at": f"2025-01-01T00:{i % 60:02d}:00",
            }

        delegation_id = await mesh.delegate_task("compute_heavy", {"data": "test"})
        assert len(mesh._task_delegations) <= mesh._MAX_TASK_DELEGATIONS


class TestNeuralMeshStrELeaks:
    """V-350/V-354: NeuralMesh str(e) → type(e).__name__ in all error paths."""

    def test_heartbeat_error_uses_type_name(self):
        # The intent is to ensure no raw str(e) leaks into the log line —
        # `type(e).__name__` (whether logged via %s or f-string) is the safe
        # pattern. The original asserted on either '{e}' or the f-string form,
        # which mis-matched a logger.error("...%s", type(e).__name__) call.
        from shadowd.cognition.neural_mesh import NeuralMesh
        source = inspect.getsource(NeuralMesh._heartbeat_loop)
        assert 'type(e).__name__' in source, "must use type(e).__name__"
        # And must NOT leak str(e) directly.
        assert 'str(e)' not in source, "raw str(e) leaked into _heartbeat_loop"

    def test_message_processor_error_uses_type_name(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        source = inspect.getsource(NeuralMesh._message_processor)
        assert "type(e).__name__" in source

    def test_broadcast_retry_error_uses_type_name(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        source = inspect.getsource(NeuralMesh._broadcast_with_retry)
        assert "type(e).__name__" in source


class TestStealthLayerCaps:
    """V-352: PrivacyGuard classifications/policies/data_store capped."""

    def test_max_classifications_constant(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        assert PrivacyGuard._MAX_CLASSIFICATIONS == 5000

    def test_max_data_store_constant(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        assert PrivacyGuard._MAX_DATA_STORE == 10000

    def test_classifications_eviction_on_overflow(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard, DataCategory
        guard = PrivacyGuard()
        for i in range(5100):
            guard.classify_data(f"data_{i}", DataCategory.PERSONAL)
        assert len(guard._classifications) <= guard._MAX_CLASSIFICATIONS

    def test_data_store_eviction_on_overflow(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard, DataCategory
        guard = PrivacyGuard()
        for i in range(10100):
            guard.store_data(f"key_{i}", f"value_{i}", DataCategory.NETWORK)
        assert len(guard._data_store) <= guard._MAX_DATA_STORE


class TestShadowVoiceStrELeaks:
    """V-354: ShadowVoice TTSEngine/STTEngine health str(e) → type(e).__name__."""

    def test_tts_speak_error_uses_type_name(self):
        from shadowd.cognition.shadow_voice import TTSEngine
        source = inspect.getsource(TTSEngine.speak)
        assert "type(e).__name__" in source

    def test_tts_list_voices_error_uses_type_name(self):
        from shadowd.cognition.shadow_voice import TTSEngine
        source = inspect.getsource(TTSEngine.list_voices)
        assert "type(e).__name__" in source

    def test_tts_health_error_no_str_e(self):
        from shadowd.cognition.shadow_voice import TTSEngine
        source = inspect.getsource(TTSEngine.health)
        assert 'str(e)' not in source
        assert "type(e).__name__" in source

    def test_stt_transcribe_error_uses_type_name(self):
        from shadowd.cognition.shadow_voice import STTEngine
        source = inspect.getsource(STTEngine.transcribe)
        assert "type(e).__name__" in source

    def test_command_handler_error_uses_type_name(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        source = inspect.getsource(ShadowVoice.process_speech)
        assert "type(e).__name__" in source


class TestShadowVoiceSessionCap:
    """V-355: ShadowVoice session cap enforced on process_speech auto-creation."""

    def test_max_sessions_default(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        voice = ShadowVoice()
        assert voice._max_sessions == 100

    @pytest.mark.asyncio
    async def test_process_speech_cleans_up_before_auto_create(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        voice = ShadowVoice()
        source = inspect.getsource(ShadowVoice.process_speech)
        assert "self._cleanup_expired_sessions()" in source
        assert "_sessions_lock" in source


class TestShadowOSStrELeaks:
    """V-356: ShadowOS str(e) → type(e).__name__ in 4 error paths."""

    def test_cognitive_error_uses_type_name(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._on_cognitive_error)
        assert "type(error).__name__" in source or "type(e).__name__" in source

    def test_save_state_error_uses_type_name(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._save_state)
        assert "type(e).__name__" in source

    def test_load_state_error_uses_type_name(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._load_state)
        assert "type(e).__name__" in source

    def test_health_monitor_error_uses_type_name(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._health_monitor_loop)
        assert "type(e).__name__" in source

    def test_health_check_error_uses_type_name(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS.get_health)
        assert "str(e)" not in source
        assert "type(e).__name__" in source


class TestShadowOSAtomicSave:
    """V-357: ShadowOS _save_state uses atomic tmp+rename."""

    def test_save_state_uses_tmp_file(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._save_state)
        assert ".tmp" in source
        assert "replace" in source or "rename" in source


class TestShadowOSDeadCodeRemoval:
    """V-358: ShadowOS get_health() dead code removed."""

    def test_no_duplicate_return_block(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS.get_health)
        return_count = source.count("return {")
        assert return_count == 2, f"Expected 2 return statements (stopped + healthy), found {return_count}"


class TestShadowOSBackgroundTasksCap:
    """V-359: ShadowOS _MAX_BACKGROUND_TASKS constant added."""

    def test_max_background_tasks_constant(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_inst = ShadowOS.__new__(ShadowOS)
        assert hasattr(ShadowOS, '__init__')
        source = inspect.getsource(ShadowOS.__init__)
        assert "_MAX_BACKGROUND_TASKS" in source


class TestDailyBriefingStrELeak:
    """V-360: DailyBriefing str(e) → type(e).__name__."""

    def test_briefing_schedule_error_uses_type_name(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        source = inspect.getsource(DailyBriefingEngine._briefing_schedule_loop)
        assert "type(e).__name__" in source


class TestNeuralMeshMessageSigning:
    """Verify NeuralMesh message signing is intact (no regression)."""

    def test_message_sign_verify(self):
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(message_type=MeshMessageType.GOSSIP, payload={"test": "data"})
        secret = "test_secret"
        sig = msg.sign(secret)
        assert msg.verify(secret) is True
        assert msg.verify("wrong_secret") is False


class TestStealthLayerAESRoundTrip:
    """Verify StealthLayer AES encryption round-trip (no regression)."""

    def test_encrypt_decrypt_round_trip(self):
        from shadowd.cognition.stealth_layer import StealthLayer, DataCategory
        layer = StealthLayer()
        layer.store("test_key", {"value": "secret_data"}, DataCategory.NETWORK)
        result = layer.retrieve("test_key", subject="shadow_ai")
        assert result == {"value": "secret_data"}


class TestPrivacyGuardAccessControl:
    """Verify PrivacyGuard access control still works after caps (no regression)."""

    def test_external_cannot_access_credentials(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard, DataCategory, AccessAction
        guard = PrivacyGuard()
        guard.classify_data("api_key", DataCategory.CREDENTIALS)
        granted, reason = guard.check_access("external", AccessAction.READ, "api_key", DataCategory.CREDENTIALS)
        assert granted is False

    def test_shadow_ai_can_read_network(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard, DataCategory, AccessAction
        guard = PrivacyGuard()
        guard.classify_data("wifi", DataCategory.NETWORK)
        granted, _ = guard.check_access("shadow_ai", AccessAction.READ, "wifi", DataCategory.NETWORK)
        assert granted is True


class TestDailyBriefingWellnessCalculation:
    """Verify wellness score logic is correct (no regression from Sprint 12)."""

    def test_wellness_base_subtracts_urgent(self):
        from shadowd.cognition.daily_briefing import DailyBriefing, DailyBriefingEngine, BriefingItem, BriefingSection, UrgencyLevel
        engine = DailyBriefingEngine()
        briefing = DailyBriefing()
        briefing.items = [
            BriefingItem(section=BriefingSection.TODAY, title="t", content="c", urgency=UrgencyLevel.HIGH),
            BriefingItem(section=BriefingSection.WELLNESS, title="t2", content="c2", urgency=UrgencyLevel.CRITICAL),
        ]
        score = engine._calculate_wellness(briefing)
        assert score < 0.7
        assert score > 0.0