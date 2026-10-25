import asyncio
import copy
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from shadowd.cognition.shadow_os import ShadowOS, MAX_OBSERVATION_KEYS
from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
from shadowd.cognition.conflict_resolution import ConflictResolution, ConflictStatus
from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode, NodeStatus


# ═══════════════════════════════════════════════════════════════════
# SOS-H1: _handle_analyze caps observations at MAX_OBSERVATION_KEYS
# ═══════════════════════════════════════════════════════════════════

class TestHandleAnalyzeObservationCap:
    @pytest.mark.asyncio
    async def test_observations_capped_at_max(self):
        os = ShadowOS()
        state = CognitiveState(mode=CognitiveMode.ANALYZE)
        for i in range(MAX_OBSERVATION_KEYS + 20):
            state.observations.append({"dummy": i})
        with patch.object(os.memory, 'get_recent', return_value=[]):
            result = await os._handle_analyze(state)
        assert len(result.observations) <= MAX_OBSERVATION_KEYS

    @pytest.mark.asyncio
    async def test_observations_dont_exceed_max_after_append(self):
        os = ShadowOS()
        fake_atoms = [MagicMock(atom_id=f"a{i}", to_dict=lambda: {"id": i}) for i in range(60)]
        for i, a in enumerate(fake_atoms):
            a.to_dict = lambda idx=i: {"id": idx}
        with patch.object(os.memory, 'get_recent', return_value=fake_atoms):
            state = CognitiveState(mode=CognitiveMode.ANALYZE)
            result = await os._handle_analyze(state)
        assert len(result.observations) <= MAX_OBSERVATION_KEYS


# ═══════════════════════════════════════════════════════════════════
# SOS-H2: process_observation uses get_peers() instead of _peers
# ═══════════════════════════════════════════════════════════════════

class TestProcessObservationPeerAccess:
    @pytest.mark.asyncio
    async def test_mesh_exception_does_not_crash(self):
        os = ShadowOS()
        os.stealth.get_sensitivity_score = MagicMock(return_value=0.1)
        os.stealth.check_access = MagicMock(return_value=(True, "ok"))
        os.stealth.transform_for_sharing = MagicMock(return_value={"safe": True})
        os.mesh.get_peers = MagicMock(side_effect=RuntimeError("mesh broken"))
        async def fake_process_event(event, trigger=None):
            return CognitiveState(mode=CognitiveMode.ANALYZE)
        os.graph.process_event = fake_process_event
        async def fake_store(*a, **kw):
            m = MagicMock()
            m.atom_id = "test"
            return m
        os.memory.store = fake_store
        os.memory._atoms = {}
        result = await os.process_observation({"type": "observation", "content": "test"})
        assert result["status"] == "observed"

    @pytest.mark.asyncio
    async def test_empty_peers_no_share_attempt(self):
        os = ShadowOS()
        os.stealth.get_sensitivity_score = MagicMock(return_value=0.1)
        os.stealth.check_access = MagicMock(return_value=(True, "ok"))
        os.mesh.get_peers = MagicMock(return_value=[])
        os.mesh.share_insight = MagicMock()
        async def fake_process_event(event, trigger=None):
            return CognitiveState(mode=CognitiveMode.ANALYZE)
        os.graph.process_event = fake_process_event
        async def fake_store(*a, **kw):
            m = MagicMock()
            m.atom_id = "test"
            return m
        os.memory.store = fake_store
        os.memory._atoms = {}
        result = await os.process_observation({"type": "observation", "content": "test"})
        assert result["status"] == "observed"
        os.mesh.share_insight.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# SOS-H3: _handle_mesh_heartbeat handles missing peer gracefully
# ═══════════════════════════════════════════════════════════════════

class TestMeshHeartbeatGraceful:
    @pytest.mark.asyncio
    async def test_heartbeat_missing_peer_no_crash(self):
        os = ShadowOS()
        os.mesh._peers = {}
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(
            message_id="test",
            message_type=MeshMessageType.HEARTBEAT,
            sender_id="unknown_peer",
            payload={"status": "online", "capabilities": ["test"]},
        )
        await os._handle_mesh_heartbeat(msg)

    @pytest.mark.asyncio
    async def test_heartbeat_exception_caught(self):
        os = ShadowOS()
        peer_mock = MagicMock()
        peer_mock.status = NodeStatus.ONLINE
        type(peer_mock).status = PropertyMock(side_effect=RuntimeError("boom"))
        os.mesh._peers = {"bad_peer": peer_mock}
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(
            message_id="test2",
            message_type=MeshMessageType.HEARTBEAT,
            sender_id="bad_peer",
            payload={"status": "online", "capabilities": ["test"]},
        )
        await os._handle_mesh_heartbeat(msg)


# ═══════════════════════════════════════════════════════════════════
# SOS-M1: get_health presence.get_stats() safe access
# ═══════════════════════════════════════════════════════════════════

class TestGetHealthPresenceSafe:
    def test_presence_missing_key_no_crash(self):
        os = ShadowOS()
        os._running = True
        os._start_time = time.time()
        os.presence.get_stats = MagicMock(return_value={})
        os.mesh._circuit_breakers = {}
        health = os.get_health()
        assert health["status"] in ("healthy", "degraded")
        pd = health["modules"].get("presence_detection", {})
        assert pd.get("status") == "healthy"

    def test_presence_returns_non_dict(self):
        os = ShadowOS()
        os._running = True
        os._start_time = time.time()
        os.presence.get_stats = MagicMock(return_value="not_a_dict")
        os.mesh._circuit_breakers = {}
        health = os.get_health()
        pd = health["modules"].get("presence_detection", {})
        assert pd.get("status") in ("healthy", "unhealthy")


# ═══════════════════════════════════════════════════════════════════
# SOS-M2: _save_state limits peers to first 50
# ═══════════════════════════════════════════════════════════════════

class TestSaveStatePeersCap:
    @pytest.mark.asyncio
    async def test_save_state_caps_peers_at_50(self, tmp_path):
        os = ShadowOS(data_dir=str(tmp_path))
        fake_peers = {}
        for i in range(100):
            mock_peer = MagicMock()
            mock_peer.to_dict.return_value = {"peer_id": f"p{i}"}
            fake_peers[f"peer_{i}"] = mock_peer
        os.mesh._peers = fake_peers
        os.mesh._circuit_breakers = {}
        await os._save_state()
        import json
        state = json.loads((tmp_path / "shadow_os_state.json").read_text())
        assert len(state["peers"]) <= 50


# ═══════════════════════════════════════════════════════════════════
# SOS-M3: analyze_communication handles None urgency_confidence
# ═══════════════════════════════════════════════════════════════════

class TestAnalyzeCommunicationUrgencyNone:
    @pytest.mark.asyncio
    async def test_urgency_confidence_none_no_crash(self):
        os = ShadowOS()
        fake_result = MagicMock()
        fake_result.urgency_confidence = None
        fake_result.to_dict.return_value = {"urgency_confidence": None}
        os.thought_sense.analyze = MagicMock(return_value=fake_result)
        os.recall_synthesis.add_observation = MagicMock()
        os.recall_synthesis.synthesize = MagicMock(return_value=MagicMock(to_dict=lambda: {}))
        await os.analyze_communication("test message")
        os.recall_synthesis.add_observation.assert_called_once()
        call_args = os.recall_synthesis.add_observation.call_args
        assert call_args.kwargs.get("importance") == 0.3 or (len(call_args.args) >= 3 and call_args.args[2] == 0.3)

    @pytest.mark.asyncio
    async def test_urgency_confidence_zero(self):
        os = ShadowOS()
        fake_result = MagicMock()
        fake_result.urgency_confidence = 0
        fake_result.to_dict.return_value = {"urgency_confidence": 0}
        os.thought_sense.analyze = MagicMock(return_value=fake_result)
        os.recall_synthesis.add_observation = MagicMock()
        os.recall_synthesis.synthesize = MagicMock(return_value=MagicMock(to_dict=lambda: {}))
        await os.analyze_communication("test message")
        call_args = os.recall_synthesis.add_observation.call_args
        assert call_args.kwargs.get("importance") == 0.3 or (len(call_args.args) >= 3 and call_args.args[2] == 0.3)


# ═══════════════════════════════════════════════════════════════════
# API-H1: person_id length validation
# ═══════════════════════════════════════════════════════════════════

class TestEmpathyPersonIdValidation:
    @pytest.mark.asyncio
    async def test_profile_rejects_long_person_id(self):
        from shadowd.cognition.api_router import empathy_profile
        with pytest.raises(Exception) as exc_info:
            await empathy_profile("x" * 101)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_communication_rejects_long_person_id(self):
        from shadowd.cognition.api_router import empathy_communication
        with pytest.raises(Exception) as exc_info:
            await empathy_communication("x" * 101)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_profile_allows_normal_person_id(self):
        from shadowd.cognition.api_router import empathy_profile, get_shadow_os
        os = get_shadow_os()
        os.empathy.get_person_profile = MagicMock(return_value=None)
        with pytest.raises(Exception) as exc_info:
            await empathy_profile("abc123")
        assert exc_info.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════
# API-H2: /conflict/escalate returns 409 for resolved conflicts
# ═══════════════════════════════════════════════════════════════════

class TestConflictEscalateResolved:
    def test_escalate_resolved_returns_409(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2, "desc")
        cr.resolve_conflict(conflict.conflict_id, "done")
        result = cr.escalate_conflict(conflict.conflict_id)
        assert result is False
        fetched = cr.get_conflict(conflict.conflict_id)
        assert fetched.status == ConflictStatus.RESOLVED

    def test_escalate_active_conflict_works(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2, "desc")
        result = cr.escalate_conflict(conflict.conflict_id)
        assert result is True
        fetched = cr.get_conflict(conflict.conflict_id)
        assert fetched.status == ConflictStatus.ESCALATED


# ═══════════════════════════════════════════════════════════════════
# API-M1-M6: Auth dependencies on previously-unauthenticated endpoints
# ═══════════════════════════════════════════════════════════════════

class TestApiAuthDependencies:
    def _get_route_deps(self, router, path, method="get"):
        for route in router.routes:
            if hasattr(route, 'path') and route.path == path:
                for m, handler in route.methods.items() if hasattr(route, 'methods') else []:
                    pass
                if hasattr(route, 'dependant') and route.dependant:
                    deps = [d.call for d in route.dependant.dependencies]
                    return deps
        return []

    def test_narrative_chapters_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/narrative/chapters" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_narrative_arcs_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/narrative/arcs" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_emotion_detect_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/emotion/detect" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_creative_synthesize_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/creative/synthesize" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_meta_think_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/meta/think" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_moral_evaluate_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/moral/evaluate" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_pipeline_stats_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/pipeline/stats" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_meta_calibration_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/meta/calibration" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_meta_biases_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/meta/biases" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_meta_alerts_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/meta/alerts" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_moral_dilemmas_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/moral/dilemmas" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_moral_principles_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/moral/principles" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_moral_evaluations_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/moral/evaluations" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_emotion_trajectory_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/emotion/trajectory" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_emotion_balance_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/emotion/balance" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_meta_load_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/meta/load" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_creative_associate_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/creative/associate" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_creative_brainstorm_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/creative/brainstorm" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_meta_resolve_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/meta/resolve" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_meta_bias_detect_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/meta/bias-detect" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_moral_principle_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/moral/principle" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_moral_resolve_dilemma_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/moral/resolve-dilemma" and hasattr(route, 'methods') and 'POST' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_creative_ideas_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/creative/ideas" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_creative_patterns_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/creative/patterns" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_procedural_habits_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/procedural/habits" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break

    def test_procedural_routines_has_auth(self):
        from shadowd.cognition.api_router import router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/procedural/routines" and hasattr(route, 'methods') and 'GET' in route.methods:
                deps = [d.call for d in route.dependant.dependencies]
                from shadowd.cognition.api_router import verify_api_key
                assert verify_api_key in deps
                break


# ═══════════════════════════════════════════════════════════════════
# API-M7: List element max_length validation
# ═══════════════════════════════════════════════════════════════════

class TestListElementValidation:
    def test_dream_memory_tags_truncated(self):
        from shadowd.cognition.api_router import DreamMemoryRequest
        req = DreamMemoryRequest(content="x", tags=["a" * 100])
        assert len(req.tags[0]) <= 50

    def test_goal_create_tags_truncated(self):
        from shadowd.cognition.api_router import GoalCreateRequest
        req = GoalCreateRequest(title="test", tags=["long_tag_" * 10])
        assert len(req.tags[0]) <= 50

    def test_skill_add_deps_truncated(self):
        from shadowd.cognition.api_router import SkillAddRequest
        req = SkillAddRequest(
            skill_id="s1", name="test", domain="career",
            dependencies=["dep_" * 20],
        )
        assert len(req.dependencies[0]) <= 50

    def test_narrative_arc_themes_truncated(self):
        from shadowd.cognition.api_router import NarrativeArcRequest
        req = NarrativeArcRequest(title="test", themes=["t" * 100])
        assert len(req.themes[0]) <= 50

    def test_procedure_add_steps_truncated(self):
        from shadowd.cognition.api_router import ProcedureAddRequest
        req = ProcedureAddRequest(name="test", category="work", steps=["s" * 100])
        assert len(req.steps[0]) <= 50

    def test_conflict_perspective_interests_truncated(self):
        from shadowd.cognition.api_router import ConflictPerspectiveRequest
        req = ConflictPerspectiveRequest(
            conflict_id="c1", party="A", position="pos",
            interests=["i" * 100],
        )
        assert len(req.interests[0]) <= 50

    def test_conflict_perspective_concerns_truncated(self):
        from shadowd.cognition.api_router import ConflictPerspectiveRequest
        req = ConflictPerspectiveRequest(
            conflict_id="c1", party="A", position="pos",
            concerns=["c" * 100],
        )
        assert len(req.concerns[0]) <= 50

    def test_empty_list_passes(self):
        from shadowd.cognition.api_router import DreamMemoryRequest
        req = DreamMemoryRequest(content="x", tags=[])
        assert req.tags == []

    def test_none_list_passes(self):
        from shadowd.cognition.api_router import DreamMemoryRequest
        req = DreamMemoryRequest(content="x")
        assert req.tags is None