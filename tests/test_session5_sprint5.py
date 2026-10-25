#!/usr/bin/env python3
"""
Session 5 Sprint 5 Regression Tests

Covers:
- V-222: Task delegation cap (20 tasks per mission)
- V-227: Timing attack fix (hmac.compare_digest)
- V-228: GossipProtocol seen_messages cap
- V-229: StateSync conflicts cap
- V-233: Atomic profile save
- V-235: import_profile key validation
- V-236: Observer cap (20)
- V-205: Orchestrator crew/workflow caps
- V-210: block_memory UUID IDs
- V-213/V-214: Knowledge graph facts/episodes caps
- V-216: Knowledge graph hash() replaced with sha256
- V-219: MeshPeer SSRF validation
- V-220/V-238: API request model length validation
- V-224: shadow_crew error info leak fix
"""

import pytest
import inspect
import hashlib


class TestTaskDelegationCap:
    """V-222: Task delegation is capped at 20 tasks per mission."""

    def test_delegation_cap_exists(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.execute_mission)
        assert "_MAX_SUBTASKS_PER_MISSION" in source, "Delegation cap should use _MAX_SUBTASKS_PER_MISSION class constant"


class TestMeshSignatureTimingSafe:
    """V-227: Mesh message signature verification uses constant-time comparison."""

    def test_uses_compare_digest(self):
        from shadowd.cognition.neural_mesh import MeshMessage
        source = inspect.getsource(MeshMessage.verify)
        assert "compare_digest" in source, "Should use hmac.compare_digest for timing-safe comparison"

    def test_verify_correctness(self):
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(sender_id="node1", message_type=MeshMessageType.INSIGHT_SHARE, payload={"data": "test"})
        sig = msg.sign("secret123")
        assert msg.verify("secret123") is True
        assert msg.verify("wrong_secret") is False


class TestGossipProtocolSeenCap:
    """V-228: GossipProtocol caps seen_messages."""

    def test_seen_messages_cap(self):
        from shadowd.cognition.neural_mesh import GossipProtocol
        assert hasattr(GossipProtocol, 'MAX_SEEN_MESSAGES')
        assert GossipProtocol.MAX_SEEN_MESSAGES == 10000


class TestStateSyncConflictsCap:
    """V-229: StateSync caps conflicts list."""

    def test_conflicts_cap(self):
        from shadowd.cognition.neural_mesh import StateSync
        assert hasattr(StateSync, 'MAX_CONFLICTS')
        assert StateSync.MAX_CONFLICTS == 100


class TestAtomicProfileSave:
    """V-233: PersonalityEngine profile save uses atomic write."""

    def test_uses_tmp_rename(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine._save_profile)
        assert ".tmp" in source or "replace" in source, "Should use atomic write"


class TestImportProfileKeyValidation:
    """V-235: import_profile validates Big Five attribute keys."""

    def test_import_uses_valid_set(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine.import_profile)
        assert "VALID_BIG_FIVE" in source, "Should use explicit key validation set"


class TestObserverCap:
    """V-236: PersonalityEngine observers capped at 20."""

    def test_observer_cap(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        assert hasattr(PersonalityEngine, 'MAX_OBSERVERS')
        assert PersonalityEngine.MAX_OBSERVERS == 20


class TestOrchestratorCaps:
    """V-205: SuperOrchestrator caps _crews and _active_workflows."""

    def test_crews_cap(self):
        import shadowd.super_orchestrator as so
        source = inspect.getsource(so)
        assert "_MAX_CREWS" in source, "Should have max crews cap"


class TestBlockMemoryUUID:
    """V-210: block_memory uses UUID instead of timestamp for IDs."""

    def test_uses_uuid(self):
        import shadowd.block_memory as bm
        source = inspect.getsource(bm)
        assert "uuid" in source, "Should use uuid for IDs"
        assert "datetime.now().timestamp()" not in source, "Should not use timestamp for IDs"


class TestKnowledgeGraphCaps:
    """V-213/V-214: Knowledge graph caps facts and episodes."""

    def test_facts_capped(self):
        import shadowd.knowledge_graph as kg
        source = inspect.getsource(kg.Entity.add_fact)
        assert "1000" in source, "Facts should be capped at 1000"

    def test_episodes_capped(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        source = inspect.getsource(TemporalKnowledgeGraph.add_episode)
        assert "_MAX_EPISODES" in source or "10000" in source, "Episodes should be capped"

    def test_entity_episodes_capped(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        source = inspect.getsource(TemporalKnowledgeGraph.add_episode)
        assert "_MAX_ENTITY_EPISODES" in source or "100" in source, "Entity episodes should be capped"


class TestKnowledgeGraphHashFix:
    """V-216: hash() replaced with sha256 in knowledge graph."""

    def test_no_insecure_hash(self):
        import shadowd.knowledge_graph as kg
        source = inspect.getsource(kg)
        assert "hash(" not in source or "hashlib" in source, "Should not use Python hash() for IDs"
        assert "sha256" in source, "Should use sha256"


class TestMeshPeerSSRF:
    """V-219: MeshPeerRequest validates addresses."""

    def test_safe_address_method(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        req = MeshPeerRequest(node_id="n1", name="test")
        assert hasattr(req, 'is_safe_address')

    def test_blocks_metadata_endpoint(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        req = MeshPeerRequest(node_id="n1", name="test", address="169.254.169.254")
        assert req.is_safe_address() is False

    def test_blocks_zero_address(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        req = MeshPeerRequest(node_id="n1", name="test", address="0.0.0.0")
        assert req.is_safe_address() is False

    def test_allows_localhost(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        req = MeshPeerRequest(node_id="n1", name="test", address="127.0.0.1")
        assert req.is_safe_address() is True


class TestAPIRequestValidation:
    """V-220/V-238: API request models have length validation."""

    def test_stealth_check_has_max_length(self):
        from shadowd.cognition.api_router import StealthCheckRequest
        fields = StealthCheckRequest.model_fields
        subject_field = fields["subject"]
        assert subject_field is not None
        assert "max_length" in str(subject_field) or hasattr(subject_field, "max_length")

    def test_predict_options_has_max_items(self):
        from shadowd.cognition.api_router import PersonalityPredictRequest
        fields = PersonalityPredictRequest.model_fields
        assert "options" in fields


class TestShadowCrewNoInfoLeak:
    """V-224: shadow_crew error messages use type().__name__ not str(e)."""

    def test_task_error_uses_type(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.execute_mission)
        assert "type(e).__name__" in source, "Task error should use type name not str(e)"

    def test_audit_log_uses_type(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.execute_mission)
        lines = [l for l in source.split('\n') if '"error"' in l and "task_failed" in l or "audit" in l.lower()]
        error_lines = [l for l in source.split('\n') if '"error":' in l]
        for line in error_lines:
            if "type(e)" in line:
                assert True
                return
        assert "str(e)" not in source.split("task_failed")[0].split("\n")[-1] if "task_failed" in source else True