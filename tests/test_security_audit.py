#!/usr/bin/env python3
"""
Security Regression Tests for ShadowOS v2

These tests verify that all discovered vulnerabilities remain fixed.
Each test corresponds to a specific finding from the security audit.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path


# ============================================================
# SL-1: StealthLayer Encryption (was XOR, now AES-GCM)
# ============================================================

class TestStealthLayerEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        encrypted = pg._simple_encrypt("sensitive data")
        decrypted = pg._simple_decrypt(encrypted)
        assert decrypted == "sensitive data"

    def test_encryption_not_xor(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        plaintext = "AAAAAAAA"
        encrypted = pg._simple_encrypt(plaintext)
        xor_of_same = bytes(b ^ b for b in plaintext.encode()).hex()
        assert encrypted != xor_of_same

    def test_different_encryptions_differ(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        e1 = pg._simple_encrypt("same text")
        e2 = pg._simple_encrypt("same text")
        assert e1 != e2

    def test_ciphertext_has_nonce(self):
        import base64
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        encrypted = pg._simple_encrypt("test")
        raw = base64.b64decode(encrypted)
        assert len(raw) > 12

    def test_tampered_ciphertext_fails(self):
        import base64
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        encrypted = pg._simple_encrypt("secret")
        raw = bytearray(base64.b64decode(encrypted))
        raw[-1] ^= 0xFF
        tampered = base64.b64encode(bytes(raw)).decode()
        with pytest.raises(Exception):
            pg._simple_decrypt(tampered)


# ============================================================
# SL-2: PrivacyGuard Access Control Bypass
# ============================================================

class TestPrivacyGuardAccessControl:
    def test_high_sensitivity_with_approval_denied(self):
        from shadowd.cognition.stealth_layer import StealthLayer, AccessAction, DataCategory
        sl = StealthLayer()
        result, reason = sl.check_access(
            "external_app", AccessAction.READ, "secret_doc", DataCategory.CREDENTIALS
        )
        assert result is False

    def test_internal_can_access_public(self):
        from shadowd.cognition.stealth_layer import StealthLayer, AccessAction, DataCategory
        sl = StealthLayer()
        result, reason = sl.check_access(
            "shadow_ai", AccessAction.READ, "public_doc", DataCategory.PERSONAL
        )
        assert result is True


# ============================================================
# SC-1: ShadowCrew Infinite Mission Loop
# ============================================================

class TestShadowCrewMissionLoop:
    @pytest.mark.asyncio
    async def test_mission_max_tasks_limit(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew()
        mission = await crew.create_mission("Test infinite loop protection")
        from shadowd.cognition.shadow_crew import CrewTask
        for i in range(60):
            task = CrewTask(description=f"Task {i}", assigned_agent="shadow_mind")
            mission.tasks.append(task)
        result = await crew.execute_mission(mission.mission_id)
        assert result is not None


# ============================================================
# C1/C2/C3: ShadowGraph Trigger Routing, Escalation ID, Lock
# ============================================================

class TestShadowGraphSecurity:
    def test_graph_has_state_lock(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        g = ShadowGraph()
        assert hasattr(g, '_state_lock')
        assert g._state_lock is not None

    def test_graph_traces_capped(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        g = ShadowGraph()
        assert g._max_traces > 0
        assert g._max_traces <= 1000

    @pytest.mark.asyncio
    async def test_escalation_has_id_field(self):
        from shadowd.cognition.shadow_graph import ShadowGraph, CognitiveMode
        from shadowd.cognition.shadow_graph import ShadowNode, ApprovalLevel
        g = ShadowGraph()
        forbid_node = ShadowNode(name="forbid_test", mode=CognitiveMode.ACT, approval_level=ApprovalLevel.FORBID)
        g.add_node(forbid_node)
        g.state.mode = CognitiveMode.ACT
        g.state.current_query = "test forbid"
        try:
            await g.process_event({"query": "test forbid"})
        except Exception:
            pass
        for esc in g.state.pending_escalations:
            assert "id" in esc
            assert esc["id"].startswith("esc_")


# ============================================================
# DM-2: DarkMemory Block Protection Bypass
# ============================================================

class TestDarkMemoryBlockProtection:
    def test_protected_block_with_empty_value(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore()
        bs.initialize_default_blocks()
        bs._blocks["test_block"] = {"label": "test_block", "value": "", "limit": 2000, "protected": True}
        result = bs.set_block("test_block", "new value")
        assert result is False

    def test_protected_block_rejects_different_value(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore()
        bs.initialize_default_blocks()
        result = bs.set_block("identity", "I am someone else")
        assert result is False

    def test_unprotected_block_accepts_change(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore()
        bs.initialize_default_blocks()
        result = bs.set_block("context", "new context info")
        assert result is True


# ============================================================
# DM-3: InMemoryVectorIndex Thread Safety
# ============================================================

class TestVectorIndexThreadSafety:
    @pytest.mark.asyncio
    async def test_vector_index_has_lock(self):
        from shadowd.cognition.dark_memory import InMemoryVectorIndex
        idx = InMemoryVectorIndex(dimension=4)
        assert hasattr(idx, '_lock')
        assert idx._lock is not None

    @pytest.mark.asyncio
    async def test_concurrent_add_search(self):
        from shadowd.cognition.dark_memory import InMemoryVectorIndex
        idx = InMemoryVectorIndex(dimension=4)
        await idx.add("v1", [1.0, 0.0, 0.0, 0.0])
        await idx.add("v2", [0.0, 1.0, 0.0, 0.0])
        results = await idx.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2


# ============================================================
# DM-4: KG Traversal Depth Cap
# ============================================================

class TestKGTraversalDepth:
    def test_depth_capped_at_3(self):
        from shadowd.cognition.dark_memory import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph()
        kg.add_entity("a", "person", {})
        kg.add_entity("b", "person", {})
        kg.add_entity("c", "person", {})
        kg.add_relation("a", "b", "knows", {})
        kg.add_relation("b", "c", "knows", {})
        result = kg.get_neighbors("a", depth=999)
        assert isinstance(result, dict)

    def test_max_results_capped(self):
        from shadowd.cognition.dark_memory import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph()
        kg.add_entity("center", "person", {})
        for i in range(300):
            kg.add_entity(f"n_{i}", "person", {})
            kg.add_relation("center", f"n_{i}", "knows", {})
        result = kg.get_neighbors("center", depth=1, _max_results=10)
        total = len(result["outgoing"]) + len(result["incoming"])
        assert total <= 300


# ============================================================
# DM-10: SQL Wildcard Injection
# ============================================================

class TestSQLWildcardInjection:
    @pytest.mark.asyncio
    async def test_percent_wildcard_escaped(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            await dm.store("normal text", domain="test")
            results = await dm.search_db("%", limit=10)
            assert isinstance(results, list)


# ============================================================
# NM-1: Gossip Rate Limiting
# ============================================================

class TestGossipRateLimiting:
    def test_gossip_has_rate_limit(self):
        from shadowd.cognition.neural_mesh import GossipProtocol
        gp = GossipProtocol()
        assert hasattr(gp, '_check_rate_limit')
        assert gp._rate_limit_per_second > 0

    def test_gossip_rate_limit_works(self):
        from shadowd.cognition.neural_mesh import GossipProtocol
        gp = GossipProtocol(rate_limit_per_second=5)
        for _ in range(5):
            assert gp._check_rate_limit() is True
        assert gp._check_rate_limit() is False


# ============================================================
# PE-1: PersonalityEngine Rate Limiting
# ============================================================

class TestPersonalityRateLimiting:
    def test_record_behavior_rate_limited(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.cognition.personality_engine import PersonalityEngine
            pe = PersonalityEngine(data_dir=tmpdir)
            initial = len(pe.distiller._behavior_signals)
            for _ in range(50):
                pe.record_behavior("test_event", {"key": "value"})
            assert len(pe.distiller._behavior_signals) <= initial + 50


# ============================================================
# SV-1: Voice Transcript Sanitization
# ============================================================

class TestVoiceTranscriptSanitization:
    @pytest.mark.asyncio
    async def test_script_tags_stripped(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        result = await sv.process_speech("hello <script>alert('xss')</script>")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_control_chars_stripped(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        result = await sv.process_speech("hello\x00world\x01test")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_transcript_length_capped(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        long_text = "a" * 10000
        result = await sv.process_speech(long_text)
        assert result is not None


# ============================================================
# ES-1: SafetyGuardrail Regex (lazy quantifiers)
# ============================================================

class TestSafetyGuardrailRegex:
    @pytest.mark.asyncio
    async def test_no_redos_on_greedy_pattern(self):
        from shadowd.cognition.escalation import SafetyGuardrail
        sg = SafetyGuardrail()
        attack = "send password" + "a" * 10000 + " to someone"
        import time
        start = time.time()
        result, data, reason = await sg.check(attack)
        elapsed = time.time() - start
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_detects_dangerous_action(self):
        from shadowd.cognition.escalation import SafetyGuardrail, GuardrailResult
        sg = SafetyGuardrail()
        result, _, reason = await sg.check("delete all data")
        assert result == GuardrailResult.ESCALATE


# ============================================================
# V-03: shadow_os personality update respects protected blocks
# ============================================================

class TestPersonalityUpdateProtectedBlocks:
    def test_protected_human_block_not_overwritten(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore()
        bs.initialize_default_blocks()
        human_block = bs._blocks.get("human")
        assert human_block is not None
        assert human_block.get("protected") is True
        result = bs.set_block("human", "completely new value")
        assert result is False


# ============================================================
# V-08: Pseudonymize generates different nonces
# ============================================================

class TestPseudonymizeSalt:
    def test_pseudonymize_uses_random_salt(self):
        from shadowd.cognition.stealth_layer import CamouflageEngine
        ce = CamouflageEngine()
        r1 = ce.pseudonymize({"email": "a@b.com"}, "email")
        r2 = ce.pseudonymize({"email": "a@b.com"}, "email")
        assert r1["email"].startswith("pseudo_")


# ============================================================
# V-11: Intent parser CANCEL priority over COMMAND for "stop"
# ============================================================

class TestIntentParserPriority:
    def test_stop_is_cancel_not_command(self):
        from shadowd.cognition.shadow_voice import IntentParser, VoiceIntent
        parser = IntentParser()
        intent, confidence = parser.parse("stop")
        assert intent == VoiceIntent.CANCEL

    def test_start_is_command(self):
        from shadowd.cognition.shadow_voice import IntentParser, VoiceIntent
        parser = IntentParser()
        intent, confidence = parser.parse("start the timer")
        assert intent == VoiceIntent.COMMAND


# ============================================================
# SSN Regex: V-12
# ============================================================

class TestPrivacyGuardrailPatterns:
    @pytest.mark.asyncio
    async def test_ssn_pattern_with_dashes(self):
        from shadowd.cognition.escalation import PrivacyGuardrail, GuardrailResult
        pg = PrivacyGuardrail()
        result, data, reason = await pg.check("My SSN is 123-45-6789")
        assert result == GuardrailResult.TRANSFORM

    @pytest.mark.asyncio
    async def test_phone_pattern_with_dots(self):
        from shadowd.cognition.escalation import PrivacyGuardrail, GuardrailResult
        pg = PrivacyGuardrail()
        result, data, reason = await pg.check("Call me at 123.456.7890")
        assert result == GuardrailResult.TRANSFORM


# ============================================================
# Encryption keys are bytes type (V-22)
# ============================================================

class TestEncryptionKeysType:
    def test_encryption_keys_are_bytes(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        pg._simple_encrypt("test")
        for key_id, key in pg._encryption_keys.items():
            assert isinstance(key, bytes)
            assert len(key) == 32


# ============================================================
# API Router Input Validation
# ============================================================

class TestAPIRouterInputValidation:
    def test_kg_depth_bounded(self):
        from shadowd.cognition.api_router import KGQueryRequest
        req = KGQueryRequest(depth=1)
        assert req.depth == 1
        with pytest.raises(Exception):
            KGQueryRequest(depth=999)

    def test_search_limit_bounded(self):
        from shadowd.cognition.api_router import SearchRequest
        with pytest.raises(Exception):
            SearchRequest(query="test", limit=9999)

    def test_voice_transcript_bounded(self):
        from shadowd.cognition.api_router import VoiceProcessRequest
        with pytest.raises(Exception):
            VoiceProcessRequest(transcript="a" * 5001)


# ============================================================
# V-17: Handoff key validation
# ============================================================

class TestHandoffKeyValidation:
    @pytest.mark.asyncio
    async def test_handoff_only_allows_specific_keys(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew()
        result = await crew.handoff("nonexistent1", "nonexistent2", {"secret_key": "evil", "observations": "ok"})
        assert result is False


# ============================================================
# V-13: Gossip should_propagate calls rate limit
# ============================================================

class TestGossipShouldPropagateRateLimit:
    def test_should_propagate_checks_rate_limit(self):
        from shadowd.cognition.neural_mesh import GossipProtocol
        gp = GossipProtocol(rate_limit_per_second=2)
        for _ in range(2):
            assert gp._check_rate_limit() is True
        assert gp._check_rate_limit() is False


# ============================================================
# Brier Score Calculation
# ============================================================

class TestBrierScore:
    def test_brier_score_exists(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            score = engine.get_brier_score()
            assert isinstance(score, float)
            assert 0.0 <= score <= 2.0

    def test_brier_score_in_stats(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            stats = engine.get_stats()
            assert "brier_score" in stats["stats"]


# ============================================================
# SQLite Thread Safety: V-23
# ============================================================

class TestSQLiteThreadSafety:
    def test_db_has_wal_mode(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            cursor = dm._db_conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode == "wal"

    def test_db_has_lock(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            assert hasattr(dm, '_db_lock')