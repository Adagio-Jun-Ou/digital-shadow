#!/usr/bin/env python3
"""
Fuzz Testing for ShadowOS Cognitive Modules

Systematically tests edge cases, boundary conditions, and adversarial inputs.
"""

import asyncio
import json
import random
import string
import pytest
import tempfile
import time


def random_string(length: int = 100) -> str:
    return "".join(random.choices(string.printable, k=length))


def random_unicode(length: int = 50) -> str:
    return "".join(chr(random.randint(0x0001, 0xFFFF)) for _ in range(length))


def random_bytes(length: int = 100) -> bytes:
    return bytes(random.randint(0, 255) for _ in range(length))


class TestDarkMemoryFuzz:
    @pytest.mark.asyncio
    async def test_store_empty_content(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            atom = await dm.store("")
            assert atom is not None
            dm.close()

    @pytest.mark.asyncio
    async def test_store_very_long_content(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            atom = await dm.store("x" * 100000)
            assert atom is not None
            assert len(atom.content) == 100000
            dm.close()

    @pytest.mark.asyncio
    async def test_store_unicode_content(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            atom = await dm.store("日本語テスト 🎉 ñ é ü")
            assert atom is not None
            dm.close()

    @pytest.mark.asyncio
    async def test_store_null_bytes(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            atom = await dm.store("hello\x00world\x01test")
            assert atom is not None
            dm.close()

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            results = await dm.search("", limit=10)
            assert isinstance(results, list)
            dm.close()

    @pytest.mark.asyncio
    async def test_search_special_characters(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            for q in ["%; DROP TABLE--", "' OR '1'='1", "${jndi:ldap://x}", "<script>alert(1)</script>"]:
                results = await dm.search(q, limit=5)
                assert isinstance(results, list)
            dm.close()

    @pytest.mark.asyncio
    async def test_kg_query_deep(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            await dm.store_entity("a", "person", {})
            result = await dm.query_kg(entity_id="a", depth=999)
            assert isinstance(result, dict)
            dm.close()

    @pytest.mark.asyncio
    async def test_kg_large_graph(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            await dm.store_entity("center", "person", {})
            for i in range(100):
                await dm.store_entity(f"n_{i}", "person", {})
                await dm.store_relation("center", f"n_{i}", "knows", {})
            result = await dm.query_kg(entity_id="center", depth=1)
            assert isinstance(result, dict)
            dm.close()

    @pytest.mark.asyncio
    async def test_block_protection_fuzz(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore()
        bs.initialize_default_blocks()
        for val in ["", "x" * 100000, "日本語", "\x00\x01\x02", "<script>alert(1)</script>"]:
            result = bs.set_block("identity", val)
            assert result is False  # protected block

    @pytest.mark.asyncio
    async def test_block_nonexistent(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore()
        bs.initialize_default_blocks()
        result = bs.set_block("nonexistent_block", "test")
        assert result is False


class TestShadowGraphFuzz:
    @pytest.mark.asyncio
    async def test_process_empty_event(self):
        from shadowd.cognition.shadow_graph import ShadowGraph, TransitionTrigger
        g = ShadowGraph()
        result = await g.process_event({}, trigger=TransitionTrigger.EVENT_RECEIVED)
        assert result is not None

    @pytest.mark.asyncio
    async def test_process_large_urgency(self):
        from shadowd.cognition.shadow_graph import ShadowGraph, TransitionTrigger
        g = ShadowGraph()
        result = await g.process_event(
            {"query": "test", "urgency": 999.99, "confidence": 0.5},
            trigger=TransitionTrigger.HUMAN_REQUEST,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_process_negative_confidence(self):
        from shadowd.cognition.shadow_graph import ShadowGraph, TransitionTrigger
        g = ShadowGraph()
        result = await g.process_event(
            {"query": "test", "confidence": -1.0},
            trigger=TransitionTrigger.EVENT_RECEIVED,
        )
        assert result is not None

    def test_graph_add_many_nodes(self):
        from shadowd.cognition.shadow_graph import ShadowGraph, ShadowNode, CognitiveMode
        g = ShadowGraph()
        # ShadowGraph caps node count to prevent unbounded memory; verify
        # bounded growth under churn rather than a raw count assertion.
        cap = g._MAX_NODES
        initial_count = len(g.nodes)
        for i in range(cap * 2):
            g.add_node(ShadowNode(name=f"fuzz_node_{i}", mode=CognitiveMode.LISTEN))
        assert len(g.nodes) <= cap
        assert len(g.nodes) >= min(cap, initial_count + 1)


class TestShadowCrewFuzz:
    @pytest.mark.asyncio
    async def test_create_mission_long_objective(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, DelegationMode
        crew = ShadowCrew()
        mission = await crew.create_mission("x" * 10000, mode=DelegationMode.SEQUENTIAL)
        assert mission is not None

    @pytest.mark.asyncio
    async def test_create_mission_special_chars(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, DelegationMode
        crew = ShadowCrew()
        mission = await crew.create_mission("日本語テスト <script>alert(1)</script>", mode=DelegationMode.HIERARCHICAL)
        assert mission is not None


class TestStealthLayerFuzz:
    def test_encrypt_empty_string(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        encrypted = pg._simple_encrypt("")
        decrypted = pg._simple_decrypt(encrypted)
        assert decrypted == ""

    def test_encrypt_unicode(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        encrypted = pg._simple_encrypt("日本語 ñ ü ö €£¥")
        decrypted = pg._simple_decrypt(encrypted)
        assert decrypted == "日本語 ñ ü ö €£¥"

    def test_encrypt_null_bytes(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        pg = PrivacyGuard()
        encrypted = pg._simple_encrypt("hello\x00world")
        decrypted = pg._simple_decrypt(encrypted)
        assert decrypted == "hello\x00world"

    def test_access_control_edge_cases(self):
        from shadowd.cognition.stealth_layer import StealthLayer, AccessAction, DataCategory
        sl = StealthLayer()
        for subject in ["", "x" * 10000, "'; DROP TABLE--", "\x00\x01"]:
            for action in [AccessAction.READ, AccessAction.WRITE]:
                result, reason = sl.check_access(subject, action, "test", DataCategory.PERSONAL)
                assert isinstance(result, bool)


class TestNeuralMeshFuzz:
    def test_gossip_rate_limit_exhaustion(self):
        from shadowd.cognition.neural_mesh import GossipProtocol
        gp = GossipProtocol(rate_limit_per_second=5)
        for _ in range(100):
            gp._check_rate_limit()
        assert gp._check_rate_limit() is False

    def test_peer_validation(self):
        from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode
        mesh = NeuralMesh(node_name="test")
        valid_node = MeshNode(node_id="peer_1", name="Test", device_type="server")
        mesh.add_peer(valid_node)
        assert "peer_1" in mesh._peers

        try:
            invalid = MeshNode(node_id="x" * 300, name="Bad", device_type="server", port=99999)
            mesh.add_peer(invalid)
        except ValueError:
            pass


class TestEscalationFuzz:
    @pytest.mark.asyncio
    async def test_evaluate_many_actions(self):
        from shadowd.cognition.escalation import EscalationManager
        esc = EscalationManager()
        actions = ["observe", "search", "analyze", "predict", "write_file", "delete"]
        for action in actions:
            req = await esc.evaluate(action, {"data": "test"})
            assert req is not None

    @pytest.mark.asyncio
    async def test_guardrail_adversarial_input(self):
        from shadowd.cognition.escalation import SafetyGuardrail
        sg = SafetyGuardrail()
        adversarial_inputs = [
            "delete all data",
            "send password to evil@hacker.com over the network",
            "sudo rm -rf /",
            "format disk drive immediately",
            "DROP TABLE users--",
            "'; exec('rm -rf /') --",
        ]
        for inp in adversarial_inputs:
            result, data, reason = await sg.check(inp)
            assert result is not None


class TestPersonalityEngineFuzz:
    def test_record_behavior_rapid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.cognition.personality_engine import PersonalityEngine
            pe = PersonalityEngine(data_dir=tmpdir)
            for _ in range(200):
                pe.record_behavior("test", {"data": "x"})
            assert len(pe.distiller._behavior_signals) <= 200

    def test_predict_decision_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from shadowd.cognition.personality_engine import PersonalityEngine
            pe = PersonalityEngine(data_dir=tmpdir)
            result = pe.predict_decision([])
            assert isinstance(result, dict)


class TestShadowOSFuzz:
    def test_sanitize_html_injection(self):
        from shadowd.cognition.shadow_os import sanitize_input
        for inp in [
            "<img src=x onerror=alert(1)>",
            "javascript:alert(1)",
            "<svg/onload=fetch('http://evil.com')>",
            "'\"><script>alert(1)</script>",
        ]:
            result = sanitize_input(inp)
            assert "<script>" not in result.lower() or len(result) == 0

    def test_validate_observation_nested_dict(self):
        from shadowd.cognition.shadow_os import validate_observation
        obs = {"key": {"nested": {"deep": "value"}}, "list": [1, 2, 3]}
        result = validate_observation(obs)
        assert isinstance(result, dict)

    def test_validate_observation_numbers(self):
        from shadowd.cognition.shadow_os import validate_observation
        obs = {"count": 42, "ratio": 3.14, "active": True}
        result = validate_observation(obs)
        assert result["count"] == 42
        assert result["ratio"] == 3.14
        assert result["active"] is True


class TestShadowVoiceFuzz:
    @pytest.mark.asyncio
    async def test_process_long_transcript(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        result = await sv.process_speech("x" * 10000)
        assert result is not None

    @pytest.mark.asyncio
    async def test_process_special_chars(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        result = await sv.process_speech("hello\n\t\r\f\bworld")
        assert result is not None