#!/usr/bin/env python3
"""
Comprehensive tests for ShadowOS cognitive modules.

Tests:
- ShadowGraph: Node/edge creation, state transitions, checkpoints, execution
- DarkMemory: Store/retrieve/search, vector index, temporal KG, blocks
- ShadowCrew: Agent registration, mission planning/execution, delegation
- NeuralMesh: Peer management, gossip, sync, consensus
- StealthLayer: Privacy guard, camouflage, data minimization
- PersonalityEngine: Trait updates, pattern detection, decision prediction
- ShadowVoice: Session management, intent parsing, command routing
- ShadowOS: Integration tests across all modules
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowd.cognition.shadow_graph import (
    ShadowGraph, ShadowNode, ShadowEdge, CognitiveState, CognitiveMode,
    TransitionTrigger, ApprovalLevel, Checkpoint, ExecutionTrace,
)
from shadowd.cognition.dark_memory import (
    DarkMemory, MemoryAtom, AtomType, MemoryTier, TemporalKnowledgeGraph,
    InMemoryVectorIndex, BlockStore, MemoryCompactor,
)
from shadowd.cognition.shadow_crew import (
    ShadowCrew, ShadowAgent, AgentRole, CrewMission, CrewTask,
    DelegationMode, MissionStatus, AgentCapability, AgentRegistry,
)
from shadowd.cognition.neural_mesh import (
    NeuralMesh, MeshNode, MeshMessage, MeshMessageType, NodeStatus,
    GossipProtocol, StateSync, MeshConsensus,
)
from shadowd.cognition.stealth_layer import (
    StealthLayer, PrivacyGuard, CamouflageEngine, DataMinimizer,
    SensitivityLevel, DataCategory, AccessAction, CamouflageMode,
    AuditEntry, DataClassification, AccessPolicy,
)
from shadowd.cognition.personality_engine import (
    PersonalityEngine, PersonalityProfile, BigFiveProfile, BehaviorPattern,
    TraitDimension, DecisionStyle, CommunicationStyle, ValuePriority,
    PersonaDistiller,
)
from shadowd.cognition.shadow_voice import (
    ShadowVoice, VoiceSession, VoiceCommand, VoiceIntent, VoiceState,
    IntentParser as VoiceIntentParser, TTSEngine, STTEngine, VoicePersonality,
)
from shadowd.cognition.shadow_os import ShadowOS


# ============================================================================
# ShadowGraph Tests
# ============================================================================

class TestCognitiveState:
    def test_default_state(self):
        state = CognitiveState()
        assert state.mode == CognitiveMode.IDLE
        assert state.confidence == 0.0
        assert state.urgency == 0.0
        assert isinstance(state.observations, list)
        assert isinstance(state.context, dict)

    def test_transition_to(self):
        state = CognitiveState()
        state.transition_to(CognitiveMode.LISTEN)
        assert state.mode == CognitiveMode.LISTEN
        assert state.previous_mode == CognitiveMode.IDLE

    def test_to_dict(self):
        state = CognitiveState(shadow_id="test", mode=CognitiveMode.ANALYZE)
        d = state.to_dict()
        assert d["shadow_id"] == "test"
        assert d["mode"] == "analyze"

    def test_checkpoint_hash(self):
        state = CognitiveState(shadow_id="test")
        h = state.checkpoint_hash()
        assert isinstance(h, str)
        assert len(h) == 16


class TestShadowNode:
    def test_node_creation(self):
        node = ShadowNode(name="test", mode=CognitiveMode.LISTEN)
        assert node.name == "test"
        assert node.mode == CognitiveMode.LISTEN
        assert node.approval_level == ApprovalLevel.AUTO

    def test_node_hash_equality(self):
        n1 = ShadowNode(name="test", mode=CognitiveMode.LISTEN)
        n2 = ShadowNode(name="test", mode=CognitiveMode.ANALYZE)
        assert n1 == n2
        assert hash(n1) == hash(n2)


class TestShadowEdge:
    def test_edge_creation(self):
        edge = ShadowEdge(source="listen", target="analyze")
        assert edge.source == "listen"
        assert edge.target == "analyze"
        assert edge.priority == 5

    def test_edge_with_condition(self):
        edge = ShadowEdge(
            source="reason", target="respond",
            condition=lambda s: s.confidence > 0.5
        )
        state = CognitiveState(confidence=0.7)
        assert edge.should_traverse(state) is True
        state.confidence = 0.3
        assert edge.should_traverse(state) is False

    def test_edge_with_trigger(self):
        edge = ShadowEdge(source="listen", target="analyze", trigger=TransitionTrigger.EVENT_RECEIVED)
        assert edge.should_traverse(CognitiveState(), TransitionTrigger.EVENT_RECEIVED) is True
        assert edge.should_traverse(CognitiveState(), TransitionTrigger.ERROR) is False


class TestShadowGraph:
    def test_default_graph_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = ShadowGraph(graph_id="test", data_dir=tmpdir)
            assert len(graph.nodes) == 10
            assert "listen" in graph.nodes
            assert "analyze" in graph.nodes
            assert "reason" in graph.nodes

    def test_add_custom_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = ShadowGraph(graph_id="test", data_dir=tmpdir)
            custom = ShadowNode(name="custom", mode=CognitiveMode.IDLE)
            graph.add_node(custom)
            assert "custom" in graph.nodes

    def test_add_custom_edge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = ShadowGraph(graph_id="test", data_dir=tmpdir)
            graph.add_edge("listen", "reflect", priority=10)
            assert len(graph.edges) > 0

    def test_register_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = ShadowGraph(graph_id="test", data_dir=tmpdir)
            called = []
            def handler(state):
                called.append(True)
                return state
            graph.register_handler("analyze", handler)
            assert "analyze" in graph._handlers

    @pytest.mark.asyncio
    async def test_process_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = ShadowGraph(graph_id="test", data_dir=tmpdir)
            result = await graph.process_event(
                event={"query": "test query"},
                trigger=TransitionTrigger.HUMAN_REQUEST,
            )
            assert isinstance(result, CognitiveState)

    def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = ShadowGraph(graph_id="test", data_dir=tmpdir)
            status = graph.get_status()
            assert status["graph_id"] == "test"
            assert status["current_mode"] is not None
            assert "nodes" in status

    def test_get_graph_visualization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = ShadowGraph(graph_id="test", data_dir=tmpdir)
            dot = graph.get_graph_visualization()
            assert "digraph" in dot
            assert "listen" in dot

    def test_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = ShadowGraph(graph_id="test", data_dir=tmpdir)
            graph.state.confidence = 0.9
            graph.reset()
            assert graph.state.confidence == 0.0
            assert graph.state.mode == CognitiveMode.IDLE

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = ShadowGraph(graph_id="test", data_dir=tmpdir)
            result = await graph.resume_from_checkpoint()
            assert isinstance(result, CognitiveState)


# ============================================================================
# DarkMemory Tests
# ============================================================================

class TestMemoryAtom:
    def test_atom_creation(self):
        atom = MemoryAtom(content="test content", atom_type=AtomType.OBSERVATION)
        assert atom.atom_id.startswith("atom_")
        assert atom.tier == MemoryTier.WORKING
        assert atom.importance == 0.5

    def test_atom_touch(self):
        atom = MemoryAtom(content="test")
        initial_count = atom.access_count
        atom.touch()
        assert atom.access_count == initial_count + 1

    def test_atom_to_dict(self):
        atom = MemoryAtom(content="test", atom_type=AtomType.DECISION)
        d = atom.to_dict()
        assert d["atom_type"] == "decision"
        assert "atom_id" in d


class TestInMemoryVectorIndex:
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        index = InMemoryVectorIndex(dimension=4)
        await index.add("doc1", [1.0, 0.0, 0.0, 0.0])
        await index.add("doc2", [0.0, 1.0, 0.0, 0.0])
        results = await index.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == "doc1"
        assert results[0][1] > 0.99

    @pytest.mark.asyncio
    async def test_remove(self):
        index = InMemoryVectorIndex(dimension=4)
        await index.add("doc1", [1.0, 0.0, 0.0, 0.0])
        await index.remove("doc1")
        assert index.size() == 0

    def test_cosine_similarity(self):
        index = InMemoryVectorIndex(dimension=4)
        sim = index._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 0.01
        sim = index._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 0.01


class TestTemporalKnowledgeGraph:
    def test_add_entity(self):
        kg = TemporalKnowledgeGraph()
        eid = kg.add_entity("person_1", "person", {"name": "Alice"})
        assert eid == "person_1"
        entity = kg.get_entity("person_1")
        assert entity is not None
        assert entity["type"] == "person"

    def test_add_relation(self):
        kg = TemporalKnowledgeGraph()
        kg.add_entity("a", "person", {"name": "Alice"})
        kg.add_entity("b", "person", {"name": "Bob"})
        rid = kg.add_relation("a", "b", "knows")
        assert rid.startswith("rel_")

    def test_get_relations(self):
        kg = TemporalKnowledgeGraph()
        kg.add_entity("a", "person", {})
        kg.add_entity("b", "person", {})
        kg.add_relation("a", "b", "works_with")
        relations = kg.get_relations(entity_id="a")
        assert len(relations) == 1
        assert relations[0]["type"] == "works_with"

    def test_get_neighbors(self):
        kg = TemporalKnowledgeGraph()
        kg.add_entity("a", "person", {})
        kg.add_entity("b", "person", {})
        kg.add_relation("a", "b", "knows")
        neighbors = kg.get_neighbors("a")
        assert len(neighbors["outgoing"]) == 1

    def test_search_entities(self):
        kg = TemporalKnowledgeGraph()
        kg.add_entity("alice", "person", {"name": "Alice"})
        kg.add_entity("bob", "person", {"name": "Bob"})
        results = kg.search_entities("alice")
        assert len(results) == 1

    def test_get_timeline(self):
        kg = TemporalKnowledgeGraph()
        kg.add_entity("alice", "person", {"name": "Alice"})
        timeline = kg.get_timeline("alice")
        assert len(timeline) >= 1

    def test_get_stats(self):
        kg = TemporalKnowledgeGraph()
        kg.add_entity("a", "person", {})
        kg.add_entity("b", "person", {})
        kg.add_relation("a", "b", "knows")
        stats = kg.get_stats()
        assert stats["entities"] == 2
        assert stats["relations"] == 1


class TestBlockStore:
    def test_initialize_defaults(self):
        store = BlockStore()
        store.initialize_default_blocks()
        blocks = store.list_blocks()
        assert len(blocks) >= 7
        labels = [b["label"] for b in blocks]
        assert "identity" in labels
        assert "persona" in labels
        assert "human" in labels

    def test_get_set_block(self):
        store = BlockStore()
        store.initialize_default_blocks()
        store.set_block("context", "Current context: testing")
        block = store.get_block("context")
        assert block is not None
        assert "testing" in block["value"]

    def test_protected_block(self):
        store = BlockStore()
        store.initialize_default_blocks()
        store.set_block("identity", "New identity value")
        result = store.set_block("identity", "Another value")
        assert result is False

    def test_replace_in_block(self):
        store = BlockStore()
        store.initialize_default_blocks()
        store.set_block("context", "Hello world")
        result = store.replace_in_block("context", "world", "universe")
        assert result is True
        block = store.get_block("context")
        assert "universe" in block["value"]

    def test_insert_in_block(self):
        store = BlockStore()
        store.initialize_default_blocks()
        store.set_block("context", "Line 1\nLine 3")
        result = store.insert_in_block("context", 1, "Line 2")
        assert result is True

    def test_get_core_context(self):
        store = BlockStore()
        store.initialize_default_blocks()
        context = store.get_core_context()
        assert "<identity>" in context
        assert "</identity>" in context


class TestDarkMemory:
    @pytest.fixture
    def memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = DarkMemory(data_dir=tmpdir)
            yield mem
            mem.close()

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, memory):
        atom = await memory.store("Test observation", atom_type=AtomType.OBSERVATION)
        assert atom.atom_id is not None
        retrieved = await memory.retrieve(atom.atom_id)
        assert retrieved is not None
        assert retrieved.content == "Test observation"

    @pytest.mark.asyncio
    async def test_search(self, memory):
        await memory.store("Python programming language", atom_type=AtomType.OBSERVATION, tags=["tech"])
        await memory.store("Machine learning models", atom_type=AtomType.OBSERVATION, tags=["tech"])
        results = await memory.search("Python", limit=5)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_store_with_domain(self, memory):
        await memory.store("Work task completed", domain="career", importance=0.8)
        results = await memory.get_by_domain("career")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_kg_operations(self, memory):
        await memory.store_entity("person_alice", "person", {"name": "Alice"})
        await memory.store_entity("person_bob", "person", {"name": "Bob"})
        await memory.store_relation("person_alice", "person_bob", "works_with")
        result = await memory.query_kg(entity_id="person_alice")
        assert result["entity"] is not None

    def test_blocks(self, memory):
        blocks = memory.list_blocks()
        assert len(blocks) >= 7

    def test_get_core_context(self, memory):
        context = memory.get_core_context()
        assert "<identity>" in context

    @pytest.mark.asyncio
    async def test_compact(self, memory):
        for i in range(10):
            await memory.store(f"Item {i}", importance=0.1)
        removed = await memory.compact()
        assert isinstance(removed, int)

    def test_get_stats(self, memory):
        stats = memory.get_stats()
        assert "total_atoms" in stats
        assert "vector_index_size" in stats


# ============================================================================
# ShadowCrew Tests
# ============================================================================

class TestShadowAgent:
    def test_agent_creation(self):
        agent = ShadowAgent(
            agent_id="test_agent",
            name="TestAgent",
            role=AgentRole.OBSERVER,
            goal="Test things",
            backstory="I am a test agent",
        )
        assert agent.agent_id == "test_agent"
        assert agent.role == AgentRole.OBSERVER

    def test_can_handle(self):
        agent = ShadowAgent(
            agent_id="test",
            name="TestAgent",
            role=AgentRole.OBSERVER,
            goal="Test",
            backstory="Test",
            capabilities=[AgentCapability(name="scan", description="Scan")],
        )
        assert agent.can_handle("scan") is True
        assert agent.can_handle("fly") is False

    def test_to_dict(self):
        agent = ShadowAgent(
            agent_id="test", name="T", role=AgentRole.ACTOR,
            goal="G", backstory="B",
        )
        d = agent.to_dict()
        assert d["agent_id"] == "test"
        assert d["role"] == "observer" or d["role"] == "actor"


class TestShadowCrew:
    def test_default_agents(self):
        crew = ShadowCrew()
        agents = crew.list_agents()
        assert len(agents) >= 6
        names = [a["name"] for a in agents]
        assert "ShadowEye" in names
        assert "ShadowMind" in names

    @pytest.mark.asyncio
    async def test_create_mission(self):
        crew = ShadowCrew()
        mission = await crew.create_mission("Analyze the environment")
        assert mission.mission_id is not None
        assert mission.status == MissionStatus.PENDING
        assert len(mission.tasks) > 0

    @pytest.mark.asyncio
    async def test_execute_mission(self):
        crew = ShadowCrew()
        mission = await crew.create_mission("Observe and report", DelegationMode.HIERARCHICAL)
        result = await crew.execute_mission(mission.mission_id)
        assert "mission_id" in result
        assert "status" in result

    @pytest.mark.asyncio
    async def test_predict_mission(self):
        crew = ShadowCrew()
        mission = await crew.create_mission("Predict future events", DelegationMode.HIERARCHICAL)
        task_descriptions = [t.description for t in mission.tasks]
        has_predict = any("predict" in d.lower() or "forecast" in d.lower() for d in task_descriptions)
        assert has_predict or len(mission.tasks) > 0

    @pytest.mark.asyncio
    async def test_execute_mission(self):
        crew = ShadowCrew()
        mission = await crew.create_mission("Observe and respond")
        result = await crew.execute_mission(mission.mission_id)
        assert result["status"] in ["completed", "failed", "escalated"]

    def test_get_agent(self):
        crew = ShadowCrew()
        agent = crew.get_agent("shadow_mind")
        assert agent is not None
        assert agent.name == "ShadowMind"

    def test_get_stats(self):
        crew = ShadowCrew()
        stats = crew.get_stats()
        assert stats["agents_count"] >= 6


# ============================================================================
# NeuralMesh Tests
# ============================================================================

class TestMeshNode:
    def test_node_creation(self):
        node = MeshNode(node_id="n1", name="Test", device_type="server")
        assert node.node_id == "n1"
        assert node.status == NodeStatus.ONLINE

    def test_is_alive(self):
        node = MeshNode(node_id="n1", name="Test", status=NodeStatus.ONLINE)
        assert node.is_alive() is True
        node.status = NodeStatus.OFFLINE
        assert node.is_alive() is False

    def test_to_dict(self):
        node = MeshNode(node_id="n1", name="Test")
        d = node.to_dict()
        assert d["node_id"] == "n1"


class TestGossipProtocol:
    def test_should_propagate_new(self):
        gossip = GossipProtocol()
        msg = MeshMessage(sender_id="n1", payload={"test": True})
        assert gossip.should_propagate(msg) is True

    def test_should_not_propagate_duplicate(self):
        gossip = GossipProtocol()
        msg = MeshMessage(sender_id="n1", payload={"test": True})
        gossip.should_propagate(msg)
        assert gossip.should_propagate(msg) is False

    def test_select_peers(self):
        gossip = GossipProtocol(fanout=2)
        peers = [
            MeshNode(node_id=f"p{i}", name=f"Peer{i}", status=NodeStatus.ONLINE)
            for i in range(5)
        ]
        selected = gossip.select_peers(peers)
        assert len(selected) <= 2


class TestStateSync:
    def test_update_and_get(self):
        sync = StateSync()
        sync.update_local("key1", "value1")
        assert sync.get_local("key1") == "value1"

    def test_merge_remote_newer(self):
        sync = StateSync()
        sync.update_local("key1", "old_value")
        remote = {"value": "new_value", "version": 2, "updated_at": datetime.now(timezone.utc).isoformat()}
        result = sync.merge_remote("key1", remote, "remote_node")
        assert result is True

    def test_get_sync_payload(self):
        sync = StateSync()
        sync.update_local("key1", "value1")
        payload = sync.get_sync_payload()
        assert "updates" in payload
        assert "key1" in payload["updates"]


class TestNeuralMesh:
    def test_init(self):
        mesh = NeuralMesh(node_name="test-mesh")
        assert mesh.local_node.name == "test-mesh"
        assert len(mesh.local_node.capabilities) > 0

    def test_add_peer(self):
        mesh = NeuralMesh()
        peer = MeshNode(node_id="peer1", name="TestPeer")
        mesh.add_peer(peer)
        peers = mesh.get_peers()
        assert len(peers) == 1

    def test_remove_peer(self):
        mesh = NeuralMesh()
        peer = MeshNode(node_id="peer1", name="TestPeer")
        mesh.add_peer(peer)
        mesh.remove_peer("peer1")
        assert len(mesh.get_peers()) == 0

    def test_local_state(self):
        mesh = NeuralMesh()
        mesh.update_local_state("test_key", "test_value")
        assert mesh.get_local_state("test_key") == "test_value"

    @pytest.mark.asyncio
    async def test_propose_consensus(self):
        mesh = NeuralMesh()
        proposal = await mesh.propose_consensus("key1", "value1")
        assert proposal["proposal_id"] is not None
        assert proposal["key"] == "key1"

    def test_get_stats(self):
        mesh = NeuralMesh()
        stats = mesh.get_stats()
        assert "local_node" in stats
        assert "peers_count" in stats


# ============================================================================
# StealthLayer Tests
# ============================================================================

class TestPrivacyGuard:
    def test_classify_data(self):
        guard = PrivacyGuard()
        cls = guard.classify_data("test", DataCategory.PERSONAL)
        assert cls.sensitivity == SensitivityLevel.CONFIDENTIAL

    def test_check_access_allowed(self):
        guard = PrivacyGuard()
        granted, reason = guard.check_access("shadow_ai", AccessAction.READ, "test", DataCategory.NETWORK)
        assert granted is True

    def test_check_access_denied_no_policy(self):
        guard = PrivacyGuard()
        granted, reason = guard.check_access("unknown_subject", AccessAction.READ, "test", DataCategory.PERSONAL)
        assert granted is False
        assert "No access policy" in reason

    def test_store_and_retrieve(self):
        guard = PrivacyGuard()
        guard.store_data("test_data", {"secret": "value"}, DataCategory.PERSONAL)
        result = guard.retrieve_data("test_data", "shadow_ai")
        assert result is not None

    def test_audit_log(self):
        guard = PrivacyGuard()
        guard.check_access("shadow_ai", AccessAction.READ, "test", DataCategory.NETWORK)
        log = guard.get_audit_log()
        assert len(log) > 0


class TestCamouflageEngine:
    def test_anonymize(self):
        engine = CamouflageEngine()
        data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
        result = engine.anonymize(data)
        assert result["age"] == 30
        assert result["name"].startswith("anon_")
        assert result["email"].startswith("anon_")

    def test_pseudonymize(self):
        engine = CamouflageEngine()
        data = {"id": "user123", "data": "value"}
        result = engine.pseudonymize(data, "id")
        assert result["id"].startswith("pseudo_")

    def test_aggregate(self):
        engine = CamouflageEngine()
        data_points = [{"value": 10}, {"value": 20}, {"value": 30}]
        result = engine.aggregate(data_points, "value")
        assert result["count"] == 3
        assert result["mean"] == 20.0

    def test_differential_privacy(self):
        engine = CamouflageEngine(epsilon=1.0)
        original = 100.0
        noisy = engine.apply_differential_privacy(original)
        assert abs(noisy - original) < 50  # With epsilon=1.0, noise should be reasonable

    def test_sensitivity_score(self):
        engine = CamouflageEngine()
        low_sens = engine.get_sensitivity_score({"name": "Alice", "age": 30})
        high_sens = engine.get_sensitivity_score({"ssn": "123-45-6789", "password": "secret"})
        assert high_sens > low_sens


class TestDataMinimizer:
    def test_minimize(self):
        minimizer = DataMinimizer()
        minimizer.register_purpose("test", ["name", "age"], 30)
        data = {"name": "Alice", "age": 30, "ssn": "123-45-6789", "password": "secret"}
        result = minimizer.minimize(data, "test")
        assert "name" in result
        assert "age" in result
        assert "ssn" not in result
        assert "password" not in result


class TestStealthLayer:
    def test_store_and_retrieve(self):
        stealth = StealthLayer()
        stealth.store("test1", {"data": "value"}, DataCategory.PERSONAL)
        result = stealth.retrieve("test1", "shadow_ai")
        assert result is not None

    def test_access_denied(self):
        stealth = StealthLayer()
        granted, reason = stealth.check_access("external", AccessAction.WRITE, "test", DataCategory.CREDENTIALS)
        assert granted is False

    def test_transform_for_sharing(self):
        stealth = StealthLayer()
        data = {"name": "Alice", "email": "a@b.com"}
        result = stealth.transform_for_sharing(data, CamouflageMode.ANONYMIZE)
        assert result["name"].startswith("anon_")

    def test_get_sensitivity_score(self):
        stealth = StealthLayer()
        score = stealth.get_sensitivity_score({"name": "Alice"})
        assert 0.0 <= score <= 1.0

    def test_audit_log(self):
        stealth = StealthLayer()
        stealth.check_access("shadow_ai", AccessAction.READ, "test", DataCategory.PERSONAL)
        stealth.store("audit_test", {"data": "value"}, DataCategory.PERSONAL)
        log = stealth.get_audit_log()
        assert len(log) > 0

    def test_get_stats(self):
        stealth = StealthLayer()
        stats = stealth.get_stats()
        assert "stealth_stats" in stats
        assert "guard_stats" in stats


# ============================================================================
# PersonalityEngine Tests
# ============================================================================

class TestBigFiveProfile:
    def test_default(self):
        profile = BigFiveProfile()
        assert profile.openness == 0.5
        assert profile.conscientiousness == 0.5

    def test_update(self):
        profile = BigFiveProfile()
        profile.update(TraitDimension.OPENNESS, 0.1)
        assert profile.openness == 0.6

    def test_clamping(self):
        profile = BigFiveProfile()
        profile.update(TraitDimension.OPENNESS, 1.0)
        assert profile.openness == 1.0
        profile.update(TraitDimension.OPENNESS, 0.5)
        assert profile.openness == 1.0  # Capped

    def test_to_prompt_fragment(self):
        profile = BigFiveProfile(openness=0.8, conscientiousness=0.7)
        fragment = profile.to_prompt_fragment()
        assert isinstance(fragment, str)
        assert len(fragment) > 0


class TestBehaviorPattern:
    def test_creation(self):
        pattern = BehaviorPattern(name="test_pattern", description="A test pattern")
        assert pattern.pattern_id.startswith("pat_")
        assert pattern.confidence == 0.0

    def test_reinforce(self):
        pattern = BehaviorPattern(name="test", description="test")
        pattern.reinforce("example")
        assert pattern.evidence_count == 1
        assert pattern.confidence > 0
        assert len(pattern.examples) == 1


class TestPersonalityProfile:
    def test_creation(self):
        profile = PersonalityProfile()
        assert profile.profile_id.startswith("pers_")
        assert profile.decision_style == DecisionStyle.DELIBERATIVE

    def test_add_decision(self):
        profile = PersonalityProfile()
        profile.add_decision({"choice": "A", "reason": "test"})
        assert len(profile.decision_history) == 1

    def test_find_or_add_pattern(self):
        profile = PersonalityProfile()
        pattern = profile.add_or_reinforce_pattern("test", "A test pattern", "example")
        assert pattern.name == "test"
        same = profile.add_or_reinforce_pattern("test", "A test pattern", "example2")
        assert same.evidence_count == 2

    def test_to_system_prompt(self):
        profile = PersonalityProfile()
        prompt = profile.to_system_prompt()
        assert "<personality_profile>" in prompt

    def test_predict_decision(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonalityEngine(data_dir=tmpdir)
            options = [
                {"name": "safe_option", "reward": 0.5, "risk_level": 0.1, "novelty": 0.2},
                {"name": "risky_option", "reward": 0.9, "risk_level": 0.8, "novelty": 0.9},
            ]
            result = engine.predict_decision(options)
            assert "predicted_choice" in result
            assert "scores" in result


class TestPersonaDistiller:
    def test_record_behavior(self):
        distiller = PersonaDistiller()
        distiller.record_behavior("question_asked", {"complexity": "complex"})
        assert len(distiller._behavior_signals) == 1

    def test_distill_trait_updates(self):
        distiller = PersonaDistiller()
        distiller.record_behavior("question_asked", {"complexity": "complex"})
        updates = distiller.distill_trait_updates()
        assert len(updates) > 0


class TestPersonalityEngine:
    @pytest.fixture
    def engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eng = PersonalityEngine(data_dir=tmpdir)
            yield eng

    def test_record_behavior(self, engine):
        engine.record_behavior("decision_made", {"speed": "fast", "risk_level": "high"})
        assert engine._stats["behaviors_recorded"] == 1

    @pytest.mark.asyncio
    async def test_distill(self, engine):
        engine.record_behavior("question_asked", {"complexity": "complex"})
        result = await engine.distill()
        assert "trait_updates" in result
        assert "profile_version" in result

    def test_predict_decision(self, engine):
        options = [{"name": "A", "reward": 0.7, "risk_level": 0.3}]
        result = engine.predict_decision(options)
        assert "predicted_choice" in result
        assert "scores" in result

    def test_get_profile(self, engine):
        profile = engine.get_profile()
        assert "big_five" in profile
        assert "decision_style" in profile

    def test_get_patterns(self, engine):
        engine.record_behavior("decision_made", {"speed": "fast"})
        patterns = engine.get_patterns()
        assert isinstance(patterns, list)

    def test_get_stats(self, engine):
        stats = engine.get_stats()
        assert "profile_version" in stats
        assert "big_five" in stats


# ============================================================================
# ShadowVoice Tests
# ============================================================================

class TestVoiceIntentParser:
    def test_parse_query(self):
        parser = VoiceIntentParser()
        intent, confidence = parser.parse("What's the weather?")
        assert intent == VoiceIntent.QUERY
        assert confidence > 0

    def test_parse_command(self):
        parser = VoiceIntentParser()
        intent, confidence = parser.parse("Start the analysis")
        assert intent == VoiceIntent.COMMAND

    def test_parse_greeting(self):
        parser = VoiceIntentParser()
        intent, confidence = parser.parse("Hello shadow")
        assert intent == VoiceIntent.GREETING

    def test_parse_emergency(self):
        parser = VoiceIntentParser()
        intent, confidence = parser.parse("Emergency alert!")
        assert intent == VoiceIntent.EMERGENCY

    def test_parse_cancel(self):
        parser = VoiceIntentParser()
        intent, confidence = parser.parse("Cancel that")
        assert intent == VoiceIntent.CANCEL

    def test_extract_entities(self):
        parser = VoiceIntentParser()
        entities = parser.extract_entities("What will happen tomorrow with oracle?")
        assert entities.get("time_reference") == "tomorrow"
        assert entities.get("ability") == "oracle"


class TestVoiceSession:
    def test_creation(self):
        session = VoiceSession()
        assert session.session_id.startswith("vsess_")
        assert session.state == VoiceState.IDLE

    def test_add_transcript(self):
        session = VoiceSession()
        session.add_transcript("user", "Hello")
        session.add_transcript("shadow", "Hi there")
        assert len(session.transcript) == 2

    def test_end(self):
        session = VoiceSession()
        session.end()
        assert session.ended_at is not None
        assert session.duration_seconds >= 0


class TestVoicePersonality:
    def test_adapt_response(self):
        vp = VoicePersonality()
        text = "This is a very long response that should be shortened for concise mode. " * 5
        adapted = vp.adapt_response(text, {"verbosity": "concise"})
        assert len(adapted) <= len(text) or len(text) < 200

    def test_wake_response(self):
        vp = VoicePersonality(communication_style="casual")
        response = vp.get_wake_response()
        assert isinstance(response, str)
        assert len(response) > 0


class TestShadowVoice:
    def test_init(self):
        voice = ShadowVoice()
        assert voice._state == VoiceState.IDLE

    @pytest.mark.asyncio
    async def test_process_speech(self):
        voice = ShadowVoice()
        result = await voice.process_speech("What's the weather today?")
        assert "command" in result
        assert result["command"]["intent"] in ["query", "unknown", "conversation"]

    def test_get_state(self):
        voice = ShadowVoice()
        state = voice.get_state()
        assert state["state"] == "idle"

    def test_get_stats(self):
        voice = ShadowVoice()
        stats = voice.get_stats()
        assert "state" in stats
        assert "stats" in stats


# ============================================================================
# ShadowOS Integration Tests
# ============================================================================

class TestShadowOS:
    @pytest.fixture
    def os_instance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os_inst = ShadowOS(data_dir=tmpdir)
            yield os_inst

    def test_init(self, os_instance):
        assert os_instance.graph is not None
        assert os_instance.memory is not None
        assert os_instance.crew is not None
        assert os_instance.mesh is not None
        assert os_instance.stealth is not None
        assert os_instance.personality is not None
        assert os_instance.voice is not None

    @pytest.mark.asyncio
    async def test_start_stop(self, os_instance):
        await os_instance.start()
        assert os_instance._running is True
        await os_instance.stop()
        assert os_instance._running is False

    @pytest.mark.asyncio
    async def test_process_query(self, os_instance):
        await os_instance.start()
        result = await os_instance.process_query("What patterns do you see?")
        assert "query" in result
        assert "cognitive_mode" in result
        await os_instance.stop()

    @pytest.mark.asyncio
    async def test_process_observation(self, os_instance):
        await os_instance.start()
        result = await os_instance.process_observation({
            "type": "terminal",
            "content": "User ran: git commit",
            "domain": "career",
        })
        assert result["status"] == "observed"
        assert "atom_id" in result
        await os_instance.stop()

    @pytest.mark.asyncio
    async def test_execute_action(self, os_instance):
        await os_instance.start()
        result = await os_instance.execute_action("analyze", {"target": "recent_data"})
        assert isinstance(result, dict)
        await os_instance.stop()

    def test_get_health(self, os_instance):
        health = os_instance.get_health()
        assert health["status"] in ["healthy", "stopped"]
        assert "modules" in health

    def test_get_status(self, os_instance):
        status = os_instance.get_status()
        assert "graph_status" in status
        assert "memory_stats" in status
        assert "personality_stats" in status

    @pytest.mark.asyncio
    async def test_memory_blocks(self, os_instance):
        await os_instance.start()
        blocks = os_instance.get_memory_blocks()
        assert len(blocks) >= 7
        os_instance.set_memory_block("context", "Testing context block")
        block = os_instance.get_memory_block("context")
        assert block is not None
        await os_instance.stop()

    @pytest.mark.asyncio
    async def test_search_memory(self, os_instance):
        await os_instance.start()
        await os_instance.memory.store("Python programming", atom_type=AtomType.OBSERVATION)
        results = await os_instance.search_memory("Python")
        assert isinstance(results, list)
        await os_instance.stop()

    @pytest.mark.asyncio
    async def test_personality_integration(self, os_instance):
        await os_instance.start()
        profile = os_instance.get_personality()
        assert "big_five" in profile
        result = await os_instance.personality.distill()
        assert "trait_updates" in result
        await os_instance.stop()

    @pytest.mark.asyncio
    async def test_crew_integration(self, os_instance):
        await os_instance.start()
        agents = os_instance.get_crew_agents()
        assert len(agents) >= 6
        await os_instance.stop()

    @pytest.mark.asyncio
    async def test_stealth_integration(self, os_instance):
        await os_instance.start()
        log = os_instance.get_audit_log()
        assert isinstance(log, list)
        stats = os_instance.stealth.get_stats()
        assert "stealth_stats" in stats
        await os_instance.stop()

    @pytest.mark.asyncio
    async def test_graph_visualization(self, os_instance):
        dot = os_instance.get_graph_visualization()
        assert "digraph" in dot
        assert "listen" in dot

    @pytest.mark.asyncio
    async def test_voice_process(self, os_instance):
        await os_instance.start()
        result = await os_instance.process_voice("What's going on?")
        assert "command" in result
        await os_instance.stop()


# ============================================================================
# Additional Edge Case / Stress Tests
# ============================================================================

class TestEdgeCases:
    def test_empty_memory_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = DarkMemory(data_dir=tmpdir)
            results = asyncio.new_event_loop().run_until_complete(mem.search("nonexistent"))
            assert len(results) == 0
            mem.close()

    @pytest.mark.asyncio
    async def test_large_vector_index(self):
        index = InMemoryVectorIndex(dimension=8)
        for i in range(100):
            vec = [float(i % 8 == j) for j in range(8)]
            await index.add(f"doc_{i}", vec)
        assert index.size() == 100
        results = await index.search([1, 0, 0, 0, 0, 0, 0, 0], top_k=5)
        assert len(results) == 5

    def test_personality_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonalityEngine(data_dir=tmpdir)
            result = engine.predict_decision([])
            assert "error" in result

    def test_mesh_no_peers(self):
        mesh = NeuralMesh()
        stats = mesh.get_stats()
        assert stats["peers_count"] == 0
        alive = mesh.get_alive_peers()
        assert len(alive) == 0

    def test_crew_invalid_mission(self):
        crew = ShadowCrew()
        result = asyncio.new_event_loop().run_until_complete(
            crew.execute_mission("nonexistent")
        )
        assert "error" in result

    def test_dark_memory_duplicate_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = DarkMemory(data_dir=tmpdir)
            atom1 = asyncio.new_event_loop().run_until_complete(
                mem.store("Test", atom_type=AtomType.OBSERVATION)
            )
            atom2 = asyncio.new_event_loop().run_until_complete(
                mem.store("Test again", atom_type=AtomType.OBSERVATION)
            )
            assert atom1.atom_id != atom2.atom_id
            mem.close()

    def test_steganography_empty_data(self):
        engine = CamouflageEngine()
        result = engine.aggregate([], "value")
        assert result["count"] == 0

    def test_personality_profile_versioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonalityEngine(data_dir=tmpdir)
            initial_version = engine.profile.version
            engine.record_behavior("decision_made", {"speed": "fast"})
            asyncio.new_event_loop().run_until_complete(engine.distill())
            assert engine.profile.version >= initial_version


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])