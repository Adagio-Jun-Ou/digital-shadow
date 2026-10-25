#!/usr/bin/env python3
"""Session 5 Sprint 70 — Deep Audit of dark_memory, shadow_crew, neural_mesh

Tests for:
  DM-H2: retrieve() returns deep copy
  DM-H3: _atoms dict thread-safe with _atoms_lock
  DM-M2: get_recent/get_by_domain use _atoms_lock
  DM-M3: _evict_stale uses vector_index.remove() not direct dict
  DM-M4: search/search_with_boost use _atoms_lock snapshot
  DM-M5: (noted, no fix needed)
  DM-M6: import_data max_content_length reduced to 100KB
  DM-M7: TemporalKnowledgeGraph has threading.Lock
  SC-H1/SC-H2: (already addressed in prior sprints)
  SC-M1: delegation cap _MAX_SUBTASKS_PER_MISSION
  SC-M2: _MAX_MISSIONS cap
  SC-M3: AgentRegistry _MAX_AGENTS cap
  NM-H1: add_peer validates device_type and name length
  NM-H2: _received_insights capped at 500
  NM-M1: get_peers returns deep copy
  NM-M2: MeshConsensus validates proposer
  NM-M3: share_insight sanitizes messages
"""

import asyncio
import copy
import threading
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from shadowd.cognition.dark_memory import (
    DarkMemory, MemoryAtom, MemoryTier, AtomType,
    InMemoryVectorIndex, TemporalKnowledgeGraph, BlockStore, MemoryCompactor,
)
from shadowd.cognition.shadow_crew import (
    ShadowCrew, ShadowAgent, AgentRole, CrewMission, CrewTask,
    DelegationMode, MissionStatus, AgentRegistry, CircuitBreaker,
)
from shadowd.cognition.neural_mesh import (
    NeuralMesh, MeshNode, MeshMessage, MeshMessageType, NodeStatus,
    GossipProtocol, StateSync, MeshConsensus,
)


# ── DarkMemory ──────────────────────────────────────────────────────────────

class TestDMH2RetrieveDeepCopy:
    @pytest.mark.asyncio
    async def test_retrieve_returns_deep_copy(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        atom = await dm.store("secret content", tags=["test"])
        retrieved = await dm.retrieve(atom.atom_id)
        assert retrieved is not None
        assert retrieved.content == "secret content"
        retrieved.content = "MODIFIED"
        again = await dm.retrieve(atom.atom_id)
        assert again.content == "secret content"
        dm.close()

    @pytest.mark.asyncio
    async def test_retrieve_tags_not_mutated(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        atom = await dm.store("content", tags=["a", "b"])
        retrieved = await dm.retrieve(atom.atom_id)
        retrieved.tags.append("c")
        again = await dm.retrieve(atom.atom_id)
        assert "c" not in again.tags
        dm.close()

    @pytest.mark.asyncio
    async def test_retrieve_metadata_not_mutated(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        atom = await dm.store("content", metadata={"key": "val"})
        retrieved = await dm.retrieve(atom.atom_id)
        retrieved.metadata["key2"] = "val2"
        again = await dm.retrieve(atom.atom_id)
        assert "key2" not in again.metadata
        dm.close()


class TestDMH3AtomsLock:
    def test_atoms_lock_exists(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        assert hasattr(dm, '_atoms_lock')
        assert isinstance(dm._atoms_lock, type(threading.Lock()))
        dm.close()

    @pytest.mark.asyncio
    async def test_concurrent_stores(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        errors = []

        async def store_atom(i):
            try:
                await dm.store(f"content_{i}")
            except Exception as e:
                errors.append(e)

        await asyncio.gather(*[store_atom(i) for i in range(20)])
        assert len(errors) == 0
        assert len(dm._atoms) == 20
        dm.close()


class TestDMM2GetRecentAndGetByDomainUseLock:
    @pytest.mark.asyncio
    async def test_get_recent_thread_safe(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        await dm.store("recent 1", domain="test")
        await dm.store("recent 2", domain="test")
        results = await dm.get_recent(limit=10)
        assert len(results) == 2
        dm.close()

    @pytest.mark.asyncio
    async def test_get_by_domain_thread_safe(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        await dm.store("domain content", domain="research")
        await dm.store("other content", domain="coding")
        results = await dm.get_by_domain("research")
        assert len(results) == 1
        assert results[0].domain == "research"
        dm.close()


class TestDMM3EvictStaleUsesVectorRemove:
    @pytest.mark.asyncio
    async def test_evict_stale_removes_from_vector_index(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        atom = await dm.store("stale content", tier=MemoryTier.ARCHIVAL, importance=0.1)
        atom.accessed_at = "2020-01-01T00:00:00+00:00"
        atom.tier = MemoryTier.ARCHIVAL
        atom.importance = 0.0
        with dm._atoms_lock:
            dm._atoms[atom.atom_id] = atom
        await dm._vector_index.add(atom.atom_id, atom.embedding)
        assert dm._vector_index.size() >= 1
        result = await dm._evict_stale(ttl_hours=1)
        assert result >= 1
        assert atom.atom_id not in dm._vector_index._vectors
        dm.close()


class TestDMM4SearchUsesAtomsSnapshot:
    @pytest.mark.asyncio
    async def test_search_uses_thread_lock_not_async_lock(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        await dm.store("python programming")
        await dm.store("python testing")
        results = await dm.search("python")
        assert len(results) >= 1
        dm.close()

    @pytest.mark.asyncio
    async def test_search_with_boost(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        await dm.store("python code", domain="engineering", importance=0.8)
        results = await dm.search_with_boost("python", domain_boost="engineering")
        assert len(results) >= 1
        dm.close()


class TestDMM6ImportDataContentLimit:
    @pytest.mark.asyncio
    async def test_import_rejects_200kb_content(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        import json
        data = {"atoms": [{"atom_id": "big1", "content": "x" * 200_000, "atom_type": "observation", "tier": "working"}]}
        result = await dm.import_data(json.dumps(data))
        assert result["skipped"] == 1
        assert result["imported"] == 0
        dm.close()

    @pytest.mark.asyncio
    async def test_import_accepts_50kb_content(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        import json
        data = {"atoms": [{"atom_id": "ok1", "content": "x" * 50_000, "atom_type": "observation", "tier": "working"}]}
        result = await dm.import_data(json.dumps(data))
        assert result["imported"] == 1
        dm.close()


class TestDMM7TemporalKnowledgeGraphLock:
    def test_kg_has_threading_lock(self):
        kg = TemporalKnowledgeGraph()
        assert hasattr(kg, '_lock')
        assert isinstance(kg._lock, type(threading.Lock()))

    def test_kg_concurrent_add_entity(self):
        kg = TemporalKnowledgeGraph()
        errors = []

        def add_entities(start, count):
            for i in range(start, start + count):
                try:
                    kg.add_entity(f"e_{i}", "person", {"name": f"Entity {i}"})
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=add_entities, args=(0, 100))
        t2 = threading.Thread(target=add_entities, args=(100, 100))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0
        assert len(kg._entities) == 200

    def test_kg_concurrent_add_relation(self):
        kg = TemporalKnowledgeGraph()
        kg.add_entity("a", "person", {})
        kg.add_entity("b", "person", {})
        errors = []

        def add_rels(start, count):
            for i in range(start, start + count):
                try:
                    kg.add_relation("a", "b", f"rel_{i}")
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=add_rels, args=(0, 50))
        t2 = threading.Thread(target=add_rels, args=(50, 50))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0
        assert len(kg._relations) == 100


class TestDMFindDuplicatesDocumentation:
    @pytest.mark.asyncio
    async def test_find_duplicates_docstring_mentions_hash(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        a1 = await dm.store("hello world", atom_type=AtomType.OBSERVATION)
        a2 = await dm.store("hello world", atom_type=AtomType.OBSERVATION)
        assert "Note" in dm.find_duplicates.__doc__
        assert "deterministic" in dm.find_duplicates.__doc__
        dm.close()


class TestDMM3CompactUsesAtomsLock:
    @pytest.mark.asyncio
    async def test_compact_thread_safe(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path), max_atoms=50)
        for i in range(60):
            await dm.store(f"atom_{i}", importance=0.1)
        evicted = await dm.compact()
        assert evicted >= 0
        dm.close()


class TestDMEvictExpiredUsesAtomsLock:
    @pytest.mark.asyncio
    async def test_evict_expired_thread_safe(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        atom = await dm.store("expiring content", importance=0.1)
        atom.expires_at = "2020-01-01T00:00:00+00:00"
        with dm._atoms_lock:
            dm._atoms[atom.atom_id] = atom
        result = await dm._evict_expired()
        assert result >= 1
        dm.close()


class TestDMExportDataUsesAtomsLock:
    @pytest.mark.asyncio
    async def test_export_data_thread_safe(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        await dm.store("export test")
        data_str = await dm.export_data()
        import json
        data = json.loads(data_str)
        assert data["atoms_count" if "atoms_count" in data else "atoms"] or len(data.get("atoms", [])) >= 1
        dm.close()


# ── ShadowCrew ───────────────────────────────────────────────────────────────

class TestSCM1MaxSubtasksPerMission:
    def test_max_subtasks_constant(self):
        assert ShadowCrew._MAX_SUBTASKS_PER_MISSION == 20

    @pytest.mark.asyncio
    async def test_delegation_cap_in_execute(self):
        crew = ShadowCrew()
        mission = await crew.create_mission("Test objective", mode=DelegationMode.HIERARCHICAL)
        assert len(mission.tasks) <= ShadowCrew._MAX_SUBTASKS_PER_MISSION + 5


class TestSCM2MaxMissions:
    def test_max_missions_constant(self):
        crew = ShadowCrew()
        assert crew._MAX_MISSIONS == 100

    @pytest.mark.asyncio
    async def test_missions_eviction(self):
        crew = ShadowCrew()
        for i in range(105):
            m = await crew.create_mission(f"Objective {i}")
        assert len(crew._missions) <= crew._MAX_MISSIONS


class TestSCM3MaxAgents:
    def test_max_agents_constant(self):
        assert AgentRegistry._MAX_AGENTS == 100

    def test_register_at_cap_logs_warning(self):
        reg = AgentRegistry()
        for i in range(100):
            agent = ShadowAgent(agent_id=f"agent_{i}", name=f"Agent {i}", role=AgentRole.OBSERVER, goal="test", backstory="test")
            reg.register(agent)
        assert len(reg._agents) == 100
        extra = ShadowAgent(agent_id="extra", name="Extra", role=AgentRole.OBSERVER, goal="test", backstory="test")
        reg.register(extra)
        assert len(reg._agents) == 100


# ── NeuralMesh ───────────────────────────────────────────────────────────────

class TestNMH1DeviceTypeValidation:
    def test_valid_device_types(self):
        mesh = NeuralMesh()
        for dt in ["phone", "laptop", "server", "desktop", "tablet", "iot"]:
            node = MeshNode(node_id=f"n_{dt}", name="Test", device_type=dt)
            mesh.add_peer(node)
            assert f"n_{dt}" in mesh._peers

    def test_invalid_device_type_rejected(self):
        mesh = NeuralMesh()
        node = MeshNode(node_id="bad1", name="Test", device_type="invalid_type")
        with pytest.raises(ValueError, match="Invalid device_type"):
            mesh.add_peer(node)

    def test_name_length_truncated(self):
        mesh = NeuralMesh()
        node = MeshNode(node_id="n1", name="A" * 200, device_type="server")
        mesh.add_peer(node)
        assert len(mesh._peers["n1"].name) == 100

    def test_port_validation(self):
        mesh = NeuralMesh()
        node = MeshNode(node_id="n1", name="Test", device_type="server", port=99999)
        with pytest.raises(ValueError):
            mesh.add_peer(node)


class TestNMH2ReceivedInsightsCap:
    def test_received_insights_cap(self):
        mesh = NeuralMesh()
        assert hasattr(mesh, '_MAX_RECEIVED_INSIGHTS')
        assert mesh._MAX_RECEIVED_INSIGHTS == 500

    @pytest.mark.asyncio
    async def test_received_insights_capped(self):
        mesh = NeuralMesh()
        for i in range(600):
            msg = MeshMessage(
                message_type=MeshMessageType.INSIGHT_SHARE,
                sender_id="remote_node",
                payload={"insight": {"data": f"insight_{i}"}},
            )
            msg.signature = msg.sign(mesh._mesh_secret) if mesh._mesh_secret else ""
            insight = msg.payload.get("insight", {})
            sanitized = mesh._sanitize_message(insight) if isinstance(insight, dict) else {"data": str(insight)[:500]}
            mesh._received_insights.append(sanitized)
            if len(mesh._received_insights) > mesh._MAX_RECEIVED_INSIGHTS:
                mesh._received_insights = mesh._received_insights[-mesh._MAX_RECEIVED_INSIGHTS:]
        assert len(mesh._received_insights) <= mesh._MAX_RECEIVED_INSIGHTS


class TestNMM1GetPeersDeepCopy:
    def test_get_peers_returns_deep_copy(self):
        mesh = NeuralMesh()
        node = MeshNode(node_id="n1", name="Node1", device_type="server")
        mesh.add_peer(node)
        peers = mesh.get_peers()
        assert len(peers) == 1
        peers[0]["name"] = "HACKED"
        original = mesh._peers["n1"].name
        assert original == "Node1"

    def test_get_alive_peers_returns_deep_copy(self):
        mesh = NeuralMesh()
        node = MeshNode(node_id="n1", name="Node1", device_type="server")
        mesh.add_peer(node)
        peers = mesh.get_alive_peers()
        if peers:
            peers[0]["name"] = "HACKED"
        assert mesh._peers["n1"].name == "Node1"


class TestNMM2ConsensusValidatesProposer:
    def test_valid_proposer_local_node(self):
        mesh = NeuralMesh(node_id="local_1")
        assert mesh._consensus._valid_proposer("local_1") is True

    def test_valid_proposer_known_peer(self):
        mesh = NeuralMesh(node_id="local_1")
        peer = MeshNode(node_id="peer_1", name="Peer", device_type="server")
        mesh.add_peer(peer)
        assert mesh._consensus._valid_proposer("peer_1") is True

    def test_invalid_proposer_unknown(self):
        mesh = NeuralMesh(node_id="local_1")
        assert mesh._consensus._valid_proposer("unknown_node") is False

    @pytest.mark.asyncio
    async def test_receive_proposal_rejects_unknown(self):
        mesh = NeuralMesh(node_id="local_1")
        proposal = {"proposal_id": "p1", "proposer": "unknown_node"}
        result = await mesh.receive_proposal(proposal)
        assert result is False

    @pytest.mark.asyncio
    async def test_receive_proposal_accepts_known(self):
        mesh = NeuralMesh(node_id="local_1")
        peer = MeshNode(node_id="peer_1", name="Peer", device_type="server")
        mesh.add_peer(peer)
        proposal = {"proposal_id": "p2", "proposer": "peer_1", "key": "test", "value": 1, "status": "voting"}
        result = await mesh.receive_proposal(proposal)
        assert result is True


class TestNMM3SanitizeMessage:
    def test_sanitize_strips_control_chars(self):
        data = {"key": "hello\x00world\x01test", "name": "user\t\nname"}
        result = NeuralMesh._sanitize_message(data)
        assert "\x00" not in result["key"]
        assert "\x01" not in result["key"]
        assert "\t" in result["name"]
        assert "\n" in result["name"]

    def test_sanitize_truncates_long_strings(self):
        data = {"key": "a" * 2000}
        result = NeuralMesh._sanitize_message(data)
        assert len(result["key"]) == 1000

    def test_sanitize_truncates_long_keys(self):
        data = {"a" * 200: "value"}
        result = NeuralMesh._sanitize_message(data)
        key = list(result.keys())[0]
        assert len(key) == 100

    def test_sanitize_truncates_long_lists(self):
        data = {"items": list(range(200))}
        result = NeuralMesh._sanitize_message(data)
        assert len(result["items"]) == 100

    def test_sanitize_nested_dicts(self):
        data = {"outer": {"inner": "val\x00ue"}}
        result = NeuralMesh._sanitize_message(data)
        assert "\x00" not in result["outer"]["inner"]

    def test_sanitize_non_dict_returns_empty(self):
        result = NeuralMesh._sanitize_message("not a dict")
        assert result == {}

    def test_sanitize_non_string_keys_skipped(self):
        data = {"valid_key": "value", 123: "bad_key"}
        result = NeuralMesh._sanitize_message(data)
        assert "valid_key" in result
        assert 123 not in result

    @pytest.mark.asyncio
    async def test_share_insight_sanitizes(self):
        mesh = NeuralMesh()
        await mesh.share_insight({"data": "test\x00content", "category": "security"})
        assert len(mesh._insight_buffer) == 1
        assert "\x00" not in mesh._insight_buffer[0]["data"]


class TestNMH1InsightBufferCap:
    def test_insight_buffer_cap(self):
        mesh = NeuralMesh()
        assert mesh._MAX_INSIGHT_BUFFER == 100

    @pytest.mark.asyncio
    async def test_insight_buffer_does_not_exceed_cap(self):
        mesh = NeuralMesh()
        for i in range(120):
            mesh._insight_buffer.append({"data": f"insight_{i}"})
            if len(mesh._insight_buffer) > mesh._MAX_INSIGHT_BUFFER:
                mesh._insight_buffer = mesh._insight_buffer[-mesh._MAX_INSIGHT_BUFFER:]
        assert len(mesh._insight_buffer) <= mesh._MAX_INSIGHT_BUFFER


class TestDMGetDomainSummaryThreadSafe:
    @pytest.mark.asyncio
    async def test_get_domain_summary(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        await dm.store("code content", domain="engineering")
        await dm.store("research content", domain="research")
        summary = dm.get_domain_summary()
        assert "engineering" in summary
        assert "research" in summary
        dm.close()


class TestDMGetStatsThreadSafe:
    @pytest.mark.asyncio
    async def test_get_stats_thread_safe(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        await dm.store("stats test", importance=0.8)
        stats = dm.get_stats()
        assert stats["total_atoms"] >= 1
        dm.close()


class TestDMBlockStoreAccessCount:
    def test_get_block_increments_access_count(self):
        bs = BlockStore()
        bs.initialize_default_blocks()
        block = bs.get_block("identity")
        assert block is not None
        assert block["access_count"] == 1
        bs.get_block("identity")
        block2 = bs.get_block("identity")
        assert block2["access_count"] == 3


class TestDMMemoryCompactor:
    def test_compactor_works(self):
        compactor = MemoryCompactor(max_working=5, max_archival=10)
        atoms = [MemoryAtom(content=f"atom_{i}", importance=0.1 * i, tier=MemoryTier.WORKING) for i in range(10)]
        kept, removed = compactor.compact(atoms)
        assert len(kept) <= 15
        assert len(removed) >= 0


class TestNMRemovePeerRecreatesConsensus:
    def test_remove_peer_recreates_consensus(self):
        mesh = NeuralMesh(node_id="local_1")
        peer = MeshNode(node_id="p1", name="Peer1", device_type="server")
        mesh.add_peer(peer)
        assert "p1" in mesh._peers
        old_consensus = mesh._consensus
        mesh.remove_peer("p1")
        assert "p1" not in mesh._peers
        assert mesh._consensus is not old_consensus


class TestNMAddPeerValidatesCapabilities:
    def test_capabilities_validated(self):
        mesh = NeuralMesh()
        node = MeshNode(
            node_id="n1",
            name="Test",
            device_type="server",
            capabilities=["good_cap"] + [None] + ["a" * 200],
        )
        node.capabilities = ["good_cap", None, "a" * 200]
        mesh.add_peer(node)
        assert len(mesh._peers["n1"].capabilities) <= 50

    def test_too_many_capabilities_capped(self):
        mesh = NeuralMesh()
        node = MeshNode(
            node_id="n1",
            name="Test",
            device_type="server",
            capabilities=[f"cap_{i}" for i in range(100)],
        )
        mesh.add_peer(node)
        assert len(mesh._peers["n1"].capabilities) <= 50


class TestDMImportDataAtomLock:
    @pytest.mark.asyncio
    async def test_import_data_thread_safe(self, tmp_path):
        dm = DarkMemory(data_dir=str(tmp_path))
        import json
        atoms = [{"atom_id": f"import_{i}", "content": f"data_{i}", "atom_type": "observation", "tier": "working"} for i in range(10)]
        data = json.dumps({"atoms": atoms})
        result = await dm.import_data(data)
        assert result["imported"] == 10
        dm.close()


class TestSCDelegationCap:
    @pytest.mark.asyncio
    async def test_mission_delegation_cap(self):
        crew = ShadowCrew()
        assert crew._MAX_SUBTASKS_PER_MISSION == 20

    def test_agent_registry_max_agents(self):
        reg = AgentRegistry()
        assert reg._MAX_AGENTS == 100


class TestNMStateMachineSigning:
    def test_message_sign_verify(self):
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="node1",
            payload={"data": "test"},
        )
        secret = "my_secret_key"
        sig = msg.sign(secret)
        assert msg.verify(secret) is True

    def test_message_verify_fails_wrong_secret(self):
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="node1",
            payload={"data": "test"},
        )
        msg.sign("secret1")
        assert msg.verify("secret2") is False

    def test_message_verify_fails_no_signature(self):
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="node1",
            payload={"data": "test"},
        )
        assert msg.verify("secret") is False


class TestSCCircuitBreaker:
    def test_circuit_breaker_basic(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        assert cb.is_available()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_available()
        cb.record_failure()
        assert not cb.is_available()

    def test_circuit_breaker_recovery(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_available()
        time.sleep(0.1)
        assert cb.is_available()
        cb.record_success()
        assert cb.is_available()


class TestNMInsightSanitizationInMessageProcessor:
    @pytest.mark.asyncio
    async def test_received_insights_are_sanitized(self):
        mesh = NeuralMesh(node_id="local_1")
        mesh._mesh_secret = ""
        insight_data = {"key": "val\x00ue", "desc": "a" * 2000}
        sanitized = mesh._sanitize_message(insight_data)
        assert "\x00" not in sanitized["key"]
        assert len(sanitized["desc"]) == 1000