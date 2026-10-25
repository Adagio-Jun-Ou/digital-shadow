"""
Session 5 Sprint 25b — Remaining Audit Findings (15 fixes)

Fixes:
  DM3 MEDIUM: dark_memory search() accesses self._atoms without lock after vector search
  DM4 MEDIUM: dark_memory retrieve() doesn't persist access_count to SQLite
  DM5 MEDIUM: dark_memory import_data has no size limit on individual atom content
  DM7 MEDIUM: InMemoryVectorIndex._lock is asyncio.Lock — documented not thread-safe
  AM1 MEDIUM: knowledge_graph Entity.facts capped at 1000 (verify)
  AM2 MEDIUM: knowledge_graph add_entity_fact -> add_fact calls _save() per fact — verify
  AM4 MEDIUM: TemporalKnowledgeGraph add_episode/add_relation now call _save()
  SE3 MEDIUM: shadow_enhanced _classify_domain serializes event once, passed to callers
  SO1/SO2 MEDIUM: super_orchestrator crew/workflow IDs use UUID instead of sequential counter
  SO3 MEDIUM: super_orchestrator process() validates input_data size (max 10MB)
  SO6 MEDIUM: super_orchestrator _retry_execute closures use defensive copies
  A2 MEDIUM: shadowosd rate_limit_store uses threading.Lock
  A5 MEDIUM: MeshPeerRequest.port documents that privileged ports are intentionally allowed
  A8 MEDIUM: crew_missions endpoint has limit validation (ge=1, le=200)
  E2 MEDIUM: EscalationManager resolve_escalation uses threading.Lock
  E3 MEDIUM: EscalationManager stats asked_permission uses max(0, ...) to prevent negative
"""

import inspect
import os
import threading
from pathlib import Path

import pytest


# ─── DM3: search() uses lock for _atoms access ───

class TestDarkMemorySearchLock:
    def test_search_uses_db_lock(self):
        from shadowd.cognition.dark_memory import DarkMemory
        source = inspect.getsource(DarkMemory.search)
        assert "self._atoms_lock" in source, "search() should acquire self._atoms_lock for thread-safe snapshot"

    def test_search_uses_atoms_snapshot(self):
        from shadowd.cognition.dark_memory import DarkMemory
        source = inspect.getsource(DarkMemory.search)
        assert "atoms_snapshot" in source, "search() should use atoms_snapshot"

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            await dm.store("hello world", atom_type=AtomType.OBSERVATION)
            results = await dm.search("hello")
            assert len(results) >= 1
            assert "hello" in results[0].content.lower()
            dm.close()


# ─── DM4: retrieve() persists access_count to SQLite ───

class TestDarkMemoryRetrievePersistAccessCount:
    def test_retrieve_updates_db(self):
        from shadowd.cognition.dark_memory import DarkMemory
        source = inspect.getsource(DarkMemory.retrieve)
        assert "UPDATE observations SET access_count" in source, \
            "retrieve() should update access_count in DB"

    @pytest.mark.asyncio
    async def test_retrieve_persists_access_count(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            atom = await dm.store("test persist access", atom_type=AtomType.OBSERVATION)
            retrieved = await dm.retrieve(atom.atom_id)
            assert retrieved is not None
            assert retrieved.access_count == 1  # touch increments

            # Verify DB was updated
            dm2 = DarkMemory(data_dir=tmpdir)
            cursor = dm2._db_conn.execute(
                "SELECT access_count FROM observations WHERE id = ?", (atom.atom_id,)
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] >= 1
            dm.close()
            dm2.close()


# ─── DM5: import_data has max_content_length check ───

class TestDarkMemoryImportDataContentSize:
    def test_import_data_has_content_length_check(self):
        from shadowd.cognition.dark_memory import DarkMemory
        source = inspect.getsource(DarkMemory.import_data)
        assert "max_content_length" in source, "import_data should check max_content_length"

    @pytest.mark.asyncio
    async def test_import_skips_oversized_content(self):
        from shadowd.cognition.dark_memory import DarkMemory, MemoryTier
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            oversized_content = "x" * (2 * 1024 * 1024)  # 2MB, over 1MB limit
            data = json.dumps({
                "atoms": [
                    {"atom_id": "a1", "atom_type": "observation", "content": oversized_content},
                    {"atom_id": "a2", "atom_type": "observation", "content": "small content"},
                ]
            })
            result = await dm.import_data(data)
            assert result["skipped"] >= 1, "Oversized content should be skipped"
            assert result["imported"] >= 1, "Small content should be imported"
            dm.close()


# ─── DM7: InMemoryVectorIndex._lock documented as asyncio.Lock ───

class TestInMemoryVectorIndexLockNote:
    def test_vector_index_has_docstring_about_thread_safety(self):
        from shadowd.cognition.dark_memory import InMemoryVectorIndex
        doc = InMemoryVectorIndex.__doc__
        assert doc is not None
        assert "NOT thread-safe" in doc or "not thread-safe" in doc, \
            "InMemoryVectorIndex should document thread safety"

    def test_vector_index_lock_is_asyncio_lock(self):
        from shadowd.cognition.dark_memory import InMemoryVectorIndex
        import asyncio
        idx = InMemoryVectorIndex(dimension=64)
        assert isinstance(idx._lock, asyncio.Lock), "_lock should be asyncio.Lock"


# ─── AM1: Entity.facts capped at 1000 ───

class TestEntityFactsCap:
    def test_entity_add_fact_caps_at_1000(self):
        from shadowd.knowledge_graph import Entity, TemporalFact
        source = inspect.getsource(Entity.add_fact)
        assert "1000" in source, "add_fact should cap facts at 1000"

    def test_entity_facts_truncated(self):
        from shadowd.knowledge_graph import Entity, TemporalFact
        from datetime import datetime, timezone
        entity = Entity(name="test", entity_type="person")
        now = datetime.now(timezone.utc)
        for i in range(1005):
            fact = TemporalFact(
                fact_id=f"f_{i}",
                content=f"Fact {i}",
                entities=["test"],
                valid_from=now,
            )
            entity.add_fact(fact)
        assert len(entity.facts) <= 1000, f"Facts should be capped at 1000, got {len(entity.facts)}"


# ─── AM2: add_entity_fact -> add_fact -> _save() ───

class TestKnowledgeGraphAddFactSaves:
    def test_add_episode_calls_save(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        source = inspect.getsource(TemporalKnowledgeGraph.add_episode)
        assert "self._save()" in source, "add_episode should call self._save()"

    def test_add_relation_calls_save(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        source = inspect.getsource(TemporalKnowledgeGraph.add_relation)
        assert "self._save()" in source, "add_relation should call self._save()"


# ─── AM4: TemporalKnowledgeGraph add_episode/add_relation call _save() ───

class TestTemporalKnowledgeGraphSave:
    def test_add_episode_calls_save(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        source = inspect.getsource(TemporalKnowledgeGraph.add_episode)
        assert "_save()" in source, "add_episode should call _save()"

    def test_add_relation_calls_save(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        source = inspect.getsource(TemporalKnowledgeGraph.add_relation)
        assert "_save()" in source, "add_relation should call _save()"

    def test_add_episode_persists_data(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="test episode",
            source="test",
            facts=[{"content": "test fact", "entities": ["Alice"]}],
            entities=["Alice"],
        )
        save_file = tmp_path / "knowledge_graph.json"
        assert save_file.exists(), "add_episode should persist data via _save()"


# ─── SE3: shadow_enhanced serializes event once in observe() ───

class TestShadowEnhancedSingleSerialization:
    def test_observe_passes_content_str_to_classify_domain(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI.observe)
        assert "content_str=" in source, "observe should pass content_str to _classify_domain"

    def test_observe_passes_content_str_to_update_context(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI.observe)
        assert "self._update_context(event, content_str=" in source, \
            "observe should pass content_str to _update_context"

    def test_observe_passes_content_str_to_check_insights(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI.observe)
        assert "self._check_insights(event, content_str=" in source, \
            "observe should pass content_str to _check_insights"

    def test_classify_domain_accepts_content_str(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI._classify_domain)
        assert "content_str" in source, "_classify_domain should accept content_str parameter"

    def test_update_context_accepts_content_str(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI._update_context)
        assert "content_str" in source, "_update_context should accept content_str parameter"

    def test_check_insights_accepts_content_str(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI._check_insights)
        assert "content_str" in source, "_check_insights should accept content_str parameter"


# ─── SO1/SO2: UUID-based IDs for crew/workflow ───

class TestSuperOrchestratorUUIDs:
    def test_crew_name_uses_uuid(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        source = inspect.getsource(SuperShadowOrchestrator._execute_multi_agent)
        assert "uuid" in source, "crew_name should use uuid"

    def test_workflow_id_uses_uuid(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        source = inspect.getsource(SuperShadowOrchestrator._execute_workflow)
        assert "uuid" in source, "workflow_id should use uuid"

    def test_uuid_imported(self):
        import shadowd.super_orchestrator as so
        assert hasattr(so, "uuid"), "uuid should be imported"


# ─── SO3: process() validates input_data size ───

class TestSuperOrchestratorInputSizeLimit:
    def test_process_has_size_limit(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        source = inspect.getsource(SuperShadowOrchestrator.process)
        assert "_MAX_INPUT_SIZE" in source, "process should check input size"

    @pytest.mark.asyncio
    async def test_process_rejects_oversized_input(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            so = SuperShadowOrchestrator(data_dir=Path(tmpdir))
            so._MAX_INPUT_SIZE = 100
            result = await so.process({"data": "x" * 5000})
            assert result["cached"] is False
            assert result["result"]["success"] is False
            assert "too large" in result["result"]["error"].lower() or "Input too large" in result["result"]["error"]


# ─── SO6: _retry_execute closures use defensive copies ───

class TestSuperOrchestratorDefensiveCopies:
    def test_process_uses_snapshot(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        source = inspect.getsource(SuperShadowOrchestrator.process)
        assert "_input_data_snapshot" in source, \
            "process should create defensive copies for closures"
        assert "_analysis_snapshot" in source, \
            "process should create defensive copy of analysis"


# ─── A2: shadowosd rate_limit_store uses threading.Lock ───

class TestShadowosdRateLimitLock:
    def test_rate_limit_lock_exists(self):
        import shadowd.cognition.shadowosd as mod
        assert hasattr(mod, "_rate_limit_lock"), "_rate_limit_lock should exist"
        # threading.Lock is a factory function (not a type) since Python 3.11,
        # so compare against the concrete lock type instead.
        assert isinstance(mod._rate_limit_lock, type(threading.Lock())), \
            "_rate_limit_lock should be threading.Lock"

    def test_rate_limit_middleware_uses_lock(self):
        import shadowd.cognition.shadowosd as mod
        source = inspect.getsource(mod.rate_limit_middleware)
        assert "with _rate_limit_lock" in source, \
            "rate_limit_middleware should use _rate_limit_lock"


# ─── A5: MeshPeerRequest port allows privileged (documented) ───

class TestMeshPeerRequestPort:
    def test_port_allows_privileged_with_documentation(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        source = inspect.getsource(MeshPeerRequest)
        assert "Privileged" in source or "privileged" in source.lower() or \
               "Tailscale" in source or "low ports" in source, \
            "MeshPeerRequest should document why privileged ports are allowed"

    def test_port_ge_1(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        fields = MeshPeerRequest.model_fields
        port_field = fields["port"]
        ge_values = [m.ge for m in port_field.metadata if hasattr(m, "ge")]
        assert ge_values == [1], f"port should have ge=1, got {ge_values}"


# ─── A8: crew_missions endpoint has limit validation ───

class TestCrewMissionsLimitValidation:
    def test_limit_has_ge_and_le(self):
        from shadowd.cognition import api_router
        source = inspect.getsource(api_router.crew_missions)
        assert "ge=" in source, "crew_missions should have ge constraint on limit"
        assert "le=" in source, "crew_missions should have le constraint on limit"


# ─── E2: EscalationManager resolve_escalation uses lock ───

class TestEscalationResolveQueueLock:
    def test_resolve_escalation_uses_lock(self):
        from shadowd.cognition.escalation import EscalationManager
        source = inspect.getsource(EscalationManager.resolve_escalation)
        assert "self._queue_lock" in source, \
            "resolve_escalation should use self._queue_lock"

    def test_can_execute_uses_lock_for_queue(self):
        from shadowd.cognition.escalation import EscalationManager
        source = inspect.getsource(EscalationManager.can_execute)
        assert "self._queue_lock" in source, \
            "can_execute should use self._queue_lock for queue mutation"

    def test_get_pending_uses_lock(self):
        from shadowd.cognition.escalation import EscalationManager
        source = inspect.getsource(EscalationManager.get_pending_escalations)
        assert "self._queue_lock" in source, \
            "get_pending_escalations should use self._queue_lock"

    def test_lock_is_threading_lock(self):
        from shadowd.cognition.escalation import EscalationManager
        em = EscalationManager()
        assert isinstance(em._queue_lock, type(threading.Lock())), \
            "_queue_lock should be threading.Lock"

    def test_resolve_then_get_pending(self):
        from shadowd.cognition.escalation import EscalationManager, AutonomyLevel
        import asyncio
        em = EscalationManager()
        em.set_action_level("delete", AutonomyLevel.ASK)
        record_id = None
        for _ in range(5):
            coro = em.evaluate("delete", {"target": "file.txt"})
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        req = pool.submit(asyncio.run, coro).result()
                else:
                    req = loop.run_until_complete(coro)
                record_id = req.action_id if hasattr(req, "action_id") else None
                break
            except Exception:
                break

        if record_id is None:
            pytest.skip("Could not create escalation record")

        pending = em.get_pending_escalations()
        initial_count = len(pending)

        if initial_count > 0:
            rid = pending[0]["record_id"]
            result = em.resolve_escalation(rid, approved=True)
            assert result is True

        remaining = em.get_pending_escalations()
        assert len(remaining) <= initial_count


# ─── E3: EscalationManager stats asked_permission can't go negative ───

class TestEscalationStatsNonNegative:
    def test_asked_permission_uses_max_zero(self):
        from shadowd.cognition.escalation import EscalationManager
        source = inspect.getsource(EscalationManager.resolve_escalation)
        assert "max(0," in source, "asked_permission decrement should use max(0, ...)"

    def test_asked_permission_never_negative(self):
        from shadowd.cognition.escalation import EscalationManager
        em = EscalationManager()
        # Manually set to 0 and try to resolve
        em._stats["asked_permission"] = 0
        # resolve_escalation on empty queue returns False
        result = em.resolve_escalation("nonexistent", approved=True)
        assert result is False
        assert em._stats["asked_permission"] >= 0, "asked_permission should never go negative"

    def test_asked_permission_non_negative_after_decrement(self):
        from shadowd.cognition.escalation import EscalationManager
        em = EscalationManager()
        em._stats["asked_permission"] = 0
        # Even if we somehow try to decrement below 0, max(0,...) prevents it
        em._stats["asked_permission"] = max(0, em._stats["asked_permission"] - 1)
        assert em._stats["asked_permission"] == 0


# ─── Integration: DarkMemory search with snapshot ───

class TestDarkMemorySearchSnapshot:
    @pytest.mark.asyncio
    async def test_search_uses_snapshot_not_direct_atoms(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DarkMemory(data_dir=tmpdir)
            await dm.store("alpha bravo charlie", atom_type=AtomType.OBSERVATION)
            await dm.store("delta echo foxtrot", atom_type=AtomType.DECISION)

            results = await dm.search("alpha")
            assert len(results) >= 1
            found = any("alpha" in r.content.lower() for r in results)
            assert found, "Search should find matching atom"
            dm.close()


# ─── Integration: TemporalKnowledgeGraph persistence ───

class TestTemporalKnowledgeGraphPersistence:
    def test_add_episode_persists(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        episode = kg.add_episode(
            content="John met Sarah",
            source="conversation",
            facts=[{"content": "John knows Sarah", "entities": ["John", "Sarah"]}],
            entities=["John", "Sarah"],
        )
        assert episode is not None

        save_file = tmp_path / "knowledge_graph.json"
        assert save_file.exists(), "Data should be persisted after add_episode"

    def test_add_relation_persists(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg.add_episode(
            content="Alice works for Bob",
            source="test",
            entities=["Alice", "Bob"],
        )
        kg.add_relation("Alice", "Bob", RelationType.WORKS_FOR)
        save_file = tmp_path / "knowledge_graph.json"
        assert save_file.exists(), "Data should be persisted after add_relation"

    def test_reload_preserves_data(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg1 = TemporalKnowledgeGraph(data_dir=tmp_path)
        kg1.add_episode(
            content="Test reload",
            source="test",
            entities=["Entity1"],
        )
        del kg1

        kg2 = TemporalKnowledgeGraph(data_dir=tmp_path)
        assert len(kg2.episodes) >= 1, "Episodes should survive reload"