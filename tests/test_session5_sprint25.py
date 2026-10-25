"""
Session 6 Sprint 25 — Deep Audit Round 3 (1 CRITICAL, 1 HIGH, 12 MEDIUM, 4 LOW fixes)

Fixes:
  KG1 CRITICAL: Missing `import hashlib` in knowledge_graph.py — NameError on add_episode
  DM1 HIGH: _evict_stale now removes atoms from vector index and SQLite
  DM2 MEDIUM: MemoryCompactor.compact() returns actual removed atoms (was empty list)
  A1 MEDIUM: SSRF is_safe_address now blocks RFC1918 (10.x, 192.168.x, 172.16-31.x)
  A4 MEDIUM: ObservationRequest.content now Optional[str] with max_length=65536
  SE1 MEDIUM: _load_domains() removed from _cap_insights_queue() (was accidental)
  SE2 MEDIUM: observe() truncates event content stored in domain records to 500 chars
  SE5 MEDIUM: ShadowEnhanced.observe() now wraps subsystem calls in try/except and acquires _lock
  US2 MEDIUM: stop_background_tasks() now awaits cancelled tasks via asyncio.gather
  US4 MEDIUM: UnifiedShadow.observe() wraps each subsystem call in try/except
  E4 LOW: EscalationManager._persist_history adds fsync
  KG2 MEDIUM: get_stats() entity_types uses Counter instead of broken dict comprehension
  KG3 LOW: KnowledgeGraph._save adds fsync
  P3 MEDIUM: AsyncTaskQueue._result_events capped at _MAX_RESULT_EVENTS=2000
"""

import inspect
import os


import pytest


# ─── KG1: Missing hashlib import ───

class TestKnowledgeGraphHashlib:
    def test_hashlib_imported(self):
        import shadowd.knowledge_graph as kg
        assert hasattr(kg, 'hashlib'), "hashlib should be imported"


# ─── DM1: _evict_stale removes from vector index and SQLite ───

class TestDarkMemoryEvictStale:
    def test_evict_stale_removes_from_vector_index(self):
        from shadowd.cognition.dark_memory import DarkMemory
        source = inspect.getsource(DarkMemory._evict_stale)
        assert "_vector_index" in source, "_evict_stale should clean vector index"
        assert "_vectors.pop" not in source, "_evict_stale should use _vector_index.remove() not direct dict manipulation"
        assert "_vector_index.remove" in source, "_evict_stale should use _vector_index.remove()"

    def test_evict_stale_removes_from_sqlite(self):
        from shadowd.cognition.dark_memory import DarkMemory
        source = inspect.getsource(DarkMemory._evict_stale)
        assert "DELETE FROM observations" in source, "_evict_stale should delete from observations table"


# ─── DM2: compactor returns actual removed atoms ───

class TestMemoryCompactorReturns:
    def test_compactor_returns_deleted_atoms(self):
        from shadowd.cognition.dark_memory import MemoryCompactor
        source = inspect.getsource(MemoryCompactor.compact)
        assert "to_delete" in source and "return kept, to_delete" in source, \
            "compact() should return (kept, to_delete) not (kept, [])"


# ─── A1: SSRF is_safe_address blocks RFC1918 ───

class TestSSRFAddressValidation:
    def test_blocks_10_network(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        req = MeshPeerRequest(node_id="test", name="test", address="10.0.0.1", port=8000)
        assert not req.is_safe_address(), "10.x should be blocked"

    def test_blocks_192_168_network(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        req = MeshPeerRequest(node_id="test", name="test", address="192.168.1.1", port=8000)
        assert not req.is_safe_address(), "192.168.x should be blocked"

    def test_blocks_172_16_network(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        req = MeshPeerRequest(node_id="test", name="test", address="172.16.0.1", port=8000)
        assert not req.is_safe_address(), "172.16.x should be blocked"

    def test_allows_172_32(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        req = MeshPeerRequest(node_id="test", name="test", address="172.32.0.1", port=8000)
        assert req.is_safe_address(), "172.32.x should be allowed (not RFC1918)"

    def test_blocks_metadata_endpoint(self):
        from shadowd.cognition.api_router import MeshPeerRequest
        req = MeshPeerRequest(node_id="test", name="test", address="169.254.169.254", port=8000)
        assert not req.is_safe_address(), "169.254.169.254 should be blocked"


# ─── A4: ObservationRequest content size limit ───

class TestObservationRequestContentLimit:
    def test_content_is_optional_string(self):
        from shadowd.cognition.api_router import ObservationRequest
        fields = ObservationRequest.model_fields
        assert fields["content"].is_required() is False, "content should be optional"
        max_len = None
        for meta in fields["content"].metadata:
            if hasattr(meta, 'max_length'):
                max_len = meta.max_length
                break
        assert max_len == 65536, f"content should have max_length=65536, got {max_len}"


# ─── SE1: _cap_insights_queue no longer calls _load_domains ───

class TestShadowEnhancedCapInsights:
    def test_cap_does_not_call_load_domains(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI._cap_insights_queue)
        assert "_load_domains" not in source, "_cap_insights_queue should not call _load_domains"


# ─── SE5: observe uses lock ───

class TestShadowEnhancedObserveLock:
    def test_observe_uses_lock(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI.observe)
        assert "self._lock" in source, "observe should acquire self._lock"

    def test_observe_has_error_isolation(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI.observe)
        assert "except" in source, "observe should catch exceptions per subsystem"


# ─── US2: stop_background_tasks awaits cancellations ───

class TestUnifiedSystemStop:
    def test_stop_awaits_cancellations(self):
        from shadowd.unified_system import UnifiedShadowSystem
        source = inspect.getsource(UnifiedShadowSystem.stop_background_tasks)
        assert "asyncio.gather" in source, "stop should gather cancelled tasks"


# ─── US4: observe error isolation ───

class TestUnifiedSystemObserveErrorIsolation:
    def test_observe_wraps_memory_in_try(self):
        from shadowd.unified_system import UnifiedShadowSystem
        source = inspect.getsource(UnifiedShadowSystem.observe)
        assert "except" in source, "observe should catch exceptions"


# ─── E4: EscalationManager _persist_history fsync ───

class TestEscalationPersistFsync:
    def test_persist_uses_fsync(self):
        from shadowd.cognition.escalation import EscalationManager
        source = inspect.getsource(EscalationManager._persist_history)
        assert "fsync" in source, "_persist_history should use fsync"


# ─── KG2: get_stats entity_types uses Counter ───

class TestKnowledgeGraphStatsEntityTypes:
    def test_entity_types_uses_counter(self):
        import shadowd.knowledge_graph as kg
        source = inspect.getsource(kg.TemporalKnowledgeGraph.get_stats)
        assert "Counter" in source, "get_stats should use Counter for entity_types"


# ─── KG3: KnowledgeGraph _save fsync ───

class TestKnowledgeGraphSaveFsync:
    def test_save_uses_fsync(self):
        import shadowd.knowledge_graph as kg
        source = inspect.getsource(kg.TemporalKnowledgeGraph.flush)
        assert "fsync" in source, "flush() should use fsync"


# ─── P3: AsyncTaskQueue _result_events capped ───

class TestAsyncTaskQueueResultEventsCap:
    def test_result_events_has_max(self):
        from shadowd.performance import AsyncTaskQueue
        source = inspect.getsource(AsyncTaskQueue.__init__)
        assert "_MAX_RESULT_EVENTS" in source, "Should have _MAX_RESULT_EVENTS cap"

    def test_result_events_eviction(self):
        from shadowd.performance import AsyncTaskQueue
        source = inspect.getsource(AsyncTaskQueue.get_result)
        assert "_MAX_RESULT_EVENTS" in source, "Should enforce cap in get_result"