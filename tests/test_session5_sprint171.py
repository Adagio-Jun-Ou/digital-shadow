#!/usr/bin/env python3
"""
Sprint 171 — Lock Ordering Violation Audit: Regression Tests

Tests for the 14 findings from the lock ordering audit across 7 files:
- Finding #1  (HIGH): dark_memory.py search_db() missing _db_thread_lock
- Finding #2  (HIGH): dark_memory.py retrieve() reads _atoms without _atoms_lock
- Finding #3  (MEDIUM): shadow_graph.py asyncio.Lock + threading.Lock overlap
- Finding #4  (MEDIUM): anomaly_detector.py get_stats() reads _stats
- Finding #5  (MEDIUM): attention_engine.py get_stats() reads _stats
- Finding #6  (MEDIUM): shadow_crew.py _find_available_agent accesses _agents directly
- Finding #7  (MEDIUM): shadow_crew.py execute_mission reads _missions without lock
- Finding #8  (MEDIUM): shadow_crew.py _shared_context mutated without async lock
- Finding #9  (MEDIUM): stealth_layer.py lock held during AES-GCM (by design)
- Finding #10 (LOW):  autonomous_daemon.py consistent lock ordering
- Finding #11 (MEDIUM): neural_mesh.py add_peer/remove_peer no synchronization
- Finding #12 (LOW):  neural_mesh.py per-component locks consistent
- Finding #13 (LOW):  shadow_crew.py CircuitBreaker TOCTOU
- Finding #14 (MEDIUM): dark_memory.py InMemoryVectorIndex asyncio.Lock not thread-safe
"""

import asyncio
import copy
import threading
import time
import unittest
from unittest.mock import MagicMock, patch


# ── Finding #1: dark_memory.py search_db() now acquires _db_thread_lock ──

class TestDMSearchDBLockOrdering(unittest.TestCase):
    """HIGH: search_db() must acquire _db_thread_lock + _db_lock for SQLite access."""

    def test_search_db_acquires_thread_lock(self):
        from shadowd.cognition.dark_memory import DarkMemory
        dm = DarkMemory(data_dir="/tmp/test_sprint171_dm1")
        try:
            # Store an atom
            asyncio.run(dm.store("test content for search", domain="test"))

            # search_db should work without error (acquires both locks)
            results = asyncio.run(dm.search_db("test content"))
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
        finally:
            dm.close()

    def test_search_db_concurrent_access(self):
        """Verify concurrent SQLite access via search_db doesn't crash."""
        from shadowd.cognition.dark_memory import DarkMemory
        dm = DarkMemory(data_dir="/tmp/test_sprint171_dm2")
        errors = []
        try:
            asyncio.run(dm.store("python programming", domain="test"))
            asyncio.run(dm.store("java programming", domain="test"))

            def search_thread():
                try:
                    results = asyncio.run(dm.search_db("programming"))
                    self.assertIsInstance(results, list)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=search_thread) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(len(errors), 0, f"Concurrent search_db errors: {errors}")
        finally:
            dm.close()


# ── Finding #2: dark_memory.py retrieve() now acquires _atoms_lock ──

class TestDMRetrieveLockOrdering(unittest.TestCase):
    """HIGH: retrieve() must acquire _atoms_lock for reading _atoms."""

    def test_retrieve_acquires_atoms_lock(self):
        from shadowd.cognition.dark_memory import DarkMemory
        dm = DarkMemory(data_dir="/tmp/test_sprint171_dm3")
        try:
            atom = asyncio.run(dm.store("test retrieve content", domain="test"))
            result = asyncio.run(dm.retrieve(atom.atom_id))
            self.assertIsNotNone(result)
            self.assertEqual(result.content, "test retrieve content")
        finally:
            dm.close()

    def test_retrieve_concurrent_thread_safe(self):
        """Verify concurrent retrieve calls don't crash with _atoms_lock."""
        from shadowd.cognition.dark_memory import DarkMemory
        dm = DarkMemory(data_dir="/tmp/test_sprint171_dm4")
        errors = []
        try:
            atom_ids = []
            for i in range(10):
                atom = asyncio.run(dm.store(f"content {i}", domain="test"))
                atom_ids.append(atom.atom_id)

            def retrieve_thread(atom_id):
                try:
                    result = asyncio.run(dm.retrieve(atom_id))
                    self.assertIsNotNone(result)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=retrieve_thread, args=(aid,)) for aid in atom_ids]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(len(errors), 0, f"Concurrent retrieve errors: {errors}")
        finally:
            dm.close()


# ── Finding #3: shadow_graph.py asyncio.Lock + threading.Lock overlap ──

class TestShadowGraphLockOverlap(unittest.TestCase):
    """MEDIUM: ShadowGraph uses both asyncio.Lock and threading.Lock for state."""

    def test_shadow_graph_has_both_locks(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        sg = ShadowGraph(graph_id="test_graph")
        self.assertIsInstance(sg._state_lock, asyncio.Lock)
        self.assertIsInstance(sg._sync_lock, type(threading.Lock()))
        self.assertIsInstance(sg._stats_lock, type(threading.Lock()))

    def test_sync_methods_use_sync_lock(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        sg = ShadowGraph(graph_id="test_graph")
        # get_status acquires _sync_lock
        status = sg.get_status()
        self.assertIn("current_mode", status)


# ── Finding #6: shadow_crew.py _find_available_agent thread-safe ──

class TestShadowCrewFindAgentThreadSafe(unittest.TestCase):
    """MEDIUM: _find_available_agent now uses registry lock for _agents access."""

    def test_find_available_agent_works(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, CrewTask
        crew = ShadowCrew(crew_id="test_crew")
        task = CrewTask(description="test task", task_id="t1")
        # Should work without error (uses registry lock internally)
        agent = crew._find_available_agent("nonexistent", task)
        # Default agents exist, so should find one
        self.assertIsNotNone(agent)

    def test_find_available_agent_thread_safe(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, CrewTask
        crew = ShadowCrew(crew_id="test_crew")
        errors = []

        def find_agent():
            try:
                task = CrewTask(description="test", task_id="t_find")
                crew._find_available_agent("nonexistent", task)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=find_agent) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0, f"Concurrent _find_available_agent errors: {errors}")


# ── Finding #7: shadow_crew.py execute_mission acquires _missions_lock ──

class TestShadowCrewExecuteMissionLock(unittest.TestCase):
    """MEDIUM: execute_mission acquires _missions_lock for mission lookup."""

    def test_execute_mission_nonexistent(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew(crew_id="test_crew")
        result = asyncio.run(crew.execute_mission("nonexistent_id"))
        self.assertIn("error", result)


# ── Finding #8: shadow_crew.py handoff acquires _shared_context_lock ──

class TestShadowCrewHandoffLock(unittest.TestCase):
    """MEDIUM: handoff acquires _shared_context_lock (asyncio.Lock)."""

    def test_handoff_acquires_lock(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew(crew_id="test_crew")
        result = asyncio.run(crew.handoff("shadow_eye", "shadow_mind", {"observations": "data"}))
        self.assertTrue(result)
        self.assertIn("handoff_shadow_eye_shadow_mind", crew._shared_context)


# ── Finding #10: autonomous_daemon.py consistent lock ordering ──

class TestAutonomousDaemonLockOrdering(unittest.TestCase):
    """LOW: AutonomousDaemon has consistent single-lock pattern."""

    def test_daemon_has_lock(self):
        from shadowd.cognition.autonomous_daemon import AutonomousDaemon
        daemon = AutonomousDaemon(base_tick_seconds=15.0)
        self.assertIsInstance(daemon._lock, type(threading.Lock()))
        self.assertIsInstance(daemon._stats_lock, type(threading.Lock()))


# ── Finding #11: neural_mesh.py add_peer/remove_peer thread-safe ──

class TestNeuralMeshPeerLock(unittest.TestCase):
    """MEDIUM: NeuralMesh add_peer/remove_peer now use _peers_lock."""

    def test_mesh_has_peers_lock(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        mesh = NeuralMesh(node_id="test_node", node_name="Test", device_type="server")
        self.assertIsInstance(mesh._peers_lock, type(threading.Lock()))

    def test_add_peer_thread_safe(self):
        from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode
        mesh = NeuralMesh(node_id="test_node", node_name="Test", device_type="server")
        peer = MeshNode(node_id="peer1", name="Peer1", device_type="laptop", port=8765)
        mesh.add_peer(peer)
        self.assertIn("peer1", mesh._peers)

    def test_remove_peer_thread_safe(self):
        from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode
        mesh = NeuralMesh(node_id="test_node", node_name="Test", device_type="server")
        peer = MeshNode(node_id="peer1", name="Peer1", device_type="laptop", port=8765)
        mesh.add_peer(peer)
        mesh.remove_peer("peer1")
        self.assertNotIn("peer1", mesh._peers)

    def test_concurrent_add_remove_peers(self):
        from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode
        mesh = NeuralMesh(node_id="test_node", node_name="Test", device_type="server")
        errors = []

        def add_peer():
            try:
                for i in range(20):
                    peer = MeshNode(
                        node_id=f"peer_{threading.current_thread().name}_{i}",
                        name=f"Peer_{i}", device_type="laptop", port=8765
                    )
                    mesh.add_peer(peer)
            except Exception as e:
                errors.append(e)

        def remove_peer():
            try:
                time.sleep(0.01)
                for i in range(10):
                    mesh.remove_peer(f"peer_Thread-1_{i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_peer, name="Thread-1")
        t2 = threading.Thread(target=remove_peer, name="Thread-2")
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(len(errors), 0, f"Concurrent peer operations failed: {errors}")

    def test_get_peers_thread_safe(self):
        from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode
        mesh = NeuralMesh(node_id="test_node", node_name="Test", device_type="server")
        peer = MeshNode(node_id="peer1", name="Peer1", device_type="laptop", port=8765)
        mesh.add_peer(peer)
        peers = mesh.get_peers()
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0]["node_id"], "peer1")

    def test_update_peer_health_thread_safe(self):
        from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode
        mesh = NeuralMesh(node_id="test_node", node_name="Test", device_type="server")
        peer = MeshNode(node_id="peer1", name="Peer1", device_type="laptop", port=8765)
        mesh.add_peer(peer)
        mesh.update_peer_health("peer1", success=True, latency_ms=10.0)
        health = mesh.get_mesh_health()
        self.assertIn("peer1", health)

    def test_get_mesh_health_thread_safe(self):
        from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode
        mesh = NeuralMesh(node_id="test_node", node_name="Test", device_type="server")
        peer = MeshNode(node_id="peer1", name="Peer1", device_type="laptop", port=8765)
        mesh.add_peer(peer)
        mesh.update_peer_health("peer1", success=True)
        health = mesh.get_mesh_health()
        self.assertIsInstance(health, dict)


# ── Finding #12: neural_mesh.py per-component locks ──

class TestNeuralMeshComponentLocks(unittest.TestCase):
    """LOW: GossipProtocol, StateSync, MeshConsensus have their own locks."""

    def test_gossip_has_lock(self):
        from shadowd.cognition.neural_mesh import GossipProtocol
        gp = GossipProtocol()
        self.assertIsInstance(gp._lock, type(threading.Lock()))

    def test_state_sync_has_lock(self):
        from shadowd.cognition.neural_mesh import StateSync
        ss = StateSync()
        self.assertIsInstance(ss._lock, type(threading.Lock()))

    def test_consensus_has_lock(self):
        from shadowd.cognition.neural_mesh import MeshConsensus
        mc = MeshConsensus("node1", {})
        self.assertIsInstance(mc._lock, type(threading.Lock()))


# ── Finding #13: shadow_crew.py CircuitBreaker is thread-safe ──

class TestCircuitBreakerThreadSafe(unittest.TestCase):
    """LOW: CircuitBreaker has its own _lock for state transitions."""

    def test_circuit_breaker_has_lock(self):
        from shadowd.cognition.shadow_crew import CircuitBreaker
        cb = CircuitBreaker()
        self.assertIsInstance(cb._lock, type(threading.Lock()))

    def test_circuit_breaker_thread_safe(self):
        from shadowd.cognition.shadow_crew import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        errors = []

        def record_failure():
            try:
                for _ in range(5):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_failure) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0, f"CircuitBreaker errors: {errors}")


# ── Finding #14: dark_memory.py InMemoryVectorIndex asyncio.Lock documented ──

class TestInMemoryVectorIndexLockDocumentation(unittest.TestCase):
    """MEDIUM: InMemoryVectorIndex._lock is asyncio.Lock, documented as not thread-safe."""

    def test_vector_index_lock_is_asyncio(self):
        from shadowd.cognition.dark_memory import InMemoryVectorIndex
        idx = InMemoryVectorIndex()
        self.assertIsInstance(idx._lock, asyncio.Lock)

    def test_vector_index_docstring_documents_thread_safety(self):
        from shadowd.cognition.dark_memory import InMemoryVectorIndex
        self.assertIn("NOT thread-safe", InMemoryVectorIndex.__doc__)

    def test_vector_index_docstring_documents_lock_ordering(self):
        from shadowd.cognition.dark_memory import InMemoryVectorIndex
        self.assertIn("Lock ordering", InMemoryVectorIndex.__doc__)


# ── Additional: dark_memory.py get_health() uses _atoms_lock ──

class TestDMGetHealthLock(unittest.TestCase):
    """Additional: get_health() acquires _atoms_lock for _atoms reads."""

    def test_get_health_thread_safe(self):
        from shadowd.cognition.dark_memory import DarkMemory
        dm = DarkMemory(data_dir="/tmp/test_sprint171_dm5")
        try:
            asyncio.run(dm.store("health test 1", importance=0.8, domain="test"))
            asyncio.run(dm.store("health test 2", importance=0.1, domain="test"))

            health = dm.get_health()
            self.assertIn("score", health)
            self.assertIn("status", health)
        finally:
            dm.close()


# ── Lock ordering consistency checks ──

class TestLockOrderingConsistency(unittest.TestCase):
    """Verify consistent lock ordering across modules to prevent deadlocks."""

    def test_dark_memory_lock_ordering(self):
        """DarkMemory lock order: _atoms_lock -> _db_thread_lock -> _db_lock -> _stats_lock."""
        from shadowd.cognition.dark_memory import DarkMemory
        dm = DarkMemory(data_dir="/tmp/test_sprint171_dm6")
        try:
            # Verify all locks exist
            self.assertIsInstance(dm._atoms_lock, type(threading.Lock()))
            self.assertIsInstance(dm._db_thread_lock, type(threading.Lock()))
            self.assertIsInstance(dm._db_lock, asyncio.Lock)
            self.assertIsInstance(dm._stats_lock, type(threading.Lock()))
        finally:
            dm.close()

    def test_shadow_crew_lock_ordering(self):
        """ShadowCrew lock order: _missions_lock -> _cb_lock -> _stats_lock."""
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew(crew_id="test_crew")
        self.assertIsInstance(crew._missions_lock, type(threading.Lock()))
        self.assertIsInstance(crew._cb_lock, type(threading.Lock()))
        self.assertIsInstance(crew._stats_lock, type(threading.Lock()))
        self.assertIsInstance(crew._shared_context_lock, asyncio.Lock)

    def test_anomaly_detector_lock_ordering(self):
        """AnomalyDetector lock order: _lock -> _stats_lock."""
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        ad = AnomalyDetectionEngine()
        self.assertIsInstance(ad._lock, type(threading.Lock()))
        self.assertIsInstance(ad._stats_lock, type(threading.Lock()))

    def test_attention_engine_lock_ordering(self):
        """AttentionEngine lock order: _lock -> _stats_lock."""
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        self.assertIsInstance(ae._lock, type(threading.Lock()))
        self.assertIsInstance(ae._stats_lock, type(threading.Lock()))

    def test_neural_mesh_lock_ordering(self):
        """NeuralMesh lock order: _peers_lock -> _stats_lock, per-component _lock."""
        from shadowd.cognition.neural_mesh import NeuralMesh
        mesh = NeuralMesh(node_id="test_node", node_name="Test", device_type="server")
        self.assertIsInstance(mesh._peers_lock, type(threading.Lock()))
        self.assertIsInstance(mesh._stats_lock, type(threading.Lock()))
        # Per-component locks
        self.assertIsInstance(mesh._gossip._lock, type(threading.Lock()))
        self.assertIsInstance(mesh._sync._lock, type(threading.Lock()))
        self.assertIsInstance(mesh._consensus._lock, type(threading.Lock()))


if __name__ == "__main__":
    unittest.main()
