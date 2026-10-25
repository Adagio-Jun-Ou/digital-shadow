#!/usr/bin/env python3
"""
Sprint 141 regression tests:
- neural_mesh.py: _stats_lock for thread safety
- shadow_crew.py: _stats already under _stats_lock (verify)
"""

import threading
import pytest
from unittest.mock import MagicMock, AsyncMock

from shadowd.cognition.neural_mesh import NeuralMesh, MeshNode
from shadowd.cognition.shadow_crew import ShadowCrew


class TestNeuralMeshStatsLock:
    def test_stats_lock_exists(self):
        nm = NeuralMesh.__new__(NeuralMesh)
        nm._stats = {"messages_sent": 0, "messages_received": 0, "gossip_propagations": 0, "syncs_completed": 0, "consensuses_reached": 0, "discoveries": 0}
        nm._stats_lock = threading.Lock()
        assert hasattr(nm, '_stats_lock')

    def test_concurrent_stats_increment(self):
        nm = NeuralMesh.__new__(NeuralMesh)
        nm._stats = {"messages_sent": 0, "messages_received": 0, "gossip_propagations": 0, "syncs_completed": 0, "consensuses_reached": 0, "discoveries": 0}
        nm._stats_lock = threading.Lock()
        errors = []
        def increment():
            try:
                for _ in range(100):
                    with nm._stats_lock:
                        nm._stats["messages_sent"] += 1
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert nm._stats["messages_sent"] == 400

    def test_get_stats_returns_copy(self):
        nm = NeuralMesh.__new__(NeuralMesh)
        nm._stats = {"messages_sent": 42, "messages_received": 10, "gossip_propagations": 0, "syncs_completed": 0, "consensuses_reached": 0, "discoveries": 0}
        nm._stats_lock = threading.Lock()
        # get_stats() also acquires _peers_lock; the original test only stubbed
        # _stats_lock, which broke after _peers_lock was added.
        nm._peers_lock = threading.Lock()
        nm._peers = {}
        nm.local_node = MagicMock()
        nm.local_node.to_dict.return_value = {}
        nm._message_queue = MagicMock()
        nm._message_queue.qsize.return_value = 0
        nm._gossip = MagicMock()
        nm._gossip.get_stats.return_value = {}
        nm._sync = MagicMock()
        nm._sync.get_stats.return_value = {}
        nm._consensus = MagicMock()
        nm._consensus.get_stats.return_value = {}
        stats = nm.get_stats()
        stats["stats"]["messages_sent"] = 9999
        assert nm._stats["messages_sent"] == 42


class TestShadowCrewStatsAlreadyLocked:
    def test_crew_stats_lock_exists(self):
        crew = ShadowCrew(crew_id="test_lock")
        assert hasattr(crew, '_stats_lock')
        assert isinstance(crew._stats_lock, type(threading.Lock()))

    def test_crew_stats_increment_under_lock(self):
        crew = ShadowCrew(crew_id="test_increment")
        errors = []
        def increment():
            try:
                for _ in range(100):
                    with crew._stats_lock:
                        crew._stats["missions_created"] += 1
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert crew._stats["missions_created"] == 400