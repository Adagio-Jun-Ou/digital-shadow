"""
Session 5 Sprint 29 — Feature Improvement Regression Tests

Features:
  1. DarkMemory TTL-based auto-expiration daemon
  2. GoalPlanner goal dependencies visualization
  3. AnomalyDetector persistence integration with ShadowOS
  4. WebSocket manager ping/pong health check
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
# Feature 1: DarkMemory TTL-based auto-expiration daemon
# ═══════════════════════════════════════════════════════════════════

class TestDarkMemoryTTLExpiration:
    def test_default_ttl_parameter(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=720.0)
            assert dm._default_ttl_hours == 720.0

    def test_default_ttl_default_value(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td)
            assert dm._default_ttl_hours == 720.0

    def test_custom_ttl_parameter(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=48.0)
            assert dm._default_ttl_hours == 48.0

    @pytest.mark.asyncio
    async def test_low_importance_atoms_get_expires_at(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=48.0)
            atom = await dm.store("low importance note", importance=0.1)
            assert atom.expires_at is not None
            expires = datetime.fromisoformat(atom.expires_at)
            now = datetime.now(timezone.utc)
            expected = now + timedelta(hours=48)
            assert abs((expires - expected).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_high_importance_atoms_no_expires_at(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=48.0)
            atom = await dm.store("important insight", importance=0.8)
            assert atom.expires_at is None

    @pytest.mark.asyncio
    async def test_medium_importance_atoms_no_expires_at(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=48.0)
            atom = await dm.store("medium note", importance=0.5)
            assert atom.expires_at is None

    @pytest.mark.asyncio
    async def test_boundary_importance_0_3_no_expires(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=48.0)
            atom = await dm.store("boundary note", importance=0.3)
            assert atom.expires_at is None

    @pytest.mark.asyncio
    async def test_evict_expired_removes_expired_atoms(self):
        from shadowd.cognition.dark_memory import DarkMemory, MemoryAtom, AtomType, MemoryTier
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=720)
            past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            atom = MemoryAtom(
                content="expired atom",
                importance=0.1,
                expires_at=past,
            )
            atom.embedding = [0.0] * 64
            dm._atoms[atom.atom_id] = atom
            await dm._vector_index.add(atom.atom_id, atom.embedding)
            with dm._db_thread_lock:
                dm._db_conn.execute(
                    "INSERT OR REPLACE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (atom.atom_id, "observation", atom.content, "{}", "[]",
                     None, atom.importance, 0, atom.created_at, atom.accessed_at)
                )
                dm._db_conn.commit()

            count = await dm._evict_expired()
            assert count == 1
            assert atom.atom_id not in dm._atoms

    @pytest.mark.asyncio
    async def test_evict_expired_keeps_non_expired(self):
        from shadowd.cognition.dark_memory import DarkMemory, MemoryAtom, AtomType, MemoryTier
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=720)
            future = (datetime.now(timezone.utc) + timedelta(hours=100)).isoformat()
            atom = MemoryAtom(
                content="still valid",
                importance=0.1,
                expires_at=future,
            )
            atom.embedding = [0.0] * 64
            dm._atoms[atom.atom_id] = atom
            await dm._vector_index.add(atom.atom_id, atom.embedding)

            count = await dm._evict_expired()
            assert count == 0
            assert atom.atom_id in dm._atoms

    @pytest.mark.asyncio
    async def test_evict_expired_removes_from_vector_and_db(self):
        from shadowd.cognition.dark_memory import DarkMemory, MemoryAtom
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=720)
            past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            atom = MemoryAtom(content="expired", importance=0.1, expires_at=past)
            atom.embedding = [0.0] * 64
            dm._atoms[atom.atom_id] = atom
            await dm._vector_index.add(atom.atom_id, atom.embedding)
            with dm._db_thread_lock:
                dm._db_conn.execute(
                    "INSERT OR REPLACE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (atom.atom_id, "observation", atom.content, "{}", "[]",
                     None, atom.importance, 0, atom.created_at, atom.accessed_at)
                )
                dm._db_conn.commit()

            await dm._evict_expired()
            assert dm._vector_index.size() == 0
            cursor = dm._db_conn.execute("SELECT COUNT(*) FROM observations WHERE id = ?", (atom.atom_id,))
            assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_atoms_without_expires_at_kept(self):
        from shadowd.cognition.dark_memory import DarkMemory, MemoryAtom
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=720)
            atom = MemoryAtom(content="no expiry", importance=0.8)
            atom.expires_at = None
            atom.embedding = [0.0] * 64
            dm._atoms[atom.atom_id] = atom
            await dm._vector_index.add(atom.atom_id, atom.embedding)

            count = await dm._evict_expired()
            assert count == 0
            assert atom.atom_id in dm._atoms

    @pytest.mark.asyncio
    async def test_ttl_eviction_loop_cancels(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=720, compaction_interval=9999)
            dm._ttl_eviction_interval = 0.05
            await dm.start_ttl_eviction()
            assert dm._ttl_eviction_task is not None
            assert not dm._ttl_eviction_task.done()
            dm._ttl_eviction_task.cancel()
            try:
                await dm._ttl_eviction_task
            except asyncio.CancelledError:
                pass
            dm.close()

    @pytest.mark.asyncio
    async def test_close_cancels_ttl_eviction_task(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=720, compaction_interval=9999)
            dm._ttl_eviction_interval = 9999
            await dm.start_ttl_eviction()
            assert dm._ttl_eviction_task is not None
            dm.close()
            dm._ttl_eviction_task.cancel()
            try:
                await dm._ttl_eviction_task
            except asyncio.CancelledError:
                pass


# ═══════════════════════════════════════════════════════════════════
# Feature 2: GoalPlanner goal dependencies visualization
# ═══════════════════════════════════════════════════════════════════

class TestGoalPlannerDependencyGraph:
    def test_get_dependency_graph_empty(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        graph = gp.get_dependency_graph()
        assert graph == {}

    def test_get_dependency_graph_single_goal(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        goal = gp.create_goal("Root goal")
        graph = gp.get_dependency_graph()
        assert goal.goal_id in graph
        entry = graph[goal.goal_id]
        assert entry["goal"]["goal_id"] == goal.goal_id
        assert "depends_on" in entry
        assert "depended_on_by" in entry

    def test_get_dependency_graph_with_parent(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        parent = gp.create_goal("Parent goal")
        child = gp.create_goal("Child goal")
        gp.add_sub_goal(parent.goal_id, child.goal_id)
        graph = gp.get_dependency_graph()
        assert parent.goal_id in graph[child.goal_id]["depends_on"]
        assert child.goal_id in graph[parent.goal_id]["depended_on_by"]

    def test_get_dependency_graph_step_dependencies(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        goal = gp.create_goal("Dependent goal")
        gp.decompose_goal(goal.goal_id, [
            {"description": "step 1"},
            {"description": "step 2", "dependencies": ["step_a"]},
        ])
        graph = gp.get_dependency_graph()
        entry = graph[goal.goal_id]
        assert "step_a" in entry["depends_on"]

    def test_get_dependency_graph_no_cross_refs(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        g1 = gp.create_goal("Independent 1")
        g2 = gp.create_goal("Independent 2")
        graph = gp.get_dependency_graph()
        assert graph[g1.goal_id]["depended_on_by"] == []
        assert graph[g2.goal_id]["depended_on_by"] == []


class TestGoalPlannerCriticalPath:
    def test_get_critical_path_empty(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        path = gp.get_critical_path()
        assert path == []

    def test_get_critical_path_single_incomplete(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        goal = gp.create_goal("Only goal")
        path = gp.get_critical_path()
        assert len(path) == 1
        assert path[0]["goal_id"] == goal.goal_id

    def test_get_critical_path_completed_excluded(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner, GoalStatus
        gp = CognitiveGoalPlanner()
        g1 = gp.create_goal("Done goal")
        g2 = gp.create_goal("Open goal")
        gp._goals[g1.goal_id].status = GoalStatus.COMPLETED
        path = gp.get_critical_path()
        assert len(path) == 1
        assert path[0]["goal_id"] == g2.goal_id

    def test_get_critical_path_chain(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        g1 = gp.create_goal("Root")
        g2 = gp.create_goal("Child")
        g3 = gp.create_goal("Grandchild")
        gp.add_sub_goal(g1.goal_id, g2.goal_id)
        gp.add_sub_goal(g2.goal_id, g3.goal_id)
        path = gp.get_critical_path()
        assert len(path) == 3
        ids = [p["goal_id"] for p in path]
        assert g1.goal_id in ids
        assert g2.goal_id in ids
        assert g3.goal_id in ids

    def test_get_critical_path_finds_longest(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        g1 = gp.create_goal("Root")
        g2 = gp.create_goal("Short branch")
        g3 = gp.create_goal("Long branch child")
        g4 = gp.create_goal("Long branch grandchild")
        gp.add_sub_goal(g1.goal_id, g2.goal_id)
        gp.add_sub_goal(g1.goal_id, g3.goal_id)
        gp.add_sub_goal(g3.goal_id, g4.goal_id)
        path = gp.get_critical_path()
        assert len(path) >= 3

    def test_get_blocked_goals_returns_blocked(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner, GoalStatus
        gp = CognitiveGoalPlanner()
        g1 = gp.create_goal("Blocked goal")
        gp._goals[g1.goal_id].status = GoalStatus.BLOCKED
        blocked = gp.get_blocked_goals()
        assert len(blocked) == 1
        assert blocked[0]["goal_id"] == g1.goal_id

    def test_get_blocked_goals_ignores_active(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner()
        gp.create_goal("Active goal")
        blocked = gp.get_blocked_goals()
        assert len(blocked) == 0


# ═══════════════════════════════════════════════════════════════════
# Feature 3: AnomalyDetector persistence integration
# ═══════════════════════════════════════════════════════════════════

class TestAnomalyDetectorMonitoring:
    def test_start_monitoring_creates_task(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        assert engine._monitoring_task is None
        assert engine._monitoring is False

    @pytest.mark.asyncio
    async def test_start_monitoring_sets_flag(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine._monitoring_interval = 0.05
        await engine.start_monitoring()
        assert engine._monitoring is True
        assert engine._monitoring_task is not None
        await engine.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_monitoring_cancels_task(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine._monitoring_interval = 9999
        await engine.start_monitoring()
        task = engine._monitoring_task
        await engine.stop_monitoring()
        assert engine._monitoring is False
        assert task.done() or task.cancelled()

    @pytest.mark.asyncio
    async def test_start_monitoring_idempotent(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine._monitoring_interval = 9999
        await engine.start_monitoring()
        task1 = engine._monitoring_task
        await engine.start_monitoring()
        task2 = engine._monitoring_task
        assert task1 is task2
        await engine.stop_monitoring()

    @pytest.mark.asyncio
    async def test_monitoring_detects_anomalies(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        for i in range(20):
            engine.observe("test_metric", 10.0 + i * 0.1)
        engine.observe("test_metric", 100.0)
        for i in range(20):
            engine.observe("test_metric2", 5.0 + i * 0.1)
        engine.observe("test_metric2", 500.0)
        for i in range(20):
            engine.observe("test_metric3", 3.0 + i * 0.1)
        engine.observe("test_metric3", 300.0)
        anomalies = engine.detect_correlation_anomaly(
            [], {"test_metric": 100.0, "test_metric2": 500.0, "test_metric3": 300.0}
        )
        assert len(anomalies) >= 1

    def test_monitoring_interval_default(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        assert engine._monitoring_interval == 60.0


# ═══════════════════════════════════════════════════════════════════
# Feature 4: WebSocket manager ping/pong health check
# ═══════════════════════════════════════════════════════════════════

class TestWebSocketManagerHealthCheck:
    def test_client_last_seen_initialized(self):
        from shadowd.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        assert hasattr(mgr, '_client_last_seen')
        assert isinstance(mgr._client_last_seen, dict)
        assert len(mgr._client_last_seen) == 0

    def test_get_client_health_empty(self):
        from shadowd.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        health = mgr.get_client_health()
        assert health == {}

    @pytest.mark.asyncio
    async def test_get_client_health_with_client(self):
        from shadowd.websocket_manager import ConnectionManager, WSClient, WSMessage, WSMessageType
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        client = WSClient(websocket=mock_ws, client_id="test-client-1")
        client.subscriptions = {"topic1", "topic2"}
        mgr.clients["test-client-1"] = client
        mgr._client_last_seen["test-client-1"] = time.time()
        health = mgr.get_client_health()
        assert "test-client-1" in health
        assert health["test-client-1"]["connected"] is True
        assert "last_seen" in health["test-client-1"]
        assert "topic1" in health["test-client-1"]["subscriptions"]
        assert "topic2" in health["test-client-1"]["subscriptions"]

    @pytest.mark.asyncio
    async def test_connect_sets_last_seen(self):
        from shadowd.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        client = await mgr.connect(mock_ws, "test-health-client")
        assert "test-health-client" in mgr._client_last_seen
        assert mgr._client_last_seen["test-health-client"] > 0
        await mgr.disconnect("test-health-client")

    @pytest.mark.asyncio
    async def test_disconnect_removes_last_seen(self):
        from shadowd.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        client = await mgr.connect(mock_ws, "test-remove-client")
        assert "test-remove-client" in mgr._client_last_seen
        await mgr.disconnect("test-remove-client")
        assert "test-remove-client" not in mgr._client_last_seen

    @pytest.mark.asyncio
    async def test_check_client_health_sends_ping(self):
        from shadowd.websocket_manager import ConnectionManager, WSMessageType
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        client = await mgr.connect(mock_ws, "test-ping-client")
        responses = await mgr.check_client_health()
        assert "test-ping-client" in responses
        assert isinstance(responses["test-ping-client"], bool)
        await mgr.disconnect("test-ping-client")

    def test_get_client_health_shows_disconnected(self):
        from shadowd.websocket_manager import ConnectionManager, WSClient
        mgr = ConnectionManager()
        dead_client = WSClient(websocket=AsyncMock(), client_id="dead-client")
        dead_client.is_alive = False
        mgr.clients["dead-client"] = dead_client
        mgr._client_last_seen["dead-client"] = time.time() - 300
        health = mgr.get_client_health()
        assert health["dead-client"]["connected"] is False

    @pytest.mark.asyncio
    async def test_handle_client_loop_updates_last_seen(self):
        from shadowd.websocket_manager import ConnectionManager, WSClient
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        client = await mgr.connect(mock_ws, "test-loop-client-2")
        assert mgr._client_last_seen.get("test-loop-client-2") is not None
        ts_before = mgr._client_last_seen["test-loop-client-2"]
        mgr._client_last_seen["test-loop-client-2"] = ts_before - 100
        mgr._client_last_seen["test-loop-client-2"] = time.time()
        assert mgr._client_last_seen["test-loop-client-2"] >= ts_before
        await mgr.disconnect("test-loop-client-2")


# ═══════════════════════════════════════════════════════════════════
# Integration: DarkMemory TTL + store interaction
# ═══════════════════════════════════════════════════════════════════

class TestDarkMemoryTTLIntegration:
    @pytest.mark.asyncio
    async def test_store_then_expire(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=1.0)
            atom = await dm.store("low importance", importance=0.1)
            assert atom.expires_at is not None
            past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            dm._atoms[atom.atom_id].expires_at = past
            count = await dm._evict_expired()
            assert count == 1
            retrieved = await dm.retrieve(atom.atom_id)
            assert retrieved is None

    @pytest.mark.asyncio
    async def test_store_high_importance_no_expiry(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=24.0)
            atom = await dm.store("critical info", importance=0.9)
            assert atom.expires_at is None
            count = await dm._evict_expired()
            assert count == 0
            retrieved = await dm.retrieve(atom.atom_id)
            assert retrieved is not None

    @pytest.mark.asyncio
    async def test_mixed_importance_expiry(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=24.0)
            low = await dm.store("low", importance=0.1)
            high = await dm.store("high", importance=0.8)
            assert low.expires_at is not None
            assert high.expires_at is None
            past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            dm._atoms[low.atom_id].expires_at = past
            count = await dm._evict_expired()
            assert count == 1
            assert low.atom_id not in dm._atoms
            assert high.atom_id in dm._atoms

    @pytest.mark.asyncio
    async def test_ttl_eviction_loop_runs(self):
        from shadowd.cognition.dark_memory import DarkMemory
        with tempfile.TemporaryDirectory() as td:
            dm = DarkMemory(data_dir=td, default_ttl_hours=720, compaction_interval=9999)
            dm._ttl_eviction_interval = 0.1
            atom = await dm.store("expiring", importance=0.1)
            past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            dm._atoms[atom.atom_id].expires_at = past
            await dm.start_ttl_eviction()
            await asyncio.sleep(0.3)
            assert atom.atom_id not in dm._atoms
            dm._ttl_eviction_task.cancel()
            try:
                await dm._ttl_eviction_task
            except asyncio.CancelledError:
                pass
            dm.close()