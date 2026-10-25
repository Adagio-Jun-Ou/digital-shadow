#!/usr/bin/env python3
"""
Session 5 Sprint 148 — time.time() → time.monotonic() Regression Tests

Verifies that replacing `time.time()` with `time.monotonic()` for
duration/rate-limit/monotonic-tracking uses doesn't break:
1. Rate limiting still works with monotonic timestamps
2. Duration checks still work
3. Timestamp-based IDs still get reasonable values
"""

import asyncio
import time
import threading
import unittest
from unittest.mock import patch, MagicMock

from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree, SkillDomain
from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine, MetricStream, AnomalyType
from shadowd.cognition.escalation import (
    EscalationManager, BudgetGuardrail, AutonomyLevel, GuardrailResult,
)
from shadowd.cognition.neural_mesh import (
    NeuralMesh, GossipProtocol, MeshNode, NodeStatus, MeshMessage, MeshMessageType,
)
from shadowd.cognition.personality_engine import PersonalityEngine, PersonaDistiller
from shadowd.cognition.dark_memory import MemoryAtom, AtomType, MemoryTier


class TestAdaptiveSkillTreeMonotonic(unittest.TestCase):
    def test_recent_usage_uses_monotonic(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("sk1", "Test Skill", "technical")
        before = time.monotonic()
        tree.use_skill("sk1")
        after = time.monotonic()
        recent = tree._recent_usage
        self.assertGreater(len(recent), 0)
        self.assertGreaterEqual(recent[-1], before)
        self.assertLessEqual(recent[-1], after)

    def test_usage_counting_unchanged(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("sk1", "Test Skill", "technical")
        for _ in range(5):
            tree.use_skill("sk1", success=True)
        skill = tree.get_skill("sk1")
        self.assertEqual(skill.usage_count, 5)

    def test_stats_reflect_usage(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("sk1", "Test Skill", "technical")
        tree.use_skill("sk1")
        stats = tree.get_stats()
        self.assertEqual(stats["recent_usage_count"], 1)


class TestAnomalyDetectorMonotonic(unittest.TestCase):
    def test_metric_stream_timestamps_monotonic(self):
        stream = MetricStream()
        before = time.monotonic()
        stream.add(42.0)
        after = time.monotonic()
        self.assertGreater(len(stream._timestamps), 0)
        self.assertGreaterEqual(stream._timestamps[-1], before)
        self.assertLessEqual(stream._timestamps[-1], after)

    def test_anomaly_detection_still_works(self):
        engine = AnomalyDetectionEngine()
        engine.register_metric("cpu")
        for i in range(50):
            engine.observe("cpu", 50.0 + i * 0.1)
        result = engine.observe("cpu", 500.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.anomaly_type, AnomalyType.STATISTICAL)

    def test_rate_limiting_duration_check(self):
        stream = MetricStream()
        t1 = time.monotonic()
        stream.add(1.0)
        t2 = stream._timestamps[-1]
        elapsed = t2 - t1
        self.assertGreaterEqual(elapsed, 0)
        self.assertLess(elapsed, 1.0)


class TestEscalationMonotonic(unittest.TestCase):
    def test_budget_guardrail_rate_limit(self):
        bg = BudgetGuardrail(max_requests_per_minute=3)

        async def _check():
            return await bg.check("test")

        for _ in range(3):
            asyncio.run(_check())
        result, _, reason = asyncio.run(_check())
        self.assertEqual(result, GuardrailResult.REJECT)
        self.assertIn("Rate limit", reason)

    def test_escalation_rate_limit(self):
        em = EscalationManager()
        em._rate_limit_per_minute = 2

        async def _eval():
            return await em.evaluate("observe", {})

        asyncio.run(_eval())
        asyncio.run(_eval())
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(_eval())
        self.assertIn("Rate limit", str(ctx.exception))

    def test_budget_guardrail_timestamps_are_monotonic(self):
        bg = BudgetGuardrail(max_requests_per_minute=100)
        before = time.monotonic()

        async def _check():
            return await bg.check("test")

        asyncio.run(_check())
        after = time.monotonic()
        self.assertGreater(len(bg._request_times), 0)
        self.assertGreaterEqual(bg._request_times[-1], before)
        self.assertLessEqual(bg._request_times[-1], after)

    def test_escalation_rate_requests_are_monotonic(self):
        em = EscalationManager()
        em._rate_limit_per_minute = 100
        before = time.monotonic()

        async def _eval():
            return await em.evaluate("observe", {})

        asyncio.run(_eval())
        after = time.monotonic()
        self.assertGreater(len(em._rate_requests), 0)
        self.assertGreaterEqual(em._rate_requests[-1], before)
        self.assertLessEqual(em._rate_requests[-1], after)

    def test_rate_limit_expires_after_window(self):
        em = EscalationManager()
        em._rate_limit_per_minute = 1

        async def _eval():
            return await em.evaluate("observe", {})

        asyncio.run(_eval())
        with self.assertRaises(ValueError):
            asyncio.run(_eval())
        em._rate_requests = []
        result = asyncio.run(_eval())
        self.assertIsNotNone(result)


class TestNeuralMeshMonotonic(unittest.TestCase):
    def test_gossip_rate_limit_uses_monotonic(self):
        gp = GossipProtocol(rate_limit_per_second=100)
        before = time.monotonic()
        gp._check_rate_limit()
        after = time.monotonic()
        self.assertGreater(len(gp._rate_limiter), 0)
        self.assertGreaterEqual(gp._rate_limiter[-1], before)
        self.assertLessEqual(gp._rate_limiter[-1], after)

    def test_gossip_rate_limit_enforced(self):
        gp = GossipProtocol(rate_limit_per_second=2)
        self.assertTrue(gp._check_rate_limit())
        self.assertTrue(gp._check_rate_limit())
        self.assertFalse(gp._check_rate_limit())

    def test_seen_messages_timestamp_monotonic(self):
        gp = GossipProtocol()
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="node1",
            payload={"test": 1},
        )
        before = int(time.monotonic())
        gp.should_propagate(msg)
        fp = msg.fingerprint()
        ts = gp._seen_messages.get(fp)
        self.assertIsNotNone(ts)
        self.assertGreaterEqual(ts, before)

    def test_cleanup_uses_monotonic(self):
        gp = GossipProtocol()
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="node1",
            payload={"test": 1},
        )
        gp.should_propagate(msg)
        fp = msg.fingerprint()
        self.assertIn(fp, gp._seen_messages)
        gp._seen_messages[fp] = int(time.monotonic()) - 100
        gp.cleanup(max_age_seconds=10)
        self.assertNotIn(fp, gp._seen_messages)

    def test_peer_health_last_seen_monotonic(self):
        mesh = NeuralMesh(node_id="local", node_name="test")
        peer = MeshNode(node_id="peer1", name="Test", port=8765)
        before = time.monotonic()
        mesh.add_peer(peer)
        after = time.monotonic()
        self.assertIn("peer1", mesh._peer_health)
        ls = mesh._peer_health["peer1"]["last_seen"]
        self.assertGreaterEqual(ls, before)
        self.assertLessEqual(ls, after)

    def test_update_peer_health_monotonic(self):
        mesh = NeuralMesh(node_id="local", node_name="test")
        peer = MeshNode(node_id="peer1", name="Test", port=8765)
        mesh.add_peer(peer)
        before = time.monotonic()
        mesh.update_peer_health("peer1", success=True)
        after = time.monotonic()
        ls = mesh._peer_health["peer1"]["last_seen"]
        self.assertGreaterEqual(ls, before)
        self.assertLessEqual(ls, after)

    def test_message_id_still_uses_wall_clock(self):
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="node1",
        )
        now_wall = int(time.time())
        parts = msg.message_id.split("_")
        ts_part = int(parts[-1])
        self.assertAlmostEqual(ts_part, now_wall, delta=2)

    def test_mesh_health_comparison_works(self):
        mesh = NeuralMesh(node_id="local", node_name="test")
        peer = MeshNode(node_id="peer1", name="Test", port=8765)
        mesh.add_peer(peer)
        mesh.update_peer_health("peer1", success=True, latency_ms=10.0)
        health = mesh.get_mesh_health()
        self.assertIn("peer1", health)
        self.assertGreater(health["peer1"]["health_score"], 0)


class TestPersonalityEngineMonotonic(unittest.TestCase):
    def test_distiller_rate_limit_uses_monotonic(self):
        distiller = PersonaDistiller()
        before = time.monotonic()
        distiller.record_behavior("test_event", {"key": "value"})
        after = time.monotonic()
        self.assertGreater(len(distiller._signal_timestamps), 0)
        ts = distiller._signal_timestamps[-1]
        self.assertGreaterEqual(ts, before)
        self.assertLessEqual(ts, after)

    def test_distiller_rate_limit_enforced(self):
        distiller = PersonaDistiller()
        distiller._max_signals_per_second = 3
        for _ in range(3):
            distiller.record_behavior("test_event", {"key": "value"})
        count_before = len(distiller._behavior_signals)
        distiller.record_behavior("test_event", {"key": "value"})
        count_after = len(distiller._behavior_signals)
        self.assertEqual(count_after, count_before)

    def test_rate_limit_resets_after_window(self):
        distiller = PersonaDistiller()
        distiller._max_signals_per_second = 2
        distiller.record_behavior("test1", {"k": "v"})
        distiller.record_behavior("test2", {"k": "v"})
        self.assertEqual(len(distiller._behavior_signals), 2)
        distiller._signal_timestamps.clear()
        distiller.record_behavior("test3", {"k": "v"})
        self.assertEqual(len(distiller._behavior_signals), 3)


class TestIDGenerationStillUsesWallClock(unittest.TestCase):
    def test_atom_id_uses_wall_clock_not_monotonic(self):
        atom = MemoryAtom(
            content="test content",
            atom_type=AtomType.OBSERVATION,
            tier=MemoryTier.WORKING,
        )
        now_wall = int(time.time())
        ts_part = int(atom.atom_id.split("_")[-1])
        self.assertAlmostEqual(ts_part, now_wall, delta=2)

    def test_message_id_uses_wall_clock_not_monotonic(self):
        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="test",
        )
        now_wall = int(time.time())
        ts_part = int(msg.message_id.split("_")[-1])
        self.assertAlmostEqual(ts_part, now_wall, delta=2)

    def test_id_values_are_reasonable(self):
        atom = MemoryAtom(
            content="test",
            atom_type=AtomType.OBSERVATION,
            tier=MemoryTier.WORKING,
        )
        ts_part = int(atom.atom_id.split("_")[-1])
        self.assertGreater(ts_part, 1_700_000_000)
        self.assertLess(ts_part, 2_000_000_000)

        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="test",
        )
        ts_part = int(msg.message_id.split("_")[-1])
        self.assertGreater(ts_part, 1_700_000_000)
        self.assertLess(ts_part, 2_000_000_000)


class TestMixedClockConsistency(unittest.TestCase):
    def test_neural_mesh_uses_mixed_clocks_correctly(self):
        mesh = NeuralMesh(node_id="local", node_name="test")
        peer = MeshNode(node_id="peer1", name="Test", port=8765)
        mesh.add_peer(peer)

        msg = MeshMessage(
            message_type=MeshMessageType.GOSSIP,
            sender_id="local",
        )

        now_wall = int(time.time())
        msg_ts = int(msg.message_id.split("_")[-1])
        self.assertAlmostEqual(msg_ts, now_wall, delta=2)

        health_ts = mesh._peer_health["peer1"]["last_seen"]
        mono_ts = time.monotonic()
        self.assertLess(abs(health_ts - mono_ts), 1.0)

    def test_system_clock_not_mixed_with_monotonic(self):
        gp = GossipProtocol()
        gp._check_rate_limit()
        for ts in gp._rate_limiter:
            mono_now = time.monotonic()
            self.assertLess(ts, mono_now + 1.0)
            self.assertGreater(ts, mono_now - 10.0)


if __name__ == "__main__":
    unittest.main()