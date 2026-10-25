"""
Session 12 — FeedLoop dispatcher regression tests.

The FeedLoop had producers (ShadowOS emits on observations) and a bus, but no
consumer: signals accumulated in a deque and were never translated into real
module calls. These tests pin the dispatcher: every routed signal must land in
the target module's real intake API exactly once (dedup), with low-intensity
signals dropped and stats surfaced in get_status().
"""

import os
import tempfile
import unittest
from pathlib import Path


def _make_os(tmp: Path):
    from shadowd.cognition.shadow_os import ShadowOS
    return ShadowOS(data_dir=str(tmp / "shadow_os"))


def _emit_and_dispatch(sos, source, signal_type, intensity, content):
    ids = sos.feed_loop.emit(source, signal_type, intensity, content)
    sos._dispatch_feed_signals()
    return ids


class TestFeedDispatch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.sos = _make_os(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_anomaly_signal_reaches_attention(self):
        ids = self.sos.feed_loop.emit(
            "anomaly_detector", "anomaly", 0.9,
            {"topic": "disk filling up", "domain": "system"},
        )
        # emit() returns one signal per routed target (anomaly_detector fans
        # out to 5 targets), not one per call.
        self.assertGreaterEqual(len(ids), 1)
        self.sos._dispatch_feed_signals()
        priorities = self.sos.attention.get_priorities(limit=10)
        self.assertTrue(any("disk filling up" in p.topic for p in priorities),
                        f"attention priorities: {priorities}")

    def test_behavior_signal_reaches_personality(self):
        self.sos.feed_loop.emit(
            "emotional_intelligence", "behavior", 0.9,
            {"description": "late-night work burst"},
        )
        self.sos._dispatch_feed_signals()
        stats = self.sos.personality.get_stats()
        # behavior was recorded (pattern/behavior counters grew)
        self.assertGreaterEqual(
            stats.get("behaviors_recorded", stats.get("behavior_signals", 0)), 1,
            f"personality stats: {stats}")

    def test_anomaly_signal_reaches_evolution(self):
        self.sos.feed_loop.emit(
            "anomaly_detector", "anomaly", 0.8,
            {"topic": "sleep schedule broken", "domain": "health"},
        )
        self.sos._dispatch_feed_signals()
        stats = self.sos.evolution.get_stats()
        self.assertGreaterEqual(stats.get("observations_added", 0), 1,
                                f"evolution stats: {stats}")

    def test_pattern_signal_reaches_dream(self):
        self.sos.feed_loop.emit(
            "narrative_memory", "pattern", 0.9,
            {"topic": "weekly review rhythm", "domain": "habits"},
        )
        self.sos._dispatch_feed_signals()
        stats = self.sos.dream.get_stats()
        self.assertGreaterEqual(stats.get("total_memories", 0), 1,
                                f"dream stats: {stats}")

    def test_pattern_signal_reaches_narrative(self):
        self.sos.feed_loop.emit(
            "recall_synthesis", "pattern", 0.9,
            {"topic": "cross-domain project overlap", "domain": "work"},
        )
        self.sos._dispatch_feed_signals()
        stats = self.sos.narrative.get_stats()
        self.assertGreaterEqual(stats.get("total_events", 0), 1,
                                f"narrative stats: {stats}")

    def test_high_intensity_cognitive_reaches_trinity(self):
        # thought_sense -> cognitive_trinity weight 0.9: 0.9 * 0.9 = 0.81 >= 0.7
        self.sos.feed_loop.emit(
            "thought_sense", "cognitive", 0.9,
            {"topic": "should I refactor the scheduler?"},
        )
        self.sos._dispatch_feed_signals()
        stats = self.sos.trinity.get_stats()
        self.assertGreaterEqual(stats.get("total_analyses", 0), 1,
                                f"trinity stats: {stats}")

    def test_low_intensity_trinity_signal_dropped(self):
        # 0.5 * 0.9 = 0.45 < 0.7 threshold -> dropped, no trinity analysis
        self.sos.feed_loop.emit(
            "thought_sense", "cognitive", 0.5,
            {"topic": "minor musing"},
        )
        self.sos._dispatch_feed_signals()
        stats = self.sos.trinity.get_stats()
        analyses = stats.get("total_analyses", 0)
        self.assertEqual(analyses, 0, f"trinity stats: {stats}")

    def test_dispatch_is_idempotent(self):
        self.sos.feed_loop.emit(
            "anomaly_detector", "anomaly", 0.9,
            {"topic": "unique topic xyz"},
        )
        self.sos._dispatch_feed_signals()
        first = len(self.sos.attention.get_priorities(limit=50))
        dispatched_after_first = self.sos._stats["feed_signals_dispatched"]
        self.sos._dispatch_feed_signals()  # same signals still in the deque
        second = len(self.sos.attention.get_priorities(limit=50))
        self.assertEqual(first, second)
        self.assertEqual(self.sos._stats["feed_signals_dispatched"], dispatched_after_first)

    def test_stats_surfaced_in_get_status(self):
        status = self.sos.get_status()
        self.assertIn("feed_dispatch", status)
        fd = status["feed_dispatch"]
        self.assertIn("signals_dispatched", fd)
        self.assertIn("signals_dropped", fd)
        self.assertIn("cycles", fd)

    def test_dispatch_cycle_counted(self):
        before = self.sos._stats["feed_dispatch_cycles"]
        self.sos._dispatch_feed_signals()
        self.assertEqual(self.sos._stats["feed_dispatch_cycles"], before + 1)

    def test_dedup_set_bounded(self):
        # Simulate > maxlen signal ids; set must not grow unboundedly.
        sos = self.sos
        for i in range(5010):
            sos._feed_mark_processed(f"sig_{i}")
        self.assertLessEqual(len(sos._processed_signal_ids), 5000)
        self.assertLessEqual(len(sos._processed_signal_order), 5000)


if __name__ == "__main__":
    unittest.main()
