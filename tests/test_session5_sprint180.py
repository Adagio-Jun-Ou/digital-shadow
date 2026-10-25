"""Sprint 180 regression tests — cognitive_reflection + attention_engine round 3 audit.

Fixes applied:
1. cognitive_reflection.py: get_stats() reads _stats under _stats_lock (separate from _lock)
2. attention_engine.py: get_stats() reads _stats under _stats_lock (not _lock)
3. attention_engine.py: Error messages no longer leak user input
"""

import threading
import unittest

from shadowd.cognition.cognitive_reflection import CognitiveReflection, ReflectionType
from shadowd.cognition.attention_engine import (
    AttentionEngine, AttentionLevel, DistractionType, FocusState
)


class TestSprint180CognitiveReflectionStatsLock(unittest.TestCase):
    def setUp(self):
        self.cr = CognitiveReflection()

    def test_get_stats_uses_stats_lock(self):
        for i in range(3):
            self.cr.add_reflection(f"topic_{i}", description="test reflection")
        stats = self.cr.get_stats()
        self.assertEqual(stats["reflections_added"], 3)
        self.assertIn("current_reflections", stats)
        self.assertIn("current_insights", stats)

    def test_concurrent_stats_read_write(self):
        errors = []

        def writer():
            for i in range(50):
                self.cr.add_reflection(f"topic_{i}", description="concurrent test")

        def reader():
            for _ in range(50):
                try:
                    stats = self.cr.get_stats()
                    self.assertIsInstance(stats["reflections_added"], int)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint180AttentionEngineStatsLock(unittest.TestCase):
    def setUp(self):
        self.ae = AttentionEngine()

    def test_get_stats_uses_stats_lock(self):
        self.ae.add_attention_item("test topic", level="high")
        self.ae.start_focus("focus topic")
        self.ae.end_focus()
        stats = self.ae.get_stats()
        self.assertEqual(stats["focus_sessions"], 1)
        self.assertEqual(stats["items_tracked"], 1)

    def test_concurrent_stats_read_write(self):
        errors = []

        def writer():
            for i in range(30):
                self.ae.add_attention_item(f"topic_{i}", level="medium")

        def reader():
            for _ in range(30):
                try:
                    stats = self.ae.get_stats()
                    self.assertIsInstance(stats["items_tracked"], int)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint180AttentionEngineErrorMessages(unittest.TestCase):
    def test_invalid_attention_level_generic(self):
        ae = AttentionEngine()
        with self.assertRaises(ValueError) as ctx:
            ae.add_attention_item("test", level="INVALID_LEVEL")
        self.assertNotIn("INVALID_LEVEL", str(ctx.exception))

    def test_invalid_distraction_type_generic(self):
        ae = AttentionEngine()
        with self.assertRaises(ValueError) as ctx:
            ae.record_distraction("INVALID_TYPE")
        self.assertNotIn("INVALID_TYPE", str(ctx.exception))


class TestSprint180Integration(unittest.TestCase):
    def test_reflection_extract_insights(self):
        cr = CognitiveReflection()
        entry = cr.add_reflection(
            "project review",
            description="We should have planned better",
            what_went_well=["team collaboration"],
            what_could_improve=["planning"],
            lessons_learned=["need better estimates"],
        )
        insights = cr.extract_insights(entry.entry_id)
        self.assertGreater(len(insights), 0)

    def test_attention_focus_session(self):
        ae = AttentionEngine()
        ae.add_attention_item("urgent task", level="critical", urgency=0.9)
        session_id = ae.start_focus("urgent task")
        self.assertIsNotNone(session_id)
        result = ae.end_focus(session_id)
        self.assertIsNotNone(result)
        profile = ae.get_attention_profile()
        self.assertIsInstance(profile.current_state, FocusState)

    def test_distraction_tracking(self):
        ae = AttentionEngine()
        ae.record_distraction("notification", "phone buzz", recovery_time_seconds=120)
        ae.record_distraction("interruption", "colleague question", recovery_time_seconds=300)
        stats = ae.get_stats()
        self.assertEqual(stats["distractions_detected"], 2)


if __name__ == "__main__":
    unittest.main()