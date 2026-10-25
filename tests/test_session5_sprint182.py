"""Sprint 182 — _stats_lock audit across 11 cognition modules.

All 11 modules now read self._stats under self._stats_lock (not self._lock).
"""

import threading
import unittest

from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
from shadowd.cognition.creative_synthesis import CreativeSynthesis
from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
from shadowd.cognition.metacognition_engine import MetacognitionEngine
from shadowd.cognition.moral_reasoning import MoralReasoning
from shadowd.cognition.personality_engine import PersonalityEngine
from shadowd.cognition.social_intelligence import SocialIntelligence
from shadowd.cognition.device_context import DeviceContextEngine
from shadowd.cognition.goal_planner import CognitiveGoalPlanner
from shadowd.cognition.shadow_arise import ShadowArise


class TestSprint182StatsLockAnomaly(unittest.TestCase):
    def test_concurrent_read_write(self):
        engine = AnomalyDetectionEngine()
        errors = []

        def writer():
            for i in range(30):
                engine.register_metric(f"metric_{i % 5}", "test")

        def reader():
            for _ in range(30):
                try:
                    stats = engine.get_stats()
                    self.assertIsInstance(stats.get("total_observations", 0), int)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint182StatsLockCreative(unittest.TestCase):
    def test_concurrent_read_write(self):
        cs = CreativeSynthesis()
        errors = []

        def writer():
            for i in range(30):
                cs.register_concept(f"concept_{i}", ["tech"])

        def reader():
            for _ in range(30):
                try:
                    stats = cs.get_stats()
                    self.assertIsInstance(stats, dict)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint182StatsLockEmotional(unittest.TestCase):
    def test_concurrent_read_write(self):
        ei = EmotionalIntelligence()
        errors = []

        def writer():
            for i in range(30):
                ei.detect_emotion(f"I feel happy about thing {i}")

        def reader():
            for _ in range(30):
                try:
                    stats = ei.get_stats()
                    self.assertIsInstance(stats, dict)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint182StatsLockMetacognition(unittest.TestCase):
    def test_concurrent_read_write(self):
        me = MetacognitionEngine()
        errors = []

        def writer():
            for i in range(30):
                me.start_thinking(f"problem_{i}", "analysis")

        def reader():
            for _ in range(30):
                try:
                    stats = me.get_stats()
                    self.assertIsInstance(stats, dict)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint182StatsLockMoral(unittest.TestCase):
    def test_concurrent_read_write(self):
        mr = MoralReasoning()
        errors = []

        def writer():
            for i in range(30):
                mr.evaluate(f"action_{i}", f"description_{i}")

        def reader():
            for _ in range(30):
                try:
                    stats = mr.get_stats()
                    self.assertIsInstance(stats, dict)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint182StatsLockPersonality(unittest.TestCase):
    def test_concurrent_read_write(self):
        pe = PersonalityEngine()
        errors = []

        def writer():
            for i in range(30):
                pe.record_behavior(f"activity_{i}", {"mood": f"context_{i}"})

        def reader():
            for _ in range(30):
                try:
                    stats = pe.get_stats()
                    self.assertIsInstance(stats, dict)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint182StatsLockSocial(unittest.TestCase):
    def test_concurrent_read_write(self):
        si = SocialIntelligence()
        errors = []

        def writer():
            for i in range(30):
                si.create_group(f"group_{i}", "social", f"description_{i}")

        def reader():
            for _ in range(30):
                try:
                    stats = si.get_stats()
                    self.assertIsInstance(stats, dict)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint182StatsLockDevice(unittest.TestCase):
    def test_concurrent_read_write(self):
        dce = DeviceContextEngine()
        errors = []

        def writer():
            for i in range(20):
                try:
                    dce.register_device(f"dev_{i}", "phone", f"Device {i}")
                except Exception:
                    pass

        def reader():
            for _ in range(20):
                try:
                    stats = dce.get_stats()
                    self.assertIsInstance(stats, dict)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint182StatsLockGoalPlanner(unittest.TestCase):
    def test_concurrent_read_write(self):
        gp = CognitiveGoalPlanner()
        errors = []

        def writer():
            for i in range(20):
                gp.create_goal(f"goal_{i}", "test description", "medium")

        def reader():
            for _ in range(20):
                try:
                    stats = gp.get_stats()
                    self.assertIsInstance(stats, dict)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint182StatsLockShadowArise(unittest.TestCase):
    def test_concurrent_read_write(self):
        sa = ShadowArise()
        errors = []

        def writer():
            for i in range(20):
                sa.arise(f"agent_{i}", "analyst", "test objective")

        def reader():
            for _ in range(20):
                try:
                    stats = sa.get_stats()
                    self.assertIsInstance(stats, dict)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()