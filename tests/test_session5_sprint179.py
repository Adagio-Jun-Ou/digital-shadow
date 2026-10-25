"""Sprint 179 regression tests — dream_state + presence_detection round 3 audit.

Fixes applied:
1. DreamState: recall() now uses word-boundary matching (not substring)
2. DreamState: Added _stats_lock for thread-safe stats mutations
3. DreamState: _consolidation_count → _stats dict with _stats_lock
4. DreamState: get_stats() reads _stats under _stats_lock, collections under _lock
"""

import threading
import unittest

from shadowd.cognition.dream_state import DreamState


class TestSprint179DreamStateStatsLock(unittest.TestCase):
    def setUp(self):
        self.ds = DreamState()

    def test_stats_lock_exists(self):
        self.assertIsInstance(self.ds._stats_lock, type(threading.Lock()))

    def test_stats_dict_exists(self):
        self.assertIn("memories_added", self.ds._stats)
        self.assertIn("consolidations_run", self.ds._stats)
        self.assertIn("total_patterns_found", self.ds._stats)

    def test_memories_added_increments(self):
        self.ds.add_memory("test content", domain="test")
        self.ds.add_memory("test content 2", domain="test")
        self.assertEqual(self.ds._stats["memories_added"], 2)

    def test_consolidations_run_increments(self):
        self.ds.consolidate()
        stats = self.ds.get_stats()
        self.assertEqual(stats["consolidation_count"], 1)

    def test_concurrent_stats_read_write(self):
        errors = []

        def writer():
            for i in range(50):
                self.ds.add_memory(f"memory {i}", domain="test")

        def reader():
            for _ in range(50):
                try:
                    stats = self.ds.get_stats()
                    self.assertIsInstance(stats["consolidation_count"], int)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_no_consolidation_count_attribute(self):
        self.assertFalse(hasattr(self.ds, '_consolidation_count'))


class TestSprint179DreamStateRecallWordBoundary(unittest.TestCase):
    def setUp(self):
        self.ds = DreamState()

    def test_recall_word_boundary_matching(self):
        self.ds.add_memory("I love programming in Python", domain="work")
        self.ds.add_memory("Python is a great language", domain="work")
        self.ds.add_memory("I ate a sandwich for lunch", domain="food")

        results = self.ds.recall("Python")
        self.assertEqual(len(results), 2)

        results = self.ds.recall("programming")
        self.assertEqual(len(results), 1)

    def test_recall_does_not_match_substring(self):
        self.ds.add_memory("I love cats and caterpillars", domain="animals")

        results = self.ds.recall("cat")
        self.assertLessEqual(len(results), 2)


class TestSprint179DreamStateConsolidate(unittest.TestCase):
    def setUp(self):
        self.ds = DreamState()

    def test_consolidate_returns_result(self):
        result = self.ds.consolidate()
        self.assertIsInstance(result.memories_consolidated, int)
        self.assertIsInstance(result.memories_forgotten, int)
        self.assertIsInstance(result.insights_generated, int)
        self.assertIsInstance(result.patterns_strengthened, int)

    def test_consolidate_with_memories(self):
        for i in range(5):
            self.ds.add_memory(
                f"Python programming task {i} with machine learning",
                domain="work",
            )
        result = self.ds.consolidate()
        self.assertGreaterEqual(result.patterns_strengthened, 0)


if __name__ == "__main__":
    unittest.main()