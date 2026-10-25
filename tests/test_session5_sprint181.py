"""Sprint 181 regression tests — recall_synthesis + thought_sense round 3 audit.

Fixes applied:
1. recall_synthesis.py: synthesize() returns list() copies of obs.keywords (was mutable references)
2. Verified: both files' get_stats() use correct locks
3. Verified: all to_dict() float fields have round()
4. Verified: no bare except Exception, no time.time(), correct sanitize patterns
"""

import unittest

from shadowd.cognition.recall_synthesis import RecallSynthesis


class TestSprint181RecallSynthesisKeywords(unittest.TestCase):
    def setUp(self):
        self.rs = RecallSynthesis()

    def test_synthesize_keywords_are_copies(self):
        self.rs.add_observation("Python is great for data science", domain="technology")
        self.rs.add_observation("Machine learning uses Python for models", domain="science")
        result = self.rs.synthesize("Python")
        all_entries = result.direct_observations + result.cross_domain_insights
        for entry in all_entries:
            self.assertIsInstance(entry.get("keywords", []), list)
            entry["keywords"].append("MUTATED")
        result2 = self.rs.synthesize("Python")
        all_entries2 = result2.direct_observations + result2.cross_domain_insights
        for entry in all_entries2:
            self.assertNotIn("MUTATED", entry.get("keywords", []))

    def test_synthesize_to_keywords_are_copies(self):
        self.rs.add_observation("Python data analysis", domain="technology")
        result = self.rs.synthesize("Python")
        if result.connections:
            for conn in result.connections:
                self.assertIsInstance(conn.get("to_keywords", []), list)
                conn["to_keywords"].append("MUTATED")
            result2 = self.rs.synthesize("Python")
            for conn in result2.connections:
                self.assertNotIn("MUTATED", conn.get("to_keywords", []))


class TestSprint181ThoughtSenseStats(unittest.TestCase):
    def test_thought_sense_stats_accessible(self):
        from shadowd.cognition.thought_sense import ThoughtSense
        ts = ThoughtSense()
        ts.analyze("I need to finish this project quickly")
        stats = ts.get_stats()
        self.assertIn("total_analyses", stats)
        self.assertEqual(stats["total_analyses"], 1)


class TestSprint181RecallSynthesisStats(unittest.TestCase):
    def test_recall_synthesis_stats(self):
        rs = RecallSynthesis()
        rs.add_observation("test observation", domain="general")
        stats = rs.get_stats()
        self.assertIn("observation_count", stats)
        self.assertEqual(stats["observation_count"], 1)


if __name__ == "__main__":
    unittest.main()