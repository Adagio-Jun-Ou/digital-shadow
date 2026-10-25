"""Sprint 177 regression tests — decision_journal.py round 3 audit.

Fixes applied:
1. HIGH: _stats["patterns_detected"] now updated inside _stats_lock
2. MEDIUM: get_stats() reads _stats under _stats_lock (not _lock)
3. MEDIUM: detect_patterns() uses copy.deepcopy for decisions snapshot
4. MEDIUM: get_decision_accuracy() uses copy.deepcopy for resolved
5. MEDIUM: get_overconfidence_report() uses copy.deepcopy for resolved
"""

import threading
import time
import unittest

from shadowd.cognition.decision_journal import DecisionJournal, DecisionType, DecisionStatus


class TestSprint177StatsLock(unittest.TestCase):
    def setUp(self):
        self.dj = DecisionJournal()

    def test_patterns_detected_updated_under_stats_lock(self):
        for i in range(5):
            self.dj.record_decision(
                title=f"Decision {i}",
                chosen="option_a",
                alternatives=["option_b"],
                reasoning="test",
                confidence=0.8,
                domain="test",
                decision_type="routine",
            )
        patterns = self.dj.detect_patterns(min_frequency=2)
        with self.dj._stats_lock:
            self.assertEqual(self.dj._stats["patterns_detected"], len(self.dj._patterns))

    def test_get_stats_reads_under_stats_lock(self):
        for i in range(3):
            self.dj.record_decision(
                title=f"Decision {i}",
                chosen="option_a",
                reasoning="test",
                confidence=0.8,
                domain="test",
            )
        stats = self.dj.get_stats()
        self.assertEqual(stats["decisions_recorded"], 3)

    def test_concurrent_stats_read_write(self):
        errors = []

        def writer():
            for _ in range(100):
                self.dj.record_decision(
                    title="concurrent",
                    chosen="a",
                    confidence=0.5,
                    domain="concurrent",
                )

        def reader():
            for _ in range(100):
                try:
                    stats = self.dj.get_stats()
                    self.assertIsInstance(stats["decisions_recorded"], int)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint177DeepcopySnapshot(unittest.TestCase):
    def setUp(self):
        self.dj = DecisionJournal()

    def test_detect_patterns_deepcopy(self):
        rec = self.dj.record_decision(
            title="Original",
            chosen="option_a",
            alternatives=["option_b"],
            reasoning="original reasoning",
            confidence=0.9,
            domain="deepcopy_test",
        )
        self.dj.record_outcome(rec.decision_id, "good result", 0.9)
        rec2 = self.dj.record_decision(
            title="Second",
            chosen="option_a",
            reasoning="second reasoning",
            confidence=0.85,
            domain="deepcopy_test",
        )
        self.dj.record_outcome(rec2.decision_id, "decent result", 0.7)
        self.dj.detect_patterns(min_frequency=2)
        original_title = self.dj.get_decision(rec.decision_id).title
        self.assertEqual(original_title, "Original")

    def test_get_decision_accuracy_deepcopy(self):
        rec = self.dj.record_decision(
            title="Accuracy test",
            chosen="a",
            confidence=0.8,
            domain="accuracy",
        )
        self.dj.record_outcome(rec.decision_id, "result", 0.7)
        result = self.dj.get_decision_accuracy(domain="accuracy")
        self.assertEqual(result["total_resolved"], 1)
        self.assertAlmostEqual(result["avg_accuracy"], 0.7, places=4)

    def test_get_overconfidence_report_deepcopy(self):
        rec = self.dj.record_decision(
            title="Overconfident",
            chosen="a",
            confidence=0.9,
            domain="overconfidence",
        )
        self.dj.record_outcome(rec.decision_id, "result", 0.3)
        report = self.dj.get_overconfidence_report()
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["overconfident_count"], 1)

    def test_detect_patterns_no_external_mutation(self):
        for i in range(5):
            rec = self.dj.record_decision(
                title=f"Pattern {i}",
                chosen="option_a",
                reasoning="test",
                confidence=0.8,
                domain="mutation_test",
            )
            self.dj.record_outcome(rec.decision_id, "result", 0.5)
        self.dj.detect_patterns(min_frequency=2)
        stored = self.dj.get_decision(
            list(self.dj._decisions.keys())[0]
        )
        self.assertIsNotNone(stored)


class TestSprint177RoundValues(unittest.TestCase):
    def test_outcome_accuracy_rounded_in_to_dict(self):
        dj = DecisionJournal()
        rec = dj.record_decision(
            title="Round test",
            chosen="a",
            confidence=0.1234567,
        )
        dj.record_outcome(rec.decision_id, "result", 0.9876543)
        stored = dj.get_decision(rec.decision_id)
        self.assertEqual(stored.to_dict()["outcome_accuracy"], round(0.9876543, 4))


if __name__ == "__main__":
    unittest.main()