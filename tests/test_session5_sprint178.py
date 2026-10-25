"""Sprint 178 regression tests — goal_reasoning.py round 3 deep audit.

Fixes applied:
1. MEDIUM: get_stats() now reads _stats under _stats_lock (not _lock)
2. MEDIUM: suggest_replan() now returns copy.deepcopy(action) instead of direct reference
3. MEDIUM: Eviction log messages use %s format (not f-string injection)
4. MEDIUM: detect_obstacle() metadata values sanitized via _sanitize_content
"""

import threading
import unittest

from shadowd.cognition.goal_reasoning import (
    GoalReasoning, ObstacleSeverity, ConflictType, ReplanType, AssessmentOutcome
)


class TestSprint178StatsLock(unittest.TestCase):
    def setUp(self):
        self.gr = GoalReasoning()

    def test_get_stats_reads_under_stats_lock(self):
        for i in range(5):
            self.gr.assess_goal(f"goal_{i}", progress=0.5 * i)
        stats = self.gr.get_stats()
        self.assertEqual(stats["assessments_made"], 5)
        self.assertIn("total_assessments", stats)
        self.assertIn("goal_priorities_tracked", stats)

    def test_concurrent_stats_read_write(self):
        errors = []

        def writer():
            for i in range(50):
                self.gr.assess_goal(f"concurrent_{i}", progress=0.5)

        def reader():
            for _ in range(50):
                try:
                    stats = self.gr.get_stats()
                    self.assertIsInstance(stats["assessments_made"], int)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestSprint178DeepcopySuggestReplan(unittest.TestCase):
    def setUp(self):
        self.gr = GoalReasoning()

    def test_suggest_replan_returns_deepcopy(self):
        action = self.gr.suggest_replan("goal_1", "blocked by resource limits")
        action.reason = "MUTATED"
        action2 = self.gr.get_replan_action(action.action_id)
        self.assertNotEqual(action2["reason"], "MUTATED")


class TestSprint178LogFormat(unittest.TestCase):
    def test_eviction_log_no_fstring(self):
        import inspect
        from shadowd.cognition.goal_reasoning import GoalReasoning
        source = inspect.getsource(GoalReasoning._evict_assessments)
        self.assertNotIn('f"Evicted', source)
        source = inspect.getsource(GoalReasoning._evict_obstacles)
        self.assertNotIn('f"Evicted', source)


class TestSprint178MetadataSanitization(unittest.TestCase):
    def setUp(self):
        self.gr = GoalReasoning()

    def test_obstacle_metadata_strips_control_chars(self):
        obs = self.gr.detect_obstacle(
            goal_id="goal_1",
            description="test obstacle",
            severity=ObstacleSeverity.HIGH,
            mitigation="fix it",
            metadata={"key": "value\x00with\x01control", "html": "<script>alert(1)</script>"},
        )
        d = obs.to_dict()
        self.assertNotIn("\x00", d["metadata"]["key"])
        self.assertIn("<script>", d["metadata"]["html"])

    def test_obstacle_metadata_truncates_long_values(self):
        obs = self.gr.detect_obstacle(
            goal_id="goal_1",
            description="test",
            severity=ObstacleSeverity.MEDIUM,
            metadata={"long_val": "x" * 500},
        )
        self.assertEqual(len(obs.to_dict()["metadata"]["long_val"]), 200)


class TestSprint178Integration(unittest.TestCase):
    def setUp(self):
        self.gr = GoalReasoning()

    def test_assess_goal_and_get_stats(self):
        a = self.gr.assess_goal(
            "project_alpha", progress=0.3, deadline="2026-12-31T00:00:00+00:00",
            effort_spent=5.0, effort_total=10.0, blockers=["Budget pending"],
        )
        self.assertIsNotNone(a)
        self.assertEqual(a.outcome, AssessmentOutcome.BLOCKED)
        stats = self.gr.get_stats()
        self.assertEqual(stats["assessments_made"], 1)

    def test_detect_obstacle_and_resolve(self):
        obs = self.gr.detect_obstacle(
            "goal_1", "Budget not approved", ObstacleSeverity.HIGH, "Request budget approval"
        )
        self.assertFalse(obs.resolved)
        self.assertTrue(self.gr.resolve_obstacle(obs.obstacle_id))
        resolved = self.gr.get_obstacle(obs.obstacle_id)
        self.assertTrue(resolved["resolved"])

    def test_analyze_conflicts(self):
        goals = [
            {"goal_id": "g1", "priority": "critical", "progress": 0.5,
             "resource_needs": {"budget": 0.8}, "deadline": "2026-06-01T00:00:00+00:00"},
            {"goal_id": "g2", "priority": "critical", "progress": 0.6,
             "resource_needs": {"budget": 0.5}, "deadline": "2026-06-03T00:00:00+00:00"},
        ]
        conflicts = self.gr.analyze_conflicts(goals)
        self.assertIsInstance(conflicts, list)

    def test_suggest_and_apply_replan(self):
        action = self.gr.suggest_replan("goal_1", "need to defer this goal")
        self.assertEqual(action.replan_type, ReplanType.DEFER)
        self.assertFalse(action.applied)
        self.assertTrue(self.gr.apply_replan(action.action_id))
        applied = self.gr.get_replan_action(action.action_id)
        self.assertTrue(applied["applied"])

    def test_prioritize_goals(self):
        goals = [
            {"goal_id": "g1", "priority": "low", "progress": 0.8},
            {"goal_id": "g2", "priority": "high", "progress": 0.1},
        ]
        result = self.gr.prioritize_goals(goals)
        self.assertEqual(len(result), 2)
        self.assertIn("_priority_score", result[0])


if __name__ == "__main__":
    unittest.main()