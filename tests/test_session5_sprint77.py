#!/usr/bin/env python3
"""Sprint 77 — GoalReasoning integration tests + ShadowOS/API wiring audit."""

import copy
import threading
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from shadowd.cognition.goal_reasoning import (
    GoalReasoning, GoalAssessment, Obstacle, GoalConflict, ReplanAction,
    ConflictType, ObstacleSeverity, ReplanType, AssessmentOutcome,
)
import shadowd.cognition.goal_reasoning as gr_module
from shadowd.cognition.shadow_os import ShadowOS


class TestGoalReasoningCore(unittest.TestCase):
    def test_assess_goal_on_track(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.7, effort_spent=3.0, effort_total=5.0)
        self.assertEqual(a.outcome, AssessmentOutcome.ON_TRACK)
        self.assertGreater(a.confidence, 0.3)

    def test_assess_goal_blocked(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g2", progress=0.3, blockers=["dependency missing", "team unavailable"])
        self.assertEqual(a.outcome, AssessmentOutcome.BLOCKED)

    def test_assess_goal_completed(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g3", progress=1.0)
        self.assertEqual(a.outcome, AssessmentOutcome.COMPLETED)

    def test_assess_goal_at_risk(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g4", progress=0.2, effort_spent=8.0, effort_total=5.0)
        self.assertEqual(a.outcome, AssessmentOutcome.AT_RISK)

    def test_assess_goal_with_deadline(self):
        gr = GoalReasoning()
        future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        a = gr.assess_goal("g5", progress=0.3, deadline=future)
        self.assertIsInstance(a.urgency_score, float)

    def test_assess_goal_past_deadline(self):
        gr = GoalReasoning()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        a = gr.assess_goal("g6", progress=0.5, deadline=past)
        self.assertEqual(a.urgency_score, 1.0)

    def test_detect_obstacle(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "Server down", ObstacleSeverity.HIGH, "Restart server")
        self.assertEqual(o.goal_id, "g1")
        self.assertEqual(o.severity, ObstacleSeverity.HIGH)
        self.assertFalse(o.resolved)

    def test_resolve_obstacle(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "Blocked", ObstacleSeverity.MEDIUM)
        result = gr.resolve_obstacle(o.obstacle_id)
        self.assertTrue(result)
        result2 = gr.resolve_obstacle("nonexistent")
        self.assertFalse(result2)

    def test_resolve_obstacle_already_resolved(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "Blocked")
        gr.resolve_obstacle(o.obstacle_id)
        result = gr.resolve_obstacle(o.obstacle_id)
        self.assertTrue(result)

    def test_analyze_conflicts_resource(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "a", "priority": "high", "resource_needs": {"gpu": 0.8}},
            {"goal_id": "b", "priority": "high", "resource_needs": {"gpu": 0.6}},
        ]
        conflicts = gr.analyze_conflicts(goals)
        self.assertGreater(len(conflicts), 0)

    def test_analyze_no_conflicts(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "a", "priority": "low", "resource_needs": {"cpu": 0.3}},
            {"goal_id": "b", "priority": "low", "resource_needs": {"ram": 0.2}},
        ]
        conflicts = gr.analyze_conflicts(goals)
        self.assertEqual(len(conflicts), 0)

    def test_analyze_conflicts_priority(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "a", "priority": "critical", "progress": 0.5, "resource_needs": {}},
            {"goal_id": "b", "priority": "critical", "progress": 0.6, "resource_needs": {}},
        ]
        conflicts = gr.analyze_conflicts(goals)
        conflict_types = [c.conflict_type for c in conflicts]
        self.assertIn(ConflictType.PRIORITY, conflict_types)

    def test_suggest_replan(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "blocked by infrastructure failure")
        self.assertEqual(action.replan_type, ReplanType.ESCALATE)
        self.assertFalse(action.applied)

    def test_suggest_replan_defer(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "delay the project until next quarter")
        self.assertEqual(action.replan_type, ReplanType.DEFER)

    def test_apply_replan(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "need to change approach")
        result = gr.apply_replan(action.action_id)
        self.assertTrue(result)

    def test_apply_replan_already_applied(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "rethink strategy")
        gr.apply_replan(action.action_id)
        result = gr.apply_replan(action.action_id)
        self.assertFalse(result)

    def test_apply_replan_nonexistent(self):
        gr = GoalReasoning()
        result = gr.apply_replan("nonexistent_id")
        self.assertFalse(result)

    def test_prioritize_goals(self):
        gr = GoalReasoning()
        future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
        goals = [
            {"goal_id": "low", "priority": "low", "progress": 0.8, "deadline": future},
            {"goal_id": "critical", "priority": "critical", "progress": 0.1, "deadline": future},
            {"goal_id": "medium", "priority": "medium", "progress": 0.5},
        ]
        result = gr.prioritize_goals(goals)
        self.assertEqual(len(result), 3)
        self.assertIn("_priority_score", result[0])
        self.assertGreater(result[0]["_priority_score"], result[-1]["_priority_score"])

    def test_prioritize_goals_deepcopy(self):
        gr = GoalReasoning()
        goals = [{"goal_id": "g1", "priority": "high"}]
        result = gr.prioritize_goals(goals)
        result[0]["_priority_score"] = 999
        self.assertNotEqual(goals[0].get("_priority_score"), 999)

    def test_get_set_methods(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.5)
        retrieved = gr.get_assessment(a.assessment_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["assessment_id"], a.assessment_id)

    def test_get_assessment_nonexistent(self):
        gr = GoalReasoning()
        self.assertIsNone(gr.get_assessment("nonexistent"))

    def test_list_assessments_by_goal(self):
        gr = GoalReasoning()
        gr.assess_goal("g1", progress=0.5)
        gr.assess_goal("g1", progress=0.6)
        gr.assess_goal("g2", progress=0.3)
        result = gr.list_assessments(goal_id="g1")
        self.assertEqual(len(result), 2)

    def test_list_obstacles_filter_resolved(self):
        gr = GoalReasoning()
        o1 = gr.detect_obstacle("g1", "Blocker 1")
        gr.resolve_obstacle(o1.obstacle_id)
        gr.detect_obstacle("g1", "Blocker 2")
        unresolved = gr.list_obstacles(resolved=False)
        resolved = gr.list_obstacles(resolved=True)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(len(unresolved), 1)

    def test_get_obstacle(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "Test obstacle")
        result = gr.get_obstacle(o.obstacle_id)
        self.assertIsNotNone(result)
        self.assertEqual(result["obstacle_id"], o.obstacle_id)

    def test_get_conflict(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "a", "priority": "critical", "progress": 0.5, "resource_needs": {"cpu": 0.8}},
            {"goal_id": "b", "priority": "critical", "progress": 0.4, "resource_needs": {"cpu": 0.6}},
        ]
        conflicts = gr.analyze_conflicts(goals)
        if conflicts:
            result = gr.get_conflict(conflicts[0].conflict_id)
            self.assertIsNotNone(result)

    def test_list_conflicts_by_goal(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "a", "priority": "critical", "progress": 0.5, "resource_needs": {"cpu": 0.8}},
            {"goal_id": "b", "priority": "critical", "progress": 0.4, "resource_needs": {"cpu": 0.6}},
        ]
        conflicts = gr.analyze_conflicts(goals)
        if conflicts:
            result = gr.list_conflicts(goal_id="a")
            self.assertGreater(len(result), 0)

    def test_list_replan_actions(self):
        gr = GoalReasoning()
        gr.suggest_replan("g1", "blocked")
        gr.suggest_replan("g1", "delay")
        actions = gr.list_replan_actions(goal_id="g1")
        self.assertEqual(len(actions), 2)

    def test_list_replan_actions_filter_applied(self):
        gr = GoalReasoning()
        a = gr.suggest_replan("g1", "blocked")
        gr.apply_replan(a.action_id)
        gr.suggest_replan("g1", "delay")
        applied = gr.list_replan_actions(applied=True)
        unapplied = gr.list_replan_actions(applied=False)
        self.assertEqual(len(applied), 1)
        self.assertEqual(len(unapplied), 1)

    def test_stats(self):
        gr = GoalReasoning()
        gr.assess_goal("g1", progress=0.5)
        gr.detect_obstacle("g1", "Test")
        stats = gr.get_stats()
        self.assertEqual(stats["assessments_made"], 1)
        self.assertEqual(stats["obstacles_detected"], 1)
        self.assertIn("total_assessments", stats)
        self.assertIn("total_obstacles", stats)

    def test_assess_goal_sanitization(self):
        gr = GoalReasoning()
        a = gr.assess_goal("goal<script>", progress=0.5, blockers=["block\x00er"])
        self.assertNotIn("<script>", a.goal_id)
        self.assertNotIn("\x00", a.risk_factors[0] if a.risk_factors else "")

    def test_assess_goal_metadata_cap(self):
        gr = GoalReasoning()
        big_context = {f"key_{i}": f"val_{i}" for i in range(50)}
        a = gr.assess_goal("g1", progress=0.5, context=big_context)
        self.assertLessEqual(len(a.metadata), 20)

    def test_list_limit_clamping(self):
        gr = GoalReasoning()
        result = gr.list_assessments(limit=500)
        self.assertIsInstance(result, list)

    def test_goal_conflict_to_dict(self):
        c = GoalConflict(
            conflict_id="conf_abc",
            goal_id_a="g1",
            goal_id_b="g2",
            conflict_type=ConflictType.RESOURCE,
            description="Resource conflict",
            resolution_hint="Stagger execution",
            detected_at="2024-01-01T00:00:00Z",
        )
        d = c.to_dict()
        self.assertEqual(d["conflict_id"], "conf_abc")
        self.assertEqual(d["conflict_type"], "resource")


class TestGoalReasoningDeepCopy(unittest.TestCase):
    def test_get_assessment_returns_deepcopy(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.5)
        result = gr.get_assessment(a.assessment_id)
        result["outcome"] = "MODIFIED"
        result2 = gr.get_assessment(a.assessment_id)
        self.assertNotEqual(result2["outcome"], "MODIFIED")

    def test_list_assessments_returns_deepcopy(self):
        gr = GoalReasoning()
        gr.assess_goal("g1", progress=0.5)
        results = gr.list_assessments()
        results[0]["goal_id"] = "MODIFIED"
        results2 = gr.list_assessments()
        self.assertNotEqual(results2[0]["goal_id"], "MODIFIED")

    def test_get_obstacle_returns_deepcopy(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "Test")
        result = gr.get_obstacle(o.obstacle_id)
        result["description"] = "MODIFIED"
        result2 = gr.get_obstacle(o.obstacle_id)
        self.assertNotEqual(result2["description"], "MODIFIED")

    def test_list_obstacles_returns_deepcopy(self):
        gr = GoalReasoning()
        gr.detect_obstacle("g1", "Test")
        results = gr.list_obstacles()
        results[0]["description"] = "MODIFIED"
        results2 = gr.list_obstacles()
        self.assertNotEqual(results2[0]["description"], "MODIFIED")

    def test_list_conflicts_returns_deepcopy(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "a", "priority": "critical", "progress": 0.5, "resource_needs": {"cpu": 0.8}},
            {"goal_id": "b", "priority": "critical", "progress": 0.4, "resource_needs": {"cpu": 0.6}},
        ]
        conflicts = gr.analyze_conflicts(goals)
        if conflicts:
            clist = gr.list_conflicts()
            clist[0]["description"] = "MODIFIED"
            clist2 = gr.list_conflicts()
            self.assertNotEqual(clist2[0]["description"], "MODIFIED")


class TestGoalReasoningEviction(unittest.TestCase):
    def test_assessment_eviction(self):
        gr = GoalReasoning()
        max_assessments = gr._MAX_ASSESSMENTS
        for i in range(max_assessments + 10):
            gr.assess_goal(f"g_{i}", progress=0.5)
        with gr._lock:
            self.assertLessEqual(len(gr._assessments), max_assessments)

    def test_obstacle_eviction(self):
        gr = GoalReasoning()
        max_obstacles = gr._MAX_OBSTACLES
        for i in range(max_obstacles + 10):
            gr.detect_obstacle("g1", f"Obstacle {i}")
        with gr._lock:
            self.assertLessEqual(len(gr._obstacles), max_obstacles)

    def test_conflict_eviction(self):
        gr = GoalReasoning()
        max_conflicts = gr._MAX_CONFLICTS
        goals = [
            {"goal_id": "a", "priority": "critical", "progress": 0.5, "resource_needs": {"cpu": 0.8}},
            {"goal_id": "b", "priority": "critical", "progress": 0.4, "resource_needs": {"cpu": 0.6}},
        ]
        for _ in range(max_conflicts + 10):
            gr.analyze_conflicts(goals)
        with gr._lock:
            self.assertLessEqual(len(gr._conflicts), max_conflicts)

    def test_replan_eviction(self):
        gr = GoalReasoning()
        max_replans = gr._MAX_REPLAN_ACTIONS
        for i in range(max_replans + 10):
            gr.suggest_replan("g1", f"reason_{i}")
        with gr._lock:
            self.assertLessEqual(len(gr._replan_actions), max_replans)


class TestGoalReasoningThreadSafety(unittest.TestCase):
    def test_concurrent_assess(self):
        gr = GoalReasoning()
        errors = []

        def worker():
            try:
                for i in range(100):
                    gr.assess_goal(f"g_{i}", progress=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        stats = gr.get_stats()
        self.assertEqual(stats["assessments_made"], 500)

    def test_concurrent_reads_and_writes(self):
        gr = GoalReasoning()
        gr.assess_goal("g1", progress=0.5)
        errors = []

        def writer():
            try:
                for i in range(50):
                    gr.assess_goal(f"g_{i}", progress=0.5)
                    gr.detect_obstacle(f"g_{i}", f"Blocker {i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    gr.get_stats()
                    gr.list_assessments()
                    gr.list_obstacles()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


class TestShadowOSGoalReasoningWiring(unittest.TestCase):
    def test_shadowos_has_goal_reasoning(self):
        os = ShadowOS()
        self.assertTrue(hasattr(os, 'goal_reasoning'))
        self.assertIsInstance(os.goal_reasoning, GoalReasoning)

    def test_shadowos_health_includes_goal_reasoning(self):
        os = ShadowOS()
        os._running = True
        health = os.get_health()
        self.assertIn("modules", health)
        self.assertIn("goal_reasoning", health["modules"])

    def test_shadowos_status_includes_goal_reasoning(self):
        os = ShadowOS()
        os._running = True
        status = os.get_status()
        self.assertIn("goal_reasoning_stats", status)

    def test_goal_reasoning_module_count(self):
        os = ShadowOS()
        os._running = True
        health = os.get_health()
        module_names = list(health["modules"].keys())
        expected_modules = [
            "shadow_graph", "dark_memory", "shadow_crew", "neural_mesh",
            "stealth_layer", "personality_engine", "thought_sense",
            "recall_synthesis", "temporal_reasoning", "empathy_mapping",
            "attention_engine", "emotional_intelligence", "creative_synthesis",
            "metacognition", "moral_reasoning", "cognitive_reflection",
            "goal_reasoning",
        ]
        for m in expected_modules:
            self.assertIn(m, module_names, f"Missing module: {m}")
        self.assertGreaterEqual(len(module_names), 30)


class TestGoalReasoningEdgeCases(unittest.TestCase):
    def test_assess_goal_with_no_blockers_no_deadline(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.6)
        self.assertEqual(a.outcome, AssessmentOutcome.ON_TRACK)
        self.assertAlmostEqual(a.urgency_score, 0.5, places=1)

    def test_assess_goal_with_zero_effort(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.0, effort_spent=0.0, effort_total=1.0)
        self.assertIsNotNone(a.confidence)

    def test_invalid_deadline_falls_back(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.5, deadline="not-a-date")
        self.assertEqual(a.urgency_score, 0.5)

    def test_prioritize_empty_goals(self):
        gr = GoalReasoning()
        result = gr.prioritize_goals([])
        self.assertEqual(result, [])

    def test_analyze_conflicts_empty(self):
        gr = GoalReasoning()
        result = gr.analyze_conflicts([])
        self.assertEqual(result, [])

    def test_get_conflict_nonexistent(self):
        gr = GoalReasoning()
        self.assertIsNone(gr.get_conflict("nonexistent"))

    def test_get_replan_action_nonexistent(self):
        gr = GoalReasoning()
        self.assertIsNone(gr.get_replan_action("nonexistent"))

    def test_obstacle_metadata_sanitization(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "desc\x00ription<script>", ObstacleSeverity.HIGH, "mitigation")
        self.assertNotIn("\x00", o.description)
        # Content-safe sanitize preserves <> for descriptions
        self.assertIn("description", o.description)
        self.assertIn("<script>", o.description)

    def test_assessment_id_is_12_chars(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.5)
        self.assertEqual(len(a.assessment_id), 12)
        self.assertTrue(a.assessment_id.isalnum())

    def test_obstacle_id_is_12_chars(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "Test")
        self.assertEqual(len(o.obstacle_id), 12)

    def test_conflict_id_is_12_chars(self):
        goals = [
            {"goal_id": "a", "priority": "critical", "progress": 0.5, "resource_needs": {"cpu": 0.8}},
            {"goal_id": "b", "priority": "critical", "progress": 0.4, "resource_needs": {"cpu": 0.6}},
        ]
        gr = GoalReasoning()
        conflicts = gr.analyze_conflicts(goals)
        if conflicts:
            self.assertEqual(len(conflicts[0].conflict_id), 12)

    def test_replan_action_id_is_12_chars(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "blocked")
        self.assertEqual(len(action.action_id), 12)

    def test_time_conflict_detection(self):
        gr = GoalReasoning()
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        goals = [
            {"goal_id": "a", "priority": "high", "deadline": future, "resource_needs": {}},
            {"goal_id": "b", "priority": "high", "deadline": future, "resource_needs": {}},
        ]
        conflicts = gr.analyze_conflicts(goals)
        conflict_types = [c.conflict_type for c in conflicts]
        self.assertIn(ConflictType.TIME, conflict_types)

    def test_replan_classify_escalate(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "we are stuck and can't proceed at all")
        self.assertEqual(action.replan_type, ReplanType.ESCALATE)

    def test_replan_classify_urgent(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "this is urgent and needs immediate action")
        self.assertEqual(action.replan_type, ReplanType.RESEQUENCE)

    def test_replan_classify_change(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "we need to reconsider our approach")
        self.assertEqual(action.replan_type, ReplanType.DECOMPOSE_FURTHER)

    def test_replan_default_classify(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "normal recheck")
        self.assertEqual(action.replan_type, ReplanType.RESEQUENCE)

    def test_get_replan_action(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "blocked")
        result = gr.get_replan_action(action.action_id)
        self.assertIsNotNone(result)
        self.assertEqual(result["action_id"], action.action_id)

    def test_prioritize_goals_with_known_priorities(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "low", "priority": "low", "progress": 0.9},
            {"goal_id": "critical", "priority": "critical", "progress": 0.1},
            {"goal_id": "medium", "priority": "medium", "progress": 0.5},
        ]
        result = gr.prioritize_goals(goals)
        self.assertEqual(result[0]["goal_id"], "critical")

    def test_sanitization_func(self):
        self.assertEqual(gr_module._sanitize("hello<script>world\x00test", 50), "helloscriptworldtest")
        self.assertEqual(gr_module._sanitize("a" * 100, 10), "a" * 10)

    def test_reassess_patterns(self):
        self.assertTrue(any(p.search("delay the project") for p in gr_module._REASSESS_PATTERNS))
        self.assertTrue(any(p.search("blocked and stuck") for p in gr_module._REASSESS_PATTERNS))
        self.assertTrue(any(p.search("reconsider strategy") for p in gr_module._REASSESS_PATTERNS))
        self.assertTrue(any(p.search("urgent ASAP") for p in gr_module._REASSESS_PATTERNS))
        self.assertTrue(any(p.search("complete the task") for p in gr_module._REASSESS_PATTERNS))


if __name__ == "__main__":
    unittest.main()