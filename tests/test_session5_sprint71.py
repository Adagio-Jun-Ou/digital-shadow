#!/usr/bin/env python3
"""Sprint 71 tests: GoalReasoning module — goal assessment, obstacle detection, conflict analysis, replanning, prioritization"""

import pytest
import time
import threading
from datetime import datetime, timezone, timedelta

from shadowd.cognition.goal_reasoning import (
    GoalReasoning, ConflictType, ObstacleSeverity,
    AssessmentOutcome, ReplanType,
    GoalAssessment, Obstacle, GoalConflict, ReplanAction,
)


class TestGoalAssessment:
    def test_assess_goal_on_track(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.6, effort_spent=5.0, effort_total=10.0)
        assert a.outcome == AssessmentOutcome.ON_TRACK
        assert a.goal_id == "g1"
        assert 0.0 <= a.confidence <= 1.0
        assert a.assessment_id

    def test_assess_goal_completed(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g2", progress=1.0)
        assert a.outcome == AssessmentOutcome.COMPLETED

    def test_assess_goal_blocked(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g3", progress=0.3, blockers=["resource unavailable"])
        assert a.outcome == AssessmentOutcome.BLOCKED
        assert "blocker" in a.risk_factors[0].lower() or "Resolve blockers" in a.recommendations[0]

    def test_assess_goal_at_risk(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g4", progress=0.2, effort_spent=100.0, effort_total=10.0, blockers=[])
        assert a.outcome == AssessmentOutcome.AT_RISK
        assert a.confidence < 0.5

    def test_assess_goal_with_deadline(self):
        gr = GoalReasoning()
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        a = gr.assess_goal("g5", progress=0.1, deadline=future)
        assert a.urgency_score > 0.0

    def test_assess_goal_past_deadline(self):
        gr = GoalReasoning()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        a = gr.assess_goal("g6", progress=0.3, deadline=past)
        assert a.urgency_score == 1.0

    def test_assess_goal_metadata_capped(self):
        gr = GoalReasoning()
        big_ctx = {f"k{i}": f"v{i}" for i in range(50)}
        a = gr.assess_goal("g7", progress=0.5, context=big_ctx)
        assert len(a.metadata) <= 20

    def test_assess_goal_multiple_blockers(self):
        gr = GoalReasoning()
        blockers = [f"blocker {i}" for i in range(5)]
        a = gr.assess_goal("g8", progress=0.1, blockers=blockers)
        assert a.outcome == AssessmentOutcome.BLOCKED
        assert len(a.risk_factors) > 0

    def test_assess_goal_confidence_clamped(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g9", progress=0.5, effort_spent=1.0, effort_total=1.0)
        assert 0.0 <= a.confidence <= 1.0

    def test_assess_goal_progress_clamped(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g10", progress=5.0)
        assert a.outcome == AssessmentOutcome.COMPLETED

    def test_list_assessments(self):
        gr = GoalReasoning()
        for i in range(5):
            gr.assess_goal(f"goal_{i}", progress=0.5)
        result = gr.list_assessments()
        assert len(result) == 5

    def test_list_assessments_by_goal(self):
        gr = GoalReasoning()
        gr.assess_goal("target", progress=0.5)
        gr.assess_goal("other", progress=0.3)
        result = gr.list_assessments(goal_id="target")
        assert all(a["goal_id"] == "target" for a in result)
        assert len(result) == 1

    def test_get_assessment(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.5)
        retrieved = gr.get_assessment(a.assessment_id)
        assert retrieved is not None
        assert retrieved["goal_id"] == "g1"

    def test_get_assessment_not_found(self):
        gr = GoalReasoning()
        assert gr.get_assessment("nonexistent") is None


class TestObstacleDetection:
    def test_detect_obstacle(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "Server down", ObstacleSeverity.HIGH, "Restart server")
        assert o.goal_id == "g1"
        assert o.description == "Server down"
        assert o.severity == ObstacleSeverity.HIGH
        assert not o.resolved

    def test_detect_obstacle_default_severity(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g2", "Minor delay")
        assert o.severity == ObstacleSeverity.MEDIUM

    def test_detect_obstacle_sanitization(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g3", "x" * 1000, mitigation="y" * 1000)
        assert len(o.description) <= 500
        assert len(o.mitigation) <= 500

    def test_resolve_obstacle(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g4", "Blocker")
        assert gr.resolve_obstacle(o.obstacle_id) is True
        retrieved = gr.get_obstacle(o.obstacle_id)
        assert retrieved["resolved"] is True

    def test_resolve_obstacle_not_found(self):
        gr = GoalReasoning()
        assert gr.resolve_obstacle("nonexistent") is False

    def test_list_obstacles(self):
        gr = GoalReasoning()
        gr.detect_obstacle("g1", "A", ObstacleSeverity.LOW)
        gr.detect_obstacle("g1", "B", ObstacleSeverity.HIGH)
        gr.detect_obstacle("g2", "C", ObstacleSeverity.CRITICAL)
        result = gr.list_obstacles()
        assert len(result) == 3

    def test_list_obstacles_by_goal(self):
        gr = GoalReasoning()
        gr.detect_obstacle("g1", "A")
        gr.detect_obstacle("g2", "B")
        result = gr.list_obstacles(goal_id="g1")
        assert len(result) == 1

    def test_list_obstacles_by_resolved(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "A")
        gr.resolve_obstacle(o.obstacle_id)
        gr.detect_obstacle("g1", "B")
        unresolved = gr.list_obstacles(resolved=False)
        resolved = gr.list_obstacles(resolved=True)
        assert len(unresolved) == 1
        assert len(resolved) == 1


class TestConflictAnalysis:
    def test_detect_resource_conflict(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "high", "resource_needs": {"compute": 0.8}, "progress": 0.5, "deadline": None},
            {"goal_id": "g2", "priority": "high", "resource_needs": {"compute": 0.6}, "progress": 0.4, "deadline": None},
        ]
        conflicts = gr.analyze_conflicts(goals)
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == ConflictType.RESOURCE

    def test_detect_priority_conflict(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "critical", "progress": 0.5, "deadline": None},
            {"goal_id": "g2", "priority": "critical", "progress": 0.5, "deadline": None},
        ]
        conflicts = gr.analyze_conflicts(goals)
        assert any(c.conflict_type == ConflictType.PRIORITY for c in conflicts)

    def test_no_conflict(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "low", "progress": 0.9, "deadline": None},
            {"goal_id": "g2", "priority": "medium", "progress": 0.1, "deadline": None},
        ]
        conflicts = gr.analyze_conflicts(goals)
        assert len(conflicts) == 0

    def test_list_conflicts(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "critical", "progress": 0.5, "deadline": None},
            {"goal_id": "g2", "priority": "critical", "progress": 0.5, "deadline": None},
        ]
        gr.analyze_conflicts(goals)
        result = gr.list_conflicts()
        assert len(result) >= 1

    def test_list_conflicts_by_goal(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "critical", "progress": 0.5, "deadline": None},
            {"goal_id": "g2", "priority": "critical", "progress": 0.5, "deadline": None},
        ]
        gr.analyze_conflicts(goals)
        result = gr.list_conflicts(goal_id="g1")
        assert len(result) >= 1

    def test_get_conflict(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "critical", "progress": 0.5, "deadline": None},
            {"goal_id": "g2", "priority": "critical", "progress": 0.5, "deadline": None},
        ]
        conflicts = gr.analyze_conflicts(goals)
        retrieved = gr.get_conflict(conflicts[0].conflict_id)
        assert retrieved is not None

    def test_conflict_ids_are_12char_hex(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "critical", "progress": 0.5, "deadline": None},
            {"goal_id": "g2", "priority": "critical", "progress": 0.5, "deadline": None},
        ]
        conflicts = gr.analyze_conflicts(goals)
        assert len(conflicts[0].conflict_id) == 12


class TestReplanning:
    def test_suggest_replan_blocked(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "blocked by resource constraint")
        assert action.replan_type == ReplanType.ESCALATE
        assert not action.applied

    def test_suggest_replan_delay(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g2", "delay the project timeline")
        assert action.replan_type == ReplanType.DEFER

    def test_suggest_replan_urgent(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g3", "urgent deadline approaching")
        assert action.replan_type == ReplanType.RESEQUENCE

    def test_suggest_replan_change(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g4", "reconsider the approach")
        assert action.replan_type == ReplanType.DECOMPOSE_FURTHER

    def test_apply_replan(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g5", "need to pivot")
        assert gr.apply_replan(action.action_id) is True
        result = gr.list_replan_actions(applied=True)
        assert len(result) >= 1
        assert result[0]["applied"] is True

    def test_apply_replan_twice(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g6", "rethink strategy")
        gr.apply_replan(action.action_id)
        assert gr.apply_replan(action.action_id) is False

    def test_apply_replan_not_found(self):
        gr = GoalReasoning()
        assert gr.apply_replan("nonexistent") is False

    def test_list_replan_actions(self):
        gr = GoalReasoning()
        gr.suggest_replan("g1", "blocked")
        gr.suggest_replan("g2", "delay")
        result = gr.list_replan_actions()
        assert len(result) == 2

    def test_list_replan_actions_by_goal(self):
        gr = GoalReasoning()
        gr.suggest_replan("target", "blocked")
        gr.suggest_replan("other", "delay")
        result = gr.list_replan_actions(goal_id="target")
        assert len(result) == 1

    def test_list_replan_actions_by_applied(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "blocked")
        gr.apply_replan(action.action_id)
        applied = gr.list_replan_actions(applied=True)
        unapplied = gr.list_replan_actions(applied=False)
        assert len(applied) == 1
        assert len(unapplied) == 0


class TestPrioritization:
    def test_prioritize_goals(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "low", "progress": 0.9, "deadline": None},
            {"goal_id": "g2", "priority": "critical", "progress": 0.1, "deadline": None},
            {"goal_id": "g3", "priority": "medium", "progress": 0.5, "deadline": None},
        ]
        result = gr.prioritize_goals(goals)
        assert len(result) == 3
        assert result[0]["goal_id"] == "g2"

    def test_prioritize_with_deadline(self):
        gr = GoalReasoning()
        soon = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        later = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        goals = [
            {"goal_id": "urgent", "priority": "high", "progress": 0.3, "deadline": soon},
            {"goal_id": "relaxed", "priority": "high", "progress": 0.3, "deadline": later},
        ]
        result = gr.prioritize_goals(goals)
        assert result[0]["goal_id"] == "urgent"

    def test_prioritize_preserves_data(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "medium", "progress": 0.5, "deadline": None, "custom": "data"},
        ]
        result = gr.prioritize_goals(goals)
        assert result[0]["custom"] == "data"
        assert "_priority_score" in result[0]

    def test_prioritize_empty(self):
        gr = GoalReasoning()
        result = gr.prioritize_goals([])
        assert result == []


class TestUrgencyAndConfidence:
    def test_urgency_no_deadline(self):
        gr = GoalReasoning()
        assert gr._calculate_urgency(0.5, None) == 0.5

    def test_urgency_past_deadline(self):
        gr = GoalReasoning()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        assert gr._calculate_urgency(0.3, past) == 1.0

    def test_urgency_future_deadline(self):
        gr = GoalReasoning()
        future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        u = gr._calculate_urgency(0.1, future)
        assert 0.0 < u <= 1.0

    def test_confidence_high_progress_no_blockers(self):
        gr = GoalReasoning()
        c = gr._calculate_confidence(0.8, 5.0, 10.0, [])
        assert c > 0.5

    def test_confidence_low_progress_high_effort(self):
        gr = GoalReasoning()
        c = gr._calculate_confidence(0.2, 100.0, 10.0, [])
        assert c < 0.5

    def test_confidence_with_blockers(self):
        gr = GoalReasoning()
        c_no = gr._calculate_confidence(0.5, 5.0, 10.0, [])
        c_yes = gr._calculate_confidence(0.5, 5.0, 10.0, ["blocker"])
        assert c_yes < c_no

    def test_risk_factors_deadline_passed(self):
        gr = GoalReasoning()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        risks = gr._identify_risk_factors(0.3, past, 5.0, 10.0, [])
        assert any("deadline passed" in r.lower() for r in risks)

    def test_risk_factors_effort_overrun(self):
        gr = GoalReasoning()
        risks = gr._identify_risk_factors(0.2, None, 15.0, 10.0, [])
        assert any("overrun" in r.lower() for r in risks)

    def test_risk_factors_blockers(self):
        gr = GoalReasoning()
        risks = gr._identify_risk_factors(0.5, None, 5.0, 10.0, ["blocker 1", "blocker 2"])
        assert any("blocker" in r.lower() for r in risks)


class TestRecommendations:
    def test_blocked_recommendations(self):
        gr = GoalReasoning()
        recs = gr._generate_recommendations(AssessmentOutcome.BLOCKED, 0.5, ["blockers active"], ["blocker 1"])
        assert any("resolve blockers" in r.lower() or "address" in r.lower() for r in recs)

    def test_at_risk_recommendations(self):
        gr = GoalReasoning()
        recs = gr._generate_recommendations(AssessmentOutcome.AT_RISK, 0.7, ["high risk"], [])
        assert any("review" in r.lower() or "adjust" in r.lower() or "prioritize" in r.lower() for r in recs)

    def test_on_track_recommendations(self):
        gr = GoalReasoning()
        recs = gr._generate_recommendations(AssessmentOutcome.ON_TRACK, 0.3, [], [])
        assert any("continue" in r.lower() for r in recs)

    def test_on_track_urgent_recommendations(self):
        gr = GoalReasoning()
        recs = gr._generate_recommendations(AssessmentOutcome.ON_TRACK, 0.9, [], [])
        assert any("monitor" in r.lower() for r in recs)

    def test_high_risk_recommendations(self):
        gr = GoalReasoning()
        recs = gr._generate_recommendations(AssessmentOutcome.AT_RISK, 0.5, ["r1", "r2", "r3"], [])
        assert any("risk" in r.lower() for r in recs)

    def test_recommendations_capped(self):
        gr = GoalReasoning()
        recs = gr._generate_recommendations(AssessmentOutcome.BLOCKED, 0.5, [], ["b1", "b2", "b3", "b4"])
        assert len(recs) <= 5


class TestEviction:
    def test_assessment_eviction(self):
        gr = GoalReasoning()
        gr._MAX_ASSESSMENTS = 10
        for i in range(15):
            gr.assess_goal(f"g_{i}", progress=1.0)
        assert len(gr._assessments) <= gr._MAX_ASSESSMENTS

    def test_obstacle_eviction(self):
        gr = GoalReasoning()
        gr._MAX_OBSTACLES = 5
        for i in range(8):
            gr.detect_obstacle(f"g_{i}", f"Obstacle {i}")
        assert len(gr._obstacles) <= gr._MAX_OBSTACLES

    def test_conflict_eviction(self):
        gr = GoalReasoning()
        gr._MAX_CONFLICTS = 3
        for i in range(5):
            g1 = {"goal_id": f"g_{i}", "priority": "critical", "progress": 0.5, "deadline": None, "resource_needs": {"cpu": 0.8}}
            g2 = {"goal_id": f"g_{i}_b", "priority": "critical", "progress": 0.5, "deadline": None, "resource_needs": {"cpu": 0.8}}
            gr.analyze_conflicts([g1, g2])
        assert len(gr._conflicts) <= gr._MAX_CONFLICTS + 2

    def test_replan_eviction(self):
        gr = GoalReasoning()
        gr._MAX_REPLAN_ACTIONS = 5
        for i in range(8):
            gr.suggest_replan(f"g_{i}", f"Reason {i}")
        assert len(gr._replan_actions) <= gr._MAX_REPLAN_ACTIONS


class TestThreadSafety:
    def test_concurrent_assessments(self):
        gr = GoalReasoning()
        errors = []

        def worker(i):
            try:
                for j in range(20):
                    a = gr.assess_goal(f"g_{i}_{j}", progress=float(j) / 20.0)
                    assert a.assessment_id
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_obstacles(self):
        gr = GoalReasoning()
        errors = []

        def worker(i):
            try:
                o = gr.detect_obstacle(f"g_{i}", f"Obstacle {i}", ObstacleSeverity.HIGH)
                assert o.obstacle_id
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_reads_writes(self):
        gr = GoalReasoning()
        errors = []

        def writer():
            for i in range(50):
                gr.assess_goal(f"wg_{i}", progress=0.5)
                gr.detect_obstacle(f"wg_{i}", f"Obs {i}")

        def reader():
            for _ in range(50):
                gr.list_assessments()
                gr.list_obstacles()
                gr.get_stats()

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestDeepCopy:
    def test_assessment_deepcopy(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", progress=0.5, blockers=["b1"])
        d = gr.get_assessment(a.assessment_id)
        d["confidence"] = 0.99
        original = gr.get_assessment(a.assessment_id)
        assert original["confidence"] != 0.99

    def test_obstacle_deepcopy(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "Test obstacle")
        d = gr.get_obstacle(o.obstacle_id)
        d["description"] = "modified"
        original = gr.get_obstacle(o.obstacle_id)
        assert original["description"] == "Test obstacle"

    def test_conflict_deepcopy(self):
        gr = GoalReasoning()
        goals = [
            {"goal_id": "g1", "priority": "critical", "progress": 0.5, "deadline": None, "resource_needs": {"cpu": 0.8}},
            {"goal_id": "g2", "priority": "critical", "progress": 0.5, "deadline": None, "resource_needs": {"cpu": 0.8}},
        ]
        conflicts = gr.analyze_conflicts(goals)
        if conflicts:
            c = gr.get_conflict(conflicts[0].conflict_id)
            c["description"] = "modified"
            original = gr.get_conflict(conflicts[0].conflict_id)
            assert original["description"] != "modified"

    def test_replan_deepcopy(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "blocked")
        d = gr.get_replan_action(action.action_id)
        d["suggestion"] = "modified"
        original = gr.get_replan_action(action.action_id)
        assert original["suggestion"] != "modified"


class TestStats:
    def test_stats_initial(self):
        gr = GoalReasoning()
        stats = gr.get_stats()
        assert stats["assessments_made"] == 0
        assert stats["obstacles_detected"] == 0
        assert stats["conflicts_found"] == 0

    def test_stats_after_operations(self):
        gr = GoalReasoning()
        gr.assess_goal("g1", progress=0.5)
        gr.detect_obstacle("g1", "Blocker")
        gr.suggest_replan("g1", "need change")
        stats = gr.get_stats()
        assert stats["assessments_made"] == 1
        assert stats["obstacles_detected"] == 1
        assert stats["replans_suggested"] == 1

    def test_stats_priority_tracking(self):
        gr = GoalReasoning()
        gr.assess_goal("g1", progress=0.5)
        stats = gr.get_stats()
        assert stats["goal_priorities_tracked"] == 1


class TestSanitization:
    def test_goal_id_sanitized(self):
        gr = GoalReasoning()
        a = gr.assess_goal("goal\x00with\x08control", progress=0.5)
        assert "\x00" not in a.goal_id
        assert "\x08" not in a.goal_id

    def test_obstacle_description_truncated(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "x" * 1000)
        assert len(o.description) <= 500

    def test_replan_reason_truncated(self):
        gr = GoalReasoning()
        action = gr.suggest_replan("g1", "r" * 1000)
        assert len(action.reason) <= 500

    def test_assessment_metadata_capped(self):
        gr = GoalReasoning()
        ctx = {f"key_{i}": f"val_{i}" for i in range(30)}
        a = gr.assess_goal("g1", progress=0.5, context=ctx)
        assert len(a.metadata) <= 20