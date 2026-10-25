#!/usr/bin/env python3
"""Sprint 135 regression tests — goal_planner + anomaly_detector second-pass audit."""

import pytest
import threading
import copy
from dataclasses import dataclass


def test_goal_planner_import():
    from shadowd.cognition.goal_planner import (
        CognitiveGoalPlanner, Goal, PlanStep, GoalStatus, GoalPriority,
        _sanitize, _sanitize_strict,
    )


def test_goal_planner_sanitize_strip():
    from shadowd.cognition.goal_planner import _sanitize, _sanitize_strict
    assert _sanitize("  hello  ") == "hello"
    assert _sanitize_strict("  hello  ") == "hello"
    assert _sanitize("  <tag>  ") == "<tag>"
    assert _sanitize_strict("  <tag>  ") == "tag"


def test_goal_planner_sanitize_control_chars():
    from shadowd.cognition.goal_planner import _sanitize, _sanitize_strict
    assert _sanitize("hel\x00lo") == "hello"
    assert _sanitize("hello\nworld") == "hello\nworld"
    assert _sanitize_strict("hel\x00lo") == "hello"
    assert _sanitize_strict("hello\nworld") == "helloworld"


def test_goal_planner_create_goal_strip():
    from shadowd.cognition.goal_planner import CognitiveGoalPlanner
    gp = CognitiveGoalPlanner()
    goal = gp.create_goal(title="  Test Goal  ", description="  A description  ")
    assert goal.title == "Test Goal"
    assert goal.description == "A description"


def test_goal_planner_create_goal_tags_strip():
    from shadowd.cognition.goal_planner import CognitiveGoalPlanner
    gp = CognitiveGoalPlanner()
    goal = gp.create_goal(title="G", description="D", tags=["  tag1  ", "tag2"])
    assert goal.tags[0] == "tag1"
    assert goal.tags[1] == "tag2"


def test_goal_planner_create_goal_domain_strip():
    from shadowd.cognition.goal_planner import CognitiveGoalPlanner
    gp = CognitiveGoalPlanner()
    goal = gp.create_goal(title="G", description="D", domain="  work  ")
    assert goal.domain == "work"


def test_goal_planner_create_goal_deadline_strip():
    from shadowd.cognition.goal_planner import CognitiveGoalPlanner
    gp = CognitiveGoalPlanner()
    goal = gp.create_goal(title="G", description="D", deadline="  2025-12-31  ")
    assert goal.deadline == "2025-12-31"


def test_goal_planner_goal_to_dict_strip_title():
    from shadowd.cognition.goal_planner import Goal
    g = Goal(title="  padded  ", description="  padded desc  ")
    d = g.to_dict()
    assert d["title"] == "padded"
    assert d["description"] == "padded desc"


def test_goal_planner_step_to_dict_round():
    from shadowd.cognition.goal_planner import PlanStep
    s = PlanStep(description="test", estimated_minutes=5.123456)
    d = s.to_dict()
    assert d["estimated_minutes"] == round(5.123456, 2)


def test_goal_planner_decompose_goal_sanitize():
    from shadowd.cognition.goal_planner import CognitiveGoalPlanner
    gp = CognitiveGoalPlanner()
    goal = gp.create_goal(title="Test", description="Desc")
    gp.decompose_goal(goal.goal_id, [
        {"description": "  Step one  ", "estimated_minutes": 10},
        {"description": "Step\x00two", "estimated_minutes": 20},
    ])
    goal_dict = gp.get_goal(goal.goal_id)
    assert goal_dict["steps"][0]["description"] == "Step one"
    assert goal_dict["steps"][1]["description"] == "Steptwo"


def test_goal_planner_fail_step_sanitize():
    from shadowd.cognition.goal_planner import CognitiveGoalPlanner, StepStatus
    gp = CognitiveGoalPlanner()
    goal = gp.create_goal(title="Test", description="Desc")
    gp.decompose_goal(goal.goal_id, [{"description": "Step 1"}])
    step = goal.steps[0]
    gp.start_step(goal.goal_id, step.step_id)
    gp.fail_step(goal.goal_id, step.step_id, "  Something failed\x00badly  ")
    goal_dict = gp.get_goal(goal.goal_id)
    assert goal_dict["steps"][0]["result"]["error"] == "Something failedbadly"


def test_anomaly_detector_import():
    from shadowd.cognition.anomaly_detector import (
        AnomalyDetectionEngine, Anomaly, AnomalyType, AnomalySeverity, MetricStream,
        _sanitize, _sanitize_content,
    )


def test_anomaly_detector_sanitize_strip():
    from shadowd.cognition.anomaly_detector import _sanitize, _sanitize_content
    assert _sanitize("  hello  ") == "hello"
    assert _sanitize_content("  hello  ") == "hello"
    assert _sanitize("<script>") == "script"
    assert _sanitize_content("<script>") == "<script>"


def test_anomaly_detector_sanitize_control_chars():
    from shadowd.cognition.anomaly_detector import _sanitize, _sanitize_content
    assert _sanitize("hel\x00lo") == "hello"
    assert _sanitize("hello\nworld") == "helloworld"
    assert _sanitize_content("hello\nworld") == "hello\nworld"
    assert _sanitize_content("hello\tworld") == "hello\tworld"


def test_anomaly_to_dict_expected_range_round():
    from shadowd.cognition.anomaly_detector import Anomaly, AnomalyType, AnomalySeverity
    a = Anomaly(
        anomaly_type=AnomalyType.STATISTICAL,
        severity=AnomalySeverity.HIGH,
        metric="cpu_usage",
        value=98.7654321,
        expected_range=(10.123456789, 90.987654321),
        z_score=3.14159265,
        description="Test anomaly",
    )
    d = a.to_dict()
    assert d["expected_range"][0] == round(10.123456789, 4)
    assert d["expected_range"][1] == round(90.987654321, 4)
    assert d["value"] == round(98.7654321, 4)
    assert d["z_score"] == round(3.14159265, 4)


def test_anomaly_to_dict_context_deepcopy():
    from shadowd.cognition.anomaly_detector import Anomaly, AnomalyType
    ctx = {"key": [1, 2, 3], "nested": {"a": "b"}}
    a = Anomaly(
        anomaly_type=AnomalyType.STATISTICAL,
        metric="test",
        context=ctx,
    )
    d = a.to_dict()
    assert d["context"] == ctx
    d["context"]["key"].append(4)
    assert a.context["key"] == [1, 2, 3]  # original unchanged


def test_anomaly_detector_register_strip():
    from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
    engine = AnomalyDetectionEngine()
    engine.register_metric("  cpu_usage  ", description="  CPU usage  ")
    engine.observe("cpu_usage", 50.0)
    stats = engine.get_stream_stats()
    assert "cpu_usage" in stats


def test_anomaly_detector_observe_strip():
    from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
    engine = AnomalyDetectionEngine()
    engine.observe("  memory  ", 50.0)
    for _ in range(10):
        engine.observe("memory", 50.0)
    assert "memory" in engine.get_stream_stats()


def test_anomaly_detector_observe_temporal_strip():
    from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
    from datetime import datetime, timezone
    engine = AnomalyDetectionEngine()
    engine.observe_temporal("  login  ", timestamp=datetime.now(timezone.utc))
    assert "temporal_login" in engine.get_stream_stats()


def test_goal_planner_create_goal_xss_strip():
    from shadowd.cognition.goal_planner import CognitiveGoalPlanner
    gp = CognitiveGoalPlanner()
    goal = gp.create_goal(title="<script>alert(1)</script>", description="normal")
    assert "<script>" not in goal.title
    assert "alert(1)" in goal.title


def test_goal_planner_progress_round():
    from shadowd.cognition.goal_planner import Goal, GoalStatus
    g = Goal(title="Test", progress=0.3333333)
    d = g.to_dict()
    assert d["progress"] == round(0.3333333, 3)


def test_anomaly_detector_thread_safety():
    from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
    engine = AnomalyDetectionEngine()
    errors = []

    def register_and_observe():
        try:
            for i in range(50):
                engine.register_metric(f"metric_{i}", description=f"Test {i}")
                engine.observe(f"metric_{i}", float(i))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=register_and_observe) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 0


def test_goal_planner_thread_safety():
    from shadowd.cognition.goal_planner import CognitiveGoalPlanner
    gp = CognitiveGoalPlanner()
    errors = []

    def create_goals():
        try:
            for i in range(10):
                gp.create_goal(title=f"Goal {i}", description=f"Description {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=create_goals) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 0


def test_goal_planner_tags_truncation():
    from shadowd.cognition.goal_planner import CognitiveGoalPlanner
    gp = CognitiveGoalPlanner()
    tags = [f"tag{i}" for i in range(15)]
    goal = gp.create_goal(title="G", description="D", tags=tags)
    d = goal.to_dict()
    assert len(d["tags"]) <= 10


def test_anomaly_detector_stats_copy():
    from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
    engine = AnomalyDetectionEngine()
    stats1 = engine.get_stats()
    stats2 = engine.get_stats()
    assert stats1 is not stats2


def test_anomaly_detector_stream_stats_copy():
    from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
    engine = AnomalyDetectionEngine()
    engine.register_metric("test", description="Test")
    engine.observe("test", 42.0)
    s1 = engine.get_stream_stats()
    s2 = engine.get_stream_stats()
    assert s1 is not s2


def test_anomaly_detector_behavioral_baseline_copy():
    from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
    engine = AnomalyDetectionEngine()
    engine.set_baseline("user1", {"metric1": {"mean": 50.0, "std": 5.0}})
    stats = engine.get_stats()
    assert stats["baselines"] == 1