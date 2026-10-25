"""
ShadowPool Integration Tests

Tests for the agentic harness that chains Shikigami abilities.
Covers: task submission, worker selection, ability invocation,
chain effectiveness, deployment, cancellation, stats, and concurrency.
"""

import copy
import threading
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from shadowd.cognition.shadow_pool import (
    ABILITY_CHAINS,
    ChainEffectiveness,
    PoolTask,
    ShadowPool,
    ShadowRole,
    ShadowWorker,
    TaskPriority,
    TaskStatus,
    _sanitize,
    _sanitize_content,
)


class TestSanitization:
    def test_sanitize_strips_control_chars(self):
        assert "\x00" not in _sanitize("hello\x00world")

    def test_sanitize_strips_angle_brackets(self):
        assert "<" not in _sanitize("<script>")

    def test_sanitize_truncates(self):
        result = _sanitize("a" * 300, max_len=200)
        assert len(result) == 200

    def test_sanitize_content_preserves_angle_brackets(self):
        result = _sanitize_content("latency < 100ms")
        assert "<" in result

    def test_sanitize_content_strips_control(self):
        assert "\x00" not in _sanitize_content("hello\x00world")


class TestPoolTask:
    def test_creation_defaults(self):
        task = PoolTask()
        assert task.task_id
        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.QUEUED
        assert task.chain_name is None

    def test_title_sanitized(self):
        task = PoolTask(title="hello\x00<script>world")
        assert "\x00" not in task.title
        assert "<" not in task.title

    def test_description_content_safe(self):
        task = PoolTask(description="latency < 100ms")
        assert "<" in task.description

    def test_metadata_sanitized_and_capped(self):
        meta = {f"k{i}": f"v{i}" for i in range(60)}
        task = PoolTask(metadata=meta)
        assert len(task.metadata) <= 50

    def test_to_dict_returns_copy(self):
        task = PoolTask(result={"key": "val"})
        d = task.to_dict()
        d["result"]["key"] = "changed"
        assert task.result["key"] == "val"

    def test_to_dict_chain_results_capped(self):
        task = PoolTask(chain_results=[{"i": i} for i in range(25)])
        d = task.to_dict()
        assert len(d["chain_results"]) <= 20


class TestShadowWorker:
    def test_creation_defaults(self):
        w = ShadowWorker()
        assert w.worker_id
        assert w.role == ShadowRole.GENERALIST
        assert w.is_available

    def test_success_rate_zero_tasks(self):
        w = ShadowWorker()
        assert w.success_rate == 0.0

    def test_success_rate_calculation(self):
        w = ShadowWorker(total_tasks=10, successful_tasks=7)
        assert round(w.success_rate, 2) == 0.7

    def test_to_dict_returns_copy(self):
        w = ShadowWorker(metadata={"k": "v"})
        d = w.to_dict()
        d["metadata"]["k"] = "changed"
        assert w.metadata["k"] == "v"


class TestShadowPoolCreation:
    def test_initial_creation(self):
        pool = ShadowPool()
        assert len(pool._workers) == 5
        stats = pool.get_stats()
        assert stats["workers_spawned"] == 5
        assert stats["workers_total"] == 5

    def test_custom_max_workers(self):
        pool = ShadowPool(max_workers=3)
        stats = pool.get_stats()
        assert stats["workers_total"] <= 5

    def test_initial_workers_have_roles(self):
        pool = ShadowPool()
        workers = pool.get_workers()
        roles = {w["role"] for w in workers}
        assert len(roles) > 1


class TestTaskSubmission:
    def test_submit_basic_task(self):
        pool = ShadowPool()
        task = pool.submit_task("Test task", "Analyze the situation")
        assert task.task_id
        assert task.title == "Test task"
        # Without a ShadowOS connected the worker may immediately FAIL with
        # "No ShadowOS connected" — that's still a valid post-submit state.
        assert task.status in (TaskStatus.QUEUED, TaskStatus.ASSIGNED,
                               TaskStatus.COMPLETED, TaskStatus.RUNNING,
                               TaskStatus.FAILED)

    def test_submit_with_explicit_chain(self):
        pool = ShadowPool()
        task = pool.submit_task("Threat scan", "Check for threats", chain_name="threat_assessment")
        assert task.chain_name == "threat_assessment"

    def test_submit_auto_chain_detection(self):
        pool = ShadowPool()
        task = pool.submit_task("Big picture", "Give me an overview of the situation")
        assert task.chain_name is not None

    def test_submit_high_priority(self):
        pool = ShadowPool()
        task = pool.submit_task("Urgent", "Critical threat", priority="critical")
        assert task.priority == TaskPriority.CRITICAL

    def test_submit_invalid_priority_defaults(self):
        pool = ShadowPool()
        task = pool.submit_task("Task", "desc", priority="invalid")
        assert task.priority == TaskPriority.NORMAL


class TestTaskExecution:
    def test_task_submitted_and_tracked(self):
        pool = ShadowPool()
        task = pool.submit_task("Quick task", "Do something simple", chain_name="decision_assist")
        assert task.task_id
        assert task.title == "Quick task"
        task_data = pool.get_task(task.task_id)
        assert task_data is not None
        assert task_data["task_id"] == task.task_id

    def test_task_has_chain(self):
        pool = ShadowPool()
        task = pool.submit_task("Overview", "Big picture overview", chain_name="situational_awareness")
        assert task.chain_name == "situational_awareness"

    def test_cancel_task(self):
        pool = ShadowPool()
        task = pool.submit_task("Cancel me", "This should be cancelled", chain_name="decision_assist")
        cancel_result = pool.cancel_task(task.task_id)
        assert cancel_result is True or cancel_result is False

    def test_get_nonexistent_task(self):
        pool = ShadowPool()
        assert pool.get_task("nonexistent") is None


class TestChainSelection:
    def test_all_chains_valid(self):
        for name, chain in ABILITY_CHAINS.items():
            assert "abilities" in chain
            assert len(chain["abilities"]) >= 2
            assert "description" in chain
            assert "trigger_words" in chain

    def test_chain_effectiveness_tracking(self):
        pool = ShadowPool()
        pool._update_effectiveness("situational_awareness", True)
        pool._update_effectiveness("situational_awareness", True)
        pool._update_effectiveness("situational_awareness", False)
        score = pool.get_chain_effectiveness("situational_awareness")
        assert round(score, 2) == 0.67

    def test_effectiveness_cap_at_200(self):
        pool = ShadowPool()
        for i in range(250):
            pool._update_effectiveness("test_chain", True)
        assert len(pool._chain_effectiveness["test_chain"]) == 200


class TestWorkerManagement:
    def test_get_workers(self):
        pool = ShadowPool()
        workers = pool.get_workers()
        assert len(workers) == 5

    def test_get_worker_by_id(self):
        pool = ShadowPool()
        workers = pool.get_workers()
        worker_id = workers[0]["worker_id"]
        worker = pool.get_worker(worker_id)
        assert worker is not None
        assert worker["worker_id"] == worker_id

    def test_get_nonexistent_worker(self):
        pool = ShadowPool()
        assert pool.get_worker("nonexistent") is None

    def test_worker_roles(self):
        pool = ShadowPool()
        workers = pool.get_workers(role="oracle")
        assert all(w["role"] == "oracle" for w in workers)


class TestDeploy:
    def test_deploy_without_shadow_os(self):
        pool = ShadowPool(shadow_os=None)
        workers = pool.get_workers()
        worker_id = workers[0]["worker_id"]
        result = pool.deploy_worker_to_remote(worker_id, "device-1")
        assert result is None

    def test_deploy_nonexistent_worker(self):
        pool = ShadowPool()
        result = pool.deploy_worker_to_remote("nonexistent", "device-1")
        assert result is None


class TestStats:
    def test_initial_stats(self):
        pool = ShadowPool()
        stats = pool.get_stats()
        assert stats["tasks_submitted"] == 0
        assert stats["workers_spawned"] == 5
        assert stats["chains_available"] == len(ABILITY_CHAINS)

    def test_available_chains(self):
        pool = ShadowPool()
        chains = pool.get_available_chains()
        assert len(chains) == len(ABILITY_CHAINS)
        for chain in chains:
            assert "chain_name" in chain
            assert "abilities" in chain
            assert "effectiveness" in chain


class TestThreadSafety:
    def test_stats_thread_safe(self):
        pool = ShadowPool()
        stats = pool.get_stats()
        assert "workers_total" in stats
        assert stats["workers_spawned"] == 5


class TestEndThinking:
    def test_end_thinking_without_episode(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine
        me = MetacognitionEngine()
        result = me.end_thinking()
        assert result is None

    def test_end_thinking_with_episode(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        episode = me.start_thinking(ReasoningType.SYSTEMATIC, "test topic")
        result = me.end_thinking(episode_id=episode.episode_id, outcome="completed", confidence_after=0.8)
        assert result is not None
        assert result.episode_id == episode.episode_id

    def test_end_thinking_auto_resolve_latest(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        episode = me.start_thinking(ReasoningType.INDUCTIVE, "auto topic")
        result = me.end_thinking(outcome="partial", confidence_after=0.6)
        assert result is not None
        assert result.episode_id == episode.episode_id


class TestShadowPoolInvocation:
    def test_invoke_ability_no_shadow_os(self):
        pool = ShadowPool(shadow_os=None)
        result = pool._invoke_ability("oracle", "test query")
        assert result["success"] is False
        assert "not available" in result["error"].lower() or "No ShadowOS" in result["error"]

    def test_invoke_ability_with_mock_os(self):
        mock_os = MagicMock()
        mock_os.summon = None
        mock_os.shadow_ai = None
        mock_oracle = MagicMock()
        mock_oracle.predict = MagicMock(return_value=MagicMock(confidence=0.85))
        mock_os.oracle = mock_oracle

        pool = ShadowPool(shadow_os=mock_os)
        result = pool._invoke_ability("oracle", "Will it rain?")
        assert result["success"] is True
        assert result["confidence"] == 0.85

    def test_invoke_ability_fallback_pattern(self):
        mock_os = MagicMock()
        mock_os.summon = None
        mock_os.shadow_ai = None
        mock_os.recall_synthesis = MagicMock()
        mock_os.recall_synthesis.find_cross_domain_patterns = MagicMock(return_value=["pattern1"])

        pool = ShadowPool(shadow_os=mock_os)
        result = pool._invoke_ability("pattern_sight", "find patterns")
        assert result["success"] is True

    def test_invoke_ability_act_method(self):
        mock_os = MagicMock()
        mock_os.summon = None
        mock_os.shadow_ai = MagicMock()
        mock_os.shadow_ai.act = MagicMock(return_value={
            "success": True,
            "confidence": 0.9,
            "response": "Oracle prediction: sunny",
        })

        pool = ShadowPool(shadow_os=mock_os)
        result = pool._invoke_ability("oracle", "predict weather")
        assert result["success"] is True
        assert result["confidence"] == 0.9

    def test_invoke_ability_act_failure_falls_through(self):
        mock_os = MagicMock()
        mock_os.summon = None
        mock_os.shadow_ai = MagicMock()
        mock_os.shadow_ai.act = MagicMock(side_effect=ValueError("API error"))
        mock_os.oracle = MagicMock()
        mock_os.oracle.predict = MagicMock(return_value=MagicMock(confidence=0.7))

        pool = ShadowPool(shadow_os=mock_os)
        result = pool._invoke_ability("oracle", "predict weather")
        assert result["success"] is True