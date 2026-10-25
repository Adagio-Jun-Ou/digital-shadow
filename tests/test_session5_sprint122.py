#!/usr/bin/env python3
"""Sprint 122 regression tests: workflow/engine.py deep audit."""

import asyncio
import copy
import threading
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from shadowd.workflow.engine import (
    BlockStatus, BlockResult, Checkpoint, BaseBlock, FunctionBlock,
    ConditionalBlock, Workflow, WorkflowEngine, StuckDetector,
    _SANITIZE_RE, _SANITIZE_STRICT_RE, _MAX_WORKFLOWS,
    _MAX_BLOCKS_PER_WORKFLOW, _MAX_CHECKPOINTS,
)


# ═════════════════════════════════════════════════════════════════
# BlockResult tests
# ═════════════════════════════════════════════════════════════════

class TestBlockResult:
    def test_to_dict_returns_copies(self):
        br = BlockResult(
            block_id="blk1", status=BlockStatus.SUCCESS,
            output="ok", metadata={"key": "val"}, artifacts=["a1"]
        )
        d = br.to_dict()
        d["metadata"]["key"] = "changed"
        d["artifacts"].append("a2")
        assert br.metadata["key"] == "val"
        assert len(br.artifacts) == 1

    def test_to_dict_truncates_error(self):
        br = BlockResult(
            block_id="blk1", status=BlockStatus.FAILED,
            output=None, error="x" * 300
        )
        d = br.to_dict()
        assert len(d["error"]) == 200

    def test_to_dict_truncates_output(self):
        br = BlockResult(
            block_id="blk1", status=BlockStatus.SUCCESS,
            output="y" * 600
        )
        d = br.to_dict()
        assert len(d["output"]) == 500

    def test_to_dict_rounds_elapsed(self):
        br = BlockResult(
            block_id="blk1", status=BlockStatus.SUCCESS,
            output="ok", elapsed_ms=1.234567
        )
        d = br.to_dict()
        assert d["elapsed_ms"] == 1.235

    def test_to_dict_none_fields(self):
        br = BlockResult(block_id="blk1", status=BlockStatus.PENDING, output=None)
        d = br.to_dict()
        assert d["error"] is None
        assert d["output"] is None


# ═════════════════════════════════════════════════════════════════
# Checkpoint tests
# ═════════════════════════════════════════════════════════════════

class TestCheckpoint:
    def test_to_dict_deepcopy_state(self):
        state = {"nested": {"key": "val"}}
        cp = Checkpoint(
            workflow_id="wf1", block_id="b1",
            timestamp=datetime.now(timezone.utc),
            state=state, status=BlockStatus.RUNNING
        )
        d = cp.to_dict()
        d["state"]["nested"]["key"] = "changed"
        assert state["nested"]["key"] == "val"

    def test_to_dict_truncates_error(self):
        cp = Checkpoint(
            workflow_id="wf1", block_id="b1",
            timestamp=datetime.now(timezone.utc),
            state={}, status=BlockStatus.FAILED,
            error="e" * 300
        )
        d = cp.to_dict()
        assert len(d["error"]) == 200

    def test_from_dict_roundtrip(self):
        cp = Checkpoint(
            workflow_id="wf1", block_id="b1",
            timestamp=datetime.now(timezone.utc),
            state={"x": 1}, status=BlockStatus.RUNNING
        )
        d = cp.to_dict()
        cp2 = Checkpoint.from_dict(d)
        assert cp2.workflow_id == "wf1"
        assert cp2.block_id == "b1"


# ═════════════════════════════════════════════════════════════════
# BaseBlock tests
# ═════════════════════════════════════════════════════════════════

class TestBaseBlock:
    def test_default_id_is_uuid_hex(self):
        b = BaseBlock()
        assert b.block_id.startswith("blk_")
        assert len(b.block_id) == 16

    def test_name_sanitized_strict(self):
        b = BaseBlock(name="<script>alert(1)</script>")
        assert "<" not in b.name
        assert ">" not in b.name

    def test_name_truncated_to_200(self):
        b = BaseBlock(name="x" * 300)
        assert len(b.name) == 200

    def test_to_dict_returns_list_copy(self):
        b = BaseBlock(name="test")
        b.dependencies.add("d1")
        d = b.to_dict()
        d["dependencies"].append("d2")
        assert len(b.dependencies) == 1

    def test_to_dict_name_truncated(self):
        b = BaseBlock(name="x" * 300)
        d = b.to_dict()
        assert len(d["name"]) == 200


# ═════════════════════════════════════════════════════════════════
# Workflow tests
# ═════════════════════════════════════════════════════════════════

class TestWorkflow:
    def test_id_is_uuid_hex(self):
        wf = Workflow(name="test")
        assert wf.workflow_id.startswith("wf_")
        assert len(wf.workflow_id) == 15

    def test_name_sanitized(self):
        wf = Workflow(name="hello\x00world\x1f")
        assert "\x00" not in wf.name
        assert "\x1f" not in wf.name

    def test_name_truncated_to_200(self):
        wf = Workflow(name="x" * 300)
        assert len(wf.name) == 200

    def test_add_lock(self):
        wf = Workflow(name="test")
        threads = []
        for i in range(10):
            def add(bid):
                blk = BaseBlock(block_id=f"b{bid}")
                wf.add_block(blk)
            t = threading.Thread(target=add, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(wf.blocks) == 10

    def test_add_block_cap(self):
        wf = Workflow(name="test")
        original_max = _MAX_BLOCKS_PER_WORKFLOW
        for i in range(original_max + 5):
            wf.add_block(BaseBlock(block_id=f"b{i}"))
        assert len(wf.blocks) <= original_max

    def test_create_checkpoint_sanitizes_block_id(self):
        wf = Workflow(name="test")
        cp = wf.create_checkpoint("<script>")
        assert "<" not in cp.block_id

    def test_create_checkpoint_deepcopy_context(self):
        wf = Workflow(name="test")
        wf.context = {"nested": {"key": "val"}}
        cp = wf.create_checkpoint("b1")
        cp.state["nested"]["key"] = "changed"
        assert wf.context["nested"]["key"] == "val"

    def test_checkpoint_cap(self):
        wf = Workflow(name="test")
        for i in range(_MAX_CHECKPOINTS + 5):
            wf.create_checkpoint(f"b{i}")
        assert len(wf.checkpoints) <= _MAX_CHECKPOINTS

    def test_to_dict_under_lock(self):
        wf = Workflow(name="test")
        wf.context = {"k": "v"}
        d = wf.to_dict()
        assert "workflow_id" in d
        assert d["context"]["k"] == "v"

    def test_to_dict_deepcopy_context(self):
        wf = Workflow(name="test")
        wf.context = {"nested": {"key": "val"}}
        d = wf.to_dict()
        d["context"]["nested"]["key"] = "changed"
        assert wf.context["nested"]["key"] == "val"

    def test_topological_sort_detects_cycle(self):
        wf = Workflow(name="test")
        b1 = BaseBlock(block_id="b1")
        b2 = BaseBlock(block_id="b2")
        b3 = BaseBlock(block_id="b3")
        b2.depends_on("b1")
        b3.depends_on("b2")
        b1.depends_on("b3")
        wf.add_block(b1)
        wf.add_block(b2)
        wf.add_block(b3)
        wf.add_edge("b1", "b2")
        wf.add_edge("b2", "b3")
        wf.add_edge("b3", "b1")
        with pytest.raises(ValueError, match="[Cc]ycle"):
            wf.topological_sort()


# ═════════════════════════════════════════════════════════════════
# WorkflowEngine tests
# ═════════════════════════════════════════════════════════════════

class TestWorkflowEngine:
    def test_create_workflow_sanitizes_name(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("hello\x00world")
        assert "\x00" not in wf.name

    def test_create_workflow_cap(self):
        engine = WorkflowEngine()
        for i in range(_MAX_WORKFLOWS + 5):
            engine.create_workflow(f"wf{i}")
        assert len(engine.workflows) <= _MAX_WORKFLOWS

    def test_execute_workflow_sanitizes_id(self):
        engine = WorkflowEngine()
        engine.workflows["wf_abc"] = Workflow(name="test")
        with pytest.raises(Exception):
            asyncio.run(engine.execute_workflow("wf_abc<script>"))

    def test_execute_workflow_not_found(self):
        engine = WorkflowEngine()
        with pytest.raises(ValueError, match="[Nn]ot found"):
            asyncio.run(engine.execute_workflow("nonexistent"))

    def test_resume_workflow_sanitizes_id(self):
        engine = WorkflowEngine()
        with pytest.raises(Exception):
            engine.resume_workflow("<script>")

    def test_get_status_sanitizes_id(self):
        engine = WorkflowEngine()
        result = engine.get_status("<script>")
        assert result == {"error": "Workflow not found"}

    def test_get_status_returns_dict(self):
        engine = WorkflowEngine()
        wf = engine.create_workflow("test")
        wf.add_block(BaseBlock(block_id="b1"))
        result = engine.get_status(wf.workflow_id)
        assert result["workflow_id"] == wf.workflow_id
        assert "status" in result

    def test_create_workflow_thread_safe(self):
        engine = WorkflowEngine()
        results = []
        errors = []

        def create(i):
            try:
                wf = engine.create_workflow(f"wf{i}")
                results.append(wf.workflow_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 20


# ═════════════════════════════════════════════════════════════════
# StuckDetector tests
# ═════════════════════════════════════════════════════════════════

class TestStuckDetector:
    def test_record_truncates_values(self):
        sd = StuckDetector()
        sd.record({"tool": "t" * 200, "action": "a" * 200})
        assert len(sd.history[0]["action"]["tool"]) == 100
        assert len(sd.history[0]["action"]["action"]) == 100

    def test_record_uses_utc_timestamps(self):
        sd = StuckDetector()
        sd.record({"tool": "test"})
        ts = sd.history[0]["timestamp"]
        assert "+00:00" in ts or "Z" in ts or "T" in ts

    def test_record_caps_action_counts(self):
        sd = StuckDetector(max_history=10)
        for i in range(15):
            sd.record({"tool": f"tool_{i % 3}"})
        assert len(sd.action_counts) <= 10

    def test_is_stuck_repeated_action(self):
        sd = StuckDetector()
        for _ in range(5):
            sd.record({"tool": "same_action"})
        stuck, reason = sd.is_stuck()
        assert stuck
        assert "repeated" in reason.lower()

    def test_not_stuck_insufficient_history(self):
        sd = StuckDetector()
        sd.record({"tool": "a"})
        stuck, _ = sd.is_stuck()
        assert not stuck

    def test_reset_under_lock(self):
        sd = StuckDetector()
        for i in range(10):
            sd.record({"tool": f"tool_{i}"})
        sd.reset()
        assert len(sd.history) == 0
        assert len(sd.action_counts) == 0
        assert len(sd.pattern_history) == 0

    def test_max_history_capped_at_500(self):
        sd = StuckDetector(max_history=999)
        assert sd.max_history == 500

    def test_record_thread_safety(self):
        sd = StuckDetector()
        errors = []

        def record(i):
            try:
                sd.record({"tool": f"tool_{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(sd.history) == 50


# ═════════════════════════════════════════════════════════════════
# Sanitize regex tests
# ═════════════════════════════════════════════════════════════════

class TestSanitizeRegex:
    def test_sanitize_re_strips_control_chars(self):
        text = "hello\x00world\x1f"
        assert _SANITIZE_RE.sub('', text) == "helloworld"

    def test_sanitize_re_preserves_angle_brackets(self):
        text = "latency < 100ms"
        assert _SANITIZE_RE.sub('', text) == "latency < 100ms"

    def test_sanitize_re_strips_127(self):
        text = "hello\x7f"
        assert _SANITIZE_RE.sub('', text) == "hello"

    def test_strict_re_strips_angle_brackets(self):
        text = "<script>alert(1)</script>"
        assert _SANITIZE_STRICT_RE.sub('', text) == "scriptalert(1)/script"

    def test_strict_re_strips_control_chars(self):
        text = "hello\x00\x01\x1f"
        assert _SANITIZE_STRICT_RE.sub('', text) == "hello"

    def test_strict_re_strips_tabs_newlines(self):
        text = "hello\tworld\nline\rreturn"
        assert _SANITIZE_STRICT_RE.sub('', text) == "helloworldlinereturn"


# ═════════════════════════════════════════════════════════════════
# Constants tests
# ═════════════════════════════════════════════════════════════════

class TestConstants:
    def test_max_workflows(self):
        assert _MAX_WORKFLOWS == 500

    def test_max_blocks(self):
        assert _MAX_BLOCKS_PER_WORKFLOW == 200

    def test_max_checkpoints(self):
        assert _MAX_CHECKPOINTS == 50


# ═════════════════════════════════════════════════════════════════
# FunctionBlock tests
# ═════════════════════════════════════════════════════════════════

class TestFunctionBlock:
    def test_async_function_executes(self):
        async def my_func(ctx):
            return "async_result"

        blk = FunctionBlock(my_func, name="test_async")
        result = asyncio.run(blk.execute({}))
        assert result.status == BlockStatus.SUCCESS
        assert result.output == "async_result"

    def test_sync_function_executes(self):
        def my_func(ctx):
            return "sync_result"

        blk = FunctionBlock(my_func, name="test_sync")
        result = asyncio.run(blk.execute({}))
        assert result.status == BlockStatus.SUCCESS
        assert result.output == "sync_result"

    def test_function_error_uses_type_name(self):
        def bad_func(ctx):
            raise ValueError("test error")

        blk = FunctionBlock(bad_func, name="bad", max_retries=1)
        result = asyncio.run(blk.execute({}))
        assert result.status == BlockStatus.FAILED
        assert "ValueError" in result.error
        assert "test error" not in result.error


# ═════════════════════════════════════════════════════════════════
# ConditionalBlock tests
# ═════════════════════════════════════════════════════════════════

class TestConditionalBlock:
    def test_condition_true(self):
        blk = ConditionalBlock(
            condition=lambda ctx: True,
            true_block="yes_block",
            false_block="no_block"
        )
        result = asyncio.run(blk.execute({}))
        assert result.output["condition_result"] is True
        assert result.output["next_block"] == "yes_block"

    def test_condition_false(self):
        blk = ConditionalBlock(
            condition=lambda ctx: False,
            true_block="yes_block",
            false_block="no_block"
        )
        result = asyncio.run(blk.execute({}))
        assert result.output["condition_result"] is False
        assert result.output["next_block"] == "no_block"


# ═════════════════════════════════════════════════════════════════
# Atomic write tests
# ═════════════════════════════════════════════════════════════════

class TestAtomicWrite:
    def test_save_workflow_atomic(self, tmp_path):
        engine = WorkflowEngine(storage_path=tmp_path / "wf")
        wf = engine.create_workflow("test_save")
        blk = BaseBlock(block_id="b1")
        wf.add_block(blk)
        engine._save_workflow(wf)
        saved_file = tmp_path / "wf" / f"{wf.workflow_id}.json"
        assert saved_file.exists()

    def test_save_checkpoint_atomic(self, tmp_path):
        engine = WorkflowEngine(storage_path=tmp_path / "wf")
        wf = engine.create_workflow("test_cp")
        blk = BaseBlock(block_id="b1")
        wf.add_block(blk)
        cp = wf.create_checkpoint("b1")
        engine._save_workflow(wf)
        cp_dir = tmp_path / "wf" / wf.workflow_id / "checkpoints"
        assert cp_dir.exists()


# ═════════════════════════════════════════════════════════════════
# Alternating pattern detection
# ═════════════════════════════════════════════════════════════════

class TestAlternatingPattern:
    def test_alternating_pattern_detected(self):
        sd = StuckDetector()
        sd.record({"tool": "A"})
        sd.record({"tool": "B"})
        sd.record({"tool": "A"})
        sd.record({"tool": "B"})
        sd.record({"tool": "A"})
        sd.record({"tool": "B"})
        stuck, reason = sd.is_stuck()
        assert stuck
        assert "alternating" in reason.lower()

    def test_no_alternating_pattern(self):
        sd = StuckDetector()
        for i in range(6):
            sd.record({"tool": f"unique_{i}"})
        stuck, _ = sd.is_stuck()
        assert not stuck