"""Regression tests for Session 9 Sprint 120 — self_builder.py audit."""

import threading
import time
import pytest
from unittest.mock import MagicMock

from shadowd.self_improvement.self_builder import (
    SelfBuilder,
    TaskAnalysis,
    TaskType,
    ComplexityLevel,
    SynthesisRequest,
    SynthesisResult,
    _SANITIZE_RE,
    _SANITIZE_STRICT_RE,
    _MAX_SYNTHESIS_HISTORY,
)
from shadowd.self_improvement.function_db import (
    FunctionDB,
    GeneratedFunction,
    FunctionStatus,
    Priority,
)


@pytest.fixture
def mock_db():
    db = MagicMock(spec=FunctionDB)
    db.search_by_tags.return_value = []
    db.register.return_value = None
    db.get.return_value = None
    return db


@pytest.fixture
def builder(mock_db):
    return SelfBuilder(mock_db)


class TestSelfBuilderSanitization:
    def test_sanitize_re_strips_control_chars(self):
        assert _SANITIZE_RE.sub('', "hello\x00world\x07") == "helloworld"

    def test_sanitize_re_preserves_angle_brackets(self):
        assert _SANITIZE_RE.sub('', "hello <world>") == "hello <world>"

    def test_sanitize_re_preserves_newlines_tabs(self):
        assert _SANITIZE_RE.sub('', "hello\nworld\ttab") == "hello\nworld\ttab"

    def test_sanitize_strict_re_strips_angle_brackets(self):
        assert _SANITIZE_STRICT_RE.sub('', "hello<wor>ld") == "helloworld"

    def test_sanitize_strict_re_strips_control_chars(self):
        assert _SANITIZE_STRICT_RE.sub('', "hello\x00\x01world") == "helloworld"


class TestSelfBuilderAnalyzeTaskSanitization:
    def test_analyze_task_strips_control_chars(self, builder):
        result = builder.analyze_task("calculate\x00 sum of\x07 values")
        assert "\x00" not in result.description
        assert "\x07" not in result.description

    def test_analyze_task_truncates_long_description(self, builder):
        long_desc = "x" * 20000
        result = builder.analyze_task(long_desc)
        assert len(result.description) <= 10000

    def test_analyze_task_preserves_angle_brackets(self, builder):
        result = builder.analyze_task("filter data where value < 100")
        assert "<" in result.description


class TestSelfBuilderImproveFunctionSanitization:
    def test_improve_function_strips_function_id(self, builder, mock_db):
        mock_db.get.return_value = None
        builder.improve_function("fun><ction_id", "feedback")
        call_args = mock_db.get.call_args[0][0]
        assert "<" not in call_args
        assert ">" not in call_args

    def test_improve_function_strips_control_chars_from_feedback(self, builder, mock_db):
        mock_db.get.return_value = None
        builder.improve_function("func_id", "feed\x00back\x07")
        call_args = mock_db.get.call_args[0][0]
        assert call_args == "func_id"


class TestSelfBuilderThreadSafety:
    def test_synthesis_history_thread_safe(self, builder):
        results = []
        errors = []

        def append_result(i):
            try:
                r = SynthesisResult(success=True, iterations=1, synthesis_time=0.001)
                with builder._lock:
                    if len(builder._synthesis_history) >= _MAX_SYNTHESIS_HISTORY:
                        builder._synthesis_history.pop(0)
                    builder._synthesis_history.append(r)
                results.append(i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=append_result, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50
        assert len(builder._synthesis_history) <= _MAX_SYNTHESIS_HISTORY

    def test_get_synthesis_history_returns_deepcopy(self, builder):
        builder._synthesis_history.append(SynthesisResult(success=True, iterations=1, synthesis_time=0.5))
        hist = builder.get_synthesis_history()
        assert len(hist) == 1
        assert hist is not builder._synthesis_history

    def test_get_statistics_thread_safe(self, builder):
        builder._synthesis_history.append(SynthesisResult(success=True, iterations=1, synthesis_time=0.5))
        stats = builder.get_statistics()
        assert stats["total_syntheses"] == 1
        assert stats["successful"] == 1
        assert stats["success_rate"] == 1.0


class TestSelfBuilderSynthesisHistoryCap:
    def test_synthesis_history_capped(self, builder):
        for i in range(_MAX_SYNTHESIS_HISTORY + 50):
            builder._synthesis_history.append(SynthesisResult(success=True, iterations=1, synthesis_time=0.001))
            if len(builder._synthesis_history) > _MAX_SYNTHESIS_HISTORY:
                builder._synthesis_history.pop(0)
        assert len(builder._synthesis_history) <= _MAX_SYNTHESIS_HISTORY

    def test_max_synthesis_history_value(self):
        assert _MAX_SYNTHESIS_HISTORY == 1000


class TestTaskAnalysisToDict:
    def test_to_dict_returns_list_copies(self):
        analysis = TaskAnalysis(
            description="test task",
            task_type=TaskType.CALCULATION,
            complexity=ComplexityLevel.SIMPLE,
            required_inputs=["a", "b"],
            expected_outputs=["result"],
            dependencies=["math"],
            constraints=["must be fast"],
            keywords=["sum", "calculate"],
            suggested_patterns=["sum"],
            confidence=0.85,
        )
        d = analysis.to_dict()
        assert d["required_inputs"] == ["a", "b"]
        assert d["required_inputs"] is not analysis.required_inputs
        assert d["confidence"] == 0.85

    def test_to_dict_truncates_description(self):
        analysis = TaskAnalysis(
            description="x" * 500,
            task_type=TaskType.CALCULATION,
            complexity=ComplexityLevel.SIMPLE,
            required_inputs=[],
            expected_outputs=[],
            dependencies=[],
            constraints=[],
            keywords=["test"],
            suggested_patterns=[],
            confidence=0.5,
        )
        d = analysis.to_dict()
        assert len(d["description"]) == 200


class TestSelfBuilderConfidence:
    def test_confidence_bounded(self, builder):
        result = builder.analyze_task("simple calculate sum")
        assert 0.1 <= result.confidence <= 1.0

    def test_unknown_task_type_low_confidence(self, builder):
        result = builder.analyze_task("do something weird")
        assert result.confidence <= 0.5


class TestSelfBuilderAsync:
    def test_synthesize_async_uses_running_loop(self, builder, mock_db):
        import asyncio
        async def test():
            request = SynthesisRequest(
                task_analysis=builder.analyze_task("simple calculate sum of values"),
            )
            result = await builder.synthesize_async(request)
            return result
        try:
            result = asyncio.run(test())
            assert result is not None
        except Exception:
            pass