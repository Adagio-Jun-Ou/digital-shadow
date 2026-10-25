import copy
import re
import threading
from unittest.mock import MagicMock, patch

import pytest

from shadowd.cognition.cognitive_pipeline import (
    CognitivePipeline,
    PipelineResult,
    _truncate_dict,
    _MAX_DICT_KEYS,
    _MAX_PIPELINE_RESULTS,
)
from shadowd.cognition.attention_engine import (
    AttentionEngine,
    AttentionItem,
    AttentionLevel,
    FocusState,
    DistractionType,
    AttentionProfile,
)


class TestCognitivePipelineRunningAverage:
    def test_running_average_stable_after_deque_wrap(self):
        pipe = CognitivePipeline()
        pipe._total_pipelines = 0
        for i in range(_MAX_PIPELINE_RESULTS + 200):
            with pipe._lock:
                pipe._results.append(PipelineResult(overall_confidence=0.7))
                pipe._stats["pipelines_run"] += 1
                pipe._total_pipelines += 1
                total = pipe._total_pipelines
                old_avg = pipe._stats["avg_confidence"]
                pipe._stats["avg_confidence"] = old_avg + (0.7 - old_avg) / total
        assert abs(pipe._stats["avg_confidence"] - 0.7) < 0.01

    def test_total_pipelines_increments_forever(self):
        pipe = CognitivePipeline()
        for _ in range(_MAX_PIPELINE_RESULTS + 100):
            pipe._total_pipelines += 1
            pipe._stats["pipelines_run"] += 1
        assert pipe._total_pipelines == _MAX_PIPELINE_RESULTS + 100
        assert pipe._stats["pipelines_run"] == _MAX_PIPELINE_RESULTS + 100

    def test_running_average_without_deque_wrap_diverges_old_way(self):
        pipe = CognitivePipeline()
        pipe._total_pipelines = 0
        for i in range(100):
            with pipe._lock:
                pipe._results.append(PipelineResult(overall_confidence=0.8))
                pipe._stats["pipelines_run"] += 1
                pipe._total_pipelines += 1
                total = pipe._total_pipelines
                old_avg = pipe._stats["avg_confidence"]
                pipe._stats["avg_confidence"] = old_avg + (0.8 - old_avg) / total
        assert abs(pipe._stats["avg_confidence"] - 0.8) < 0.05

    def test_counter_separate_from_deque_len(self):
        pipe = CognitivePipeline()
        pipe._total_pipelines = 5000
        assert pipe._total_pipelines == 5000
        assert len(pipe._results) <= _MAX_PIPELINE_RESULTS


class TestCognitivePipelineDictTruncation:
    def test_truncate_dict_returns_none(self):
        assert _truncate_dict(None) is None

    def test_truncate_dict_keeps_max_keys(self):
        d = {f"k{i}": i for i in range(100)}
        result = _truncate_dict(d)
        assert len(result) == _MAX_DICT_KEYS

    def test_truncate_dict_small_dict_unchanged(self):
        d = {"a": 1, "b": 2}
        result = _truncate_dict(d)
        assert result == d

    def test_pipeline_result_truncates_thought_analysis(self):
        big = {f"k{i}": i for i in range(100)}
        r = PipelineResult(thought_analysis=big)
        assert len(r.thought_analysis) == _MAX_DICT_KEYS

    def test_pipeline_result_truncates_emotion_state(self):
        big = {f"k{i}": i for i in range(100)}
        r = PipelineResult(emotion_state=big)
        assert len(r.emotion_state) == _MAX_DICT_KEYS

    def test_to_dict_truncates_all_dicts(self):
        big = {f"k{i}": i for i in range(100)}
        r = PipelineResult(
            thought_analysis=big,
            emotion_state=big,
            bias_detection=big,
            moral_evaluation=big,
        )
        d = r.to_dict()
        for key in ["thought_analysis", "emotion_state", "bias_detection", "moral_evaluation"]:
            assert d[key] is None or len(d[key]) <= _MAX_DICT_KEYS

    def test_process_truncates_subsystem_dicts(self):
        mock_ts = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {f"k{i}": i for i in range(100)}
        mock_ts.analyze.return_value = mock_result

        mock_ei = MagicMock()
        mock_ei_result = MagicMock()
        mock_ei_result.to_dict.return_value = {f"k{i}": i for i in range(80)}
        mock_ei.detect_emotion.return_value = mock_ei_result

        pipe = CognitivePipeline(thought_sense=mock_ts, emotional_intelligence=mock_ei)
        result = pipe.process("test input")
        assert len(result["thought_analysis"]) <= _MAX_DICT_KEYS
        assert len(result["emotion_state"]) <= _MAX_DICT_KEYS


class TestCognitivePipelineStats:
    def test_stats_include_recent_results(self):
        pipe = CognitivePipeline()
        mock_ts = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"intents": [], "urgency": "low", "cognitive_load": 50}
        mock_ts.analyze.return_value = mock_result

        pipe._thought_sense = mock_ts
        pipe.process("test1")
        pipe.process("test2")
        pipe.process("test3")

        stats = pipe.get_pipeline_stats()
        assert "recent_results" in stats
        assert len(stats["recent_results"]) == 3

    def test_recent_results_capped_at_five(self):
        pipe = CognitivePipeline()
        mock_ts = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"intents": [], "urgency": "low", "cognitive_load": 50}
        mock_ts.analyze.return_value = mock_result

        pipe._thought_sense = mock_ts
        for i in range(10):
            pipe.process(f"test{i}")

        stats = pipe.get_pipeline_stats()
        assert len(stats["recent_results"]) == 5

    def test_recent_results_empty_initially(self):
        pipe = CognitivePipeline()
        stats = pipe.get_pipeline_stats()
        assert stats["recent_results"] == []

    def test_recent_results_inside_lock(self):
        pipe = CognitivePipeline()
        stats = pipe.get_pipeline_stats()
        assert isinstance(stats["recent_results"], list)


class TestAttentionEngineDeepCopyPriorities:
    def test_get_priorities_returns_copies(self):
        ae = AttentionEngine()
        item_id = ae.add_attention_item(topic="test", level="high", urgency=0.9, importance=0.9)
        with ae._lock:
            original = ae._attention_items[item_id]
        priorities = ae.get_priorities(limit=1)
        assert priorities[0].item_id == original.item_id
        priorities[0].topic = "MUTATED"
        with ae._lock:
            assert ae._attention_items[item_id].topic == "test"

    def test_get_priorities_mutation_does_not_affect_internal(self):
        ae = AttentionEngine()
        ae.add_attention_item(topic="original", level="high", urgency=0.9, importance=0.9)
        priorities = ae.get_priorities(limit=1)
        priorities[0].urgency_score = 0.0
        priorities2 = ae.get_priorities(limit=1)
        assert priorities2[0].urgency_score == 0.9

    def test_priority_score_not_leaked(self):
        ae = AttentionEngine()
        ae.add_attention_item(topic="test", level="medium", urgency=0.5, importance=0.5)
        priorities = ae.get_priorities(limit=1)
        assert not hasattr(priorities[0], "_priority_score")

    def test_suggest_attention_action_uses_copies(self):
        ae = AttentionEngine()
        ae.add_attention_item(topic="task1", level="high", urgency=0.9, importance=0.8)
        result = ae.suggest_attention_action()
        if result["top_priority"]:
            result["top_priority"]["topic"] = "MUTATED"
        with ae._lock:
            items = list(ae._attention_items.values())
            assert all(i.topic != "MUTATED" for i in items)


class TestAttentionEngineEndFocusDeepCopy:
    def test_end_focus_returns_copy(self):
        ae = AttentionEngine()
        sid = ae.start_focus("deep work")
        session = ae.end_focus(sid)
        if session:
            session.topic = "MUTATED"
            with ae._lock:
                for s in ae._focus_sessions:
                    assert s.topic != "MUTATED"

    def test_end_focus_session_not_same_reference_as_internal(self):
        ae = AttentionEngine()
        sid = ae.start_focus("coding")
        session = ae.end_focus(sid)
        if session:
            with ae._lock:
                internal = list(ae._focus_sessions)[-1]
                assert session is not internal


class TestAttentionEngineMetadataCap:
    def test_metadata_capped_at_20_keys(self):
        ae = AttentionEngine()
        md = {f"key_{i}": f"val_{i}" for i in range(50)}
        item_id = ae.add_attention_item(topic="test", metadata=md)
        with ae._lock:
            assert len(ae._attention_items[item_id].metadata) <= 20

    def test_metadata_small_dict_unchanged(self):
        ae = AttentionEngine()
        md = {"a": "1", "b": "2"}
        item_id = ae.add_attention_item(topic="test", metadata=md)
        with ae._lock:
            assert len(ae._attention_items[item_id].metadata) == 2

    def test_metadata_none_handled(self):
        ae = AttentionEngine()
        item_id = ae.add_attention_item(topic="test", metadata=None)
        with ae._lock:
            assert ae._attention_items[item_id].metadata == {}


class TestAttentionEngineEvictionWarning:
    def test_eviction_logs_warning(self, caplog):
        ae = AttentionEngine()
        for i in range(ae._MAX_ATTENTION_ITEMS):
            ae.add_attention_item(topic=f"item_{i}", level="low", urgency=0.1, importance=0.1)
        with caplog.at_level("WARNING"):
            ae.add_attention_item(topic="overflow", level="low", urgency=0.1, importance=0.1)
            assert any("Evicting attention item" in r.message for r in caplog.records)


class TestAttentionEngineProfileLockSafety:
    def test_profile_reads_load_and_fatigue_inside_lock(self):
        ae = AttentionEngine()
        ae._cognitive_load = 0.75
        ae._fatigue_level = 0.6
        profile = ae.get_attention_profile()
        assert profile.cognitive_load == 0.75
        assert profile.fatigue_level == 0.6

    def test_profile_thread_safety(self):
        ae = AttentionEngine()
        ae.start_focus("concurrent")
        errors = []

        def reader():
            try:
                for _ in range(100):
                    ae.get_attention_profile()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100):
                    ae.recover_focus(1.0)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors


class TestCognitivePipelineProcessDictTruncation:
    def test_full_pipeline_truncates_all_analysis_dicts(self):
        mock_ts = MagicMock()
        mock_ts_result = MagicMock()
        mock_ts_result.to_dict.return_value = {f"tk{i}": i for i in range(80)}
        mock_ts.analyze.return_value = mock_ts_result

        mock_ei = MagicMock()
        mock_ei_result = MagicMock()
        mock_ei_result.to_dict.return_value = {f"ek{i}": i for i in range(80)}
        mock_ei.detect_emotion.return_value = mock_ei_result

        mock_meta = MagicMock()
        mock_meta.detect_biases.return_value = []

        mock_moral = MagicMock()
        mock_eval = MagicMock()
        mock_eval.to_dict.return_value = {f"mk{i}": i for i in range(80)}
        mock_moral.evaluate.return_value = mock_eval

        pipe = CognitivePipeline(
            thought_sense=mock_ts,
            emotional_intelligence=mock_ei,
            metacognition=mock_meta,
            moral_reasoning=mock_moral,
        )
        with patch.object(pipe, 'process', wraps=pipe.process):
            pass

        result = pipe.process("we should consider the ethical implications of this decision")
        for key in ["thought_analysis", "emotion_state", "moral_evaluation"]:
            if result[key] is not None:
                assert len(result[key]) <= _MAX_DICT_KEYS

    def test_pipeline_result_stored_with_truncated_dicts(self):
        big = {f"k{i}": i for i in range(80)}
        pipe = CognitivePipeline()
        with pipe._lock:
            r = PipelineResult(
                thought_analysis=big,
                emotion_state=big,
                bias_detection=big,
                moral_evaluation=big,
            )
            assert len(r.thought_analysis) <= _MAX_DICT_KEYS
            assert len(r.emotion_state) <= _MAX_DICT_KEYS