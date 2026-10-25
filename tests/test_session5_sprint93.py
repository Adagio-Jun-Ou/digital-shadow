import asyncio
import copy
import pytest
import time
from unittest.mock import MagicMock
from shadowd.cognition.shadow_graph import (
    ShadowGraph, ShadowNode, ShadowEdge, CognitiveMode,
    CognitiveState, TransitionTrigger, ApprovalLevel, Checkpoint,
)
from shadowd.cognition.self_improvement import (
    SelfImprovementEngine, AccuracyCalibrator, StrategyEvolver,
    ConfidenceCalibrator, Prediction, DecisionRecord, OutcomeType, FeedbackSource,
)
from shadowd.cognition.daily_briefing import BriefingItem, BriefingSection, UrgencyLevel


class TestBug4HandlerErrorEscalation:
    def test_handler_exception_transitions_to_escalate(self):
        graph = ShadowGraph(graph_id="test_b4")
        def bad_handler(state):
            raise RuntimeError("boom")
        graph.register_handler("listen", bad_handler)
        graph.state.mode = CognitiveMode.LISTEN
        result = asyncio.run(graph.process_event(
            {"query": "test"},
            trigger=TransitionTrigger.EVENT_RECEIVED,
        ))
        assert result.mode == CognitiveMode.ESCALATE
        assert graph._stats["errors"] >= 1

    def test_handler_error_creates_checkpoint(self):
        graph = ShadowGraph(graph_id="test_b4b")
        def bad_handler(state):
            raise ValueError("fail")
        graph.register_handler("listen", bad_handler)
        graph.state.mode = CognitiveMode.LISTEN
        initial_ckpt_count = len(graph.checkpoints)
        asyncio.run(graph.process_event(
            {"query": "test"},
            trigger=TransitionTrigger.EVENT_RECEIVED,
        ))
        assert len(graph.checkpoints) > initial_ckpt_count

    def test_handler_error_calls_error_hooks(self):
        graph = ShadowGraph(graph_id="test_b4c")
        hook_called = []
        def error_hook(state, exc):
            hook_called.append(type(exc).__name__)
        def bad_handler(state):
            raise RuntimeError("boom")
        graph.register_handler("listen", bad_handler)
        graph.on_error(error_hook)
        graph.state.mode = CognitiveMode.LISTEN
        asyncio.run(graph.process_event(
            {"query": "test"},
            trigger=TransitionTrigger.EVENT_RECEIVED,
        ))
        assert "RuntimeError" in hook_called


class TestBug5DirectoryFsync:
    def test_persist_checkpoint_includes_dir_fsync(self, tmp_path):
        graph = ShadowGraph(graph_id="test_b5", data_dir=str(tmp_path))
        checkpoint = Checkpoint(
            checkpoint_id="ckpt_test",
            state={"mode": "listen"},
            node_name="listen",
            timestamp="2024-01-01T00:00:00Z",
            execution_id="exec_1",
            metadata={},
        )
        graph._persist_checkpoint(checkpoint)
        checkpoint_file = tmp_path / "test_b5_latest.json"
        assert checkpoint_file.exists()
        import json
        data = json.loads(checkpoint_file.read_text())
        assert data["checkpoint_id"] == "ckpt_test"


class TestBug7ModeToNodeName:
    def test_exact_name_match_preferred(self):
        graph = ShadowGraph(graph_id="test_b7")
        result = graph._mode_to_node_name(CognitiveMode.LISTEN)
        assert result == "listen"

    def test_fallback_to_mode_value(self):
        graph = ShadowGraph(graph_id="test_b7b")
        graph.nodes["custom"] = ShadowNode(name="custom", mode=CognitiveMode.ANALYZE)
        result = graph._mode_to_node_name(CognitiveMode.ANALYZE)
        assert result == "analyze"


class TestBug11ReadMethodLocks:
    def test_get_status_is_consistent(self):
        graph = ShadowGraph(graph_id="test_b11")
        result = graph.get_status()
        assert "graph_id" in result
        assert "current_mode" in result
        assert result["graph_id"] == "test_b11"

    def test_export_graph_state_is_thread_safe(self):
        graph = ShadowGraph(graph_id="test_b11b")
        result = graph.export_graph_state()
        assert "graph_id" in result
        assert "state" in result
        assert "nodes" in result

    def test_get_node_statistics_is_thread_safe(self):
        graph = ShadowGraph(graph_id="test_b11c")
        result = graph.get_node_statistics()
        assert isinstance(result, dict)
        assert "listen" in result

    def test_get_trace_summary_under_lock(self):
        graph = ShadowGraph(graph_id="test_b11d")
        result = graph.get_trace_summary()
        assert isinstance(result, list)

    def test_get_checkpoint_summary_under_lock(self):
        graph = ShadowGraph(graph_id="test_b11e")
        result = graph.get_checkpoint_summary()
        assert isinstance(result, list)


class TestBug12ApproveEscalation:
    def test_approve_by_id_only(self):
        graph = ShadowGraph(graph_id="test_b12")
        esc_id = "esc_abc123"
        graph.state.pending_escalations.append({
            "id": esc_id,
            "node": "escalate",
            "reason": "test",
        })
        result = asyncio.run(graph.approve_escalation(esc_id, True))
        assert result is True
        assert graph.state.mode == CognitiveMode.ACT

    def test_reject_by_node_name_no_longer_works(self):
        graph = ShadowGraph(graph_id="test_b12b")
        graph.state.pending_escalations.append({
            "id": "esc_unique_123",
            "node": "escalate",
            "reason": "test",
        })
        result = asyncio.run(graph.approve_escalation("escalate", True))
        assert result is False

    def test_approve_nonexistent_returns_false(self):
        graph = ShadowGraph(graph_id="test_b12c")
        result = asyncio.run(graph.approve_escalation("esc_nonexistent", True))
        assert result is False


class TestDeadCodeRemoval:
    def test_no_interrupted_attribute(self):
        graph = ShadowGraph(graph_id="test_dc")
        assert not hasattr(graph, "_interrupted")

    def test_no_approval_queue_attribute(self):
        graph = ShadowGraph(graph_id="test_dc2")
        assert not hasattr(graph, "_approval_queue")


class TestSyncLockThreadSafety:
    def test_add_node_under_lock(self):
        graph = ShadowGraph(graph_id="test_tl")
        graph.add_node(ShadowNode(name="custom_node", mode=CognitiveMode.DORMANT))
        assert "custom_node" in graph.nodes

    def test_add_edge_under_lock(self):
        graph = ShadowGraph(graph_id="test_tl2")
        initial_count = len(graph.edges)
        graph.add_edge("listen", "reflect", trigger=TransitionTrigger.SCHEDULED)
        assert len(graph.edges) == initial_count + 1

    def test_reset_under_lock(self):
        graph = ShadowGraph(graph_id="test_tl3")
        asyncio.run(graph.process_event({"query": "test"}, trigger=TransitionTrigger.EVENT_RECEIVED))
        assert graph._stats["transitions"] > 0
        graph.reset()
        assert graph._stats["transitions"] == 0

    def test_import_graph_state_under_lock(self):
        graph = ShadowGraph(graph_id="test_tl4")
        data = {
            "graph_id": "imported",
            "state": {"mode": "reason", "confidence": 0.8, "urgency": 0.3},
            "execution_counter": 42,
        }
        graph.import_graph_state(data)
        assert graph.graph_id == "imported"
        assert graph.state.mode == CognitiveMode.REASON


class TestSelfImprovementDeepAudit:
    def test_prediction_to_dict_metadata_deepcopy(self):
        p = Prediction(content="test", metadata={"key": "val", "nested": {"a": 1}})
        d = p.to_dict()
        assert d["metadata"] == {"key": "val", "nested": {"a": 1}}
        d["metadata"]["nested"]["a"] = 999
        assert p.metadata["nested"]["a"] == 1

    def test_decision_record_to_dict_metadata_deepcopy(self):
        d = DecisionRecord(action="test", metadata={"x": [1, 2]})
        r = d.to_dict()
        assert r["metadata"] == {"x": [1, 2]}
        r["metadata"]["x"].append(3)
        assert d.metadata["x"] == [1, 2]

    def test_decision_record_to_dict_feedback_deepcopy(self):
        d = DecisionRecord(action="test")
        d.add_feedback(FeedbackSource.EXPLICIT_USER, 0.8, "great")
        r = d.to_dict()
        assert len(r["feedback"]) == 1
        r["feedback"].append({"fake": True})
        assert len(d.feedback) == 1

    def test_record_decision_stats_under_lock(self):
        engine = SelfImprovementEngine(data_dir="/tmp/test_si_audit")
        d = engine.record_decision("test action")
        assert engine._stats["decisions_recorded"] == 1

    def test_get_stats_returns_copy(self):
        engine = SelfImprovementEngine(data_dir="/tmp/test_si_audit2")
        engine.record_decision("test")
        stats = engine.get_stats()
        stats["stats"]["decisions_recorded"] = 9999
        assert engine._stats["decisions_recorded"] != 9999

    def test_improvement_history_deepcopy(self):
        engine = SelfImprovementEngine(data_dir="/tmp/test_si_hist")
        asyncio.run(engine.run_improvement_cycle())
        history = engine.get_improvement_history()
        assert len(history) >= 1
        history[0]["cycle_number"] = 9999
        real = engine.get_improvement_history()
        assert real[0]["cycle_number"] != 9999

    def test_strategy_evolver_get_weight_under_lock(self):
        evolver = StrategyEvolver()
        evolver.record_strategy_outcome("test_strategy", 0.8)
        w = evolver.get_strategy_weight("test_strategy")
        assert w > 0

    def test_confidence_calibrator_get_stats_under_lock(self):
        cc = ConfidenceCalibrator()
        cc.record(0.7, True)
        s = cc.get_stats()
        assert s["total_records"] == 1

    def test_accuracy_calibrator_get_stats_under_lock(self):
        ac = AccuracyCalibrator()
        pred = Prediction(content="test", predicted_outcome="yes")
        ac.record_prediction(pred)
        s = ac.get_stats()
        assert s["total_predictions"] >= 1

    def test_resolve_prediction_sanitize(self):
        engine = SelfImprovementEngine(data_dir="/tmp/test_si_resolve")
        pred = engine.record_prediction("will it rain", "yes")
        result = engine.resolve_prediction(pred, "hello\x00world\x01bad")
        assert "\x00" not in (pred.actual_outcome or "")
        assert "\x01" not in (pred.actual_outcome or "")

    def test_resolve_decision_sanitize(self):
        engine = SelfImprovementEngine(data_dir="/tmp/test_si_rd")
        d = engine.record_decision("test")
        result = engine.resolve_decision(d.decision_id, "hello\x00world\x01bad")
        if result:
            assert "\x00" not in result.actual_outcome
            assert "\x01" not in result.actual_outcome


class TestDailyBriefingSanitization:
    def test_briefing_item_domain_sanitized(self):
        item = BriefingItem(
            section=BriefingSection.TODAY,
            title="Test",
            content="content",
            domain="<script>evil</script>",
        )
        assert "<script>" not in item.domain

    def test_briefing_item_action_url_sanitized(self):
        item = BriefingItem(
            section=BriefingSection.TODAY,
            title="Test",
            content="content",
            action_url="click\x00here\x01now",
        )
        assert "\x00" not in (item.action_url or "")
        assert "\x01" not in (item.action_url or "")


class TestAttentionEnginePriorityFix:
    def test_get_priorities_no_mutation_of_internal(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        engine = AttentionEngine()
        item_id = engine.add_attention_item("test topic", level="high", domain="test")
        with engine._lock:
            original_item = engine._attention_items[item_id]
            assert not hasattr(original_item, "_priority_score")
        priorities = engine.get_priorities()
        with engine._lock:
            assert not hasattr(engine._attention_items[item_id], "_priority_score")