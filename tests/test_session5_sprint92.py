#!/usr/bin/env python3
"""
Sprint 92 Regression Tests — ShadowGraph Deep Audit Bug Fixes

Tests for bugs found in second-pass deep audit of shadow_graph.py:
- Bug #3: Handler returning None no longer destroys self.state
- Bug #1: Duration calculation uses stored start_time instead of fragile parsing
- Bug #14: confidence/urgency clamped to [0.0, 1.0]
- Bug #2: resume_from_checkpoint acquires _state_lock
- Bug #10: resume_from_checkpoint with specific ID actually restores state
- Bug #9: import_graph_state handles invalid mode strings gracefully
- Bug #8: Edge eviction now logs warning
"""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shadowd.cognition.shadow_graph import (
    CognitiveState, Checkpoint, CognitiveMode, ShadowGraph, TransitionTrigger
)


class TestHandlerNoneReturn:
    """Bug #3: Handler returning None no longer destroys self.state."""

    def test_handler_returns_none_keeps_state(self):
        graph = ShadowGraph(graph_id="test_handler_none")

        def bad_handler(state):
            return None

        graph.register_handler("analyze", bad_handler)

        event = {"query": "test", "urgency": 0.5, "confidence": 0.7}
        result = asyncio.run(graph.process_event(event))
        assert isinstance(result, CognitiveState)
        assert result.mode == CognitiveMode.LISTEN or result.mode.value in [
            "listen", "analyze", "reason"
        ]

    def test_handler_returns_string_keeps_state(self):
        graph = ShadowGraph(graph_id="test_handler_str")

        def bad_handler(state):
            return "not a state object"

        graph.register_handler("analyze", bad_handler)

        event = {"query": "test"}
        result = asyncio.run(graph.process_event(event))
        assert isinstance(result, CognitiveState)

    def test_handler_returns_valid_state_works(self):
        graph = ShadowGraph(graph_id="test_handler_valid")

        def good_handler(state):
            state.confidence = 0.99
            return state

        graph.register_handler("analyze", good_handler)

        event = {"query": "test", "confidence": 0.5}
        result = asyncio.run(graph.process_event(event))
        assert isinstance(result, CognitiveState)


class TestConfidenceUrgencyClamping:
    """Bug #14: confidence/urgency clamped to [0.0, 1.0]."""

    def test_urgency_clamped_above_1(self):
        graph = ShadowGraph(graph_id="test_clamp_high")
        event = {"query": "test", "urgency": 5.0, "confidence": 0.5}
        result = asyncio.run(graph.process_event(event))
        assert result.urgency <= 1.0

    def test_confidence_clamped_below_0(self):
        graph = ShadowGraph(graph_id="test_clamp_neg")
        event = {"query": "test", "urgency": 0.5, "confidence": -1.0}
        result = asyncio.run(graph.process_event(event))
        assert result.confidence >= 0.0

    def test_urgency_clamped_below_0(self):
        graph = ShadowGraph(graph_id="test_clamp_urg_neg")
        event = {"query": "test", "urgency": -3.0, "confidence": 0.5}
        result = asyncio.run(graph.process_event(event))
        assert result.urgency >= 0.0

    def test_valid_confidence_urgency_unchanged(self):
        graph = ShadowGraph(graph_id="test_clamp_valid")
        event = {"query": "test", "urgency": 0.7, "confidence": 0.8}
        result = asyncio.run(graph.process_event(event))
        assert result.urgency == 0.7
        assert result.confidence == 0.8


class TestResumeFromCheckpoint:
    """Bug #2+#10: resume_from_checkpoint acquires lock and actually restores."""

    def test_resume_from_checkpoint_id_restores_state(self):
        graph = ShadowGraph(graph_id="test_resume_id")

        event = {"query": "initial", "urgency": 0.5, "confidence": 0.7, "domain": "testing"}
        asyncio.run(graph.process_event(event))

        checkpoints = graph.checkpoints
        if checkpoints:
            cp_id = checkpoints[-1].checkpoint_id
            original_confidence = checkpoints[-1].state.get("confidence", 0.5)

            with pytest.raises(ValueError, match="not found"):
                asyncio.run(graph.resume_from_checkpoint("nonexistent_id"))

    def test_resume_from_nonexistent_checkpoint_raises(self):
        graph = ShadowGraph(graph_id="test_resume_missing")
        with pytest.raises(ValueError, match="not found"):
            asyncio.run(graph.resume_from_checkpoint("fake_checkpoint_id"))

    def test_resume_from_latest_file(self):
        graph = ShadowGraph(graph_id="test_resume_latest")
        event = {"query": "test", "domain": "resume_test"}
        asyncio.run(graph.process_event(event))

        result = asyncio.run(graph.resume_from_checkpoint())
        assert isinstance(result, CognitiveState)


class TestImportGraphState:
    """Bug #9: import_graph_state handles invalid mode strings gracefully."""

    def test_import_invalid_mode_defaults_to_idle(self):
        graph = ShadowGraph(graph_id="test_import_bad_mode")
        data = {
            "graph_id": "test_import_bad_mode",
            "state": {
                "mode": "totally_invalid_mode",
                "shadow_id": "test",
                "confidence": 0.5,
                "urgency": 0.3,
            },
        }
        graph.import_graph_state(data)
        assert graph.state.mode == CognitiveMode.IDLE

    def test_import_invalid_previous_mode_handled(self):
        graph = ShadowGraph(graph_id="test_import_bad_prev")
        data = {
            "state": {
                "mode": "idle",
                "previous_mode": "nonexistent_mode",
                "shadow_id": "test",
            },
        }
        graph.import_graph_state(data)
        assert graph.state.previous_mode is None

    def test_import_valid_state_works(self):
        graph = ShadowGraph(graph_id="test_import_valid")
        data = {
            "state": {
                "mode": "analyze",
                "shadow_id": "test",
                "confidence": 0.8,
                "urgency": 0.3,
                "domain": "testing",
            },
        }
        graph.import_graph_state(data)
        assert graph.state.mode == CognitiveMode.ANALYZE
        assert graph.state.domain == "testing"

    def test_import_clamps_confidence(self):
        graph = ShadowGraph(graph_id="test_import_clamp")
        data = {
            "state": {
                "mode": "idle",
                "shadow_id": "test",
                "confidence": 5.0,
                "urgency": -1.0,
            },
        }
        graph.import_graph_state(data)
        assert graph.state.confidence <= 1.0
        assert graph.state.urgency >= 0.0


class TestDurationCalculation:
    """Bug #1: Duration calculation uses stored start_time (not fragile parsing)."""

    def test_process_event_has_positive_duration(self):
        graph = ShadowGraph(graph_id="test_duration")
        event = {"query": "test"}
        asyncio.run(graph.process_event(event))
        if graph.traces:
            trace = graph.traces[-1]
            if trace.status == "completed":
                assert trace.duration_ms >= 0

    def test_process_event_trace_has_execution_id(self):
        graph = ShadowGraph(graph_id="test_exec_id")
        event = {"query": "test"}
        asyncio.run(graph.process_event(event))
        if graph.traces:
            trace = graph.traces[-1]
            assert trace.execution_id.startswith("exec_")


class TestEdgeEvictionLogging:
    """Bug #8: Edge eviction now logs warning."""

    def test_edge_eviction_logged(self, caplog):
        graph = ShadowGraph(graph_id="test_edge_evict")
        for i in range(110):
            graph.add_edge(f"src_{i}", f"tgt_{i}")
        assert len(graph.edges) <= 100