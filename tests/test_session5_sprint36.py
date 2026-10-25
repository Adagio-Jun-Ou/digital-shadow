#!/usr/bin/env python3
"""
Integration tests for Sprint 36 — wiring ThoughtSense, RecallSynthesis,
and TemporalReasoning into ShadowOS.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

from shadowd.cognition.shadow_os import ShadowOS
from shadowd.cognition.thought_sense import ThoughtSense, ThoughtAnalysis, Intent, UrgencyLevel
from shadowd.cognition.recall_synthesis import RecallSynthesis, Observation, SynthesisResult
from shadowd.cognition.temporal_reasoning import TemporalReasoning, TemporalPattern, Prediction, PatternType, TrendDirection


# ── ShadowOS initialization ──────────────────────────────

class TestShadowOSNewModules:
    def test_thought_sense_initialized(self):
        os = ShadowOS()
        assert hasattr(os, 'thought_sense')
        assert isinstance(os.thought_sense, ThoughtSense)

    def test_recall_synthesis_initialized(self):
        os = ShadowOS()
        assert hasattr(os, 'recall_synthesis')
        assert isinstance(os.recall_synthesis, RecallSynthesis)

    def test_temporal_reasoning_initialized(self):
        os = ShadowOS()
        assert hasattr(os, 'temporal_reasoning')
        assert isinstance(os.temporal_reasoning, TemporalReasoning)


# ── ShadowOS get_status includes new modules ─────────────

class TestShadowOSStatusHealth:
    def test_status_includes_thought_sense(self):
        os = ShadowOS()
        status = os.get_status()
        assert "thought_sense_stats" in status
        assert isinstance(status["thought_sense_stats"], dict)

    def test_status_includes_recall_synthesis(self):
        os = ShadowOS()
        status = os.get_status()
        assert "recall_synthesis_stats" in status
        assert isinstance(status["recall_synthesis_stats"], dict)

    def test_status_includes_temporal_reasoning(self):
        os = ShadowOS()
        status = os.get_status()
        assert "temporal_reasoning_stats" in status
        assert isinstance(status["temporal_reasoning_stats"], dict)

    def test_health_includes_thought_sense(self):
        os = ShadowOS()
        os._running = True
        os._start_time = 1234567890.0
        health = os.get_health()
        assert "thought_sense" in health["modules"]
        assert health["modules"]["thought_sense"]["status"] == "healthy"

    def test_health_includes_recall_synthesis(self):
        os = ShadowOS()
        os._running = True
        os._start_time = 1234567890.0
        health = os.get_health()
        assert "recall_synthesis" in health["modules"]
        assert health["modules"]["recall_synthesis"]["status"] == "healthy"

    def test_health_includes_temporal_reasoning(self):
        os = ShadowOS()
        os._running = True
        os._start_time = 1234567890.0
        health = os.get_health()
        assert "temporal_reasoning" in health["modules"]
        assert health["modules"]["temporal_reasoning"]["status"] == "healthy"


# ── ShadowOS analyze_communication ──────────────────────

class TestAnalyzeCommunication:
    def test_analyze_communication_basic(self):
        os = ShadowOS()
        result = asyncio.run(
            os.analyze_communication("I urgently need help with this project!")
        )
        assert "thought_analysis" in result
        assert "recall_synthesis" in result
        assert "intents" in result["thought_analysis"]
        assert "urgency" in result["thought_analysis"]

    def test_analyze_communication_with_domain(self):
        os = ShadowOS()
        result = asyncio.run(
            os.analyze_communication("Meeting at 3pm", domain="career")
        )
        assert "thought_analysis" in result
        assert "recall_synthesis" in result

    def test_analyze_communication_sanitizes_input(self):
        os = ShadowOS()
        result = asyncio.run(
            os.analyze_communication("<script>alert('xss')</script>Hello")
        )
        assert "thought_analysis" in result
        assert "<script>" not in result["thought_analysis"]["text"]


# ── ShadowOS get_temporal_patterns ───────────────────────

class TestTemporalIntegration:
    def test_get_temporal_patterns_empty(self):
        os = ShadowOS()
        patterns = os.get_temporal_patterns()
        assert isinstance(patterns, list)

    def test_get_temporal_patterns_with_type(self):
        os = ShadowOS()
        os.temporal_reasoning.record_event("test_event")
        patterns = os.get_temporal_patterns(event_type="test_event")
        assert isinstance(patterns, list)

    def test_predict_next_event(self):
        os = ShadowOS()
        predictions = os.predict_next_event("nonexistent_event")
        assert isinstance(predictions, list)


# ── API request models ───────────────────────────────────

class TestAPIRequestModels:
    def test_thought_analyze_request_valid(self):
        from shadowd.cognition.api_router import ThoughtAnalyzeRequest
        req = ThoughtAnalyzeRequest(text="Hello world")
        assert req.text == "Hello world"

    def test_thought_analyze_request_max_length(self):
        from shadowd.cognition.api_router import ThoughtAnalyzeRequest
        with pytest.raises(Exception):
            ThoughtAnalyzeRequest(text="x" * 10001)

    def test_recall_observe_request_defaults(self):
        from shadowd.cognition.api_router import RecallObserveRequest
        req = RecallObserveRequest(text="Observation text")
        assert req.domain is None
        assert req.importance == 0.5

    def test_recall_observe_request_importance_clamped(self):
        from shadowd.cognition.api_router import RecallObserveRequest
        with pytest.raises(Exception):
            RecallObserveRequest(text="test", importance=1.5)
        with pytest.raises(Exception):
            RecallObserveRequest(text="test", importance=-0.1)

    def test_recall_synthesize_request(self):
        from shadowd.cognition.api_router import RecallSynthesizeRequest
        req = RecallSynthesizeRequest(query="meeting")
        assert req.query == "meeting"
        assert req.domain is None

    def test_temporal_record_request(self):
        from shadowd.cognition.api_router import TemporalRecordRequest
        req = TemporalRecordRequest(event_type="standup")
        assert req.event_type == "standup"
        assert req.metadata is None

    def test_temporal_record_request_with_metadata(self):
        from shadowd.cognition.api_router import TemporalRecordRequest
        req = TemporalRecordRequest(event_type="standup", metadata={"key": "val"})
        assert req.metadata == {"key": "val"}

    def test_recall_observe_request_text_max_length(self):
        from shadowd.cognition.api_router import RecallObserveRequest
        with pytest.raises(Exception):
            RecallObserveRequest(text="x" * 5001)

    def test_temporal_record_request_event_type_max_length(self):
        from shadowd.cognition.api_router import TemporalRecordRequest
        with pytest.raises(Exception):
            TemporalRecordRequest(event_type="x" * 101)


# ── API endpoint functions ────────────────────────────────

class TestAPIEndpoints:
    def _get_os(self):
        import shadowd.cognition.api_router as router_mod
        router_mod._shadow_os = None
        return router_mod.get_shadow_os()

    def test_thought_stats_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import thought_stats
        router_mod._shadow_os = None
        result = asyncio.run(thought_stats())
        assert "total_analyses" in result

    def test_thought_analyze_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import thought_analyze, ThoughtAnalyzeRequest
        router_mod._shadow_os = None
        req = ThoughtAnalyzeRequest(text="Can you help me with this?")
        result = asyncio.run(thought_analyze(req))
        assert "intents" in result
        assert "emotions" in result

    def test_recall_observe_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import recall_observe, RecallObserveRequest
        router_mod._shadow_os = None
        req = RecallObserveRequest(text="Had a productive meeting", domain="career", importance=0.7)
        result = asyncio.run(recall_observe(req))
        assert "observation_id" in result

    def test_recall_synthesize_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import recall_observe, recall_synthesize, RecallObserveRequest, RecallSynthesizeRequest
        router_mod._shadow_os = None
        obs_req = RecallObserveRequest(text="Deployed the new feature", domain="career", importance=0.8)
        asyncio.run(recall_observe(obs_req))
        syn_req = RecallSynthesizeRequest(query="feature deployment", domain="career")
        result = asyncio.run(recall_synthesize(syn_req))
        assert "query" in result
        assert "confidence" in result

    def test_recall_patterns_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import recall_patterns
        router_mod._shadow_os = None
        result = asyncio.run(recall_patterns())
        assert "patterns" in result

    def test_recall_insights_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import recall_insights
        router_mod._shadow_os = None
        result = asyncio.run(recall_insights())
        assert "insights" in result

    def test_temporal_record_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import temporal_record, TemporalRecordRequest
        router_mod._shadow_os = None
        req = TemporalRecordRequest(event_type="standup", metadata={"location": "zoom"})
        result = asyncio.run(temporal_record(req))
        assert "event_id" in result

    def test_temporal_patterns_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import temporal_patterns
        router_mod._shadow_os = None
        result = asyncio.run(temporal_patterns())
        assert "patterns" in result

    def test_temporal_predict_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import temporal_predict
        router_mod._shadow_os = None
        result = asyncio.run(temporal_predict("standup", within_hours=48))
        assert "predictions" in result

    def test_temporal_trend_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import temporal_trend
        router_mod._shadow_os = None
        result = asyncio.run(temporal_trend("standup", window_days=30))
        assert "event_type" in result
        assert "direction" in result

    def test_temporal_stats_endpoint(self):
        import shadowd.cognition.api_router as router_mod
        from shadowd.cognition.api_router import temporal_stats
        router_mod._shadow_os = None
        result = asyncio.run(temporal_stats())
        assert "events_recorded" in result


# ── Module-level functionality ───────────────────────────

class TestThoughtSenseIntegration:
    def test_analyze_returns_thought_analysis(self):
        ts = ThoughtSense()
        result = ts.analyze("I need this done ASAP! This is urgent!")
        assert isinstance(result, ThoughtAnalysis)
        assert result.urgency in (UrgencyLevel.HIGH.value, UrgencyLevel.CRITICAL.value)
        assert len(result.intents) > 0

    def test_analyze_empty_text(self):
        ts = ThoughtSense()
        result = ts.analyze("")
        assert isinstance(result, ThoughtAnalysis)
        assert result.intents == []

    def test_get_stats(self):
        ts = ThoughtSense()
        ts.analyze("Hello world")
        stats = ts.get_stats()
        assert "total_analyses" in stats
        assert stats["total_analyses"] == 1


class TestRecallSynthesisIntegration:
    def test_add_observation_and_synthesize(self):
        rs = RecallSynthesis()
        obs_id = rs.add_observation(text="Standup meeting every Monday", domain="career", importance=0.8)
        assert obs_id is not None
        result = rs.synthesize(query="meeting", domain="career")
        assert isinstance(result, SynthesisResult)
        assert result.query == "meeting"

    def test_find_cross_domain_patterns_insufficient_data(self):
        rs = RecallSynthesis()
        rs.add_observation(text="Standup meeting", domain="career", importance=0.7)
        patterns = rs.find_cross_domain_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) == 0

    def test_get_insights(self):
        rs = RecallSynthesis()
        rs.add_observation(text="Some observation", domain="health")
        insights = rs.get_insights()
        assert isinstance(insights, list)

    def test_get_stats(self):
        rs = RecallSynthesis()
        rs.add_observation(text="Test observation", domain="career")
        stats = rs.get_stats()
        assert "observation_count" in stats
        assert stats["observation_count"] == 1


class TestTemporalReasoningIntegration:
    def test_record_event(self):
        tr = TemporalReasoning()
        event_id = tr.record_event("test_event")
        assert event_id.startswith("tevt_")

    def test_detect_patterns_insufficient_data(self):
        tr = TemporalReasoning()
        tr.record_event("test_event")
        patterns = tr.detect_patterns("test_event")
        assert patterns == []

    def test_predict_next_empty(self):
        tr = TemporalReasoning()
        predictions = tr.predict_next("nonexistent")
        assert isinstance(predictions, list)

    def test_analyze_trend_empty(self):
        tr = TemporalReasoning()
        trend = tr.analyze_trend("nonexistent")
        assert trend.direction == TrendDirection.UNKNOWN
        assert trend.confidence == 0.0

    def test_get_stats(self):
        tr = TemporalReasoning()
        tr.record_event("test_event")
        stats = tr.get_stats()
        assert "events_recorded" in stats
        assert stats["events_recorded"] == 1