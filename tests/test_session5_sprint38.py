#!/usr/bin/env python3
"""Tests for Session 5 Sprint 38 — EmpathyMapping wiring + security audit fixes"""

import pytest
import threading
from unittest.mock import patch

from shadowd.cognition.empathy_mapping import (
    EmpathyMapping, RelationshipType, InteractionType,
    PersonProfile, CommunicationStyle, ReactionPrediction, RelationshipHealth,
)
from shadowd.cognition.temporal_reasoning import TemporalReasoning
from shadowd.cognition.recall_synthesis import RecallSynthesis, _extract_keywords
from shadowd.cognition.thought_sense import ThoughtSense, _MAX_TEXT_LENGTH
from shadowd.cognition.shadow_os import ShadowOS
from shadowd.cognition.api_router import (
    router, get_shadow_os, EmpathyPersonRequest, EmpathyInteractionRequest,
    EmpathyPredictRequest,
)


# ── EmpathyMapping Unit Tests ──


class TestEmpathyMappingBasics:
    def test_add_person(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice", "friend")
        assert pid is not None
        assert len(pid) == 12
        profile = em.get_person_profile(pid)
        assert profile.name == "Alice"
        assert profile.relationship == RelationshipType.FRIEND

    def test_add_person_default_relationship(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        profile = em.get_person_profile(pid)
        assert profile.relationship == RelationshipType.ACQUAINTANCE

    def test_add_person_with_metadata(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol", "colleague", {"team": "engineering"})
        profile = em.get_person_profile(pid)
        assert profile.metadata["team"] == "engineering"

    def test_add_person_empty_name_raises(self):
        em = EmpathyMapping()
        with pytest.raises(ValueError, match="name must not be empty"):
            em.add_person("   ")

    def test_add_person_invalid_relationship_raises(self):
        em = EmpathyMapping()
        with pytest.raises(ValueError, match="invalid relationship"):
            em.add_person("Dave", "enemy")

    def test_add_person_strips_control_chars(self):
        em = EmpathyMapping()
        pid = em.add_person("Eve\x00Name")
        profile = em.get_person_profile(pid)
        assert "\x00" not in profile.name

    def test_record_interaction(self):
        em = EmpathyMapping()
        pid = em.add_person("Frank", "friend")
        iid = em.record_interaction(pid, "call", 0.7, "Great call")
        assert iid is not None
        profile = em.get_person_profile(pid)
        assert profile.interaction_count == 1
        assert profile.avg_sentiment == pytest.approx(0.7, abs=0.01)

    def test_record_interaction_invalid_type_raises(self):
        em = EmpathyMapping()
        pid = em.add_person("Gina")
        with pytest.raises(ValueError, match="invalid interaction type"):
            em.record_interaction(pid, "telepathy", 0.5)

    def test_record_interaction_nonexistent_person_raises(self):
        em = EmpathyMapping()
        with pytest.raises(KeyError):
            em.record_interaction("nonexistent", "call", 0.5)

    def test_sentiment_clamped(self):
        em = EmpathyMapping()
        pid = em.add_person("Hank")
        em.record_interaction(pid, "text", 5.0)
        profile = em.get_person_profile(pid)
        assert profile.avg_sentiment == 1.0

    def test_notes_truncated_and_sanitized(self):
        em = EmpathyMapping()
        pid = em.add_person("Iris")
        long_notes = "a" * 3000 + "\x00bad"
        em.record_interaction(pid, "text", 0.5, long_notes)
        interactions = em._person_interactions[pid]
        assert len(interactions[0].notes) <= 2000
        assert "\x00" not in interactions[0].notes

    def test_metadata_values_truncated(self):
        em = EmpathyMapping()
        pid = em.add_person("Jack", "friend", {"note": "x" * 600})
        profile = em.get_person_profile(pid)
        assert len(profile.metadata["note"]) <= 500

    def test_get_communication_style(self):
        em = EmpathyMapping()
        pid = em.add_person("Karen", "manager")
        em.record_interaction(pid, "email", 0.5, "Status update")
        em.record_interaction(pid, "email", -0.2, "Re: deadline")
        style = em.get_communication_style(pid)
        assert style.person_id == pid
        assert isinstance(style, CommunicationStyle)

    def test_get_empathy_score_no_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Lou")
        score = em.get_empathy_score(pid)
        assert score == 0.0

    def test_get_empathy_score_with_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Mona", "friend")
        for i in range(10):
            em.record_interaction(pid, "text", 0.6, "chat")
        score = em.get_empathy_score(pid)
        assert score > 0

    def test_predict_reaction(self):
        em = EmpathyMapping()
        pid = em.add_person("Nate", "colleague")
        em.record_interaction(pid, "meeting", 0.5, "Weekly sync")
        pred = em.predict_reaction(pid, "project update", "direct")
        assert pred.person_id == pid
        assert -1.0 <= pred.predicted_sentiment <= 1.0
        assert 0.0 <= pred.predicted_receptiveness <= 1.0
        assert isinstance(pred, ReactionPrediction)

    def test_get_relationship_health(self):
        em = EmpathyMapping()
        pid = em.add_person("Olivia", "friend")
        em.record_interaction(pid, "call", 0.8, "Good chat")
        health = em.get_relationship_health(pid)
        assert isinstance(health, RelationshipHealth)
        assert health.health_score >= 0

    def test_get_social_graph(self):
        em = EmpathyMapping()
        p1 = em.add_person("P1", "friend")
        p2 = em.add_person("P2", "family")
        graph = em.get_social_graph()
        assert "nodes" in graph
        assert "edges" in graph
        assert graph["total_persons"] == 2

    def test_get_stats(self):
        em = EmpathyMapping()
        pid = em.add_person("Quinn", "friend")
        em.record_interaction(pid, "call", 0.5, "Hi")
        stats = em.get_stats()
        assert "total_persons" in stats
        assert "total_interactions" in stats
        assert stats["total_persons"] == 1

    def test_person_cap_eviction(self):
        em = EmpathyMapping()
        em._MAX_PERSONS = 3
        ids = []
        for i in range(5):
            ids.append(em.add_person(f"Person{i}"))
        assert len(em._persons) == 3


# ── ShadowOS Wiring Tests ──


class TestShadowOSWiring:
    def test_empathy_attribute_exists(self):
        with patch.object(ShadowOS, '__init__', lambda self, **kw: None):
            pass
        os_instance = object.__new__(ShadowOS)
        os_instance.empathy = EmpathyMapping()
        assert hasattr(os_instance, 'empathy')
        assert isinstance(os_instance.empathy, EmpathyMapping)

    def test_add_person_delegates_to_empathy(self):
        with patch.object(ShadowOS, '__init__', lambda self, **kw: None):
            pass
        os_instance = object.__new__(ShadowOS)
        os_instance.empathy = EmpathyMapping()
        pid = os_instance.add_person("TestPerson", "friend")
        assert pid is not None
        profile = os_instance.empathy.get_person_profile(pid)
        assert profile.name == "TestPerson"

    def test_get_relationship_health_delegates(self):
        with patch.object(ShadowOS, '__init__', lambda self, **kw: None):
            pass
        os_instance = object.__new__(ShadowOS)
        os_instance.empathy = EmpathyMapping()
        pid = os_instance.add_person("HealthCheck", "friend")
        os_instance.empathy.record_interaction(pid, "call", 0.8, "Good")
        result = os_instance.get_relationship_health(pid)
        assert isinstance(result, dict)
        assert "health_score" in result
        assert "trend" in result

    def test_get_status_includes_empathy(self):
        os_instance = ShadowOS()
        status = os_instance.get_status()
        assert "empathy_stats" in status

    def test_get_health_includes_empathy(self):
        os_instance = ShadowOS()
        os_instance._running = True
        os_instance._start_time = 0
        health = os_instance.get_health()
        assert "empathy_mapping" in health["modules"]


# ── API Router Model Tests ──


class TestAPIRouterModels:
    def test_empathy_person_request_validation(self):
        req = EmpathyPersonRequest(name="Alice", relationship="friend")
        assert req.name == "Alice"
        assert req.relationship == "friend"
        assert req.metadata is None

    def test_empathy_person_request_default_relationship(self):
        req = EmpathyPersonRequest(name="Bob")
        assert req.relationship == "acquaintance"

    def test_empathy_interaction_request_validation(self):
        req = EmpathyInteractionRequest(
            person_id="abc123",
            interaction_type="call",
            sentiment=0.7,
            notes="Good call",
        )
        assert req.person_id == "abc123"
        assert req.sentiment == 0.7

    def test_empathy_interaction_sentiment_bounds(self):
        with pytest.raises(Exception):
            EmpathyInteractionRequest(
                person_id="x",
                interaction_type="call",
                sentiment=5.0,
            )

    def test_empathy_predict_request(self):
        req = EmpathyPredictRequest(
            person_id="abc",
            topic="project",
            approach="direct",
        )
        assert req.approach == "direct"


# ── Security Audit: TemporalReasoning ──


class TestTemporalReasoningSecurity:
    def test_event_type_sanitized(self):
        tr = TemporalReasoning()
        eid = tr.record_event("test\x00event")
        events = list(tr._events)
        assert events[-1]["event_type"] == "testevent"

    def test_event_type_truncated(self):
        tr = TemporalReasoning()
        long_type = "a" * 300
        eid = tr.record_event(long_type)
        events = list(tr._events)
        assert len(events[-1]["event_type"]) <= 200

    def test_event_type_empty_raises(self):
        tr = TemporalReasoning()
        with pytest.raises(ValueError, match="event_type must not be empty"):
            tr.record_event("   ")

    def test_metadata_truncated(self):
        tr = TemporalReasoning()
        eid = tr.record_event("test", metadata={"key" * 50: "val" * 200})
        events = list(tr._events)
        meta = events[-1]["metadata"]
        for k, v in meta.items():
            assert len(k) <= 100
            assert len(v) <= 500

    def test_events_capped(self):
        tr = TemporalReasoning()
        assert tr._MAX_EVENTS == 10000


# ── Security Audit: RecallSynthesis ──


class TestRecallSynthesisSecurity:
    def test_text_sanitized(self):
        rs = RecallSynthesis()
        obs_id = rs.add_observation("hello\x00world")
        obs = rs._observations[obs_id]
        assert "\x00" not in obs.text

    def test_text_truncated(self):
        rs = RecallSynthesis()
        long_text = "a" * 15000
        obs_id = rs.add_observation(long_text)
        obs = rs._observations[obs_id]
        assert len(obs.text) <= rs._MAX_TEXT_LENGTH

    def test_domain_truncated(self):
        rs = RecallSynthesis()
        long_domain = "d" * 100
        obs_id = rs.add_observation("test", domain=long_domain)
        obs = rs._observations[obs_id]
        assert obs.domain is None or len(obs.domain) <= rs._MAX_DOMAIN_LENGTH

    def test_importance_clamped(self):
        rs = RecallSynthesis()
        obs_id = rs.add_observation("test", importance=5.0)
        obs = rs._observations[obs_id]
        assert obs.importance == 1.0

    def test_observations_capped(self):
        rs = RecallSynthesis()
        assert rs._MAX_OBSERVATIONS == 5000

    def test_double_break_fix(self):
        rs = RecallSynthesis()
        rs.add_observation("python coding project", domain="career")
        rs.add_observation("python meditation practice", domain="health")
        result = rs.connect_concepts("python", "project")
        assert result is not None
        assert isinstance(result.confidence, float)


# ── Security Audit: ThoughtSense ──


class TestThoughtSenseSecurity:
    def test_text_truncation(self):
        ts = ThoughtSense()
        long_text = "I am happy. " * 2000
        result = ts.analyze(long_text)
        assert result is not None

    def test_max_text_length_constant(self):
        assert _MAX_TEXT_LENGTH == 10000

    def test_empty_text_handled(self):
        ts = ThoughtSense()
        result = ts.analyze("")
        assert result is not None
        assert result.summary != ""

    def test_thread_safety(self):
        ts = ThoughtSense()
        results = []

        def analyze(text):
            r = ts.analyze(text)
            results.append(r)

        threads = [threading.Thread(target=analyze, args=(f"Thread text {i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10


# ── Integration: EmpathyMapping + ShadowOS ──


class TestEmpathyIntegration:
    def test_full_empathy_workflow(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice", "friend", {"note": "Best friend"})
        em.record_interaction(pid, "call", 0.8, "Great conversation about projects")
        em.record_interaction(pid, "text", 0.6, "Quick check-in")
        em.record_interaction(pid, "meeting", -0.2, "Minor disagreement on timeline")

        profile = em.get_person_profile(pid)
        assert profile.interaction_count == 3
        assert profile.avg_sentiment > 0

        style = em.get_communication_style(pid)
        assert style.person_id == pid
        assert len(style.dos) > 0

        score = em.get_empathy_score(pid)
        assert score > 0

        health = em.get_relationship_health(pid)
        assert health.health_score > 0
        assert health.trend in ("improving", "stable", "declining")

        pred = em.predict_reaction(pid, "new project proposal", "supportive")
        assert pred.person_id == pid
        assert len(pred.alternative_approaches) > 0

    def test_social_graph_with_multiple_people(self):
        em = EmpathyMapping()
        p1 = em.add_person("Alice", "friend")
        p2 = em.add_person("Bob", "family")
        p3 = em.add_person("Carol", "colleague")
        em.record_interaction(p1, "call", 0.7, "Chat")
        em.record_interaction(p2, "text", 0.9, "Love you")
        em.record_interaction(p3, "email", 0.3, "Update")
        graph = em.get_social_graph()
        assert graph["total_persons"] == 3
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) > 0

    def test_shadow_os_get_status_empathy(self):
        with patch.object(ShadowOS, '__init__', lambda self, **kw: None):
            pass
        os_instance = object.__new__(ShadowOS)
        os_instance.empathy = EmpathyMapping()
        os_instance._stats = {"events_processed": 0, "queries_handled": 0, "actions_taken": 0,
                              "insights_generated": 0, "errors": 0, "uptime_seconds": 0}
        status = {"empathy_stats": os_instance.empathy.get_stats(),
                  "stats": os_instance._stats}
        assert "total_persons" in status["empathy_stats"]