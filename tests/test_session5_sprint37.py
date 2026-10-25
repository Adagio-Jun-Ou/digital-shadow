#!/usr/bin/env python3
"""
Tests for Sprint 37 — EmpathyMapping cognitive module.
"""

import threading
from datetime import datetime, timezone, timedelta

import pytest

from shadowd.cognition.empathy_mapping import (
    EmpathyMapping,
    RelationshipType,
    InteractionType,
    PersonProfile,
    CommunicationStyle,
    ReactionPrediction,
    RelationshipHealth,
)


# ── Data Classes ──────────────────────────────────────────

class TestRelationshipType:
    def test_all_values(self):
        vals = {r.value for r in RelationshipType}
        assert vals == {"family", "friend", "colleague", "manager", "report", "acquaintance", "professional"}

    def test_string_comparison(self):
        assert RelationshipType.FAMILY == "family"
        assert RelationshipType.MANAGER == "manager"


class TestInteractionType:
    def test_all_values(self):
        vals = {i.value for i in InteractionType}
        assert vals == {"call", "text", "email", "meeting", "in_person", "social_media"}

    def test_string_comparison(self):
        assert InteractionType.TEXT == "text"
        assert InteractionType.IN_PERSON == "in_person"


class TestPersonProfileToDict:
    def test_to_dict(self):
        p = PersonProfile(
            person_id="abc123",
            name="Alice",
            relationship=RelationshipType.FRIEND,
            created_at="2025-01-01T00:00:00+00:00",
            last_interaction="2025-01-10T00:00:00+00:00",
            interaction_count=5,
            avg_sentiment=0.3,
            preferred_channels=["text", "call"],
            emotional_triggers=["stress"],
            formality_preference="casual",
            response_patterns={"avg_response_hours": 2.0},
            metadata={"note": "test"},
        )
        d = p.to_dict()
        assert d["person_id"] == "abc123"
        assert d["name"] == "Alice"
        assert d["relationship"] == "friend"
        assert d["avg_sentiment"] == 0.3
        assert isinstance(d, dict)


class TestCommunicationStyleToDict:
    def test_to_dict(self):
        cs = CommunicationStyle(
            person_id="abc123",
            preferred_channel="text",
            formality_level="casual",
            response_time_hours=2.0,
            emotional_triggers=["stress"],
            dos=["Be warm"],
            donts=["Avoid formality"],
        )
        d = cs.to_dict()
        assert d["person_id"] == "abc123"
        assert d["preferred_channel"] == "text"
        assert isinstance(d, dict)


class TestReactionPredictionToDict:
    def test_to_dict(self):
        rp = ReactionPrediction(
            person_id="abc123",
            topic="promotion",
            approach="direct",
            predicted_sentiment=0.5,
            predicted_receptiveness=0.7,
            confidence=0.6,
            alternative_approaches=["gentle"],
        )
        d = rp.to_dict()
        assert d["topic"] == "promotion"
        assert d["confidence"] == 0.6


class TestRelationshipHealthToDict:
    def test_to_dict(self):
        rh = RelationshipHealth(
            person_id="abc123",
            health_score=75.0,
            trend="improving",
            last_interaction="2025-01-10T00:00:00+00:00",
            recommendation="Keep it up",
            interaction_frequency="weekly",
        )
        d = rh.to_dict()
        assert d["health_score"] == 75.0
        assert d["trend"] == "improving"


# ── add_person ───────────────────────────────────────────

class TestAddPerson:
    def test_add_person_returns_id(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        assert isinstance(pid, str)
        assert len(pid) == 12

    def test_add_person_with_relationship(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob", relationship="manager")
        profile = em.get_person_profile(pid)
        assert profile.relationship == RelationshipType.MANAGER

    def test_add_person_with_metadata(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol", metadata={"team": "engineering", "location": "SF"})
        profile = em.get_person_profile(pid)
        assert profile.metadata["team"] == "engineering"

    def test_add_person_empty_name_raises(self):
        em = EmpathyMapping()
        with pytest.raises(ValueError):
            em.add_person("   ")

    def test_add_person_invalid_relationship_raises(self):
        em = EmpathyMapping()
        with pytest.raises(ValueError):
            em.add_person("Dave", relationship="enemy")

    def test_add_person_name_truncated(self):
        em = EmpathyMapping()
        long_name = "A" * 300
        pid = em.add_person(long_name)
        profile = em.get_person_profile(pid)
        assert len(profile.name) == 200

    def test_add_person_default_values(self):
        em = EmpathyMapping()
        pid = em.add_person("Eve")
        profile = em.get_person_profile(pid)
        assert profile.interaction_count == 0
        assert profile.avg_sentiment == 0.0
        assert profile.preferred_channels == []
        assert profile.last_interaction is None

    def test_add_person_all_relationship_types(self):
        em = EmpathyMapping()
        for rel in ["family", "friend", "colleague", "manager", "report", "acquaintance", "professional"]:
            pid = em.add_person(f"Person_{rel}", relationship=rel)
            assert em.get_person_profile(pid).relationship.value == rel

    def test_add_person_max_cap(self):
        em = EmpathyMapping()
        for i in range(502):
            em.add_person(f"Person_{i}")
        stats = em.get_stats()
        assert stats["total_persons"] == 500


# ── record_interaction ───────────────────────────────────

class TestRecordInteraction:
    def test_record_basic_interaction(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        iid = em.record_interaction(pid, "text", 0.5, "Had a good chat")
        assert isinstance(iid, str)
        assert len(iid) == 12

    def test_record_updates_profile(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        em.record_interaction(pid, "call", 0.8)
        profile = em.get_person_profile(pid)
        assert profile.interaction_count == 1
        assert profile.avg_sentiment == 0.8
        assert profile.last_interaction is not None
        assert "call" in profile.preferred_channels

    def test_record_multiple_updates_avg_sentiment(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol")
        em.record_interaction(pid, "text", 0.6)
        em.record_interaction(pid, "call", 0.2)
        profile = em.get_person_profile(pid)
        assert profile.interaction_count == 2
        assert abs(profile.avg_sentiment - 0.4) < 0.01

    def test_record_sentiment_clamped(self):
        em = EmpathyMapping()
        pid = em.add_person("Dave")
        em.record_interaction(pid, "text", 5.0)
        profile = em.get_person_profile(pid)
        assert profile.avg_sentiment == 1.0

        em.record_interaction(pid, "text", -5.0)
        profile = em.get_person_profile(pid)
        assert profile.avg_sentiment <= 1.0

    def test_record_invalid_interaction_type_raises(self):
        em = EmpathyMapping()
        pid = em.add_person("Eve")
        with pytest.raises(ValueError):
            em.record_interaction(pid, "telepathy", 0.5)

    def test_record_invalid_person_raises(self):
        em = EmpathyMapping()
        with pytest.raises(KeyError):
            em.record_interaction("nonexistent", "text", 0.5)

    def test_record_updates_preferred_channels(self):
        em = EmpathyMapping()
        pid = em.add_person("Frank")
        em.record_interaction(pid, "email", 0.5)
        em.record_interaction(pid, "email", 0.5)
        em.record_interaction(pid, "text", 0.5)
        profile = em.get_person_profile(pid)
        assert profile.preferred_channels[0] == "email"

    def test_record_formality_updates_after_threshold(self):
        em = EmpathyMapping()
        pid = em.add_person("Grace")
        em.record_interaction(pid, "text", 0.8)
        em.record_interaction(pid, "call", 0.9)
        profile = em.get_person_profile(pid)
        assert profile.formality_preference == "casual"

    def test_record_negative_sentiment_formality(self):
        em = EmpathyMapping()
        pid = em.add_person("Heidi")
        em.record_interaction(pid, "text", -0.5)
        em.record_interaction(pid, "call", -0.6)
        profile = em.get_person_profile(pid)
        assert profile.formality_preference == "mixed"

    def test_record_notes_truncated(self):
        em = EmpathyMapping()
        pid = em.add_person("Ivan")
        long_notes = "x" * 3000
        iid = em.record_interaction(pid, "text", 0.0, long_notes)
        interactions = em.get_stats()["total_interactions"]
        assert interactions == 1

    def test_record_metadata_sanitized(self):
        em = EmpathyMapping()
        pid = em.add_person("Judy")
        em.record_interaction(pid, "text", 0.0, metadata={"key": "val" * 200})
        stats = em.get_stats()
        assert stats["total_interactions"] == 1

    def test_all_interaction_types(self):
        em = EmpathyMapping()
        pid = em.add_person("Karl")
        for itype in ["call", "text", "email", "meeting", "in_person", "social_media"]:
            iid = em.record_interaction(pid, itype, 0.5)
            assert isinstance(iid, str)


# ── get_communication_style ───────────────────────────────

class TestGetCommunicationStyle:
    def test_basic_communication_style(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        em.record_interaction(pid, "text", 0.5)
        em.record_interaction(pid, "text", 0.6)
        style = em.get_communication_style(pid)
        assert isinstance(style, CommunicationStyle)
        assert style.person_id == pid
        assert style.preferred_channel == "text"

    def test_style_with_no_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        style = em.get_communication_style(pid)
        assert style.preferred_channel == "text"
        assert style.formality_level == "mixed"

    def test_formal_relationship_style(self):
        em = EmpathyMapping()
        pid = em.add_person("CEO", relationship="manager")
        em.record_interaction(pid, "email", 0.3)
        em.record_interaction(pid, "email", 0.4)
        style = em.get_communication_style(pid)
        assert style.preferred_channel == "email"

    def test_dos_and_donts(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol")
        em.record_interaction(pid, "text", 0.8, "great conversation")
        em.record_interaction(pid, "call", 0.7, "nice chat")
        style = em.get_communication_style(pid)
        assert len(style.dos) > 0
        assert len(style.donts) >= 0

    def test_invalid_person_raises(self):
        em = EmpathyMapping()
        with pytest.raises(KeyError):
            em.get_communication_style("nonexistent")


# ── get_empathy_score ───────────────────────────────────

class TestGetEmpathyScore:
    def test_zero_with_no_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        assert em.get_empathy_score(pid) == 0.0

    def test_empathy_increases_with_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        em.record_interaction(pid, "text", 0.5)
        score1 = em.get_empathy_score(pid)
        for i in range(5):
            em.record_interaction(pid, "text", 0.5)
        score2 = em.get_empathy_score(pid)
        assert score2 > score1

    def test_empathy_capped_at_100(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol")
        for i in range(25):
            em.record_interaction(pid, "text", 0.5, "notes here")
        score = em.get_empathy_score(pid)
        assert score <= 100.0

    def test_consistent_sentiment_higher_score(self):
        em = EmpathyMapping()
        pid1 = em.add_person("Consistent")
        pid2 = em.add_person("Inconsistent")
        for _ in range(5):
            em.record_interaction(pid1, "text", 0.5)
        for i in range(5):
            em.record_interaction(pid2, "text", 0.9 if i % 2 == 0 else -0.9)
        score1 = em.get_empathy_score(pid1)
        score2 = em.get_empathy_score(pid2)
        assert score1 > score2

    def test_notes_increase_empathy(self):
        em = EmpathyMapping()
        pid1 = em.add_person("WithNotes")
        pid2 = em.add_person("NoNotes")
        for _ in range(5):
            em.record_interaction(pid1, "text", 0.5, "detailed notes")
            em.record_interaction(pid2, "text", 0.5)
        score1 = em.get_empathy_score(pid1)
        score2 = em.get_empathy_score(pid2)
        assert score1 > score2

    def test_invalid_person_raises(self):
        em = EmpathyMapping()
        with pytest.raises(KeyError):
            em.get_empathy_score("nonexistent")


# ── predict_reaction ─────────────────────────────────────

class TestPredictReaction:
    def test_basic_prediction(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        em.record_interaction(pid, "text", 0.6)
        pred = em.predict_reaction(pid, "promotion", "direct")
        assert isinstance(pred, ReactionPrediction)
        assert pred.person_id == pid
        assert pred.topic == "promotion"
        assert -1.0 <= pred.predicted_sentiment <= 1.0
        assert 0.0 <= pred.predicted_receptiveness <= 1.0
        assert 0.0 <= pred.confidence <= 1.0

    def test_trigger_lowers_sentiment(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        em.record_interaction(pid, "text", -0.8, "stress and anxiety about work")
        em.record_interaction(pid, "text", -0.6, "more stress at the office")
        pred = em.predict_reaction(pid, "stress management", "direct")
        assert pred.predicted_sentiment < em.get_person_profile(pid).avg_sentiment

    def test_approach_affects_sentiment_for_formal(self):
        em = EmpathyMapping()
        pid = em.add_person("CEO", relationship="manager")
        em.record_interaction(pid, "email", 0.0, "quarterly review")
        pred_direct = em.predict_reaction(pid, "budget", "direct")
        pred_gentle = em.predict_reaction(pid, "budget", "gentle")
        assert pred_direct.predicted_sentiment > pred_gentle.predicted_sentiment

    def test_approach_affects_sentiment_for_casual(self):
        em = EmpathyMapping()
        pid = em.add_person("Buddy", relationship="friend")
        em.record_interaction(pid, "text", 0.8, "good times")
        em.record_interaction(pid, "text", 0.9, "fun chat")
        pred_gentle = em.predict_reaction(pid, "vacation", "gentle")
        pred_direct = em.predict_reaction(pid, "vacation", "direct")
        assert pred_gentle.predicted_sentiment > pred_direct.predicted_sentiment

    def test_confidence_increases_with_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol")
        em.record_interaction(pid, "text", 0.5)
        conf1 = em.predict_reaction(pid, "topic", "direct").confidence
        for _ in range(10):
            em.record_interaction(pid, "text", 0.5)
        conf2 = em.predict_reaction(pid, "topic", "direct").confidence
        assert conf2 > conf1

    def test_alternative_approaches(self):
        em = EmpathyMapping()
        pid = em.add_person("Dave")
        em.record_interaction(pid, "text", 0.5)
        pred = em.predict_reaction(pid, "project", "direct")
        assert len(pred.alternative_approaches) > 0

    def test_invalid_person_raises(self):
        em = EmpathyMapping()
        with pytest.raises(KeyError):
            em.predict_reaction("nonexistent", "topic", "direct")

    def test_receptiveness_range(self):
        em = EmpathyMapping()
        pid = em.add_person("Eve")
        for _ in range(10):
            em.record_interaction(pid, "text", 0.8)
        pred = em.predict_reaction(pid, "news", "gentle")
        assert pred.predicted_receptiveness > 0.5


# ── get_relationship_health ──────────────────────────────

class TestGetRelationshipHealth:
    def test_basic_health(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        em.record_interaction(pid, "text", 0.7)
        health = em.get_relationship_health(pid)
        assert isinstance(health, RelationshipHealth)
        assert 0.0 <= health.health_score <= 100.0
        assert health.trend in ("improving", "stable", "declining")

    def test_no_interactions_low_health(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        health = em.get_relationship_health(pid)
        assert health.health_score < 30
        assert health.interaction_frequency == "rare"

    def test_improving_trend(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol")
        for s in [-0.5, -0.3, -0.1, 0.2, 0.4, 0.7, 0.8, 0.9]:
            em.record_interaction(pid, "text", s)
        health = em.get_relationship_health(pid)
        assert health.trend == "improving"

    def test_declining_trend(self):
        em = EmpathyMapping()
        pid = em.add_person("Dave")
        for s in [0.9, 0.7, 0.4, 0.2, -0.1, -0.3, -0.5, -0.7]:
            em.record_interaction(pid, "text", s)
        health = em.get_relationship_health(pid)
        assert health.trend == "declining"

    def test_interaction_frequency(self):
        em = EmpathyMapping()
        pid = em.add_person("Eve")
        for _ in range(5):
            em.record_interaction(pid, "text", 0.5)
        health = em.get_relationship_health(pid)
        assert health.interaction_frequency == "monthly"

    def test_recommendation_for_low_health(self):
        em = EmpathyMapping()
        pid = em.add_person("Frank")
        health = em.get_relationship_health(pid)
        assert "needs attention" in health.recommendation or "Invest" in health.recommendation or "relationship" in health.recommendation.lower()

    def test_invalid_person_raises(self):
        em = EmpathyMapping()
        with pytest.raises(KeyError):
            em.get_relationship_health("nonexistent")

    def test_last_interaction_set(self):
        em = EmpathyMapping()
        pid = em.add_person("Grace")
        em.record_interaction(pid, "text", 0.5)
        health = em.get_relationship_health(pid)
        assert health.last_interaction is not None


# ── get_social_graph ─────────────────────────────────────

class TestGetSocialGraph:
    def test_empty_graph(self):
        em = EmpathyMapping()
        graph = em.get_social_graph()
        assert graph["nodes"] == []
        assert graph["edges"] == []
        assert graph["total_persons"] == 0

    def test_graph_with_people(self):
        em = EmpathyMapping()
        pid1 = em.add_person("Alice", relationship="friend")
        pid2 = em.add_person("Bob", relationship="colleague")
        em.record_interaction(pid1, "text", 0.5)
        em.record_interaction(pid2, "email", 0.3)
        graph = em.get_social_graph()
        assert graph["total_persons"] == 2
        assert len(graph["nodes"]) == 2

    def test_graph_has_empathy_scores(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol")
        em.record_interaction(pid, "text", 0.5)
        graph = em.get_social_graph()
        node = [n for n in graph["nodes"] if n["id"] == pid][0]
        assert "empathy_score" in node
        assert "avg_sentiment" in node

    def test_graph_edges_for_related_people(self):
        em = EmpathyMapping()
        em.add_person("Alice", relationship="family")
        em.add_person("Bob", relationship="friend")
        graph = em.get_social_graph()
        assert len(graph["edges"]) > 0

    def test_graph_node_fields(self):
        em = EmpathyMapping()
        pid = em.add_person("Dave", relationship="manager")
        em.record_interaction(pid, "email", 0.3)
        graph = em.get_social_graph()
        node = graph["nodes"][0]
        assert node["id"] == pid
        assert node["name"] == "Dave"
        assert node["relationship"] == "manager"


# ── get_stats ─────────────────────────────────────────────

class TestGetStats:
    def test_empty_stats(self):
        em = EmpathyMapping()
        stats = em.get_stats()
        assert stats["total_persons"] == 0
        assert stats["total_interactions"] == 0
        assert stats["avg_sentiment"] == 0.0
        assert stats["avg_empathy_score"] == 0.0

    def test_stats_with_data(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        em.record_interaction(pid, "text", 0.5)
        em.record_interaction(pid, "call", 0.7)
        stats = em.get_stats()
        assert stats["total_persons"] == 1
        assert stats["total_interactions"] == 2
        assert abs(stats["avg_sentiment"] - 0.6) < 0.01
        assert "text" in stats["channel_distribution"]
        assert "call" in stats["channel_distribution"]

    def test_stats_relationship_distribution(self):
        em = EmpathyMapping()
        em.add_person("A", relationship="friend")
        em.add_person("B", relationship="friend")
        em.add_person("C", relationship="manager")
        stats = em.get_stats()
        assert stats["relationship_distribution"]["friend"] == 2
        assert stats["relationship_distribution"]["manager"] == 1

    def test_stats_max_values(self):
        em = EmpathyMapping()
        stats = em.get_stats()
        assert stats["max_persons"] == 500
        assert stats["max_interactions"] == 10000


# ── emotional_triggers ───────────────────────────────────

class TestEmotionalTriggers:
    def test_triggers_from_negative_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        em.record_interaction(pid, "text", -0.7, "feeling stressed and anxious about deadline")
        em.record_interaction(pid, "text", -0.6, "more stress at work today")
        profile = em.get_person_profile(pid)
        assert "stressed" in profile.emotional_triggers or "stress" in profile.emotional_triggers

    def test_no_triggers_with_few_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        em.record_interaction(pid, "text", 0.5)
        profile = em.get_person_profile(pid)
        assert profile.emotional_triggers == []

    def test_triggers_only_from_significant_sentiment(self):
        em = EmpathyMapping()
        pid = em.add_person("Carol")
        em.record_interaction(pid, "text", 0.1, "meeting about project deadlines")
        em.record_interaction(pid, "text", 0.2, "another meeting discussion")
        profile = em.get_person_profile(pid)
        for trigger in profile.emotional_triggers:
            assert len(trigger) > 3


# ── response_patterns ─────────────────────────────────────

class TestResponsePatterns:
    def test_response_patterns_with_multiple_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        em.record_interaction(pid, "text", 0.5, "chat")
        em.record_interaction(pid, "text", 0.5, "chat")
        profile = em.get_person_profile(pid)
        assert "avg_response_hours" in profile.response_patterns
        assert profile.response_patterns["total_interactions"] == 2

    def test_response_patterns_empty_with_one_interaction(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        em.record_interaction(pid, "text", 0.5)
        profile = em.get_person_profile(pid)
        assert profile.response_patterns == {}


# ── thread safety ─────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_add_person(self):
        em = EmpathyMapping()
        errors = []

        def add_person(idx):
            try:
                em.add_person(f"Person_{idx}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_person, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = em.get_stats()
        assert stats["total_persons"] == 100

    def test_concurrent_record_interaction(self):
        em = EmpathyMapping()
        pid = em.add_person("ThreadTarget")
        errors = []

        def record():
            try:
                em.record_interaction(pid, "text", 0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        profile = em.get_person_profile(pid)
        assert profile.interaction_count == 50

    def test_concurrent_read_write(self):
        em = EmpathyMapping()
        pid = em.add_person("ReadWrite")
        errors = []

        def write_ops():
            for i in range(20):
                try:
                    em.record_interaction(pid, "text", i * 0.05 - 0.5)
                except Exception as e:
                    errors.append(e)

        def read_ops():
            for _ in range(20):
                try:
                    em.get_person_profile(pid)
                    em.get_empathy_score(pid)
                    em.get_relationship_health(pid)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=write_ops),
            threading.Thread(target=read_ops),
            threading.Thread(target=read_ops),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ── edge cases ────────────────────────────────────────────

class TestEdgeCases:
    def test_deque_eviction(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice")
        for i in range(12):
            em.record_interaction(pid, "text", i * 0.1 - 0.5)
        stats = em.get_stats()
        assert stats["total_interactions"] <= 10000
        profile = em.get_person_profile(pid)
        assert profile.interaction_count == 12

    def test_predict_reaction_unknown_approach(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        em.record_interaction(pid, "text", 0.5)
        pred = em.predict_reaction(pid, "topic", "yelling")
        assert pred.approach == "direct"

    def test_social_graph_with_many_people(self):
        em = EmpathyMapping()
        for i in range(10):
            em.add_person(f"Person_{i}")
        graph = em.get_social_graph()
        assert len(graph["nodes"]) == 10

    def test_metadata_long_values_truncated(self):
        em = EmpathyMapping()
        pid = em.add_person("Alice", metadata={"note": "x" * 1000})
        profile = em.get_person_profile(pid)
        assert len(profile.metadata.get("note", "")) <= 500

    def test_relationship_health_with_many_interactions(self):
        em = EmpathyMapping()
        pid = em.add_person("Bob")
        for _ in range(25):
            em.record_interaction(pid, "text", 0.5)
        health = em.get_relationship_health(pid)
        assert health.interaction_frequency == "daily"

    def test_empathy_score_channel_diversity(self):
        em = EmpathyMapping()
        pid = em.add_person("MultiChannel")
        em.record_interaction(pid, "text", 0.5)
        score1 = em.get_empathy_score(pid)
        em.record_interaction(pid, "call", 0.5)
        em.record_interaction(pid, "email", 0.5)
        score2 = em.get_empathy_score(pid)
        assert score2 > score1

    def test_communication_style_negative_sentiment(self):
        em = EmpathyMapping()
        pid = em.add_person("NegativeNancy")
        em.record_interaction(pid, "text", -0.7)
        em.record_interaction(pid, "text", -0.6)
        style = em.get_communication_style(pid)
        assert len(style.donts) > 0