"""
Sprint 89b Regression Tests — to_dict() Deepcopy Fixes

Critical bug fix: to_dict() methods that used bare asdict() were returning
shallow copies — external code could mutate internal dataclass state by
modifying the returned dict/list references.

Fixes applied:
- attention_engine: AttentionItem.to_dict() copies metadata dict;
  FocusSession.to_dict() uses copy.deepcopy(asdict(self))
- empathy_mapping: PersonProfile.to_dict() copies response_patterns and
  metadata dicts; CommunicationStyle.to_dict() copies list fields;
  ReactionPrediction.to_dict() copies alternative_approaches;
  RelationshipHealth.to_dict() uses copy.deepcopy(asdict(self))
- temporal_reasoning: TemporalAnomaly.to_dict() and DurationEstimate.to_dict()
  use copy.deepcopy(asdict(self)); added 'import copy'
"""

import pytest


class TestAttentionEngineToDict:
    """Verify to_dict() returns defensive copies."""

    def test_attention_item_metadata_is_copy(self):
        from shadowd.cognition.attention_engine import AttentionItem, AttentionLevel
        item = AttentionItem(
            item_id="test1",
            topic="work",
            level=AttentionLevel.HIGH,
            domain="career",
            urgency_score=0.8,
            importance_score=0.9,
            created_at="2024-01-01T00:00:00",
            last_focused_at="2024-01-01T00:05:00",
            focus_count=1,
            metadata={"key": "value"},
        )
        d = item.to_dict()
        d["metadata"]["key"] = "mutated"
        assert item.metadata["key"] == "value"

    def test_attention_item_metadata_new_key(self):
        from shadowd.cognition.attention_engine import AttentionItem, AttentionLevel
        item = AttentionItem(
            item_id="test2",
            topic="work",
            level=AttentionLevel.MEDIUM,
            domain="career",
            urgency_score=0.5,
            importance_score=0.5,
            created_at="2024-01-01T00:00:00",
            last_focused_at="2024-01-01T00:00:00",
            focus_count=0,
            metadata={},
        )
        d = item.to_dict()
        d["metadata"]["new_key"] = "new_val"
        assert "new_key" not in item.metadata

    def test_focus_session_deepcopy(self):
        from shadowd.cognition.attention_engine import FocusSession, FocusState
        session = FocusSession(
            session_id="sess1",
            topic="deep work",
            started_at="2024-01-01T00:00:00",
            ended_at=None,
            duration_minutes=30,
            interruptions=0,
            depth_score=0.8,
        )
        d = session.to_dict()
        d["session_id"] = "mutated"
        assert session.session_id == "sess1"

    def test_distraction_event_to_dict(self):
        from shadowd.cognition.attention_engine import DistractionEvent, DistractionType
        evt = DistractionEvent(
            event_id="d1",
            distraction_type=DistractionType.NOTIFICATION,
            description="slack ping",
            timestamp="2024-01-01T00:00:00",
            recovery_time_seconds=300,
        )
        d = evt.to_dict()
        d["description"] = "mutated"
        assert evt.description == "slack ping"


class TestEmpathyMappingToDict:
    """Verify to_dict() returns defensive copies."""

    def test_person_profile_response_patterns_is_copy(self):
        from shadowd.cognition.empathy_mapping import PersonProfile, RelationshipType
        profile = PersonProfile(
            person_id="p1",
            name="Alice",
            relationship=RelationshipType.FRIEND,
            created_at="2024-01-01T00:00:00",
            last_interaction="2024-01-01T00:00:00",
            interaction_count=5,
            avg_sentiment=0.7,
            preferred_channels=["text"],
            emotional_triggers=["stress"],
            formality_preference="casual",
            response_patterns={"greeting": "warm"},
            metadata={"notes": "test"},
        )
        d = profile.to_dict()
        d["response_patterns"]["greeting"] = "cold"
        assert profile.response_patterns["greeting"] == "warm"

    def test_person_profile_metadata_is_copy(self):
        from shadowd.cognition.empathy_mapping import PersonProfile, RelationshipType
        profile = PersonProfile(
            person_id="p2",
            name="Bob",
            relationship=RelationshipType.COLLEAGUE,
            created_at="2024-01-01T00:00:00",
            last_interaction="2024-01-01T00:00:00",
            interaction_count=3,
            avg_sentiment=0.5,
            preferred_channels=["email"],
            emotional_triggers=[],
            formality_preference="formal",
            response_patterns={},
            metadata={"pref": "email"},
        )
        d = profile.to_dict()
        d["metadata"]["pref"] = "phone"
        assert profile.metadata["pref"] == "email"

    def test_communication_style_emotional_triggers_is_copy(self):
        from shadowd.cognition.empathy_mapping import CommunicationStyle
        style = CommunicationStyle(
            person_id="p1",
            preferred_channel="text",
            formality_level="casual",
            response_time_hours=1,
            emotional_triggers=["stress", "deadline"],
            dos=["be brief"],
            donts=["micromanage"],
        )
        d = style.to_dict()
        d["emotional_triggers"].append("anger")
        assert len(style.emotional_triggers) == 2

    def test_communication_style_dos_is_copy(self):
        from shadowd.cognition.empathy_mapping import CommunicationStyle
        style = CommunicationStyle(
            person_id="p1",
            preferred_channel="email",
            formality_level="formal",
            response_time_hours=4,
            emotional_triggers=[],
            dos=["be polite"],
            donts=["be late"],
        )
        d = style.to_dict()
        d["dos"].append("extra")
        assert len(style.dos) == 1

    def test_communication_style_donts_is_copy(self):
        from shadowd.cognition.empathy_mapping import CommunicationStyle
        style = CommunicationStyle(
            person_id="p1",
            preferred_channel="email",
            formality_level="formal",
            response_time_hours=4,
            emotional_triggers=[],
            dos=[],
            donts=["yell"],
        )
        d = style.to_dict()
        d["donts"].append("extra")
        assert len(style.donts) == 1

    def test_reaction_prediction_alternative_approaches_is_copy(self):
        from shadowd.cognition.empathy_mapping import ReactionPrediction
        pred = ReactionPrediction(
            person_id="p1",
            topic="project update",
            approach="direct",
            predicted_sentiment=0.8,
            predicted_receptiveness=0.7,
            confidence=0.75,
            alternative_approaches=["indirect", "passive"],
        )
        d = pred.to_dict()
        d["alternative_approaches"].append("aggressive")
        assert len(pred.alternative_approaches) == 2

    def test_relationship_health_deepcopy(self):
        from shadowd.cognition.empathy_mapping import RelationshipHealth
        health = RelationshipHealth(
            person_id="p1",
            health_score=75,
            trend="stable",
            last_interaction="2024-01-01",
            recommendation="keep in touch",
            interaction_frequency="weekly",
        )
        d = health.to_dict()
        d["health_score"] = 0
        assert health.health_score == 75


class TestTemporalReasoningToDict:
    """Verify to_dict() returns deep copies."""

    def test_temporal_anomaly_deepcopy(self):
        from shadowd.cognition.temporal_reasoning import TemporalAnomaly
        anomaly = TemporalAnomaly(
            event_type="login",
            expected_time="09:00",
            actual_time="03:00",
            deviation_score=2.5,
            description="Unusual login time",
        )
        d = anomaly.to_dict()
        d["deviation_score"] = 99.0
        assert anomaly.deviation_score == 2.5

    def test_duration_estimate_deepcopy(self):
        from shadowd.cognition.temporal_reasoning import DurationEstimate
        est = DurationEstimate(
            event_type="meeting",
            mean_minutes=30.0,
            median_minutes=25.0,
            min_minutes=5.0,
            max_minutes=120.0,
            confidence=0.85,
            sample_size=50,
        )
        d = est.to_dict()
        d["mean_minutes"] = 999.0
        assert est.mean_minutes == 30.0


class TestRecallSynthesisToDict:
    """Verify recall_synthesis to_dict() returns defensive copies."""

    def test_observation_keywords_is_copy(self):
        from shadowd.cognition.recall_synthesis import Observation
        obs = Observation(
            observation_id="o1",
            text="test observation",
            domain="work",
            timestamp="2024-01-01T00:00:00",
            importance=0.7,
            keywords=["python", "debug"],
        )
        d = obs.to_dict()
        d["keywords"].append("extra")
        assert "extra" not in obs.keywords

    def test_insight_domains_is_copy(self):
        from shadowd.cognition.recall_synthesis import Insight
        insight = Insight(
            domain="work",
            description="test insight",
            confidence=0.8,
            source_observations=["o1"],
            created_at="2024-01-01T00:00:00",
            domains=["work", "health"],
        )
        d = insight.to_dict()
        d["domains"].append("extra")
        assert "extra" not in insight.domains

    def test_cross_domain_pattern_lists_are_copies(self):
        from shadowd.cognition.recall_synthesis import CrossDomainPattern
        pattern = CrossDomainPattern(
            pattern_id="p1",
            keywords=["test"],
            domains_involved=["work"],
            observations_linked=["o1"],
            confidence=0.7,
            description="test pattern",
        )
        d = pattern.to_dict()
        d["keywords"].append("extra")
        d["domains_involved"].append("extra")
        assert "extra" not in pattern.keywords
        assert "extra" not in pattern.domains_involved

    def test_domain_connection_shared_keywords_is_copy(self):
        from shadowd.cognition.recall_synthesis import DomainConnection
        conn = DomainConnection(
            domain_a="work",
            domain_b="health",
            strength=0.8,
            shared_keywords=["stress"],
            connection_count=3,
        )
        d = conn.to_dict()
        d["shared_keywords"].append("extra")
        assert "extra" not in conn.shared_keywords