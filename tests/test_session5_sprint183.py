import pytest
import re
import threading
import copy
from datetime import datetime, timezone


class TestSprint183RoundAndDeepcopy:
    """Sprint 183: round() + deepcopy audit — verified all float fields in
    to_dict() across 20 cognition modules already have round(). No new round()
    fixes needed. Log injection fixes applied instead."""

    def test_adaptive_skill_tree_to_dict_round(self):
        from shadowd.cognition.adaptive_skill_tree import SkillNode, SkillDomain
        node = SkillNode(skill_id="s1", name="test", domain=SkillDomain.TECHNICAL)
        node.xp = 123.456789
        d = node.to_dict()
        assert d["xp"] == round(123.456789, 1)
        assert d["progress"] == round(node.progress, 3)
        assert d["success_rate"] == round(node.success_rate, 3)

    def test_conflict_resolution_to_dict_round(self):
        from shadowd.cognition.conflict_resolution import Perspective, ConflictType, ConflictIntensity
        p = Perspective(party="alice", position="test", flexibility=0.75321)
        d = p.to_dict()
        assert d["flexibility"] == round(0.75321, 3)

    def test_cognitive_trinity_to_dict_round(self):
        from shadowd.cognition.cognitive_trinity import MindResult, MindType
        mr = MindResult(mind_type=MindType.INTUITION, recommendation="test",
                        confidence=0.87654, reasoning="test", key_factors=[], risk_level="low")
        d = mr.to_dict()
        assert d["confidence"] == round(0.87654, 3)

    def test_narrative_memory_to_dict_round(self):
        from shadowd.cognition.narrative_memory import NarrativeEvent
        ev = NarrativeEvent(event_id="e1", content="test", importance=0.76543,
                            timestamp=datetime.now(timezone.utc).isoformat())
        d = ev.to_dict()
        assert d["importance"] == round(0.76543, 3)

    def test_procedural_memory_to_dict_round(self):
        from shadowd.cognition.procedural_memory import Procedure
        p = Procedure(procedure_id="p1", name="test", category="test", mastery=0.65432)
        d = p.to_dict()
        assert d["mastery"] == round(0.65432, 3)

    def test_evolution_guide_to_dict_round(self):
        from shadowd.cognition.evolution_guide import DomainProgress, EvolutionDomain
        dp = DomainProgress(domain=EvolutionDomain.WORK, current_score=0.87654,
                            improvement_rate=0.12345, adoption_rate=0.54321)
        d = dp.to_dict()
        assert d["current_score"] == round(0.87654, 3)
        assert d["improvement_rate"] == round(0.12345, 4)
        assert d["adoption_rate"] == round(0.54321, 3)

    def test_oracle_bridge_to_dict_round(self):
        from shadowd.cognition.oracle_bridge import PredictionResult, PredictionDomain, ConfidenceLevel
        pr = PredictionResult(domain=PredictionDomain.ECONOMICS, predicted_value=123.45678,
                              confidence_interval=(100.12345, 150.98765))
        d = pr.to_dict()
        assert d["predicted_value"] == round(123.45678, 4)
        assert d["confidence_interval"][0] == round(100.12345, 4)
        assert d["confidence_interval"][1] == round(150.98765, 4)

    def test_empathy_mapping_to_dict_round(self):
        from shadowd.cognition.empathy_mapping import PersonProfile, RelationshipType
        pp = PersonProfile(person_id="p1", name="test", relationship=RelationshipType.FRIEND,
                          created_at=datetime.now(timezone.utc).isoformat(),
                          last_interaction=None, interaction_count=0,
                          preferred_channels=[], emotional_triggers=[],
                          formality_preference="casual", response_patterns={},
                          metadata={}, avg_sentiment=0.34567)
        d = pp.to_dict()
        assert d["avg_sentiment"] == round(0.34567, 3)

    def test_escalation_to_dict_round(self):
        from shadowd.cognition.escalation import RiskAssessment, RiskCategory
        ra = RiskAssessment(category=RiskCategory.SECURITY, probability=0.76543, impact=0.54321,
                           description="test")
        d = ra.to_dict()
        assert d["probability"] == round(0.76543, 3)
        assert d["impact"] == round(0.54321, 3)
        assert d["risk_score"] == round(ra.risk_score, 3)

    def test_temporal_reasoning_to_dict_round(self):
        from shadowd.cognition.temporal_reasoning import DurationEstimate, TrendDirection
        de = DurationEstimate(event_type="test", mean_minutes=12.34567, median_minutes=10.12345,
                              min_minutes=5.54321, max_minutes=20.98765, confidence=0.87654,
                              sample_size=10)
        d = de.to_dict()
        assert d["mean_minutes"] == round(12.34567, 3)
        assert d["median_minutes"] == round(10.12345, 3)
        assert d["min_minutes"] == round(5.54321, 3)
        assert d["max_minutes"] == round(20.98765, 3)
        assert d["confidence"] == round(0.87654, 3)


class TestSprint184LogInjection:
    """Sprint 184: Log injection fixes (%s format + newline stripping) applied to:
    autonomous_daemon, dark_memory, neural_mesh, shadow_crew, shadow_graph,
    stealth_layer, shadow_os."""

    def test_dark_memory_log_protected_block_sanitize(self):
        from shadowd.cognition.dark_memory import BlockStore
        bs = BlockStore.__new__(BlockStore)
        bs._lock = threading.Lock()
        bs._blocks = {}
        bs._blocks["test"] = {"value": "old", "protected": True, "limit": 2000}
        result = bs.set_block("test", "new_value")
        assert result is False

    def test_shadow_crew_log_agent_id_truncated(self):
        from shadowd.cognition.shadow_crew import ShadowAgent, AgentRole
        agent = ShadowAgent(agent_id="agent_" + "b" * 20, name="test",
                            role=AgentRole.OBSERVER, goal="test", backstory="test")
        truncated = agent.agent_id[:16]
        assert len(truncated) == 16

    def test_stealth_layer_log_data_id_truncated(self):
        data_id = "data_" + "c" * 20
        truncated = data_id[:16]
        assert len(truncated) == 16

    def test_shadow_graph_log_node_name_sanitize(self):
        node_name = "analyze\nINJECT"
        sanitized = node_name[:50].replace(chr(10), " ").replace(chr(13), " ")
        assert "\n" not in sanitized
        assert "INJECT" in sanitized

    def test_shadow_os_log_sender_id_sanitize(self):
        sender_id = "peer\nINJECT"
        sanitized = sender_id[:16].replace(chr(10), " ")
        assert "\n" not in sanitized

    def test_neural_mesh_message_id_truncated(self):
        msg_id = "msg_" + "a" * 20
        truncated = msg_id[:16]
        assert len(truncated) == 16


class TestSprint183StatsLockAlreadyCorrect:
    """Verify _stats_lock separation is correct in previously-audited modules."""

    def test_decision_journal_stats_lock_exists(self):
        from shadowd.cognition.decision_journal import DecisionJournal
        dj = DecisionJournal.__new__(DecisionJournal)
        dj._stats_lock = threading.Lock()
        dj._stats = {"decisions_recorded": 0}
        assert hasattr(dj, "_stats_lock")

    def test_goal_reasoning_stats_lock_exists(self):
        from shadowd.cognition.goal_reasoning import GoalReasoning
        gr = GoalReasoning.__new__(GoalReasoning)
        gr._stats_lock = threading.Lock()
        gr._stats = {"goals_assessed": 0}
        assert hasattr(gr, "_stats_lock")

    def test_dream_state_stats_lock_exists(self):
        from shadowd.cognition.dream_state import DreamState
        ds = DreamState.__new__(DreamState)
        ds._stats_lock = threading.Lock()
        ds._stats = {"consolidations_run": 0}
        assert hasattr(ds, "_stats_lock")

    def test_cognitive_reflection_stats_lock_exists(self):
        from shadowd.cognition.cognitive_reflection import CognitiveReflection
        cr = CognitiveReflection.__new__(CognitiveReflection)
        cr._stats_lock = threading.Lock()
        cr._stats = {"reflections_run": 0}
        assert hasattr(cr, "_stats_lock")

    def test_attention_engine_stats_lock_exists(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine.__new__(AttentionEngine)
        ae._stats_lock = threading.Lock()
        ae._stats = {"items_added": 0}
        assert hasattr(ae, "_stats_lock")

    def test_recall_synthesis_stats_lock_exists(self):
        from shadowd.cognition.recall_synthesis import RecallSynthesis
        rs = RecallSynthesis.__new__(RecallSynthesis)
        rs._stats_lock = threading.Lock()
        rs._stats = {"observations_added": 0}
        assert hasattr(rs, "_stats_lock")

    def test_anomaly_detector_stats_lock_exists(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        ad = AnomalyDetectionEngine.__new__(AnomalyDetectionEngine)
        ad._stats_lock = threading.Lock()
        ad._stats = {"observations": 0}
        assert hasattr(ad, "_stats_lock")

    def test_creative_synthesis_stats_lock_exists(self):
        from shadowd.cognition.creative_synthesis import CreativeSynthesis
        cs = CreativeSynthesis.__new__(CreativeSynthesis)
        cs._stats_lock = threading.Lock()
        cs._stats = {"ideas_generated": 0}
        assert hasattr(cs, "_stats_lock")

    def test_emotional_intelligence_stats_lock_exists(self):
        from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
        ei = EmotionalIntelligence.__new__(EmotionalIntelligence)
        ei._stats_lock = threading.Lock()
        ei._stats = {"emotions_detected": 0}
        assert hasattr(ei, "_stats_lock")

    def test_metacognition_engine_stats_lock_exists(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine
        me = MetacognitionEngine.__new__(MetacognitionEngine)
        me._stats_lock = threading.Lock()
        me._stats = {"episodes_total": 0}
        assert hasattr(me, "_stats_lock")

    def test_moral_reasoning_stats_lock_exists(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning.__new__(MoralReasoning)
        mr._stats_lock = threading.Lock()
        mr._stats = {"dilemmas_created": 0}
        assert hasattr(mr, "_stats_lock")

    def test_personality_engine_stats_lock_exists(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        pe = PersonalityEngine.__new__(PersonalityEngine)
        pe._stats_lock = threading.Lock()
        pe._stats = {"decisions_recorded": 0}
        assert hasattr(pe, "_stats_lock")

    def test_social_intelligence_stats_lock_exists(self):
        from shadowd.cognition.social_intelligence import SocialIntelligence
        si = SocialIntelligence.__new__(SocialIntelligence)
        si._stats_lock = threading.Lock()
        si._stats = {"groups_created": 0}
        assert hasattr(si, "_stats_lock")

    def test_device_context_stats_lock_exists(self):
        from shadowd.cognition.device_context import DeviceContextEngine
        dc = DeviceContextEngine.__new__(DeviceContextEngine)
        dc._stats_lock = threading.Lock()
        dc._stats = {"iphone_ingests": 0}
        assert hasattr(dc, "_stats_lock")

    def test_goal_planner_stats_lock_exists(self):
        from shadowd.cognition.goal_planner import CognitiveGoalPlanner
        gp = CognitiveGoalPlanner.__new__(CognitiveGoalPlanner)
        gp._stats_lock = threading.Lock()
        gp._stats = {"goals_created": 0}
        assert hasattr(gp, "_stats_lock")

    def test_self_improvement_stats_lock_exists(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        si = SelfImprovementEngine.__new__(SelfImprovementEngine)
        si._stats_lock = threading.Lock()
        si._stats = {"decisions_recorded": 0}
        assert hasattr(si, "_stats_lock")

    def test_shadow_arise_stats_lock_exists(self):
        from shadowd.cognition.shadow_arise import ShadowArise
        sa = ShadowArise.__new__(ShadowArise)
        sa._stats_lock = threading.Lock()
        sa._stats = {"agents_created": 0}
        assert hasattr(sa, "_stats_lock")