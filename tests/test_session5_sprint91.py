#!/usr/bin/env python3
"""
Sprint 91 Regression Tests — to_dict() Shallow Copy Fixes Round 3

Tests for deepcopy fixes in:
- shadow_graph.py: CognitiveState.to_dict(), Checkpoint.to_dict()
- cognitive_pipeline.py: _truncate_dict() deep-copies dict/list values
- dark_memory.py: MemoryAtom.to_dict() metadata deepcopy
- goal_reasoning.py: Obstacle.to_dict(), GoalAssessment.to_dict() metadata deepcopy
- cognitive_reflection.py: ReflectionEntry.to_dict() metadata deepcopy
- decision_journal.py: DecisionRecord.to_dict() metadata deepcopy
"""

import copy
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shadowd.cognition.shadow_graph import CognitiveState, Checkpoint, CognitiveMode, ShadowGraph
from shadowd.cognition.cognitive_pipeline import _truncate_dict, PipelineResult, CognitivePipeline
from shadowd.cognition.dark_memory import MemoryAtom, MemoryTier, AtomType
from shadowd.cognition.goal_reasoning import Obstacle, GoalAssessment, AssessmentOutcome, ObstacleSeverity
from shadowd.cognition.cognitive_reflection import ReflectionEntry, ReflectionType, LearningInsight, ReflectionPattern, CognitiveReflection
from shadowd.cognition.decision_journal import DecisionRecord, DecisionType, DecisionStatus
from shadowd.cognition.daily_briefing import BriefingItem, DailyBriefing, BriefingSection, UrgencyLevel
from shadowd.cognition.escalation import ActionRequest, EscalationRecord, AutonomyLevel
from shadowd.cognition.attention_engine import AttentionItem, AttentionLevel


class TestShadowGraphDeepcopy:
    """Tests for shadow_graph.py to_dict() deepcopy fixes."""

    def test_cognitive_state_context_deepcopy(self):
        state = CognitiveState(shadow_id="test")
        state.context = {"key": {"nested": "value"}, "list_key": [1, 2, 3]}
        d = state.to_dict()
        assert d["context"] == {"key": {"nested": "value"}, "list_key": [1, 2, 3]}
        d["context"]["key"]["nested"] = "mutated"
        d["context"]["list_key"].append(4)
        assert state.context["key"]["nested"] == "value"
        assert state.context["list_key"] == [1, 2, 3]

    def test_cognitive_state_metadata_deepcopy(self):
        state = CognitiveState(shadow_id="test")
        state.metadata = {"meta": {"deep": "val"}}
        d = state.to_dict()
        assert d["metadata"] == {"meta": {"deep": "val"}}
        d["metadata"]["meta"]["deep"] = "changed"
        assert state.metadata["meta"]["deep"] == "val"

    def test_cognitive_state_context_empty_dict(self):
        state = CognitiveState(shadow_id="test")
        d = state.to_dict()
        assert d["context"] == {}
        d["context"]["new_key"] = "new_val"
        assert state.context == {}

    def test_cognitive_state_metadata_empty_dict(self):
        state = CognitiveState(shadow_id="test")
        d = state.to_dict()
        assert d["metadata"] == {}
        d["metadata"]["new_key"] = "new_val"
        assert state.metadata == {}

    def test_checkpoint_metadata_deepcopy(self):
        cp = Checkpoint(
            checkpoint_id="ckpt_123",
            state={"mode": "listen"},
            node_name="analyze",
            timestamp="2024-01-01T00:00:00Z",
            execution_id="exec_1",
            metadata={"info": {"nested": "data"}, "tags": ["a", "b"]},
        )
        d = cp.to_dict()
        assert d["metadata"] == {"info": {"nested": "data"}, "tags": ["a", "b"]}
        d["metadata"]["info"]["nested"] = "mutated"
        d["metadata"]["tags"].append("c")
        assert cp.metadata["info"]["nested"] == "data"
        assert cp.metadata["tags"] == ["a", "b"]

    def test_checkpoint_metadata_empty(self):
        cp = Checkpoint(
            checkpoint_id="ckpt_123",
            state={},
            node_name="listen",
            timestamp="2024-01-01T00:00:00Z",
            execution_id="exec_1",
        )
        d = cp.to_dict()
        assert d["metadata"] == {}
        d["metadata"]["injected"] = True
        assert cp.metadata == {}


class TestCognitivePipelineTruncateDict:
    """Tests for cognitive_pipeline.py _truncate_dict() deepcopy fixes."""

    def test_truncate_dict_nested_dict_deepcopy(self):
        data = {"analysis": {"scores": [1, 2, 3], "label": "test"}, "count": 5}
        result = _truncate_dict(data)
        assert result["analysis"]["scores"] == [1, 2, 3]
        result["analysis"]["scores"].append(4)
        result["analysis"]["label"] = "changed"
        assert data["analysis"]["scores"] == [1, 2, 3]
        assert data["analysis"]["label"] == "test"

    def test_truncate_dict_list_value_deepcopy(self):
        data = {"emotions": ["happy", "sad"], "count": 2}
        result = _truncate_dict(data)
        assert result["emotions"] == ["happy", "sad"]
        result["emotions"].append("angry")
        assert data["emotions"] == ["happy", "sad"]

    def test_truncate_dict_string_truncation(self):
        data = {"key": "a" * 300}
        result = _truncate_dict(data)
        assert len(result["key"]) == 200

    def test_truncate_dict_none(self):
        assert _truncate_dict(None) is None

    def test_truncate_dict_max_keys(self):
        data = {f"k{i}": i for i in range(100)}
        result = _truncate_dict(data)
        assert len(result) == 50

    def test_truncate_dict_primitive_values(self):
        data = {"int": 42, "float": 3.14, "bool": True, "none": None}
        result = _truncate_dict(data)
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None

    def test_pipeline_result_to_dict_deepcopies_dicts(self):
        result = PipelineResult(
            thought_analysis={"key": {"nested": "val"}},
            emotion_state={"emotions": ["happy"]},
            bias_detection={"biases": [{"name": "anchoring"}]},
            moral_evaluation={"score": 0.8, "details": {"reasoning": "test"}},
            modules_used=["thought_sense", "emotional_intelligence"],
        )
        d = result.to_dict()
        d["thought_analysis"]["key"]["nested"] = "mutated"
        assert result.thought_analysis["key"]["nested"] == "val"


class TestDarkMemoryDeepcopy:
    """Tests for dark_memory.py MemoryAtom.to_dict() metadata deepcopy."""

    def test_memory_atom_metadata_deepcopy(self):
        atom = MemoryAtom(
            atom_type=AtomType.OBSERVATION,
            tier=MemoryTier.WORKING,
            content="test content",
            metadata={"nested": {"deep": "value"}, "list_key": [1, 2, 3]},
        )
        d = atom.to_dict()
        assert d["metadata"]["nested"]["deep"] == "value"
        d["metadata"]["nested"]["deep"] = "mutated"
        d["metadata"]["list_key"].append(4)
        assert atom.metadata["nested"]["deep"] == "value"
        assert atom.metadata["list_key"] == [1, 2, 3]

    def test_memory_atom_tags_deepcopy(self):
        atom = MemoryAtom(
            atom_type=AtomType.OBSERVATION,
            content="test",
            tags=["tag1", "tag2"],
        )
        d = atom.to_dict()
        assert d["tags"] == ["tag1", "tag2"]
        d["tags"].append("tag3")
        assert atom.tags == ["tag1", "tag2"]

    def test_memory_atom_metadata_empty(self):
        atom = MemoryAtom(content="test")
        d = atom.to_dict()
        assert d["metadata"] == {}
        d["metadata"]["injected"] = True
        assert atom.metadata == {}


class TestGoalReasoningDeepcopy:
    """Tests for goal_reasoning.py Obstacle/GoalAssessment to_dict() metadata deepcopy."""

    def test_obstacle_metadata_deepcopy(self):
        obs = Obstacle(
            obstacle_id="obs_1",
            goal_id="goal_1",
            description="Test obstacle",
            severity=ObstacleSeverity.HIGH,
            detected_at="2024-01-01T00:00:00Z",
            mitigation="Fix it",
            metadata={"nested": {"deep": "val"}, "items": [1, 2]},
        )
        d = obs.to_dict()
        assert d["metadata"]["nested"]["deep"] == "val"
        d["metadata"]["nested"]["deep"] = "mutated"
        d["metadata"]["items"].append(3)
        assert obs.metadata["nested"]["deep"] == "val"
        assert obs.metadata["items"] == [1, 2]

    def test_goal_assessment_metadata_deepcopy(self):
        assessment = GoalAssessment(
            assessment_id="assess_1",
            goal_id="goal_1",
            outcome=AssessmentOutcome.ON_TRACK,
            confidence=0.8,
            urgency_score=0.5,
            effort_estimate=10.0,
            risk_factors=["risk1"],
            recommendations=["keep going"],
            assessed_at="2024-01-01T00:00:00Z",
            metadata={"nested": {"key": "val"}, "nums": [1, 2, 3]},
        )
        d = assessment.to_dict()
        assert d["metadata"]["nested"]["key"] == "val"
        d["metadata"]["nested"]["key"] = "mutated"
        d["metadata"]["nums"].append(4)
        assert assessment.metadata["nested"]["key"] == "val"
        assert assessment.metadata["nums"] == [1, 2, 3]

    def test_goal_assessment_metadata_empty(self):
        assessment = GoalAssessment(
            assessment_id="assess_1",
            goal_id="goal_1",
            outcome=AssessmentOutcome.ON_TRACK,
            confidence=0.8,
            urgency_score=0.5,
            effort_estimate=10.0,
            risk_factors=[],
            recommendations=[],
            assessed_at="2024-01-01T00:00:00Z",
            metadata={},
        )
        d = assessment.to_dict()
        assert d["metadata"] == {}
        d["metadata"]["injected"] = True
        assert assessment.metadata == {}

    def test_obstacle_metadata_empty(self):
        obs = Obstacle(
            obstacle_id="obs_1",
            goal_id="goal_1",
            description="Test",
            severity=ObstacleSeverity.LOW,
            detected_at="2024-01-01T00:00:00Z",
            mitigation="",
            metadata={},
        )
        d = obs.to_dict()
        assert d["metadata"] == {}
        d["metadata"]["injected"] = True
        assert obs.metadata == {}


class TestCognitiveReflectionDeepcopy:
    """Tests for cognitive_reflection.py ReflectionEntry.to_dict() metadata deepcopy."""

    def test_reflection_entry_metadata_deepcopy(self):
        entry = ReflectionEntry(
            entry_id="ref_1",
            topic="test topic",
            description="test description",
            reflection_type=ReflectionType.DAILY,
            what_went_well=["good thing"],
            what_could_improve=["bad thing"],
            lessons_learned=["lesson"],
            confidence_before=0.5,
            confidence_after=0.7,
            domain="testing",
            linked_decision_ids=[],
            sentiment=0.6,
            created_at="2024-01-01T00:00:00Z",
            metadata={"nested": {"deep": "value"}, "items": [1, 2]},
        )
        d = entry.to_dict()
        assert d["metadata"]["nested"]["deep"] == "value"
        d["metadata"]["nested"]["deep"] = "mutated"
        d["metadata"]["items"].append(3)
        assert entry.metadata["nested"]["deep"] == "value"
        assert entry.metadata["items"] == [1, 2]

    def test_reflection_entry_metadata_empty(self):
        entry = ReflectionEntry(
            entry_id="ref_1",
            topic="test",
            description="",
            reflection_type=ReflectionType.DAILY,
            what_went_well=[],
            what_could_improve=[],
            lessons_learned=[],
            confidence_before=0.5,
            confidence_after=0.5,
            domain="",
            linked_decision_ids=[],
            sentiment=0.0,
            created_at="2024-01-01T00:00:00Z",
            metadata={},
        )
        d = entry.to_dict()
        assert d["metadata"] == {}
        d["metadata"]["injected"] = True
        assert entry.metadata == {}

    def test_reflection_entry_list_fields_deepcopy(self):
        entry = ReflectionEntry(
            entry_id="ref_1",
            topic="test",
            description="",
            reflection_type=ReflectionType.DAILY,
            what_went_well=["item1"],
            what_could_improve=["item2"],
            lessons_learned=["item3"],
            confidence_before=0.5,
            confidence_after=0.5,
            domain="",
            linked_decision_ids=["dec1"],
            sentiment=0.0,
            created_at="2024-01-01T00:00:00Z",
            metadata={},
        )
        d = entry.to_dict()
        d["what_went_well"].append("new_item")
        d["what_could_improve"].append("new_item")
        d["lessons_learned"].append("new_item")
        d["linked_decision_ids"].append("new_id")
        assert entry.what_went_well == ["item1"]
        assert entry.what_could_improve == ["item2"]
        assert entry.lessons_learned == ["item3"]
        assert entry.linked_decision_ids == ["dec1"]


class TestDecisionJournalDeepcopy:
    """Tests for decision_journal.py DecisionRecord.to_dict() metadata deepcopy."""

    def test_decision_record_metadata_deepcopy(self):
        record = DecisionRecord(
            decision_id="dec_1",
            title="Test Decision",
            description="Test description",
            alternatives=["alt1", "alt2"],
            chosen="alt1",
            reasoning="Because",
            confidence=0.8,
            biases_flagged=["anchoring"],
            domain="testing",
            decision_type=DecisionType.ROUTINE,
            status=DecisionStatus.PENDING,
            outcome=None,
            outcome_accuracy=None,
            outcome_recorded_at=None,
            created_at="2024-01-01T00:00:00Z",
            metadata={"nested": {"deep": "val"}, "items": [1, 2]},
        )
        d = record.to_dict()
        assert d["metadata"]["nested"]["deep"] == "val"
        d["metadata"]["nested"]["deep"] = "mutated"
        d["metadata"]["items"].append(3)
        assert record.metadata["nested"]["deep"] == "val"
        assert record.metadata["items"] == [1, 2]

    def test_decision_record_metadata_empty(self):
        record = DecisionRecord(
            decision_id="dec_1",
            title="Test",
            description="",
            alternatives=[],
            chosen="",
            reasoning="",
            confidence=0.5,
            biases_flagged=[],
            domain="",
            decision_type=DecisionType.ROUTINE,
            status=DecisionStatus.PENDING,
            outcome=None,
            outcome_accuracy=None,
            outcome_recorded_at=None,
            created_at="2024-01-01T00:00:00Z",
            metadata={},
        )
        d = record.to_dict()
        assert d["metadata"] == {}
        d["metadata"]["injected"] = True
        assert record.metadata == {}

    def test_decision_record_list_fields_deepcopy(self):
        record = DecisionRecord(
            decision_id="dec_1",
            title="Test",
            description="",
            alternatives=["opt_a", "opt_b"],
            chosen="opt_a",
            reasoning="",
            confidence=0.5,
            biases_flagged=["bias1"],
            domain="",
            decision_type=DecisionType.ROUTINE,
            status=DecisionStatus.PENDING,
            outcome=None,
            outcome_accuracy=None,
            outcome_recorded_at=None,
            created_at="2024-01-01T00:00:00Z",
            metadata={},
        )
        d = record.to_dict()
        d["alternatives"].append("opt_c")
        d["biases_flagged"].append("bias2")
        assert record.alternatives == ["opt_a", "opt_b"]
        assert record.biases_flagged == ["bias1"]


class TestShadowGraphIntegration:
    """Integration tests for shadow_graph.py deepcopy with ShadowGraph."""

    def test_shadow_graph_process_event_metadata_deepcopy(self):
        graph = ShadowGraph(graph_id="test_deepcopy")
        import asyncio
        event = {
            "query": "test query",
            "observations": ["obs1", "obs2"],
            "metadata": {"nested": {"deep": "val"}, "tags": ["a", "b"]},
        }
        result = asyncio.run(graph.process_event(event))
        d = result.to_dict()
        d["metadata"]["nested"]["deep"] = "mutated"
        assert result.metadata["nested"]["deep"] == "val"

    def test_shadow_graph_checkpoint_metadata_deepcopy(self):
        graph = ShadowGraph(graph_id="test_checkpoint_dc")
        cp = Checkpoint(
            checkpoint_id="ckpt_dc",
            state={"mode": "idle"},
            node_name="listen",
            timestamp="2024-01-01T00:00:00Z",
            execution_id="exec_dc",
            metadata={"info": {"nested": "secret"}, "list": [1, 2]},
        )
        d = cp.to_dict()
        assert d["metadata"] == {"info": {"nested": "secret"}, "list": [1, 2]}
        d["metadata"]["info"]["nested"] = "breached"
        d["metadata"]["list"].append(3)
        assert cp.metadata["info"]["nested"] == "secret"
        assert cp.metadata["list"] == [1, 2]


class TestDailyBriefingDeepcopy:
    """Tests for daily_briefing.py to_dict() metadata deepcopy."""

    def test_briefing_item_metadata_deepcopy(self):
        item = BriefingItem(
            section=BriefingSection.OVERNIGHT,
            title="Test Item",
            content="Test content",
            metadata={"nested": {"deep": "val"}, "items": [1, 2]},
        )
        d = item.to_dict()
        assert d["metadata"]["nested"]["deep"] == "val"
        d["metadata"]["nested"]["deep"] = "mutated"
        d["metadata"]["items"].append(3)
        assert item.metadata["nested"]["deep"] == "val"
        assert item.metadata["items"] == [1, 2]

    def test_daily_briefing_metadata_deepcopy(self):
        briefing = DailyBriefing(
            briefing_id="test_brief",
            metadata={"config": {"key": "val"}, "list": ["a", "b"]},
        )
        d = briefing.to_dict()
        assert d["metadata"]["config"]["key"] == "val"
        d["metadata"]["config"]["key"] = "mutated"
        d["metadata"]["list"].append("c")
        assert briefing.metadata["config"]["key"] == "val"
        assert briefing.metadata["list"] == ["a", "b"]


class TestEscalationDeepcopy:
    """Tests for escalation.py to_dict() deepcopy fixes."""

    def test_action_request_parameters_deepcopy(self):
        req = ActionRequest(
            action_type="test",
            parameters={"nested": {"deep": "val"}, "items": [1, 2, 3]},
            domain="testing",
        )
        d = req.to_dict()
        assert d["parameters"]["nested"]["deep"] == "val"
        d["parameters"]["nested"]["deep"] = "mutated"
        d["parameters"]["items"].append(4)
        assert req.parameters["nested"]["deep"] == "val"
        assert req.parameters["items"] == [1, 2, 3]

    def test_escalation_record_metadata_deepcopy(self):
        rec = EscalationRecord(
            level=AutonomyLevel.ASK,
            metadata={"nested": {"key": "val"}, "list": [1, 2]},
        )
        d = rec.to_dict()
        assert d["metadata"]["nested"]["key"] == "val"
        d["metadata"]["nested"]["key"] = "mutated"
        d["metadata"]["list"].append(3)
        assert rec.metadata["nested"]["key"] == "val"
        assert rec.metadata["list"] == [1, 2]

    def test_escalation_record_metadata_none(self):
        rec = EscalationRecord(level=AutonomyLevel.AUTO, metadata={})
        d = rec.to_dict()
        assert d["metadata"] == {}


class TestAttentionEngineDeepcopy:
    """Tests for attention_engine.py to_dict() metadata deepcopy."""

    def test_attention_item_metadata_deepcopy(self):
        item = AttentionItem(
            item_id="att_1",
            topic="test",
            level=AttentionLevel.HIGH,
            domain="testing",
            urgency_score=0.5,
            importance_score=0.7,
            created_at="2024-01-01T00:00:00Z",
            last_focused_at=None,
            focus_count=0,
            metadata={"nested": {"deep": "val"}, "items": [1, 2]},
        )
        d = item.to_dict()
        assert d["metadata"]["nested"]["deep"] == "val"
        d["metadata"]["nested"]["deep"] = "mutated"
        d["metadata"]["items"].append(3)
        assert item.metadata["nested"]["deep"] == "val"
        assert item.metadata["items"] == [1, 2]