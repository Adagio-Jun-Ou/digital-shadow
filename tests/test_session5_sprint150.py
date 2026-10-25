#!/usr/bin/env python3
"""
Sprint 150 — Global round() Audit for cognition modules.

Tests that every to_dict() method uses round() on float values,
ensuring deterministic serialization. Float fields are set to
irrational numbers (e.g., 1/3) and verified to be rounded.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shadowd.cognition.decision_journal import DecisionRecord, BiasRetrospective, DecisionPattern, DecisionType, DecisionStatus
from shadowd.cognition.neural_mesh import MeshNode, NodeStatus
from shadowd.cognition.thought_sense import IntentResult, EmotionResult, SubtextResult, ThoughtAnalysis
from shadowd.cognition.shadow_graph import CognitiveState
from shadowd.cognition.dark_memory import MemoryAtom, MemoryTier, AtomType
from shadowd.cognition.device_context import LocationPoint
from shadowd.cognition.cognitive_reflection import ReflectionEntry, LearningInsight, ReflectionType


IRRATIONAL = 1.0 / 3.0
PI = math.pi


def _assert_rounded(value, label, max_decimals=6):
    assert value != IRRATIONAL, f"{label}: raw irrational leaked through"
    s = str(value)
    if "." in s:
        decimals = len(s.split(".", 1)[1].rstrip("0"))
        assert decimals <= max_decimals, f"{label}: {s} has {decimals} decimal places (max {max_decimals})"


class TestDecisionJournal:
    def test_bias_retrospective_accuracy_avg_rounded(self):
        br = BiasRetrospective(
            retro_id="r1",
            period_start="2024-01-01",
            period_end="2024-01-31",
            decisions_reviewed=10,
            biases_found={},
            accuracy_avg=IRRATIONAL,
            overconfidence_rate=IRRATIONAL,
            findings=[],
            recommendations=[],
            created_at="2024-01-31",
        )
        d = br.to_dict()
        _assert_rounded(d["accuracy_avg"], "accuracy_avg", 4)
        _assert_rounded(d["overconfidence_rate"], "overconfidence_rate", 4)

    def test_decision_record_outcome_accuracy_rounded(self):
        rec = DecisionRecord(
            decision_id="d1",
            title="test",
            description="test",
            alternatives=[],
            chosen="a",
            reasoning="r",
            confidence=IRRATIONAL,
            biases_flagged=[],
            domain="work",
            decision_type=DecisionType.ROUTINE,
            status=DecisionStatus.PENDING,
            outcome=None,
            outcome_accuracy=IRRATIONAL,
            outcome_recorded_at="2024-01-01",
            created_at="2024-01-01",
            metadata={},
        )
        d = rec.to_dict()
        _assert_rounded(d["confidence"], "confidence", 3)
        _assert_rounded(d["outcome_accuracy"], "outcome_accuracy", 4)

    def test_decision_pattern_confidence_rounded(self):
        dp = DecisionPattern(
            pattern_id="p1",
            pattern_type="overconfident",
            description="test",
            frequency=5,
            confidence=IRRATIONAL,
            domains=["work"],
            created_at="2024-01-01",
        )
        d = dp.to_dict()
        _assert_rounded(d["confidence"], "pattern.confidence", 3)


class TestNeuralMesh:
    def test_mesh_node_trust_level_rounded(self):
        node = MeshNode(
            node_id="n1",
            name="test",
            device_type="server",
            address="127.0.0.1",
            port=8765,
            status=NodeStatus.ONLINE,
            capabilities=[],
            trust_level=IRRATIONAL,
        )
        d = node.to_dict()
        _assert_rounded(d["trust_level"], "trust_level", 3)


class TestThoughtSense:
    def test_intent_result_confidence_rounded(self):
        ir = IntentResult(intent="QUERY", confidence=IRRATIONAL, keywords_matched=["test"])
        d = ir.to_dict()
        _assert_rounded(d["confidence"], "intent.confidence", 3)

    def test_emotion_result_intensity_rounded(self):
        er = EmotionResult(emotion="JOY", intensity=IRRATIONAL, keywords_matched=["happy"])
        d = er.to_dict()
        _assert_rounded(d["intensity"], "intensity", 3)

    def test_subtext_result_confidence_rounded(self):
        sr = SubtextResult(subtext_type="HESITATION", confidence=IRRATIONAL, evidence=["maybe"])
        d = sr.to_dict()
        _assert_rounded(d["confidence"], "subtext.confidence", 3)

    def test_thought_analysis_cognitive_load_rounded(self):
        ta = ThoughtAnalysis(
            text="test",
            intents=[IntentResult(intent="QUERY", confidence=0.8, keywords_matched=[])],
            emotions=[EmotionResult(emotion="NEUTRAL", intensity=0.5, keywords_matched=[])],
            subtexts=[SubtextResult(subtext_type="NONE", confidence=0.5, evidence=[])],
            cognitive_load=IRRATIONAL,
            urgency="LOW",
            urgency_confidence=IRRATIONAL,
            timestamp="2024-01-01",
            summary="test",
        )
        d = ta.to_dict()
        _assert_rounded(d["cognitive_load"], "cognitive_load", 3)
        _assert_rounded(d["urgency_confidence"], "urgency_confidence", 3)


class TestShadowGraph:
    def test_cognitive_state_confidence_rounded(self):
        cs = CognitiveState(confidence=IRRATIONAL)
        d = cs.to_dict()
        _assert_rounded(d["confidence"], "state.confidence", 3)


class TestDarkMemory:
    def test_memory_atom_importance_rounded(self):
        atom = MemoryAtom(
            atom_id="a1",
            content="test",
            importance=IRRATIONAL,
            atom_type=AtomType.OBSERVATION,
            tier=MemoryTier.WORKING,
        )
        d = atom.to_dict()
        _assert_rounded(d["importance"], "importance", 3)


class TestDeviceContext:
    def test_location_point_accuracy_rounded(self):
        lp = LocationPoint(latitude=37.7749, longitude=-122.4194, accuracy=IRRATIONAL)
        d = lp.to_dict()
        _assert_rounded(d["accuracy"], "location.accuracy", 3)


class TestCognitiveReflection:
    def test_reflection_entry_confidence_rounded(self):
        re = ReflectionEntry(
            entry_id="re1",
            topic="test topic",
            description="test",
            reflection_type=ReflectionType.DAILY,
            what_went_well=["a"],
            what_could_improve=["b"],
            lessons_learned=["c"],
            confidence_before=IRRATIONAL,
            confidence_after=IRRATIONAL,
            domain="work",
            linked_decision_ids=[],
            sentiment=IRRATIONAL,
            created_at="2024-01-01",
            metadata={},
        )
        d = re.to_dict()
        _assert_rounded(d["confidence_before"], "confidence_before", 3)
        _assert_rounded(d["confidence_after"], "confidence_after", 3)
        _assert_rounded(d["sentiment"], "sentiment", 3)

    def test_learning_insight_confidence_rounded(self):
        li = LearningInsight(
            insight_id="li1",
            title="test",
            description="test",
            category="pattern",
            source_reflection_ids=["re1"],
            frequency=5,
            confidence=IRRATIONAL,
            applicable_domains=["work"],
            created_at="2024-01-01",
        )
        d = li.to_dict()
        _assert_rounded(d["confidence"], "insight.confidence", 3)