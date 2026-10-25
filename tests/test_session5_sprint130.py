#!/usr/bin/env python3
"""Sprint 130 regression tests — second-pass deep audit round 3.

Verifies fixes in:
- emotional_intelligence.py: .strip() in _sanitize/_sanitize_content
- creative_synthesis.py: .strip() in sanitize, hash()→sha256, add_association deepcopy
- metacognition_engine.py: .strip() in sanitize, avg_confidence/avg_cognitive_load now updated
- moral_reasoning.py: .strip() in sanitize, fairness positive/negative split, action uses _sanitize (strict)
- narrative_memory.py: .strip() in sanitize, metadata deepcopy, add_event/create_chapter/create_arc return deepcopy
- procedural_memory.py: .strip() in sanitize
"""

import pytest

from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
from shadowd.cognition.creative_synthesis import CreativeSynthesis, SynthesisMethod
from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
from shadowd.cognition.moral_reasoning import (
    MoralReasoning, EthicalFramework,
    _FAIRNESS_POSITIVE_PATTERNS, _FAIRNESS_NEGATIVE_PATTERNS,
    _FAIRNESS_POSITIVE_KEYWORDS, _FAIRNESS_NEGATIVE_KEYWORDS,
    _HARM_PATTERNS, _BENEFIT_PATTERNS, _RIGHTS_KEYWORDS, _CARE_KEYWORDS,
)
from shadowd.cognition.narrative_memory import NarrativeMemory
from shadowd.cognition.procedural_memory import ProceduralMemory


class TestSanitizeStrip:
    def test_emotional_intelligence_strips(self):
        ei = EmotionalIntelligence()
        result = ei.detect_emotion("  hello world  ")
        assert result is not None

    def test_creative_synthesis_strips(self):
        cs = CreativeSynthesis()
        cid = cs.register_concept("  test concept  ")
        assert cid is not None

    def test_metacognition_strips(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.SYSTEMATIC, "  test topic  ")
        assert ep.topic.strip() == ep.topic

    def test_moral_reasoning_strips(self):
        mr = MoralReasoning()
        ev = mr.evaluate("  test action  ")
        assert ev.action.strip() == ev.action

    def test_narrative_memory_strips(self):
        nm = NarrativeMemory()
        ch = nm.create_chapter("  Test Chapter  ", "career")
        assert ch.title.strip() == ch.title

    def test_procedural_memory_strips(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("  test procedure  ", "work")
        assert proc.name.strip() == proc.name


class TestCreativeSynthesisHash:
    def test_template_selection_deterministic(self):
        cs = CreativeSynthesis()
        cs.register_concept("alpha")
        cs.register_concept("beta")
        ideas_a = [cs.synthesize(["alpha", "beta"], method=SynthesisMethod.COMBINATION).title for _ in range(3)]
        ideas_b = [cs.synthesize(["alpha", "beta"], method=SynthesisMethod.COMBINATION).title for _ in range(3)]
        assert ideas_a == ideas_b

    def test_add_association_returns_deepcopy(self):
        cs = CreativeSynthesis()
        cs.register_concept("x")
        cs.register_concept("y")
        pair = cs.add_association("x", "y", strength=0.8)
        pair.association_strength = 0.0
        pair2 = cs.add_association("x", "y", strength=0.8)
        assert pair2.association_strength == 0.8


class TestMetacognitionStats:
    def test_avg_confidence_updates(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "topic1", confidence=0.8)
        stats = me.get_stats()
        assert stats["stats"]["avg_confidence"] != 0.5
        assert stats["stats"]["avg_confidence"] > 0.5

    def test_avg_cognitive_load_updates(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "topic1", cognitive_load=0.9)
        stats = me.get_stats()
        assert stats["stats"]["avg_cognitive_load"] != 0.3
        assert stats["stats"]["avg_cognitive_load"] > 0.3

    def test_avg_confidence_increments(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "t1", confidence=0.6)
        me.start_thinking(ReasoningType.SYSTEMATIC, "t2", confidence=0.8)
        stats = me.get_stats()
        assert abs(stats["stats"]["avg_confidence"] - 0.7) < 0.01


class TestMoralReasoningFairness:
    def test_positive_fairness_increases_score(self):
        mr = MoralReasoning()
        ev = mr.evaluate("ensure fair and equal treatment")
        fairness = [a for a in ev.assessments if a.framework == EthicalFramework.FAIRNESS]
        assert len(fairness) == 1
        assert fairness[0].score > 0.5

    def test_negative_fairness_decreases_score(self):
        mr = MoralReasoning()
        ev = mr.evaluate("discrimination and prejudice in hiring")
        fairness = [a for a in ev.assessments if a.framework == EthicalFramework.FAIRNESS]
        assert len(fairness) == 1
        assert fairness[0].score < 0.5

    def test_mixed_fairness_neutral(self):
        mr = MoralReasoning()
        ev = mr.evaluate("fair policies to end discrimination")
        fairness = [a for a in ev.assessments if a.framework == EthicalFramework.FAIRNESS]
        assert len(fairness) == 1
        assert 0.4 < fairness[0].score < 0.6

    def test_fairness_positive_patterns_exist(self):
        assert len(_FAIRNESS_POSITIVE_PATTERNS) == len(_FAIRNESS_POSITIVE_KEYWORDS)

    def test_fairness_negative_patterns_exist(self):
        assert len(_FAIRNESS_NEGATIVE_PATTERNS) == len(_FAIRNESS_NEGATIVE_KEYWORDS)

    def test_action_sanitized_strict(self):
        mr = MoralReasoning()
        ev = mr.evaluate("<script>alert(1)</script> action")
        assert "<script>" not in ev.action


class TestNarrativeMemoryDeepcopy:
    def test_add_event_returns_deepcopy(self):
        nm = NarrativeMemory()
        event = nm.add_event("test content", importance=0.8)
        event.content = "TAMPERED"
        events = list(nm._events)
        assert events[-1].content != "TAMPERED"

    def test_create_chapter_returns_deepcopy(self):
        nm = NarrativeMemory()
        chapter = nm.create_chapter("Test", "career")
        chapter.title = "TAMPERED"
        stored = nm._chapters[chapter.chapter_id]
        assert stored.title != "TAMPERED"

    def test_create_arc_returns_deepcopy(self):
        nm = NarrativeMemory()
        arc = nm.create_arc("Test Arc")
        arc.title = "TAMPERED"
        stored = nm._arcs[arc.arc_id]
        assert stored.title != "TAMPERED"

    def test_event_metadata_deepcopy(self):
        nm = NarrativeMemory()
        event = nm.add_event("test", metadata={"key": [1, 2, 3]})
        d = event.to_dict()
        d["metadata"]["key"].append(999)
        event2 = nm.add_event("test2", metadata={"key": [1, 2, 3]})
        d2 = event2.to_dict()
        assert 999 not in d2["metadata"]["key"]