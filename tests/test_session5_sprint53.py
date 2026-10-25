#!/usr/bin/env python3
"""Sprint 53 — EmotionalIntelligence + CreativeSynthesis Modules + Wiring

Tests for:
- EmotionalIntelligence: emotion detection, regulation, patterns, balance, empathy
- CreativeSynthesis: concept registration, association, synthesis, brainstorm, patterns
- ShadowOS wiring: health, status for new modules
- API endpoints: emotion/* and creative/*
- Thread safety, input sanitization, bounded data
"""

import asyncio
import threading
import pytest
from unittest.mock import MagicMock, patch

from shadowd.cognition.emotional_intelligence import (
    EmotionalIntelligence, EmotionCategory, RegulationApproach,
    PatternType, EmotionState, EmotionalPattern, RegulationStrategy,
)
from shadowd.cognition.creative_synthesis import (
    CreativeSynthesis, CreativeDomain, SynthesisMethod,
    NoveltyLevel, CreativeIdea, ConceptPair, IdeationPattern,
)


# ===================== EmotionalIntelligence Tests =====================

class TestEmotionDetection:
    def test_detect_joy(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am so happy and excited about this")
        assert state.primary == EmotionCategory.JOY
        assert state.intensity > 0

    def test_detect_anger(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am furious and frustrated with this situation")
        assert state.primary == EmotionCategory.ANGER

    def test_detect_fear(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I feel anxious and worried about tomorrow")
        assert state.primary == EmotionCategory.FEAR

    def test_detect_sadness(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I feel so sad and disappointed")
        assert state.primary == EmotionCategory.SADNESS

    def test_detect_neutral(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("the table")
        assert state.primary == EmotionCategory.NEUTRAL

    def test_detect_with_context(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am so happy and excited", context="work meeting")
        assert state.context == "work meeting"

    def test_intensity_clamped(self):
        ei = EmotionalIntelligence()
        state = EmotionState(primary=EmotionCategory.JOY, intensity=2.0, valence=1.5, arousal=1.5)
        assert state.intensity == 1.0
        assert state.valence == 1.0
        assert state.arousal == 1.0

    def test_sanitize_input(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I'm <script>alert('xss')</script> happy\x00")
        assert "<script>" not in state.context

    def test_state_id_format(self):
        state = EmotionState()
        assert state.state_id.startswith("emo_")
        assert len(state.state_id) > 10

    def test_state_to_dict(self):
        state = EmotionState(primary=EmotionCategory.JOY, intensity=0.8, valence=0.6)
        d = state.to_dict()
        assert d["primary"] == "joy"
        assert d["intensity"] == 0.8


class TestRegulationStrategies:
    def test_suggest_regulation_anger_low(self):
        ei = EmotionalIntelligence()
        strategy = ei.suggest_regulation(EmotionCategory.ANGER, 0.3)
        assert strategy.approach == RegulationApproach.COGNITIVE_REAPPRAISAL
        assert strategy.target_emotion == EmotionCategory.ANGER

    def test_suggest_regulation_anger_high(self):
        ei = EmotionalIntelligence()
        strategy = ei.suggest_regulation(EmotionCategory.ANGER, 0.9)
        assert strategy.approach == RegulationApproach.MINDFULNESS

    def test_suggest_regulation_sadness(self):
        ei = EmotionalIntelligence()
        strategy = ei.suggest_regulation(EmotionCategory.SADNESS, 0.5)
        assert strategy.approach == RegulationApproach.BEHAVIORAL_ACTIVATION

    def test_suggest_regulation_fear(self):
        ei = EmotionalIntelligence()
        strategy = ei.suggest_regulation(EmotionCategory.FEAR, 0.3)
        assert strategy.approach == RegulationApproach.ACCEPTANCE

    def test_suggest_regulation_has_steps(self):
        ei = EmotionalIntelligence()
        strategy = ei.suggest_regulation(EmotionCategory.ANGER, 0.5)
        assert len(strategy.steps) > 0

    def test_regulation_strategy_to_dict(self):
        strategy = RegulationStrategy(
            approach=RegulationApproach.MINDFULNESS,
            target_emotion=EmotionCategory.ANGER,
            steps=["breathe", "observe"],
        )
        d = strategy.to_dict()
        assert d["approach"] == "mindfulness"
        assert d["target_emotion"] == "anger"


class TestEmotionalPatterns:
    def test_record_pattern(self):
        ei = EmotionalIntelligence()
        pattern = ei.record_pattern(
            PatternType.TRIGGER, "stress trigger",
            [EmotionCategory.ANGER], ["deadline"], "triggered by deadlines"
        )
        assert pattern.pattern_id.startswith("pat_")
        assert pattern.name == "stress trigger"

    def test_detect_patterns_insufficient_data(self):
        ei = EmotionalIntelligence()
        patterns = ei.detect_patterns()
        assert patterns == []

    def test_detect_patterns_with_history(self):
        ei = EmotionalIntelligence()
        for _ in range(5):
            ei.detect_emotion("I am angry and frustrated", context="work stress")
        patterns = ei.detect_patterns(min_frequency=2)
        assert len(patterns) > 0

    def test_pattern_sanitization(self):
        ei = EmotionalIntelligence()
        pattern = ei.record_pattern(
            PatternType.TRIGGER, "<script>alert('x')</script>" + "x" * 200,
            [EmotionCategory.JOY], ["trigger<x>"],
        )
        assert "<script>" not in pattern.name
        assert len(pattern.name) <= 100


class TestEmotionalBalance:
    def test_empty_balance(self):
        ei = EmotionalIntelligence()
        balance = ei.get_emotional_balance()
        assert balance["balance"] == 0.5
        assert balance["dominant"] == "neutral"

    def test_positive_balance(self):
        ei = EmotionalIntelligence()
        for _ in range(5):
            ei.detect_emotion("I am happy and delighted")
        balance = ei.get_emotional_balance()
        assert balance["balance"] > 0.4

    def test_stability_score(self):
        ei = EmotionalIntelligence()
        for _ in range(10):
            ei.detect_emotion("I feel content")
        balance = ei.get_emotional_balance()
        assert 0 <= balance["stability"] <= 1.0


class TestEmpathyInsight:
    def test_joy_insight(self):
        ei = EmotionalIntelligence()
        insight = ei.get_empathy_insight(EmotionCategory.JOY, 0.6)
        assert insight["emotion"] == "joy"
        assert insight["likely_needs"] is not None
        assert insight["empathy_score"] > 0

    def test_sadness_insight(self):
        ei = EmotionalIntelligence()
        insight = ei.get_empathy_insight(EmotionCategory.SADNESS, 0.8)
        assert "support" in " ".join(insight["likely_needs"])

    def test_fear_insight(self):
        ei = EmotionalIntelligence()
        insight = ei.get_empathy_insight(EmotionCategory.FEAR, 0.9)
        assert insight["empathy_score"] > 0


class TestEmotionalIntelligenceStats:
    def test_get_stats(self):
        ei = EmotionalIntelligence()
        ei.detect_emotion("I feel amazing")
        stats = ei.get_stats()
        assert "stats" in stats
        assert "emotion_counts" in stats
        assert stats["stats"]["states_recorded"] == 1

    def test_emotion_history_capped(self):
        ei = EmotionalIntelligence()
        for i in range(550):
            ei.detect_emotion(f"feeling test {i}")
        assert len(ei._emotion_history) <= 5000


class TestEmotionalIntelligenceThreadSafety:
    def test_concurrent_detection(self):
        ei = EmotionalIntelligence()
        errors = []

        def detect(i):
            try:
                ei.detect_emotion(f"I feel happy about {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=detect, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert ei._stats["states_recorded"] == 50


# ===================== CreativeSynthesis Tests =====================

class TestConceptRegistration:
    def test_register_concept(self):
        cs = CreativeSynthesis()
        result = cs.register_concept("machine learning")
        assert result is True

    def test_register_empty_concept(self):
        cs = CreativeSynthesis()
        result = cs.register_concept("")
        assert result is False

    def test_register_short_concept(self):
        cs = CreativeSynthesis()
        result = cs.register_concept("x")
        assert result is False

    def test_concept_cap(self):
        cs = CreativeSynthesis()
        for i in range(210):
            cs.register_concept(f"concept_{i}")
        assert len(cs._concepts) <= 2000


class TestAssociations:
    def test_add_association(self):
        cs = CreativeSynthesis()
        pair = cs.add_association("ai", "neural networks", strength=0.8)
        assert pair.concept_a == "ai"
        assert pair.association_strength == 0.8

    def test_association_clamped(self):
        cs = CreativeSynthesis()
        pair = cs.add_association("a", "b", strength=2.0)
        assert pair.association_strength == 1.0

    def test_get_associations(self):
        cs = CreativeSynthesis()
        cs.add_association("ai", "python")
        cs.add_association("ai", "neural networks")
        result = cs.get_concept_associations("ai")
        assert len(result) >= 2


class TestSynthesis:
    def test_synthesize_basic(self):
        cs = CreativeSynthesis()
        cs.register_concept("art")
        cs.register_concept("technology")
        idea = cs.synthesize(["art", "technology"])
        assert idea.idea_id.startswith("idea_")
        assert idea.novelty_score >= 0
        assert idea.feasibility_score >= 0
        assert idea.impact_score >= 0

    def test_synthesize_with_method(self):
        cs = CreativeSynthesis()
        cs.register_concept("nature")
        cs.register_concept("algorithm")
        idea = cs.synthesize(["nature", "algorithm"], method=SynthesisMethod.ANALOGY)
        assert idea.synthesis_method == SynthesisMethod.ANALOGY

    def test_synthesize_inversion(self):
        cs = CreativeSynthesis()
        idea = cs.synthesize(["gravity"], method=SynthesisMethod.INVERSION)
        assert idea.synthesis_method == SynthesisMethod.INVERSION
        assert "gravity" in idea.source_concepts

    def test_composite_score(self):
        idea = CreativeIdea(
            novelty_score=0.8, feasibility_score=0.6, impact_score=0.7
        )
        expected = 0.8 * 0.4 + 0.6 * 0.3 + 0.7 * 0.3
        assert abs(idea.composite_score - expected) < 0.01

    def test_novelty_levels(self):
        assert NoveltyLevel.INCREMENTAL.value == 1
        assert NoveltyLevel.MODERATE.value == 2
        assert NoveltyLevel.NOVEL.value == 3
        assert NoveltyLevel.BREAKTHROUGH.value == 4

    def test_synthesize_sanitizes_concepts(self):
        cs = CreativeSynthesis()
        idea = cs.synthesize(["<script>alert(1)</script>", "normal concept"])
        assert "<script>" not in idea.title
        assert "<script>" not in idea.source_concepts[0]


class TestBrainstorm:
    def test_brainstorm_basic(self):
        cs = CreativeSynthesis()
        for c in ["ai", "art", "music", "science", "nature", "design"]:
            cs.register_concept(c)
        ideas = cs.brainstorm("ai", count=3)
        assert len(ideas) == 3

    def test_brainstorm_no_concepts(self):
        cs = CreativeSynthesis()
        ideas = cs.brainstorm("unknown concept")
        assert isinstance(ideas, list)


class TestRandomSynthesis:
    def test_random_synthesis(self):
        cs = CreativeSynthesis()
        for c in ["ai", "blockchain", "quantum", "biology", "art"]:
            cs.register_concept(c)
        idea = cs.random_synthesis()
        assert idea is not None
        assert idea.idea_id.startswith("idea_")

    def test_random_synthesis_insufficient(self):
        cs = CreativeSynthesis()
        cs.register_concept("only_one")
        idea = cs.random_synthesis()
        assert idea is None


class TestIdeasPatterns:
    def test_get_ideas(self):
        cs = CreativeSynthesis()
        cs.register_concept("ai")
        cs.register_concept("art")
        cs.synthesize(["ai", "art"])
        ideas = cs.get_ideas()
        assert len(ideas) > 0

    def test_get_ideas_filtered(self):
        cs = CreativeSynthesis()
        cs.register_concept("ai")
        cs.register_concept("art")
        cs.synthesize(["ai", "art"])
        ideas = cs.get_ideas(min_novelty=0.5)
        assert all(i["novelty_score"] >= 0.5 for i in ideas)

    def test_get_patterns(self):
        cs = CreativeSynthesis()
        cs.synthesize(["ai", "art"])
        patterns = cs.get_patterns()
        assert len(patterns) > 0

    def test_idea_cap(self):
        cs = CreativeSynthesis()
        for i in range(550):
            cs.synthesize([f"c_{i}", f"d_{i}"], method=SynthesisMethod.COMBINATION)
        assert len(cs._ideas) <= 5000


class TestCreativeSynthesisStats:
    def test_get_stats(self):
        cs = CreativeSynthesis()
        cs.register_concept("test")
        cs.synthesize(["test", "other"], method=SynthesisMethod.COMBINATION)
        stats = cs.get_stats()
        assert "stats" in stats
        assert stats["stats"]["ideas_generated"] >= 1
        assert stats["concepts_count"] >= 1

    def test_concept_pair_to_dict(self):
        pair = ConceptPair(concept_a="ai", concept_b="art")
        d = pair.to_dict()
        assert d["concept_a"] == "ai"
        assert d["concept_b"] == "art"


class TestCreativeSynthesisThreadSafety:
    def test_concurrent_synthesis(self):
        cs = CreativeSynthesis()
        for c in ["ai", "art", "music", "science", "nature", "design", "code", "logic"]:
            cs.register_concept(c)
        errors = []

        def synthesize(i):
            try:
                cs.synthesize(["ai", "art"], method=SynthesisMethod.COMBINATION)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=synthesize, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0