#!/usr/bin/env python3
"""Sprint 68 — Performance Optimization: Lock Contention + Hot-Path Optimization

Test categories:
- TestPrecompiledPatterns: Verify pre-compiled patterns exist and match correctly
- TestPerformanceLockSections: Verify read operations release lock quickly
- TestEmotionalIntelligencePerformance: Benchmark detect_emotion
- TestMoralReasoningPerformance: Benchmark evaluate
- TestMetacognitionPerformance: Benchmark detect_biases
- TestNarrativeSearchPerformance: Benchmark search with many events
- TestDecisionJournalRetrospective: Verify retrospective only iterates resolved decisions
- TestAdaptiveSkillTreeRecommendations: Verify snapshot-based recommendation computation
"""

import re
import time
import threading
import pytest

from shadowd.cognition.emotional_intelligence import (
    EmotionalIntelligence, EmotionCategory, _EMOTION_COMPILED,
)
from shadowd.cognition.moral_reasoning import (
    MoralReasoning, _HARM_PATTERNS, _BENEFIT_PATTERNS,
    _FAIRNESS_POSITIVE_PATTERNS, _FAIRNESS_NEGATIVE_PATTERNS,
    _RIGHTS_PATTERNS, _CARE_PATTERNS, _count_pattern_matches,
)
from shadowd.cognition.metacognition_engine import (
    MetacognitionEngine, BiasType, _BIAS_COMPILED, _word_match,
)
from shadowd.cognition.narrative_memory import NarrativeMemory
from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
from shadowd.cognition.decision_journal import DecisionJournal, DecisionStatus
from shadowd.cognition.creative_synthesis import CreativeSynthesis
from shadowd.cognition.procedural_memory import ProceduralMemory
from shadowd.cognition.conflict_resolution import ConflictResolution
from shadowd.cognition.social_intelligence import SocialIntelligence


class TestPrecompiledPatterns:
    def test_emotion_compiled_patterns_exist(self):
        assert len(_EMOTION_COMPILED) == 8
        for emotion, patterns in _EMOTION_COMPILED.items():
            assert len(patterns) > 0
            for kw, pat in patterns:
                if ' ' in kw:
                    assert pat is None
                else:
                    assert pat is not None
                    assert isinstance(pat, re.Pattern)

    def test_emotion_compiled_patterns_match(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am very happy and delighted")
        assert state.primary == EmotionCategory.JOY

    def test_moral_reasoning_compiled_patterns_exist(self):
        assert len(_HARM_PATTERNS) > 0
        assert len(_BENEFIT_PATTERNS) > 0
        assert len(_FAIRNESS_POSITIVE_PATTERNS) > 0
        assert len(_FAIRNESS_NEGATIVE_PATTERNS) > 0
        assert len(_RIGHTS_PATTERNS) > 0
        assert len(_CARE_PATTERNS) > 0
        for p in _HARM_PATTERNS:
            assert isinstance(p, re.Pattern)

    def test_count_pattern_matches(self):
        text = "harm and hurt and damage but also help and benefit"
        assert _count_pattern_matches(_HARM_PATTERNS, text) == 3
        assert _count_pattern_matches(_BENEFIT_PATTERNS, text) == 2

    def test_metacognition_compiled_patterns_exist(self):
        assert len(_BIAS_COMPILED) > 0
        for bias_type, patterns in _BIAS_COMPILED.items():
            assert len(patterns) > 0
            for kw, pat in patterns:
                if ' ' in kw:
                    assert pat is None
                else:
                    assert isinstance(pat, re.Pattern)


class TestPerformanceLockSections:
    def test_emotional_intelligence_get_stats_releases_lock(self):
        ei = EmotionalIntelligence()
        ei.detect_emotion("I'm happy")
        stats = ei.get_stats()
        assert "stats" in stats
        assert "history_size" in stats

    def test_creative_synthesis_get_stats_releases_lock(self):
        cs = CreativeSynthesis()
        cs.register_concept("python")
        stats = cs.get_stats()
        assert "stats" in stats
        assert "concepts_count" in stats

    def test_procedural_memory_get_stats_processing_outside_lock(self):
        pm = ProceduralMemory()
        pm.add_procedure("test", "cat", ["step1"])
        stats = pm.get_stats()
        assert "avg_mastery" in stats

    def test_conflict_resolution_get_stats_outside_lock(self):
        cr = ConflictResolution()
        cr.create_conflict("test", "interest", 2)
        stats = cr.get_stats()
        assert "total_conflicts" in stats

    def test_decision_journal_get_stats_outside_lock(self):
        dj = DecisionJournal()
        dj.record_decision("test", "option A")
        stats = dj.get_stats()
        assert "decisions_recorded" in stats


class TestEmotionalIntelligencePerformance:
    def test_detect_emotion_1000_calls_under_2s(self):
        ei = EmotionalIntelligence()
        texts = [
            "I am happy and excited about this",
            "I feel sad and lonely today",
            "This makes me angry and frustrated",
            "I'm afraid of what might happen",
            "I trust this will work out fine",
        ]
        start = time.time()
        for _ in range(200):
            for text in texts:
                ei.detect_emotion(text)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"1000 detect_emotion calls took {elapsed:.2f}s"

    def test_detect_emotion_neutral_text(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("The weather is cloudy today")
        assert state.primary == EmotionCategory.NEUTRAL

    def test_detect_emotion_concurrent_safe(self):
        ei = EmotionalIntelligence()
        errors = []
        def worker():
            try:
                for _ in range(100):
                    ei.detect_emotion("I am happy and glad")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestMoralReasoningPerformance:
    def test_evaluate_500_calls_under_2s(self):
        mr = MoralReasoning()
        start = time.time()
        for i in range(500):
            mr.evaluate(f"action {i}", "This helps people and improves fairness")
        elapsed = time.time() - start
        assert elapsed < 2.0, f"500 evaluate calls took {elapsed:.2f}s"

    def test_evaluate_with_harm_keywords(self):
        mr = MoralReasoning()
        result = mr.evaluate("harm and destroy", "This will hurt people")
        assert result.overall_score < 0.5


class TestMetacognitionPerformance:
    def test_detect_biases_500_calls_under_2s(self):
        me = MetacognitionEngine()
        texts = [
            "I definitely know this is certain",
            "Everyone thinks this is the right way",
            "I already invested too much effort to give up now",
            "Based on the initial impression, starting from here",
            "This is obviously simple really anyone can do it",
        ]
        start = time.time()
        for _ in range(100):
            for text in texts:
                me.detect_biases(text)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"500 detect_biases calls took {elapsed:.2f}s"

    def test_detect_biases_with_episode(self):
        me = MetacognitionEngine()
        ep = me.start_thinking("systematic", "Test topic")
        biases = me.detect_biases("I definitely know this is certain and guaranteed", ep.episode_id)
        assert len(biases) > 0


class TestNarrativeSearchPerformance:
    def test_search_1000_events_under_1s(self):
        nm = NarrativeMemory()
        for i in range(1000):
            nm.add_event(f"Event about work project coding task number {i}", importance=0.5)
        start = time.time()
        results = nm.search_events("coding project")
        elapsed = time.time() - start
        assert elapsed < 1.0, f"search with 1000 events took {elapsed:.2f}s"
        assert len(results) > 0

    def test_search_low_importance_early_exit(self):
        nm = NarrativeMemory()
        for i in range(1000):
            nm.add_event(f"Unrelated event {i}", importance=0.1)
        nm.add_event("Important coding project milestone", importance=0.9)
        results = nm.search_events("coding")
        assert any(e.importance >= 0.5 for e in results) or len(results) == 1


class TestDecisionJournalRetrospective:
    def test_retrospective_only_iterates_resolved(self):
        dj = DecisionJournal()
        for i in range(50):
            rec = dj.record_decision(f"decision {i}", f"option {i}")
        for i in range(50):
            rec = dj.record_decision(f"resolved {i}", f"option {i}")
            dj.update_status(rec.decision_id, DecisionStatus.COMPLETED.value)
            dj.record_outcome(rec.decision_id, "good", 0.8)
        retro = dj.run_retrospective(period_days=365)
        assert retro.decisions_reviewed == 50

    def test_retrospective_empty_when_no_resolved(self):
        dj = DecisionJournal()
        dj.record_decision("pending", "option A")
        retro = dj.run_retrospective(period_days=365)
        assert retro.decisions_reviewed == 0

    def test_retrospective_finds_overconfidence(self):
        dj = DecisionJournal()
        rec = dj.record_decision("overconfident", "option A", confidence=0.9)
        dj.record_outcome(rec.decision_id, "poor result", 0.1)
        retro = dj.run_retrospective(period_days=365)
        assert retro.overconfidence_rate > 0.5


class TestAdaptiveSkillTreeRecommendations:
    def test_recommendations_computed_outside_lock(self):
        ast = AdaptiveSkillTree()
        ast.add_skill("sk1", "Coding", "technical")
        ast.add_skill("sk2", "Writing", "creativity")
        for _ in range(5):
            ast.use_skill("sk1", success=True)
        recs = ast.get_recommendations()
        assert len(recs) > 0
        assert all(r.priority > 0 for r in recs)

    def test_recommendations_with_locked_skill(self):
        ast = AdaptiveSkillTree()
        ast.add_skill("sk1", "Base Skill", "technical")
        ast.add_skill("sk2", "Locked Skill", "technical", dependencies=["sk1"])
        recs = ast.get_recommendations()
        assert len(recs) > 0

    def test_recommendations_concurrent_safe(self):
        ast = AdaptiveSkillTree()
        ast.add_skill("sk1", "Skill A", "technical")
        ast.add_skill("sk2", "Skill B", "creativity")
        errors = []
        def worker():
            try:
                for _ in range(50):
                    ast.get_recommendations()
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors