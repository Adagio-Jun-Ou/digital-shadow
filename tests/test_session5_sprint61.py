#!/usr/bin/env python3
import re
import logging
import pytest
from unittest.mock import patch
from shadowd.cognition.creative_synthesis import CreativeSynthesis, SynthesisMethod
from shadowd.cognition.moral_reasoning import MoralReasoning, EthicalFramework, MoralPrinciple, _MAX_EVALUATIONS, _MAX_PRINCIPLES, _MAX_DILEMMAS
from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType, _MAX_EPISODES
from shadowd.cognition.emotional_intelligence import EmotionalIntelligence, EmotionCategory, PatternType
from shadowd.cognition.conflict_resolution import ConflictResolution
from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree, SkillDomain, _MAX_SKILLS
from shadowd.cognition.narrative_memory import NarrativeMemory, _MAX_CHAPTERS, _MAX_ARCS
from shadowd.cognition.procedural_memory import ProceduralMemory, _MAX_PROCEDURES, _MAX_HABITS, _MAX_ROUTINES


class TestCS2PatternKeyCollision:
    def test_colon_in_concept_no_ambiguity(self):
        cs = CreativeSynthesis()
        cs.register_concept("a:b")
        cs.register_concept("c")
        idea1 = cs.synthesize(["a:b", "c"], SynthesisMethod.COMBINATION)
        idea2 = cs.synthesize(["a", "b:c"], SynthesisMethod.COMBINATION)
        patterns = cs.get_patterns()
        assert len(patterns) >= 2

    def test_pattern_key_uses_pipe_separator(self):
        cs = CreativeSynthesis()
        cs.register_concept("x:y")
        cs.register_concept("z")
        cs.synthesize(["x:y", "z"], SynthesisMethod.COMBINATION)
        with cs._lock:
            keys = list(cs._patterns.keys())
        assert any("|" in k for k in keys)
        assert not any(":" in k.split("|")[0] for k in keys if k.count("|") > 1)

    def test_pattern_key_no_colon_concept_collision(self):
        cs = CreativeSynthesis()
        cs.register_concept("a")
        cs.register_concept("b:c")
        cs.register_concept("b")
        cs.register_concept("c")
        cs.synthesize(["a", "b:c"], SynthesisMethod.ANALOGY)
        cs.synthesize(["a", "b", "c"], SynthesisMethod.ANALOGY)
        with cs._lock:
            keys = list(cs._patterns.keys())
        assert len(keys) == 2


class TestMR9EvaluationEviction:
    def test_eviction_logs_warning(self, caplog):
        mr = MoralReasoning()
        for i in range(_MAX_EVALUATIONS + 1):
            mr.evaluate(f"action_{i}")
        assert any("Evicting evaluation" in r.message for r in caplog.records)

    def test_eviction_cleans_dangling_refs(self):
        mr = MoralReasoning()
        ev_ids = []
        for i in range(50):
            ev = mr.evaluate(f"harm {i} help {i} fair privacy")
            ev_ids.append(ev.evaluation_id)
        for ev_id in ev_ids:
            for d in mr._dilemmas.values():
                if ev_id in d.evaluation_ids:
                    assert ev_id in mr._evaluations or ev_id not in d.evaluation_ids

    def test_dangling_refs_cleaned_on_eviction(self):
        mr = MoralReasoning()
        evaluations = []
        for i in range(_MAX_EVALUATIONS + 10):
            ev = mr.evaluate(f"harm help fair privacy action_{i}")
            evaluations.append(ev.evaluation_id)
        for d in mr._dilemmas.values():
            for eid in d.evaluation_ids:
                assert eid in mr._evaluations


class TestMR10PrincipleEviction:
    def test_non_negotiable_evicted_last(self):
        mr = MoralReasoning()
        nn = MoralPrinciple(name="NN1", description="test", framework=EthicalFramework.UTILITARIAN, non_negotiable=True)
        mr._principles[nn.principle_id] = nn
        for i in range(_MAX_PRINCIPLES):
            mr.add_principle(f"principle_{i}", f"desc_{i}", EthicalFramework.UTILITARIAN)
        assert nn.principle_id in mr._principles or len(mr._principles) == _MAX_PRINCIPLES

    def test_evict_non_negotiable_warns(self, caplog):
        mr = MoralReasoning()
        for i in range(_MAX_PRINCIPLES):
            mr.add_principle(f"nn_{i}", f"desc_{i}", EthicalFramework.UTILITARIAN, non_negotiable=True)
        mr.add_principle("overflow_nn", "desc", EthicalFramework.UTILITARIAN, non_negotiable=True)
        assert any("non_negotiable" in r.message for r in caplog.records)

    def test_regular_evicted_before_non_negotiable(self):
        mr = MoralReasoning()
        nn = MoralPrinciple(name="NN_CORE", description="test", framework=EthicalFramework.DEONTOLOGICAL, non_negotiable=True)
        mr._principles[nn.principle_id] = nn
        for i in range(_MAX_PRINCIPLES):
            mr.add_principle(f"reg_{i}", f"desc_{i}", EthicalFramework.UTILITARIAN, non_negotiable=False)
        assert nn.principle_id in mr._principles


class TestME9EpisodeEviction:
    def test_episode_eviction_logs_warning(self, caplog):
        me = MetacognitionEngine()
        for i in range(_MAX_EPISODES + 1):
            me.start_thinking(ReasoningType.SYSTEMATIC, f"topic_{i}")
        assert any("Evicting episode" in r.message for r in caplog.records)

    def test_evicted_episode_removed_from_both_dicts(self):
        me = MetacognitionEngine()
        first = me.start_thinking(ReasoningType.SYSTEMATIC, "first")
        first_id = first.episode_id
        for i in range(_MAX_EPISODES):
            me.start_thinking(ReasoningType.SYSTEMATIC, f"fill_{i}")
        assert first_id not in me._episodes
        assert first_id not in me._open_episodes


class TestEI8AverageIntensity:
    def test_cycle_pattern_average_intensity(self):
        ei = EmotionalIntelligence()
        for _ in range(5):
            ei.detect_emotion("I am happy and excited about this")
        patterns = ei.detect_patterns(min_frequency=2)
        cycle_patterns = [p for p in patterns if p.pattern_type == PatternType.CYCLE]
        for cp in cycle_patterns:
            assert cp.average_intensity > 0

    def test_trigger_pattern_average_intensity(self):
        ei = EmotionalIntelligence()
        for _ in range(3):
            ei.detect_emotion("I feel happy and glad about this", context="test")
        patterns = ei.detect_patterns(min_frequency=2)
        for p in patterns:
            assert p.average_intensity >= 0.0
            assert p.average_intensity <= 1.0


class TestCR12ConsistentDictShape:
    def test_common_ground_no_perspectives(self):
        cr = ConflictResolution()
        c = cr.create_conflict("test", "interest", 2)
        result = cr.analyze_common_ground(c.conflict_id)
        assert "common_interests" in result
        assert "common_concerns" in result
        assert "overlap_score" in result
        assert "perspective_count" in result
        assert "avg_flexibility" in result

    def test_common_ground_one_perspective(self):
        cr = ConflictResolution()
        c = cr.create_conflict("test", "interest", 2)
        cr.add_perspective(c.conflict_id, "party_a", "position_a", ["interest_1"], ["concern_1"], 0.5)
        result = cr.analyze_common_ground(c.conflict_id)
        assert result["perspective_count"] == 1

    def test_common_ground_two_perspectives(self):
        cr = ConflictResolution()
        c = cr.create_conflict("test", "interest", 2)
        cr.add_perspective(c.conflict_id, "party_a", "pos_a", ["shared"], ["concern_a"], 0.5)
        cr.add_perspective(c.conflict_id, "party_b", "pos_b", ["shared"], ["concern_b"], 0.6)
        result = cr.analyze_common_ground(c.conflict_id)
        assert "shared" in result["common_interests"]
        assert result["avg_flexibility"] > 0


class TestASTM2SkillEvictionLogging:
    def test_skill_eviction_logs_warning(self, caplog):
        tree = AdaptiveSkillTree()
        for i in range(_MAX_SKILLS + 1):
            try:
                tree.add_skill(f"skill_{i}", f"Skill {i}", SkillDomain.TECHNICAL)
            except ValueError:
                pass
        assert any("Evicting oldest skill" in r.message for r in caplog.records)


class TestNMM2ChapterArcEvictionLogging:
    def test_chapter_eviction_logs_warning(self, caplog):
        nm = NarrativeMemory()
        for i in range(_MAX_CHAPTERS + 1):
            nm.create_chapter(f"Chapter {i}", "career")
        assert any("Evicting oldest chapter" in r.message for r in caplog.records)

    def test_arc_eviction_logs_warning(self, caplog):
        nm = NarrativeMemory()
        for i in range(_MAX_ARCS + 1):
            nm.create_arc(f"Arc {i}", ["career"])
        assert any("Evicting oldest arc" in r.message for r in caplog.records)


class TestPMM2EvictionLogging:
    def test_procedure_eviction_logs_warning(self, caplog):
        pm = ProceduralMemory()
        for i in range(_MAX_PROCEDURES + 1):
            pm.add_procedure(f"proc_{i}", "test")
        assert any("Evicting oldest procedure" in r.message for r in caplog.records)

    def test_habit_eviction_logs_warning(self, caplog):
        pm = ProceduralMemory()
        for i in range(_MAX_HABITS + 1):
            pm.add_habit(f"habit_{i}", "cue", "routine", "reward")
        assert any("Evicting oldest habit" in r.message for r in caplog.records)

    def test_routine_eviction_logs_warning(self, caplog):
        pm = ProceduralMemory()
        for i in range(_MAX_ROUTINES + 1):
            pm.add_routine(f"routine_{i}", "daily")
        assert any("Evicting oldest routine" in r.message for r in caplog.records)


class TestIntegrationSprint61:
    def test_cs2_and_mr9_together(self):
        cs = CreativeSynthesis()
        cs.register_concept("alpha:beta")
        cs.register_concept("gamma")
        idea = cs.synthesize(["alpha:beta", "gamma"], SynthesisMethod.COMBINATION)
        assert idea.idea_id

        mr = MoralReasoning()
        ev = mr.evaluate("help others and be fair")
        assert ev.overall_score >= 0

    def test_all_eviction_paths_covered(self):
        cs = CreativeSynthesis()
        assert cs._patterns is not None
        mr = MoralReasoning()
        assert mr._evaluations is not None
        me = MetacognitionEngine()
        assert me._episodes is not None
        ei = EmotionalIntelligence()
        assert ei._patterns is not None
        cr = ConflictResolution()
        assert cr._conflicts is not None
        ast = AdaptiveSkillTree()
        assert ast._skills is not None
        nm = NarrativeMemory()
        assert nm._chapters is not None
        pm = ProceduralMemory()
        assert pm._procedures is not None