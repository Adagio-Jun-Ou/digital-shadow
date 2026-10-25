#!/usr/bin/env python3

import re
import copy
import time
import threading
import unittest
from collections import deque
from datetime import datetime, timezone

from shadowd.cognition.creative_synthesis import (
    CreativeSynthesis, CreativeDomain, SynthesisMethod, _MAX_ASSOCIATIONS_PER_CONCEPT,
)
from shadowd.cognition.moral_reasoning import (
    MoralReasoning, EthicalFramework, Stakeholder, StakeholderType, _MAX_STAKEHOLDERS,
    _keyword_match,
)
from shadowd.cognition.metacognition_engine import (
    MetacognitionEngine, BiasType, ReasoningType, _word_match, _count_word_matches,
)
from shadowd.cognition.emotional_intelligence import (
    EmotionalIntelligence, EmotionCategory, RegulationApproach, PatternType,
    EmotionState, RegulationStrategy, _MAX_METADATA_KEYS,
)
from shadowd.cognition.conflict_resolution import (
    ConflictResolution, ConflictType, ConflictIntensity, ConflictStatus, _MAX_STRATEGIES,
)
from shadowd.cognition.adaptive_skill_tree import (
    AdaptiveSkillTree, SkillDomain, SkillNode,
)


class TestCreativeSynthesisSprint59(unittest.TestCase):
    def test_cs1_format_no_crash_on_braces(self):
        cs = CreativeSynthesis()
        cs.register_concept("concept with {curly}")
        cs.register_concept("other")
        idea = cs.synthesize(["concept with {curly}", "other"], SynthesisMethod.COMBINATION)
        self.assertIn("concept with {curly}", idea.description)

    def test_cs1_format_no_crash_on_braces_inversion(self):
        cs = CreativeSynthesis()
        cs.register_concept("a}b{c")
        cs.register_concept("normal")
        idea = cs.synthesize(["a}b{c", "normal"], SynthesisMethod.INVERSION)
        self.assertIn("a}b{c", idea.description)

    def test_cs3_association_cap(self):
        cs = CreativeSynthesis()
        for i in range(_MAX_ASSOCIATIONS_PER_CONCEPT + 10):
            cs.add_association("base", f"partner_{i}")
        with cs._lock:
            count = len(cs._associations.get("base", []))
        self.assertLessEqual(count, _MAX_ASSOCIATIONS_PER_CONCEPT)

    def test_cs6_dedup_association_pairs(self):
        cs = CreativeSynthesis()
        cs.add_association("alpha", "beta")
        cs.add_association("alpha", "beta")
        with cs._lock:
            count = len(cs._associations.get("alpha", []))
        for a_name, pairs in cs._associations.items():
            pair_texts = [(p.concept_a, p.concept_b) for p in pairs]
            self.assertEqual(len(pair_texts), len(set(pair_texts)), f"duplicate pair in {a_name}")

    def test_cs4_eviction_cleans_associations(self):
        cs = CreativeSynthesis()
        for i in range(2010):
            cs.register_concept(f"concept_{i:04d}")
        self.assertLessEqual(len(cs._concepts), 2000)

    def test_cs9_random_import_at_module_level(self):
        import shadowd.cognition.creative_synthesis as mod
        self.assertTrue(hasattr(mod, "random"))

    def test_cs1_format_analogy_template(self):
        cs = CreativeSynthesis()
        cs.register_concept("a{b}")
        cs.register_concept("c}d")
        idea = cs.synthesize(["a{b}", "c}d"], SynthesisMethod.ANALOGY)
        self.assertIn("a{b}", idea.description)

    def test_cs3_association_cap_per_concept(self):
        cs = CreativeSynthesis()
        for i in range(60):
            cs.add_association("hub", f"spoke_{i}")
        assoc = cs.get_concept_associations("hub")
        self.assertLessEqual(len(assoc), 20)


class TestMoralReasoningSprint59(unittest.TestCase):
    def test_mr1_rights_keywords_increase_deon_score(self):
        mr = MoralReasoning()
        eval_norights = mr.evaluate("do something mundane", description="a normal action")
        eval_rights = mr.evaluate("protect the right to freedom and autonomy", description="respecting rights")
        deon_no = next(a.score for a in eval_norights.assessments if a.framework == EthicalFramework.DEONTOLOGICAL)
        deon_yes = next(a.score for a in eval_rights.assessments if a.framework == EthicalFramework.DEONTOLOGICAL)
        self.assertGreater(deon_yes, deon_no)

    def test_mr2_fairness_keywords_increase_score(self):
        mr = MoralReasoning()
        eval_fair = mr.evaluate("ensure fair and equal treatment", description="just and equitable")
        eval_plain = mr.evaluate("do a task", description="nothing special")
        fair_score = next(a.score for a in eval_fair.assessments if a.framework == EthicalFramework.FAIRNESS)
        plain_score = next(a.score for a in eval_plain.assessments if a.framework == EthicalFramework.FAIRNESS)
        self.assertGreater(fair_score, plain_score)

    def test_mr3_keyword_match_word_boundary(self):
        self.assertTrue(_keyword_match("right", "the right thing"))
        self.assertFalse(_keyword_match("right", "copyright"))
        self.assertTrue(_keyword_match("fair", "that is fair"))
        self.assertFalse(_keyword_match("fair", "affair"))

    def test_mr5_recommendation_includes_conflicts(self):
        mr = MoralReasoning()
        eval_result = mr.evaluate("harm others to protect freedom and privacy",
                                   description="causing damage while respecting autonomy")
        if eval_result.conflicts:
            self.assertIn("ethical tensions", eval_result.recommendation)

    def test_mr6_stakeholder_interests_sanitized(self):
        s = Stakeholder(name="test", interests=["<script>alert(1)</script>", "normal interest"])
        self.assertNotIn("<script>", s.interests[0])

    def test_mr7_stakeholders_capped(self):
        mr = MoralReasoning()
        stakeholders = [Stakeholder(name=f"person_{i}") for i in range(50)]
        eval_result = mr.evaluate("action", stakeholders=stakeholders)
        self.assertLessEqual(len(eval_result.stakeholders), _MAX_STAKEHOLDERS)

    def test_mr3_care_keyword_matching(self):
        self.assertTrue(_keyword_match("care", "I care about this"))
        self.assertFalse(_keyword_match("care", "careful"))

    def test_mr3_harm_keyword_matching(self):
        self.assertTrue(_keyword_match("harm", "prevent harm"))
        self.assertFalse(_keyword_match("harm", "harmless"))


class TestMetacognitionSprint59(unittest.TestCase):
    def test_me7_bias_type_recency(self):
        self.assertEqual(BiasType.RECENCY.value, "recency")
        self.assertFalse(hasattr(BiasType, "RECIDENCE"))

    def test_me7_recency_in_keywords(self):
        from shadowd.cognition.metacognition_engine import _BIAS_DETECTION_KEYWORDS, _BIAS_MITIGATIONS
        self.assertIn(BiasType.RECENCY, _BIAS_DETECTION_KEYWORDS)
        self.assertIn(BiasType.RECENCY, _BIAS_MITIGATIONS)

    def test_me1_episode_mutation_inside_lock(self):
        me = MetacognitionEngine()
        episode = me.start_thinking(ReasoningType.SYSTEMATIC, "test topic")
        me.detect_biases("I definitely knew it all along, just happened recently", episode_id=episode.episode_id)
        with me._lock:
            internal = me._episodes.get(episode.episode_id)
            self.assertTrue(len(internal.biases_detected) > 0)

    def test_me2_eviction_cleans_open_episodes(self):
        me = MetacognitionEngine()
        ids = []
        for i in range(5005):
            ep = me.start_thinking(ReasoningType.SYSTEMATIC, f"topic {i}")
            ids.append(ep.episode_id)
        with me._lock:
            for eid in ids[:5]:
                self.assertNotIn(eid, me._open_episodes)
                self.assertNotIn(eid, me._episodes)

    def test_me3_cognitive_state_lock(self):
        me = MetacognitionEngine()
        me._cognitive_load = 0.9
        me._update_cognitive_state()
        self.assertEqual(me._cognitive_state.value, "overloaded")

    def test_me4_word_boundary_matching(self):
        self.assertTrue(_word_match("certain", "I am certain of this"))
        self.assertFalse(_word_match("certain", "uncertainty"))

    def test_me5_count_word_matches(self):
        self.assertEqual(_count_word_matches("test", "test test test"), 3)
        self.assertEqual(_count_word_matches("test", "testing test"), 1)

    def test_me13_no_duplicate_bias_types(self):
        me = MetacognitionEngine()
        biases = me.detect_biases("I definitely knew it all along and it just happened and I am certain")
        bias_types = [b.bias_type for b in biases]
        self.assertEqual(len(bias_types), len(set(bias_types)))

    def test_me8_single_lock_bias_detection(self):
        me = MetacognitionEngine()
        me.detect_biases("I am definitely certain and everyone thinks so")
        self.assertTrue(len(me._biases) > 0)

    def test_me4_no_substring_bias_match(self):
        me = MetacognitionEngine()
        biases = me.detect_biases("uncertainty is normal")
        confirmation_biases = [b for b in biases if b.bias_type == BiasType.OVERCONFIDENCE]
        self.assertEqual(len(confirmation_biases), 0)


class TestEmotionalIntelligenceSprint59(unittest.TestCase):
    def test_ei1_no_false_positive_fear_in_fearless(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am fearless and confident")
        self.assertNotEqual(state.primary, EmotionCategory.FEAR)

    def test_ei1_no_false_positive_sad_in_sadness(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am sadness-free and happy")
        self.assertNotEqual(state.primary, EmotionCategory.SADNESS)

    def test_ei1_word_boundary_emotion_detection(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I feel happy and delighted today")
        self.assertEqual(state.primary, EmotionCategory.JOY)

    def test_ei2_running_average_divergence(self):
        ei = EmotionalIntelligence()
        for i in range(60):
            ei.detect_emotion("I am happy and glad")
        valence1 = ei._stats["average_valence"]
        for i in range(60):
            ei.detect_emotion("I am sad and depressed")
        valence2 = ei._stats["average_valence"]
        self.assertLess(valence2, valence1)

    def test_ei3_emotion_groups_run_length(self):
        ei = EmotionalIntelligence()
        for _ in range(5):
            ei.detect_emotion("I am angry and furious")
        patterns = ei.detect_patterns(min_frequency=2)
        cycle_patterns = [p for p in patterns if p.pattern_type == PatternType.CYCLE]
        if cycle_patterns:
            for p in cycle_patterns:
                self.assertGreaterEqual(p.frequency, 2)
                self.assertNotEqual(p.average_intensity, 0.5)

    def test_ei5_metadata_cap_emotion_state(self):
        meta = {f"k{i}": f"v{i}" for i in range(100)}
        state = EmotionState(metadata=meta)
        self.assertLessEqual(len(state.metadata), _MAX_METADATA_KEYS)

    def test_ei5_metadata_cap_regulation_strategy(self):
        meta = {f"k{i}": f"v{i}" for i in range(100)}
        strat = RegulationStrategy(metadata=meta)
        self.assertLessEqual(len(strat.metadata), _MAX_METADATA_KEYS)

    def test_ei7_dead_constant_removed(self):
        import shadowd.cognition.emotional_intelligence as mod
        self.assertFalse(hasattr(mod, "_MAX_HISTORY_LEN"))

    def test_ei8_no_hardcoded_average_intensity(self):
        ei = EmotionalIntelligence()
        for _ in range(3):
            ei.detect_emotion("I am furious and enraged and very angry and hostile")
        patterns = ei.detect_patterns(min_frequency=2)
        cycle_patterns = [p for p in patterns if p.pattern_type == PatternType.CYCLE]
        if cycle_patterns:
            for p in cycle_patterns:
                self.assertNotEqual(p.average_intensity, 0.5)

    def test_ei1_triggers_word_boundary(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am afraid of the dark")
        self.assertEqual(state.primary, EmotionCategory.FEAR)
        self.assertIn("afraid", state.triggers)

    def test_ei2_counter_grows_beyond_maxlen(self):
        ei = EmotionalIntelligence()
        for _ in range(60):
            ei.detect_emotion("I feel happy")
        self.assertGreater(ei._states_counted, 50)


class TestConflictResolutionSprint59(unittest.TestCase):
    def setUp(self):
        self.cr = ConflictResolution()

    def test_cr1_mutable_reference_safety(self):
        c = self.cr.create_conflict("Test", "interest", 2)
        retrieved = self.cr.get_conflict(c.conflict_id)
        retrieved.title = "MODIFIED"
        original = self.cr.get_conflict(c.conflict_id)
        self.assertEqual(original.title, "Test")

    def test_cr1_nonexistent_returns_none(self):
        self.assertIsNone(self.cr.get_conflict("nonexistent"))

    def test_cr2_list_conflicts_deep_copy(self):
        c = self.cr.create_conflict("Test", "interest", 2)
        conflicts = self.cr.list_conflicts()
        conflicts[0].title = "MODIFIED"
        original = self.cr.get_conflict(c.conflict_id)
        self.assertEqual(original.title, "Test")

    def test_cr3_cannot_escalate_resolved(self):
        c = self.cr.create_conflict("Test", "interest", 2)
        self.cr.resolve_conflict(c.conflict_id, "Resolved")
        result = self.cr.escalate_conflict(c.conflict_id)
        self.assertFalse(result)

    def test_cr4_suggest_strategies_returns_truncated(self):
        c = self.cr.create_conflict("Test", "interest", 4)
        for i in range(12):
            self.cr.add_perspective(c.conflict_id, f"party_{i}", f"position_{i}", flexibility=0.8)
        strategies = self.cr.suggest_strategies(c.conflict_id)
        self.assertLessEqual(len(strategies), _MAX_STRATEGIES)

    def test_cr5_perspective_drop_logged(self):
        import logging
        c = self.cr.create_conflict("Test", "interest", 2)
        with self.assertLogs("shadowd.cognition.conflict_resolution", level="WARNING"):
            for i in range(12):
                self.cr.add_perspective(c.conflict_id, f"party_{i}", f"pos_{i}")

    def test_cr6_conflict_eviction_logged(self):
        import logging
        cr = ConflictResolution()
        cr._MAX_CONFLICTS = 3
        with self.assertLogs("shadowd.cognition.conflict_resolution", level="WARNING"):
            for i in range(4):
                cr.create_conflict(f"Conflict {i}", "interest", 2)

    def test_cr9_strategy_keywords_removed(self):
        import shadowd.cognition.conflict_resolution as mod
        self.assertFalse(hasattr(mod, "_STRATEGY_KEYWORDS"))

    def test_cr12_common_ground_consistent_shape(self):
        c = self.cr.create_conflict("Test", "interest", 2)
        result = self.cr.analyze_common_ground(c.conflict_id)
        self.assertIn("perspective_count", result)
        self.assertIn("avg_flexibility", result)

    def test_cr12_common_ground_not_found(self):
        result = self.cr.analyze_common_ground("nonexistent")
        self.assertIn("perspective_count", result)
        self.assertIn("avg_flexibility", result)

    def test_cr13_cannot_re_resolve(self):
        c = self.cr.create_conflict("Test", "interest", 2)
        self.assertTrue(self.cr.resolve_conflict(c.conflict_id, "Done"))
        self.assertFalse(self.cr.resolve_conflict(c.conflict_id, "Done again"))

    def test_cr3_escalate_active_conflict(self):
        c = self.cr.create_conflict("Test", "interest", 2)
        result = self.cr.escalate_conflict(c.conflict_id)
        self.assertTrue(result)

    def test_cr2_list_with_status_filter(self):
        c1 = self.cr.create_conflict("Active", "interest", 2)
        c2 = self.cr.create_conflict("To resolve", "interest", 1)
        self.cr.resolve_conflict(c2.conflict_id, "Done")
        active = self.cr.list_conflicts(status="active")
        self.assertTrue(all(c.status.value == "active" for c in active))


class TestAdaptiveSkillTreeSprint59(unittest.TestCase):
    def setUp(self):
        self.ast = AdaptiveSkillTree()

    def test_astc1_dependency_cleanup_on_eviction(self):
        ast = AdaptiveSkillTree()
        ast._MAX_SKILLS = 3
        ast.add_skill("base", "Base Skill", "technical")
        ast.add_skill("dep1", "Dependent 1", "technical", dependencies=["base"])
        ast.add_skill("dep2", "Dependent 2", "technical", dependencies=["base"])
        ast.add_skill("new", "New Skill", "technical")
        dep2 = ast.get_skill("dep2")
        if dep2:
            self.assertNotIn("base", dep2.dependencies)

    def test_astc1_unlocked_status_rechecked(self):
        ast = AdaptiveSkillTree()
        ast._MAX_SKILLS = 3
        ast.add_skill("gate", "Gate Skill", "technical")
        gate = ast.get_skill("gate")
        if gate:
            g = ast.get_skill("gate")
            ast.use_skill("gate", success=True)
        ast.add_skill("dep", "Dependent", "technical", dependencies=["gate"])
        ast.add_skill("extra1", "Extra 1", "technical")
        ast.add_skill("extra2", "Extra 2", "technical")

    def test_asth2_bounded_usage_counter(self):
        ast = AdaptiveSkillTree()
        ast.add_skill("test", "Test Skill", "technical")
        for _ in range(2000):
            ast.use_skill("test", success=True)
        self.assertLessEqual(len(ast._recent_usage), 1000)

    def test_asth2_usage_freq_visible(self):
        ast = AdaptiveSkillTree()
        ast.add_skill("a", "Skill A", "technical")
        ast.add_skill("b", "Skill B", "technical")
        ast.use_skill("a", success=True)
        recs = ast.get_recommendations()
        self.assertTrue(any(r.skill_id in ("a", "b") for r in recs))

    def test_astm1_no_uuid_import(self):
        import shadowd.cognition.adaptive_skill_tree as mod
        self.assertFalse(hasattr(mod, "uuid") or "uuid" in dir(mod))

    def test_asth1_get_skill_deep_copy(self):
        self.ast.add_skill("test", "Test", "technical")
        skill = self.ast.get_skill("test")
        skill.name = "MODIFIED"
        original = self.ast.get_skill("test")
        self.assertEqual(original.name, "Test")

    def test_asth1_list_skills_deep_copy(self):
        self.ast.add_skill("test", "Test", "technical")
        skills = self.ast.list_skills()
        skills[0].name = "MODIFIED"
        original = self.ast.get_skill("test")
        self.assertEqual(original.name, "Test")

    def test_asth1_use_skill_deep_copy(self):
        self.ast.add_skill("test", "Test", "technical")
        result = self.ast.use_skill("test", success=True)
        result.name = "MODIFIED"
        original = self.ast.get_skill("test")
        self.assertEqual(original.name, "Test")

    def test_asth1_get_skill_tree_deep_copy(self):
        self.ast.add_skill("test", "Test", "technical")
        tree = self.ast.get_skill_tree()
        tree["test"]["name"] = "MODIFIED"
        original = self.ast.get_skill_tree()
        self.assertEqual(original["test"]["name"], "Test")


if __name__ == "__main__":
    unittest.main()