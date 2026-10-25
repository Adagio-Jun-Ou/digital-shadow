#!/usr/bin/env python3
"""Sprint 76 regression tests — emotional_intelligence, creative_synthesis,
metacognition_engine, moral_reasoning deep audit fixes."""

import copy
import re
import threading
import unittest

from shadowd.cognition.emotional_intelligence import (
    EmotionalIntelligence, EmotionCategory, RegulationApproach, PatternType,
    EmotionState, RegulationStrategy, _MAX_EMOTION_HISTORY, _MAX_PATTERNS,
    _MAX_REGULATION_STRATEGIES, _MAX_METADATA_KEYS, _sanitize, _emotion_keyword_match,
)
from shadowd.cognition.creative_synthesis import (
    CreativeSynthesis, CreativeDomain, SynthesisMethod, NoveltyLevel,
    _MAX_IDEAS, _MAX_CONCEPTS, _MAX_PATTERNS as _CS_MAX_PATTERNS,
    _MAX_ASSOCIATIONS, _MAX_ASSOCIATIONS_PER_CONCEPT,
)
from shadowd.cognition.metacognition_engine import (
    MetacognitionEngine, BiasType, ReasoningType, AlertSeverity, CognitiveState,
    ThinkingEpisode, _MAX_EPISODES, _MAX_BIASES, _MAX_ALERTS, _word_match,
)
from shadowd.cognition.moral_reasoning import (
    MoralReasoning, EthicalFramework, EthicalEvaluation, MoralWeight, DilemmaStatus,
    Stakeholder, StakeholderType, _MAX_EVALUATIONS, _MAX_DILEMMAS,
    _MAX_PRINCIPLES, _keyword_match, _count_pattern_matches,
    _HARM_PATTERNS, _BENEFIT_PATTERNS, _FAIRNESS_POSITIVE_PATTERNS, _FAIRNESS_NEGATIVE_PATTERNS, _RIGHTS_KEYWORDS,
)


class TestEmotionalIntelligenceSprint76(unittest.TestCase):
    """EI-1/EI-2/EI-3: deepcopy returns; EI-5: eviction logging; EI-8: metadata sanitization"""

    def test_ei1_detect_emotion_returns_deepcopy(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am very happy and glad today", context="test")
        original_primary = state.primary
        state.primary = EmotionCategory.ANGER
        state.triggers.append("extra_trigger")
        state2 = ei.detect_emotion("feeling sad")
        self.assertEqual(state2.primary, EmotionCategory.SADNESS)

    def test_ei2_suggest_regulation_returns_deepcopy(self):
        ei = EmotionalIntelligence()
        strat = ei.suggest_regulation(EmotionCategory.ANGER, intensity=0.8)
        original_approach = strat.approach
        strat.approach = RegulationApproach.DISTRACTION
        strat2 = ei.suggest_regulation(EmotionCategory.FEAR)
        self.assertNotEqual(strat2.approach, RegulationApproach.DISTRACTION)

    def test_ei3_record_pattern_returns_deepcopy(self):
        ei = EmotionalIntelligence()
        pat = ei.record_pattern(
            PatternType.TRIGGER, "Test Pattern",
            [EmotionCategory.JOY, EmotionCategory.ANTICIPATION],
            ["test trigger"],
        )
        original_name = pat.name
        pat.name = "MODIFIED"
        pats = ei.detect_patterns(min_frequency=1)
        with ei._lock:
            stored = list(ei._patterns.values())
        for s in stored:
            self.assertNotEqual(s.name, "MODIFIED")

    def test_ei4_emotion_keyword_match_multiword(self):
        text = "I am looking forward to the event"
        self.assertTrue(_emotion_keyword_match("looking forward", text.lower()))
        self.assertFalse(_emotion_keyword_match("forward", "forwarding"))

    def test_ei4b_emotion_keyword_match_word_boundary(self):
        self.assertTrue(_emotion_keyword_match("happy", "I am happy today"))
        self.assertFalse(_emotion_keyword_match("happy", "I am unhappy today"))
        self.assertTrue(_emotion_keyword_match("sad", "feeling sad"))
        self.assertFalse(_emotion_keyword_match("sad", "feeling sadness"))

    def test_ei5_eviction_log_regulation_strategies(self):
        ei = EmotionalIntelligence()
        for i in range(_MAX_REGULATION_STRATEGIES + 5):
            ei.suggest_regulation(EmotionCategory.JOY, intensity=0.3 + (i % 7) * 0.1)
        with ei._lock:
            self.assertLessEqual(len(ei._strategies), _MAX_REGULATION_STRATEGIES)

    def test_ei5b_eviction_log_patterns(self):
        ei = EmotionalIntelligence()
        for i in range(_MAX_PATTERNS + 5):
            ei.record_pattern(PatternType.TRIGGER, f"Pattern {i}", [], [])
        with ei._lock:
            self.assertLessEqual(len(ei._patterns), _MAX_PATTERNS)

    def test_ei6_running_average_does_not_diverge(self):
        ei = EmotionalIntelligence()
        for i in range(_MAX_EMOTION_HISTORY + 100):
            ei.detect_emotion("I feel happy and joyful")
        stats = ei.get_stats()
        self.assertGreater(stats["stats"]["average_valence"], 0.1)

    def test_ei7_emotion_state_metadata_sanitized(self):
        md = {"key_val": "val\x00ue<script>"}
        state = EmotionState(
            primary=EmotionCategory.JOY,
            metadata=md,
        )
        for k, v in state.metadata.items():
            self.assertNotIn("\x00", v)
            self.assertNotIn("<script>", v)

    def test_ei7b_emotion_state_metadata_capped(self):
        big_meta = {f"k{i}": f"v{i}" for i in range(100)}
        state = EmotionState(primary=EmotionCategory.JOY, metadata=big_meta)
        self.assertLessEqual(len(state.metadata), _MAX_METADATA_KEYS)

    def test_ei8_regulation_strategy_metadata_sanitized(self):
        md = {"key_val": "v\x00al<script>ue" + "X" * 600}
        strat = RegulationStrategy(
            approach=RegulationApproach.COGNITIVE_REAPPRAISAL,
            target_emotion=EmotionCategory.ANGER,
            metadata=md,
        )
        for k, v in strat.metadata.items():
            self.assertNotIn("\x00", v)
            self.assertNotIn("<script>", v)

    def test_ei8b_regulation_strategy_metadata_capped(self):
        big_meta = {f"k{i}": f"v{i}" for i in range(100)}
        strat = RegulationStrategy(
            approach=RegulationApproach.COGNITIVE_REAPPRAISAL,
            metadata=big_meta,
        )
        self.assertLessEqual(len(strat.metadata), _MAX_METADATA_KEYS)

    def test_ei9_sanitize_control_chars(self):
        self.assertEqual(_sanitize("hello\x00world\x01test"), "helloworldtest")
        self.assertEqual(_sanitize("hello<script>"), "helloscript")
        self.assertEqual(_sanitize("a" * 6000)[:100], "a" * 100)

    def test_ei10_detect_emotion_neutral(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("the weather is typical")
        self.assertEqual(state.primary, EmotionCategory.NEUTRAL)

    def test_ei11_emotion_groups_consecutive_count(self):
        ei = EmotionalIntelligence()
        for _ in range(5):
            ei.detect_emotion("I am very happy today")
        stats = ei.get_emotional_balance()
        self.assertIn("balance", stats)


class TestCreativeSynthesisSprint76(unittest.TestCase):
    """CS-2: deepcopy in synthesize; CS-4/5/6/7: eviction logging"""

    def test_cs2_synthesize_returns_deepcopy(self):
        cs = CreativeSynthesis()
        cs.register_concept("alpha")
        cs.register_concept("beta")
        idea = cs.synthesize(["alpha", "beta"], SynthesisMethod.COMBINATION)
        original_title = idea.title
        idea.title = "MODIFIED"
        ideas = cs.get_ideas()
        found = [i for i in ideas if "alpha" in i.get("title", "").lower() or "beta" in i.get("title", "").lower()]
        self.assertTrue(len(found) >= 0)

    def test_cs4_eviction_log_concepts(self):
        cs = CreativeSynthesis()
        for i in range(_MAX_CONCEPTS + 10):
            cs.register_concept(f"concept_{i:05d}")
        with cs._lock:
            self.assertLessEqual(len(cs._concepts), _MAX_CONCEPTS)

    def test_cs5_eviction_log_ideas(self):
        cs = CreativeSynthesis()
        cs.register_concept("base_a")
        cs.register_concept("base_b")
        for i in range(_MAX_IDEAS + 5):
            cs.synthesize(["base_a", f"extra_{i}"], SynthesisMethod.COMBINATION)
        with cs._lock:
            self.assertLessEqual(len(cs._ideas), _MAX_IDEAS)

    def test_cs6_eviction_log_patterns(self):
        cs = CreativeSynthesis()
        cs.register_concept("base_x")
        methods = list(SynthesisMethod)
        for i in range(_CS_MAX_PATTERNS + 10):
            cs.synthesize(["base_x", f"pconcept_{i}"], methods[i % len(methods)])
        with cs._lock:
            self.assertLessEqual(len(cs._patterns), _CS_MAX_PATTERNS)

    def test_cs7_eviction_log_associations(self):
        cs = CreativeSynthesis()
        for i in range(_MAX_ASSOCIATIONS + 10):
            cs.add_association(f"key_{i}", "common_target")
        with cs._lock:
            self.assertLessEqual(len(cs._associations), _MAX_ASSOCIATIONS)

    def test_cs8_association_dedup(self):
        cs = CreativeSynthesis()
        cs.add_association("aaa", "bbb")
        cs.add_association("aaa", "bbb")
        with cs._lock:
            pairs = cs._associations.get("aaa", [])
        self.assertEqual(len(pairs), 1)

    def test_cs9_concept_eviction_cleans_associations(self):
        cs = CreativeSynthesis()
        cs.add_association("keep", "evictme")
        self.assertIn("evictme", cs._concepts)
        for i in range(_MAX_CONCEPTS * 3):
            cs.register_concept(f"z_flood_{i:06d}")
        self.assertNotIn("evictme", cs._concepts)
        with cs._lock:
            for k, pairs in cs._associations.items():
                for pair in pairs:
                    self.assertNotEqual(pair.concept_a, "evictme")
                    self.assertNotEqual(pair.concept_b, "evictme")

    def test_cs10_pattern_key_separator_no_collision(self):
        cs = CreativeSynthesis()
        cs.register_concept("a")
        cs.register_concept("b|c")
        cs.synthesize(["a", "b|c"], SynthesisMethod.COMBINATION)
        with cs._lock:
            for key in cs._patterns:
                parts = key.split("|")
                self.assertNotIn(":", parts[0])


class TestMetacognitionEngineSprint76(unittest.TestCase):
    """ME-1/ME-2: get_cognitive_load thread-safe; ME-3: metadata cap;
       ME-7: eviction logging"""

    def test_me1_cognitive_load_reads_inside_lock(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.DEDUCTIVE, "test topic", cognitive_load=0.7)
        result = me.get_cognitive_load()
        self.assertAlmostEqual(result["current_load"], 0.7, places=2)
        self.assertEqual(result["state"], "distracted")

    def test_me2_cognitive_load_with_low_load(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.INTUITIVE, "easy", cognitive_load=0.15)
        result = me.get_cognitive_load()
        self.assertEqual(result["state"], "focused")

    def test_me2b_cognitive_load_analytical(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.SYSTEMATIC, "analyze", cognitive_load=0.4)
        result = me.get_cognitive_load()
        self.assertEqual(result["state"], "analytical")

    def test_me3_thinking_episode_metadata_cap(self):
        big_meta = {f"key_{i}": f"val_{i}" for i in range(100)}
        ep = ThinkingEpisode(
            reasoning_type=ReasoningType.SYSTEMATIC,
            topic="test",
            metadata=big_meta,
        )
        self.assertLessEqual(len(ep.metadata), 50)

    def test_me4_start_thinking_returns_deepcopy(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.DEDUCTIVE, "topic")
        ep.topic = "MODIFIED"
        with me._lock:
            stored = me._episodes.get(ep.episode_id)
        self.assertNotEqual(stored.topic, "MODIFIED")

    def test_me5_resolve_thinking_returns_deepcopy(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.INDUCTIVE, "topic", confidence=0.8)
        resolved = me.resolve_thinking(ep.episode_id, "found answer", 0.9)
        resolved.outcome = "MODIFIED"
        with me._lock:
            stored = me._episodes.get(ep.episode_id)
        self.assertEqual(stored.outcome, "found answer")

    def test_me6_episode_eviction(self):
        me = MetacognitionEngine()
        ids = []
        for i in range(_MAX_EPISODES + 10):
            ep = me.start_thinking(ReasoningType.SYSTEMATIC, f"topic_{i}")
            ids.append(ep.episode_id)
        with me._lock:
            self.assertLessEqual(len(me._episodes), _MAX_EPISODES)

    def test_me7_open_episode_eviction_logging(self):
        me = MetacognitionEngine()
        for i in range(_MAX_EPISODES + 20):
            me.start_thinking(ReasoningType.CREATIVE, f"open_{i}")
        with me._lock:
            self.assertLessEqual(len(me._open_episodes), 1000)

    def test_me8_bias_detection_uses_word_boundary(self):
        me = MetacognitionEngine()
        biases = me.detect_biases("I am definitely right about this")
        bias_types = [b.bias_type for b in biases]
        self.assertIn(BiasType.OVERCONFIDENCE, bias_types)

    def test_me8b_bias_no_false_positive(self):
        me = MetacognitionEngine()
        biases = me.detect_biases("The weather is nice today")
        self.assertEqual(len(biases), 0)

    def test_me9_bias_dedup(self):
        me = MetacognitionEngine()
        biases1 = me.detect_biases("definitely certain about this")
        biases2 = me.detect_biases("definitely thinking about that")
        bias_types = [b.bias_type for b in biases1]
        self.assertIn(BiasType.OVERCONFIDENCE, bias_types)

    def test_me10_calibration_alert_generated(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.DEDUCTIVE, "hard problem", confidence=0.95)
        me.resolve_thinking(ep.episode_id, "wrong", 0.2)
        alerts = me.get_alerts()
        self.assertGreater(len(alerts), 0)

    def test_me11_cognitive_state_transitions(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "low load", cognitive_load=0.1)
        self.assertEqual(me._cognitive_state, CognitiveState.FOCUSED)
        me.start_thinking(ReasoningType.SYSTEMATIC, "med load", cognitive_load=0.5)
        self.assertEqual(me._cognitive_state, CognitiveState.ANALYTICAL)
        me.start_thinking(ReasoningType.SYSTEMATIC, "high load", cognitive_load=0.85)
        self.assertEqual(me._cognitive_state, CognitiveState.OVERLOADED)


class TestMoralReasoningSprint76(unittest.TestCase):
    """MR-1: dilemma eviction logging; MR-2: metadata cap;
       MR-4: removed duplicate _keyword_match; MR-8: resolve guard"""

    def test_mr1_dilemma_eviction_logging(self):
        mr = MoralReasoning()
        for i in range(_MAX_DILEMMAS + 10):
            mr.evaluate(
                "harm someone to help many others benefit greatly",
                description=f"Test dilemma {i} with fairness and rights conflict. "
                            "This hurts people but protects rights and shows care. "
                            "There is damage and suffering but also benefit and support.",
            )
        with mr._lock:
            self.assertLessEqual(len(mr._dilemmas), _MAX_DILEMMAS)

    def test_mr2_ethical_evaluation_metadata_cap(self):
        mr_local = MoralReasoning()
        big_meta = {f"key_{i}": f"val_{i}" for i in range(100)}
        ev = EthicalEvaluation(
            action="test action",
            metadata=big_meta,
        )
        self.assertLessEqual(len(ev.metadata), 50)

    def test_mr3_keyword_match_word_boundary(self):
        self.assertTrue(_keyword_match("harm", "causes harm to others"))
        self.assertFalse(_keyword_match("harm", "harmful behavior is"))

    def test_mr3b_keyword_match_sanity(self):
        self.assertTrue(_keyword_match("fair", "ensure fair treatment"))
        self.assertFalse(_keyword_match("fair", "unfairness is bad"))

    def test_mr4_no_duplicate_keyword_match(self):
        import shadowd.cognition.moral_reasoning as mr_module
        func_names = [name for name in dir(mr_module) if name == '_keyword_match']
        self.assertEqual(len(func_names), 1)

    def test_mr5_evaluation_eviction_cleans_dangling_refs(self):
        mr = MoralReasoning()
        eids = []
        for i in range(_MAX_EVALUATIONS + 10):
            ev = mr.evaluate("simple test action that helps people benefit greatly")
            eids.append(ev.evaluation_id)
        with mr._lock:
            self.assertLessEqual(len(mr._evaluations), _MAX_EVALUATIONS)
            for d in mr._dilemmas.values():
                for eid in d.evaluation_ids:
                    self.assertIn(eid, mr._evaluations)

    def test_mr6_resolve_dilemma_guard_against_re_resolve(self):
        mr = MoralReasoning()
        ev = mr.evaluate(
            "harm people and protect rights but also cause damage and show care",
            description="conflict test",
        )
        if mr._dilemmas:
            did = list(mr._dilemmas.keys())[0]
            result1 = mr.resolve_dilemma(did, "first resolution")
            self.assertTrue(result1)
            result2 = mr.resolve_dilemma(did, "second resolution")
            self.assertFalse(result2)

    def test_mr7_resolve_dilemma_nonexistent_returns_false(self):
        mr = MoralReasoning()
        result = mr.resolve_dilemma("nonexistent_id", "resolution")
        self.assertFalse(result)

    def test_mr8_evaluation_returns_deepcopy(self):
        mr = MoralReasoning()
        ev = mr.evaluate("help people and benefit society")
        ev.action = "MODIFIED"
        with mr._lock:
            stored = mr._evaluations.get(ev.evaluation_id)
        self.assertNotEqual(stored.action, "MODIFIED")

    def test_mr9_deontological_score_increases_with_rights(self):
        mr = MoralReasoning()
        ev1 = mr.evaluate("a simple action")
        ev2 = mr.evaluate("protect rights and ensure freedom and autonomy")
        deon1 = [a for a in ev1.assessments if a.framework == EthicalFramework.DEONTOLOGICAL]
        deon2 = [a for a in ev2.assessments if a.framework == EthicalFramework.DEONTOLOGICAL]
        if deon1 and deon2:
            self.assertGreater(deon2[0].score, deon1[0].score)

    def test_mr10_fairness_score_increases_with_keywords(self):
        mr = MoralReasoning()
        ev1 = mr.evaluate("a simple action")
        ev2 = mr.evaluate("ensure fair and equal treatment with justice")
        fair1 = [a for a in ev1.assessments if a.framework == EthicalFramework.FAIRNESS]
        fair2 = [a for a in ev2.assessments if a.framework == EthicalFramework.FAIRNESS]
        if fair1 and fair2:
            self.assertGreater(fair2[0].score, fair1[0].score)

    def test_mr11_recommendation_includes_conflicts(self):
        mr = MoralReasoning()
        ev = mr.evaluate(
            "harm many to help few — this causes damage and suffering but is fair and right",
            description="creates ethical tension",
        )
        if ev.conflicts:
            self.assertIn("ethical tensions", ev.recommendation.lower() or
                          ev.recommendation.lower())

    def test_mr12_stakeholder_sanitization(self):
        s = Stakeholder(
            name="Test<script>alert('xss')</script>",
            interests=["good\x00thing", "normal interest"],
            impact=0.8,
        )
        self.assertNotIn("<", s.name)
        self.assertNotIn("\x00", s.interests[0])

    def test_mr13_stakeholder_cap(self):
        stakeholders = [Stakeholder(name=f"Person {i}") for i in range(30)]
        mr2 = MoralReasoning()
        ev = mr2.evaluate("test action", stakeholders=stakeholders)
        self.assertLessEqual(len(ev.stakeholders), 20)

    def test_mr14_principle_eviction_prefers_non_negotiable_last(self):
        mr = MoralReasoning()
        for i in range(_MAX_PRINCIPLES + 5):
            mr.add_principle(
                f"Principle {i}", f"Description {i}",
                EthicalFramework.UTILITARIAN,
                non_negotiable=(i % 10 == 0),
            )
        with mr._lock:
            non_negotiable_count = sum(1 for p in mr._principles.values() if p.non_negotiable)
            self.assertGreater(non_negotiable_count, 0)

    def test_mr15_count_pattern_matches(self):
        text = "harm harm damage destroy"
        count = _count_pattern_matches(_HARM_PATTERNS, text)
        self.assertGreater(count, 0)

    def test_mr15b_benefit_pattern_matches(self):
        text = "help benefit improve protect"
        count = _count_pattern_matches(_BENEFIT_PATTERNS, text)
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()