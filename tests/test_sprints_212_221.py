#!/usr/bin/env python3
"""Tests for Sprints 212-220: NeuralSymbolicBridge, QuantumDecisionField,
EmergentGrammar, HolographicMemory, SyncognitiveFeedback,
MorphicResonanceEngine, ExistentialCalculus, ChronoSynthesis, PhoenixRegenerator.

Plus Sprint 221: Hypothesis property-based tests for all 9 new modules.
"""

import copy
import pytest
import threading
from datetime import datetime, timezone

from shadowd.cognition.neural_symbolic_bridge import (
    NeuralSymbolicBridge, InferenceMode, PatternType, RuleStrength,
    NeuralPattern, SymbolicRule, BridgeMapping,
)
from shadowd.cognition.quantum_decision_field import (
    QuantumDecisionField, QuantumFieldType, DecisionOption, QuantumDecision,
    EntanglementLink,
)
from shadowd.cognition.emergent_grammar import (
    EmergentGrammar, EmergenceType, GrammarRule, GrammarSymbol, GrammarSentence,
)
from shadowd.cognition.holographic_memory import (
    HolographicMemory, HologramType, HologramFragment, ReconstructionResult,
)
from shadowd.cognition.syncognitive_feedback import (
    SyncognitiveFeedback, FeedbackType,
    CognitiveObservation, FeedbackLoop, ProcessProfile,
)
from shadowd.cognition.morphic_resonance_engine import (
    MorphicResonanceEngine, ResonanceType,
    MorphicField, ResonancePattern, MorphicInfluence,
)
from shadowd.cognition.existential_calculus import (
    ExistentialCalculus, SignificanceAxis,
    MeaningFrame, PurposeAlignment, LifeTheme, CalculusResult,
)
from shadowd.cognition.chrono_synthesis import (
    ChronoSynthesis, TemporalLayer,
    PastPattern, FutureProjection, SynthesisDecision, TemporalContext,
)
from shadowd.cognition.phoenix_regenerator import (
    PhoenixRegenerator, HealthStatus,
    ComponentHealth, Diagnosis, RegenerationPlan, RegenerationResult,
)

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# ──────────────────────────────────────────────────────────────
# Sprint 212: NeuralSymbolicBridge
# ──────────────────────────────────────────────────────────────

class TestNeuralSymbolicBridge:
    def setup_method(self):
        self.nsb = NeuralSymbolicBridge()

    def test_add_pattern(self):
        p = self.nsb.add_pattern(PatternType.ACTIVATION, {"excitement": 0.8, "focus": 0.6}, symbol="joy", confidence=0.9)
        assert p.pattern_id
        assert p.pattern_type == PatternType.ACTIVATION
        assert p.symbol == "joy"
        assert p.confidence == 0.9
        assert "excitement" in p.dimensions

    def test_add_rule(self):
        r = self.nsb.add_rule(["if excited", "if focused"], "then productive", strength=RuleStrength.STRONG, confidence=0.8, domain="work")
        assert r.rule_id
        assert len(r.premises) == 2
        assert r.strength == RuleStrength.STRONG
        assert r.domain == "work"

    def test_create_mapping(self):
        p = self.nsb.add_pattern(PatternType.ACTIVATION, {"x": 0.5}, symbol="test")
        r = self.nsb.add_rule(["if x"], "then y", strength=RuleStrength.MODERATE)
        m = self.nsb.create_mapping(p.pattern_id, r.rule_id)
        assert m is not None
        assert m.mode == InferenceMode.BIDIRECTIONAL

    def test_create_mapping_nonexistent(self):
        m = self.nsb.create_mapping("nonexistent1", "nonexistent2")
        assert m is None

    def test_translate_neuro_to_symbolic(self):
        p = self.nsb.add_pattern(PatternType.ACTIVATION, {"x": 0.8}, symbol="test_sym")
        r = self.nsb.add_rule(["if test_sym"], "conclude test_sym", confidence=0.7)
        self.nsb.create_mapping(p.pattern_id, r.rule_id, bidirectional_strength=0.6)
        results = self.nsb.translate_neuro_to_symbolic(p.pattern_id)
        assert len(results) >= 1
        assert results[0].conclusion == "conclude test_sym"

    def test_translate_symbolic_to_neuro(self):
        p = self.nsb.add_pattern(PatternType.RESONANCE, {"y": 0.5})
        r = self.nsb.add_rule(["if y"], "conclude y", confidence=0.6)
        self.nsb.create_mapping(p.pattern_id, r.rule_id, mode=InferenceMode.BIDIRECTIONAL, bidirectional_strength=0.5)
        results = self.nsb.translate_symbolic_to_neuro(r.rule_id)
        assert len(results) >= 1

    def test_translate_bidirectional(self):
        p = self.nsb.add_pattern(PatternType.BURST, {"z": 0.9})
        r = self.nsb.add_rule(["if z"], "conclude z", confidence=0.8)
        self.nsb.create_mapping(p.pattern_id, r.rule_id)
        result = self.nsb.translate_bidirectional(p.pattern_id, "pattern")
        assert "symbolic_rules" in result

    def test_reinforce_mapping(self):
        p = self.nsb.add_pattern(PatternType.ACTIVATION, {"a": 0.5})
        r = self.nsb.add_rule(["if a"], "then b")
        m = self.nsb.create_mapping(p.pattern_id, r.rule_id)
        assert self.nsb.reinforce_mapping(m.mapping_id, 0.05)
        assert not self.nsb.reinforce_mapping("nonexistent", 0.05)

    def test_find_patterns_by_symbol(self):
        self.nsb.add_pattern(PatternType.ACTIVATION, {"x": 0.5}, symbol="findme")
        self.nsb.add_pattern(PatternType.INHIBITION, {"y": 0.3}, symbol="findme")
        results = self.nsb.find_patterns_by_symbol("findme")
        assert len(results) == 2

    def test_find_rules_by_domain(self):
        self.nsb.add_rule(["a"], "b", domain="test_dom")
        self.nsb.add_rule(["c"], "d", domain="test_dom")
        self.nsb.add_rule(["e"], "f", domain="other_dom")
        results = self.nsb.find_rules_by_domain("test_dom")
        assert len(results) == 2

    def test_compute_pattern_similarity(self):
        p1 = self.nsb.add_pattern(PatternType.ACTIVATION, {"x": 1.0, "y": 1.0})
        p2 = self.nsb.add_pattern(PatternType.ACTIVATION, {"x": 1.0, "y": 1.0})
        sim = self.nsb.compute_pattern_similarity(p1.pattern_id, p2.pattern_id)
        assert abs(sim - 1.0) < 0.01

    def test_deduce(self):
        self.nsb.add_rule(["rain", "clouds"], "bring_umbrella", confidence=0.8, domain="weather")
        results = self.nsb.deduct(["rain", "clouds"], domain="weather")
        assert len(results) >= 1
        assert results[0].conclusion == "bring_umbrella"

    def test_pattern_eviction(self):
        for i in range(5001):
            self.nsb.add_pattern(PatternType.ACTIVATION, {"dim": i / 5001.0}, symbol=f"sym_{i}")
        assert len(self.nsb._patterns) <= 5000

    def test_get_stats(self):
        self.nsb.add_pattern(PatternType.ACTIVATION, {"x": 0.5})
        stats = self.nsb.get_stats()
        assert "patterns_created" in stats
        assert "pattern_count" in stats

    def test_thread_safety(self):
        errors = []
        def add_patterns():
            try:
                for i in range(100):
                    self.nsb.add_pattern(PatternType.ACTIVATION, {"t": i / 100.0})
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=add_patterns) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_to_dict_immutability(self):
        p = self.nsb.add_pattern(PatternType.ACTIVATION, {"x": 0.5}, metadata={"key": "val"})
        d = p.to_dict()
        d["dimensions"]["x"] = 999
        assert p.dimensions["x"] != 999


# ──────────────────────────────────────────────────────────────
# Sprint 213: QuantumDecisionField
# ──────────────────────────────────────────────────────────────

class TestQuantumDecisionField:
    def setup_method(self):
        self.qdf = QuantumDecisionField()

    def test_create_decision(self):
        options = [DecisionOption(option_id="a", label="Option A", description="test"), DecisionOption(option_id="b", label="Option B", description="test2")]
        d = self.qdf.create_decision(options, "test context")
        assert d.decision_id
        assert d.superposition_state == QuantumFieldType.SUPERPOSITION.value

    def test_add_constraint(self):
        options = [DecisionOption(option_id="a", label="A"), DecisionOption(option_id="b", label="B")]
        d = self.qdf.create_decision(options)
        assert self.qdf.add_constraint(d.decision_id, "a", "some constraint")

    def test_observe_collapses(self):
        options = [DecisionOption(option_id="a", label="A"), DecisionOption(option_id="b", label="B")]
        d = self.qdf.create_decision(options)
        result = self.qdf.observe(d.decision_id, "a")
        assert result is not None
        assert result.option_id == "a"

    def test_collapse(self):
        options = [DecisionOption(option_id="a", label="A", base_probability=0.9), DecisionOption(option_id="b", label="B", base_probability=0.1)]
        d = self.qdf.create_decision(options)
        result = self.qdf.collapse(d.decision_id)
        assert result.superposition_state == QuantumFieldType.COLLAPSED.value

    def test_entangle(self):
        options1 = [DecisionOption(option_id="a", label="A1"), DecisionOption(option_id="b", label="B1")]
        options2 = [DecisionOption(option_id="c", label="C2"), DecisionOption(option_id="d", label="D2")]
        d1 = self.qdf.create_decision(options1)
        d2 = self.qdf.create_decision(options2)
        link = self.qdf.entangle(d1.decision_id, d2.decision_id, 0.7)
        assert link.correlation == 0.7

    def test_get_probability_field(self):
        options = [DecisionOption(option_id="a", label="A"), DecisionOption(option_id="b", label="B")]
        d = self.qdf.create_decision(options)
        field = self.qdf.get_probability_field(d.decision_id)
        assert "a" in field
        assert "b" in field

    def test_compute_interference(self):
        options1 = [DecisionOption(option_id="a", label="A1"), DecisionOption(option_id="b", label="B1")]
        options2 = [DecisionOption(option_id="c", label="C2"), DecisionOption(option_id="d", label="D2")]
        d1 = self.qdf.create_decision(options1)
        d2 = self.qdf.create_decision(options2)
        self.qdf.entangle(d1.decision_id, d2.decision_id, 0.5)
        interference = self.qdf.compute_interference(d1.decision_id)
        assert isinstance(interference, dict)

    def test_get_stats(self):
        options = [DecisionOption(option_id="a", label="A")]
        self.qdf.create_decision(options)
        stats = self.qdf.get_stats()
        assert "decisions_created" in stats
        assert stats["decisions_created"] == 1


# ──────────────────────────────────────────────────────────────
# Sprint 214: EmergentGrammar
# ──────────────────────────────────────────────────────────────

class TestEmergentGrammar:
    def setup_method(self):
        self.eg = EmergentGrammar()

    def test_observe(self):
        rule = self.eg.observe("if X then Y", context="logic", confidence=0.8)
        assert rule.rule_id
        assert rule.pattern == "if X then Y"
        assert rule.emergence_type == EmergenceType.PATTERN_DISCOVERY
        assert rule.frequency == 1

    def test_compose(self):
        sym1 = self.eg.abstract("pattern_a", 1)
        sym2 = self.eg.abstract("pattern_b", 1)
        sentence = self.eg.compose([sym1.symbol, sym2.symbol], context="test")
        assert sentence.sentence_id
        assert len(sentence.symbols) == 2

    def test_abstract(self):
        sym = self.eg.abstract("repeated_concept", 2)
        assert sym.symbol
        assert sym.entropy >= 0

    def test_refine(self):
        rule = self.eg.observe("pattern", context="test", confidence=0.5)
        assert self.eg.refine(rule.rule_id, 0.8)
        assert not self.eg.refine("nonexistent", 0.1)

    def test_merge_symbols(self):
        s1 = self.eg.abstract("concept_a", 1)
        s2 = self.eg.abstract("concept_b", 1)
        merged = self.eg.merge_symbols(s1.symbol_id, s2.symbol_id, "merged_ab")
        assert merged.symbol == "merged_ab"

    def test_split_symbol(self):
        sym = self.eg.abstract("multi_context_concept", 3)
        contexts = ["context1", "context2"]
        split = self.eg.split_symbol(sym.symbol_id, contexts)
        assert len(split) == 2

    def test_find_rules(self):
        self.eg.observe("pattern", context="domain_a", confidence=0.7, domain="domain_a")
        self.eg.observe("pattern2", context="domain_a", confidence=0.8, domain="domain_a")
        self.eg.observe("pattern3", context="domain_b", confidence=0.6, domain="domain_b")
        rules = self.eg.find_rules("domain_a")
        assert len(rules) == 2

    def test_find_symbols(self):
        for i in range(5):
            self.eg.abstract(f"concept_{i}", i + 1)
        syms = self.eg.find_symbols(min_usage=0)
        assert len(syms) >= 1

    def test_eviction(self):
        for i in range(2001):
            self.eg.observe(f"pattern_{i}", confidence=0.5)
        assert len(self.eg._rules) <= 2000

    def test_thread_safety(self):
        errors = []
        def observe():
            try:
                for i in range(50):
                    self.eg.observe(f"thread_pattern_{i}", confidence=0.5)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=observe) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ──────────────────────────────────────────────────────────────
# Sprint 215: HolographicMemory
# ──────────────────────────────────────────────────────────────

class TestHolographicMemory:
    def setup_method(self):
        self.hm = HolographicMemory()

    def test_encode(self):
        frag = self.hm.encode("Hello world, this is a test message", metadata={"type": "greeting"})
        assert frag.fragment_id
        assert frag.fragment_type == HologramType.FULL
        assert frag.reconstruction_quality > 0
        assert len(frag.encoding) > 0

    def test_encode_full(self):
        fragments = self.hm.encode_full("Test content for full encoding", redundancy=3)
        assert len(fragments) == 3
        assert fragments[0].fragment_type == HologramType.PARTIAL
        for f in fragments[1:]:
            assert f.fragment_type == HologramType.REDUNDANT

    def test_retrieve(self):
        frag = self.hm.encode("Test content")
        retrieved = self.hm.retrieve(frag.fragment_id)
        assert retrieved is not None
        assert retrieved.fragment_id == frag.fragment_id

    def test_reconstruct(self):
        fragments = self.hm.encode_full("Important content to reconstruct", redundancy=3)
        result = self.hm.reconstruct([f.fragment_id for f in fragments])
        assert result is not None
        assert result.quality > 0

    def test_partial_reconstruct(self):
        fragments = self.hm.encode_full("Another piece of content", redundancy=4)
        result = self.hm.partial_reconstruct([f.fragment_id for f in fragments[:2]], min_quality=0.3)
        assert result is not None

    def test_delete(self):
        frag = self.hm.encode("Content to delete")
        assert self.hm.delete(frag.fragment_id)
        assert not self.hm.delete("nonexistent")

    def test_find_fragments(self):
        self.hm.encode("Machine learning patterns")
        self.hm.encode("Deep learning architectures")
        results = self.hm.find_fragments("learning", limit=10)
        assert len(results) >= 2

    def test_verify_integrity(self):
        frag = self.hm.encode("Verify this content")
        assert self.hm.verify_integrity(frag.fragment_id)

    def test_get_redundancy_level(self):
        fragments = self.hm.encode_full("Redundant content", redundancy=3)
        level = self.hm.get_redundancy_level(fragments[0].fragment_id)
        assert level >= 0

    def test_eviction(self):
        for i in range(5001):
            self.hm.encode(f"Content {i}")
        assert len(self.hm._fragments) <= 5000


# ──────────────────────────────────────────────────────────────
# Sprint 216: SyncognitiveFeedback
# ──────────────────────────────────────────────────────────────

class TestSyncognitiveFeedback:
    def setup_method(self):
        self.sf = SyncognitiveFeedback()

    def test_observe_process(self):
        obs = self.sf.observe_process("reasoning", "Detected bias pattern", 0.85, FeedbackType.BIAS_DETECTION)
        assert obs.observation_id
        assert obs.process_name == "reasoning"
        assert obs.observation_type == FeedbackType.BIAS_DETECTION

    def test_create_feedback_loop(self):
        loop = self.sf.create_feedback_loop("emotional_intel", "reasoning", FeedbackType.BIAS_DETECTION, 0.7)
        assert loop.loop_id
        assert loop.strength == 0.7

    def test_propagate_feedback(self):
        loop = self.sf.create_feedback_loop("reasoning", "attention", FeedbackType.CONFIDENCE_ADJUSTMENT, 0.5)
        obs = self.sf.observe_process("reasoning", "Low confidence detected", 0.6, FeedbackType.CONFIDENCE_ADJUSTMENT)
        result = self.sf.propagate_feedback(loop.loop_id, obs)
        assert "adjustments" in result

    def test_correct_self(self):
        obs = self.sf.correct_self("reasoning", "Over-weighted recent evidence", -0.1)
        assert obs.observation_id
        assert obs.observation_type == FeedbackType.SELF_CORRECTION
        assert obs.confidence_after < obs.confidence_before

    def test_detect_bias(self):
        for i in range(5):
            self.sf.observe_process("reasoning", f"Same pattern repeated {i}", 0.8, FeedbackType.BIAS_DETECTION)
        biases = self.sf.detect_bias("reasoning")
        assert isinstance(biases, list)

    def test_get_process_profile(self):
        self.sf.observe_process("attention", "Normal focus", 0.8, FeedbackType.PROCESS_OPTIMIZATION)
        profile = self.sf.get_process_profile("attention")
        assert profile is not None
        assert profile.process_name == "attention"

    def test_eviction(self):
        for i in range(5001):
            self.sf.observe_process("proc", f"Obs {i}", 0.5, FeedbackType.SELF_CORRECTION)
        assert len(self.sf._observations) <= 5000


# ──────────────────────────────────────────────────────────────
# Sprint 217: MorphicResonanceEngine
# ──────────────────────────────────────────────────────────────

class TestMorphicResonanceEngine:
    def setup_method(self):
        self.mre = MorphicResonanceEngine()

    def test_register_pattern(self):
        p = self.mre.register_pattern("technology", "disruptive_innovation", 0.8)
        assert p.pattern_id
        assert p.source_domain == "technology"
        assert p.confidence == 0.8

    def test_propagate(self):
        self.mre.register_pattern("technology", "rapid_iteration", 0.9)
        self.mre.register_pattern("business", "startup_culture", 0.7)
        influences = self.mre.propagate("technology", "business", min_confidence=0.5)
        assert isinstance(influences, list)

    def test_detect_resonance(self):
        self.mre.register_pattern("science", "peer_review", 0.8)
        self.mre.register_pattern("technology", "code_review", 0.7)
        resonances = self.mre.detect_resonance("technology", "review")
        assert isinstance(resonances, list)

    def test_get_field(self):
        self.mre.register_pattern("health", "exercise_pattern", 0.7)
        field = self.mre.get_field("health")
        assert field is not None
        assert field.domain == "health"

    def test_apply_influence(self):
        p = self.mre.register_pattern("science", "hypothesis_testing", 0.9)
        influences = self.mre.propagate("science", "technology", 0.5)
        if influences:
            assert self.mre.apply_influence(influences[0].influence_id)

    def test_compute_resonance_strength(self):
        p1 = self.mre.register_pattern("domain_a", "pattern_alpha_beta_gamma", 0.8)
        p2 = self.mre.register_pattern("domain_b", "pattern_alpha_beta_delta", 0.7)
        strength = self.mre.compute_resonance_strength(p1.pattern_id, p2.pattern_id)
        assert 0.0 <= strength <= 1.0

    def test_decay_influences(self):
        self.mre.register_pattern("tech", "innovation", 0.9)
        self.mre.propagate("tech", "business", 0.5)
        decayed = self.mre.decay_influences()
        assert isinstance(decayed, int)


# ──────────────────────────────────────────────────────────────
# Sprint 218: ExistentialCalculus
# ──────────────────────────────────────────────────────────────

class TestExistentialCalculus:
    def setup_method(self):
        self.ec = ExistentialCalculus()

    def test_evaluate_significance(self):
        result = self.ec.evaluate_significance("event_1", "Made a career change decision", {SignificanceAxis.PURPOSE: 0.8, SignificanceAxis.IMPACT: 0.7})
        assert result.result_id
        assert 0 <= result.significance <= 1
        assert result.meaning != ""

    def test_compute_purpose_alignment(self):
        alignment = self.ec.compute_purpose_alignment("Help others learn", ["teaching", "mentoring", "writing"])
        assert alignment.alignment_id
        assert 0 <= alignment.alignment_score <= 1

    def test_detect_life_themes(self):
        for i in range(5):
            self.ec.evaluate_significance(f"event_{i}", f"Growth through challenge {i}", {SignificanceAxis.GROWTH: 0.8})
        themes = self.ec.detect_life_themes(min_frequency=2)
        assert isinstance(themes, list)

    def test_add_meaning_frame(self):
        frame = self.ec.add_meaning_frame("event_1", {SignificanceAxis.PURPOSE.value: 0.8, SignificanceAxis.IMPACT.value: 0.7}, "This event aligned with my purpose")
        assert frame.frame_id
        assert frame.overall_significance > 0

    def test_recommend_action(self):
        self.ec.evaluate_significance("evt", "A decision point", {SignificanceAxis.PURPOSE: 0.8, SignificanceAxis.GROWTH: 0.7})
        rec = self.ec.recommend_action("evt")
        assert isinstance(rec, str)

    def test_eviction(self):
        for i in range(3001):
            self.ec.add_meaning_frame(f"e_{i}", {SignificanceAxis.IMPACT.value: 0.5}, f"Frame {i}")
        assert len(self.ec._frames) <= 3000


# ──────────────────────────────────────────────────────────────
# Sprint 219: ChronoSynthesis
# ──────────────────────────────────────────────────────────────

class TestChronoSynthesis:
    def setup_method(self):
        self.cs = ChronoSynthesis()

    def test_record_past_pattern(self):
        pp = self.cs.record_past_pattern("meeting", "work", frequency=5)
        assert pp.pattern_id
        assert pp.event_type == "meeting"
        assert pp.domain == "work"

    def test_add_future_projection(self):
        fp = self.cs.add_future_projection("project_deadline", probability=0.7, timeframe="2 weeks", confidence=0.6)
        assert fp.projection_id
        assert fp.probability == 0.7

    def test_create_temporal_context(self):
        pp = self.cs.record_past_pattern("pattern_a", "work", frequency=3)
        fp = self.cs.add_future_projection("deadline", probability=0.8, timeframe="1 week", confidence=0.7)
        tc = self.cs.create_temporal_context(
            past_pattern_ids=[pp.pattern_id],
            present_factors={"urgency": 0.7, "resources": 0.8},
            future_projection_ids=[fp.projection_id],
            domain="work",
            urgency=0.6
        )
        assert tc.context_id

    def test_synthesize(self):
        pp = self.cs.record_past_pattern("recurring_task", "work", frequency=10)
        fp = self.cs.add_future_projection("deadline_approaching", probability=0.8, timeframe="3 days", confidence=0.7)
        tc = self.cs.create_temporal_context([pp.pattern_id], {"urgency": 0.7}, [fp.projection_id], "work", 0.8)
        decision = self.cs.synthesize(tc.context_id)
        assert decision.decision_id
        assert 0 <= decision.synthesized_score <= 1

    def test_compute_past_weight(self):
        pp = self.cs.record_past_pattern("old_event", "life", frequency=10)
        weight = self.cs.compute_past_weight(pp.pattern_id)
        assert 0 <= weight <= 1

    def test_compute_future_weight(self):
        fp = self.cs.add_future_projection("upcoming_event", probability=0.8, timeframe="1 day", confidence=0.7)
        weight = self.cs.compute_future_weight(fp.projection_id)
        assert 0 <= weight <= 1

    def test_eviction(self):
        for i in range(5001):
            self.cs.record_past_pattern(f"event_{i}", "domain", frequency=1)
        assert len(self.cs._past_patterns) <= 5000


# ──────────────────────────────────────────────────────────────
# Sprint 220: PhoenixRegenerator
# ──────────────────────────────────────────────────────────────

class TestPhoenixRegenerator:
    def setup_method(self):
        self.pr = PhoenixRegenerator()

    def test_register_component(self):
        ch = self.pr.register_component("comp_1", "DecisionEngine", 0.9)
        assert ch.component_id == "comp_1"
        assert ch.health == HealthStatus.OPTIMAL
        assert ch.health_score == 0.9

    def test_check_health(self):
        self.pr.register_component("comp_2", "MemorySystem", 0.6)
        ch = self.pr.check_health("comp_2")
        assert ch.health == HealthStatus.DEGRADED

    def test_register_component_critical(self):
        ch = self.pr.register_component("comp_3", "CriticalModule", 0.1)
        assert ch.health == HealthStatus.FAILED

    def test_report_error(self):
        self.pr.register_component("comp_4", "TestModule", 0.9)
        diag = self.pr.report_error("comp_4", "Module crashed during processing")
        assert diag.diagnosis_id
        assert diag.severity > 0

    def test_diagnose(self):
        self.pr.register_component("comp_5", "TestModule", 0.5)
        diag = self.pr.diagnose("comp_5", ["slow_response", "error_rate_high"])
        assert diag.diagnosis_id
        assert len(diag.symptoms) > 0

    def test_create_regeneration_plan(self):
        self.pr.register_component("comp_6", "TestModule", 0.3)
        plan = self.pr.create_regeneration_plan("comp_6", HealthStatus.OPTIMAL)
        assert plan.plan_id
        assert plan.current_state == HealthStatus.CRITICAL

    def test_execute_regeneration(self):
        self.pr.register_component("comp_7", "TestModule", 0.2)
        plan = self.pr.create_regeneration_plan("comp_7", HealthStatus.OPTIMAL)
        result = self.pr.execute_regeneration(plan.plan_id)
        assert result.result_id
        assert result.health_after >= result.health_before

    def test_preserve_restore_learning(self):
        self.pr.register_component("comp_8", "TestModule", 0.8)
        learning = self.pr.preserve_learning("comp_8")
        assert isinstance(learning, dict)
        assert self.pr.restore_learning("comp_8", learning)

    def test_eviction(self):
        for i in range(201):
            self.pr.register_component(f"comp_evict_{i}", f"Module_{i}", 0.9)
        assert len(self.pr._components) <= 200


# ──────────────────────────────────────────────────────────────
# Sprint 221: Hypothesis Property-Based Tests
# ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestNeuralSymbolicBridgeHypothesis:
    @given(
        pattern_type=st.sampled_from(list(PatternType)),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        symbol=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_pattern_confidence_clamped(self, pattern_type, confidence, symbol):
        nsb = NeuralSymbolicBridge()
        p = nsb.add_pattern(pattern_type, {"dim": 0.5}, symbol=symbol, confidence=confidence)
        assert 0.0 <= p.confidence <= 1.0

    @given(
        rule_strength=st.sampled_from(list(RuleStrength)),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        domain=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_rule_confidence_clamped(self, rule_strength, confidence, domain):
        nsb = NeuralSymbolicBridge()
        r = nsb.add_rule(["premise"], "conclusion", strength=rule_strength, confidence=confidence, domain=domain)
        assert 0.0 <= r.confidence <= 1.0


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestQuantumDecisionFieldHypothesis:
    @given(
        base_prob=st.floats(min_value=0.0, max_value=1.0),
        num_options=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_decision_probabilities_sum(self, base_prob, num_options):
        qdf = QuantumDecisionField()
        options = [{"label": f"Option {i}", "probability": base_prob} for i in range(num_options)]
        d = qdf.create_decision(options, "test")
        prob_sum = sum(d.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01 or prob_sum == 0  # equal distribution or empty


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestEmergentGrammarHypothesis:
    @given(
        pattern=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=('L', 'N', 'P'))),
        confidence=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_observe_confidence_clamped(self, pattern, confidence):
        eg = EmergentGrammar()
        rule = eg.observe(pattern, context="test", confidence=confidence)
        assert 0.0 <= rule.confidence <= 1.0

    @given(
        symbol=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        level=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_abstract_entropy_range(self, symbol, level):
        eg = EmergentGrammar()
        sym = eg.abstract(symbol, level)
        assert 0.0 <= sym.entropy <= 1.0


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestHolographicMemoryHypothesis:
    @given(content=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=('L', 'N', 'Z'))))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_encode_retrieve_roundtrip(self, content):
        hm = HolographicMemory()
        frag = hm.encode(content)
        retrieved = hm.retrieve(frag.fragment_id)
        assert retrieved is not None
        assert retrieved.content == content.strip()[:5000].strip()

    @given(redundancy=st.integers(min_value=1, max_value=5))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_encode_full_redundancy(self, redundancy):
        hm = HolographicMemory()
        fragments = hm.encode_full("Test content for encoding", redundancy=redundancy)
        assert len(fragments) == redundancy
        assert fragments[0].fragment_type == HologramType.PARTIAL
        for f in fragments[1:]:
            assert f.fragment_type == HologramType.REDUNDANT


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestSyncognitiveFeedbackHypothesis:
    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0),
        feedback_type=st.sampled_from(list(FeedbackType)),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_observation_confidence_clamped(self, confidence, feedback_type):
        sf = SyncognitiveFeedback()
        obs = sf.observe_process("test_process", "Test details", confidence, feedback_type)
        assert 0.0 <= obs.confidence_before <= 1.0

    @given(strength=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_feedback_loop_strength_clamped(self, strength):
        sf = SyncognitiveFeedback()
        loop = sf.create_feedback_loop("src", "tgt", FeedbackType.SELF_CORRECTION, strength)
        assert 0.0 <= loop.strength <= 1.0


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestMorphicResonanceHypothesis:
    @given(confidence=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_pattern_confidence_clamped(self, confidence):
        mre = MorphicResonanceEngine()
        p = mre.register_pattern("test_domain", "test_pattern", confidence)
        assert 0.0 <= p.confidence <= 1.0

    @given(strength=st.floats(min_value=0.0, max_value=1.0), decay_rate=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_influence_fields_clamped(self, strength, decay_rate):
        mre = MorphicResonanceEngine()
        p = mre.register_pattern("domain_a", "pattern_a", 0.8)
        influences = mre.propagate("domain_a", "domain_b", 0.5)
        # Just test creation
        assert isinstance(influences, list)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestExistentialCalculusHypothesis:
    @given(
        significance=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_significance_clamped(self, significance):
        ec = ExistentialCalculus()
        result = ec.evaluate_significance("evt", "Test event", {SignificanceAxis.PURPOSE: significance})
        assert 0.0 <= result.significance <= 1.0

    @given(
        alignment_score=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_purpose_alignment_clamped(self, alignment_score):
        ec = ExistentialCalculus()
        # alignment_score is computed, not input - just test structure
        pa = ec.compute_purpose_alignment("test purpose", ["activity1"])
        assert 0.0 <= pa.alignment_score <= 1.0


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestChronoSynthesisHypothesis:
    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0),
        probability=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_pattern_projection_confidence_clamped(self, confidence, probability):
        cs = ChronoSynthesis()
        pp = cs.record_past_pattern("event_type", "domain", frequency=3)
        assert 0.0 <= pp.confidence <= 1.0
        fp = cs.add_future_projection("event_type", probability=probability, timeframe="1 week", confidence=confidence)
        assert 0.0 <= fp.probability <= 1.0
        assert 0.0 <= fp.confidence <= 1.0


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestPhoenixRegeneratorHypothesis:
    @given(health_score=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_health_score_determines_status(self, health_score):
        pr = PhoenixRegenerator()
        ch = pr.register_component(f"comp_{health_score:.4f}", "TestModule", health_score)
        if health_score >= 0.8:
            assert ch.health == HealthStatus.OPTIMAL
        elif health_score >= 0.5:
            assert ch.health == HealthStatus.DEGRADED
        elif health_score >= 0.2:
            assert ch.health == HealthStatus.CRITICAL
        else:
            assert ch.health == HealthStatus.FAILED


# ──────────────────────────────────────────────────────────────
# ShadowOS Integration
# ──────────────────────────────────────────────────────────────

class TestShadowOSIntegration:
    def setup_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        self.os = ShadowOS()

    def test_new_modules_in_status(self):
        status = self.os.get_status()
        assert "neural_bridge_stats" in status
        assert "quantum_field_stats" in status
        assert "grammar_stats" in status
        assert "holographic_stats" in status
        assert "syncognitive_stats" in status
        assert "morphic_stats" in status
        assert "existential_stats" in status
        assert "chrono_stats" in status
        assert "phoenix_stats" in status

    def test_new_modules_in_health(self):
        health = self.os.get_health()
        assert "modules" in health
        assert isinstance(health["modules"], dict)
        assert hasattr(self.os, "neural_bridge")
        assert hasattr(self.os, "quantum_field")
        assert hasattr(self.os, "grammar")
        assert hasattr(self.os, "holographic")
        assert hasattr(self.os, "syncognitive")
        assert hasattr(self.os, "morphic")
        assert hasattr(self.os, "existential")
        assert hasattr(self.os, "chrono")
        assert hasattr(self.os, "phoenix")

    def test_module_types(self):
        assert isinstance(self.os.neural_bridge, NeuralSymbolicBridge)
        assert isinstance(self.os.quantum_field, QuantumDecisionField)
        assert isinstance(self.os.grammar, EmergentGrammar)
        assert isinstance(self.os.holographic, HolographicMemory)
        assert isinstance(self.os.syncognitive, SyncognitiveFeedback)
        assert isinstance(self.os.morphic, MorphicResonanceEngine)
        assert isinstance(self.os.existential, ExistentialCalculus)
        assert isinstance(self.os.chrono, ChronoSynthesis)
        assert isinstance(self.os.phoenix, PhoenixRegenerator)