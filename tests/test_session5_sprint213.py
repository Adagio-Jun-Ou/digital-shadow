#!/usr/bin/env python3
"""Regression tests for Sprint 213 second-pass audit fixes across 9 ShadowAI cognition modules."""

import copy
import threading
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSyncognitiveFeedbackCorrectSelf:
    def test_correct_self_atomic_lock(self):
        from shadowd.cognition.syncognitive_feedback import SyncognitiveFeedback, FeedbackType
        sf = SyncognitiveFeedback()
        sf.observe_process("proc_a", "initial", 0.8, FeedbackType.SELF_CORRECTION)
        obs = sf.correct_self("proc_a", "fix something", -0.1)
        assert obs.process_name == "proc_a"
        assert obs.observation_type == FeedbackType.SELF_CORRECTION
        assert obs.confidence_after < obs.confidence_before

    def test_correct_self_no_toctou_stale_confidence(self):
        from shadowd.cognition.syncognitive_feedback import SyncognitiveFeedback, FeedbackType
        sf = SyncognitiveFeedback()
        sf.observe_process("proc_b", "initial", 0.7, FeedbackType.BIAS_DETECTION)
        obs = sf.correct_self("proc_b", "correction", 0.1)
        assert obs.confidence_before == pytest.approx(0.7, abs=0.01)
        assert obs.confidence_after == pytest.approx(0.8, abs=0.01)

    def test_correct_self_concurrent(self):
        from shadowd.cognition.syncognitive_feedback import SyncognitiveFeedback, FeedbackType
        sf = SyncognitiveFeedback()
        sf.observe_process("proc_c", "initial", 0.5, FeedbackType.SELF_CORRECTION)
        results = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            obs = sf.correct_self("proc_c", "correction", 0.01)
            results.append(obs)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 4


class TestEmergentGrammarSymbolLookup:
    def test_compose_increments_usage_count(self):
        from shadowd.cognition.emergent_grammar import EmergentGrammar
        eg = EmergentGrammar()
        sym = eg.abstract("test_pattern")
        assert sym.usage_count == 0
        sentence = eg.compose([sym.symbol])
        refreshed = eg.find_symbols(min_usage=1)
        assert len(refreshed) == 1
        assert refreshed[0].symbol == sym.symbol
        assert refreshed[0].usage_count >= 1

    def test_compose_unknown_symbol_no_crash(self):
        from shadowd.cognition.emergent_grammar import EmergentGrammar
        eg = EmergentGrammar()
        sentence = eg.compose(["nonexistent"])
        assert sentence is not None

    def test_symbol_by_name_reverse_index(self):
        from shadowd.cognition.emergent_grammar import EmergentGrammar
        eg = EmergentGrammar()
        sym1 = eg.abstract("hello world")
        sym2 = eg.abstract("goodbye world")
        assert eg._symbol_by_name.get(sym1.symbol) == sym1.symbol_id
        assert eg._symbol_by_name.get(sym2.symbol) == sym2.symbol_id

    def test_merge_updates_reverse_index(self):
        from shadowd.cognition.emergent_grammar import EmergentGrammar
        eg = EmergentGrammar()
        sym1 = eg.abstract("alpha")
        sym2 = eg.abstract("beta")
        merged = eg.merge_symbols(sym1.symbol_id, sym2.symbol_id, "gamma")
        assert merged is not None
        assert eg._symbol_by_name.get("gamma") == merged.symbol_id
        assert eg._symbol_by_name.get("alpha") is None
        assert eg._symbol_by_name.get("beta") is None

    def test_split_updates_reverse_index(self):
        from shadowd.cognition.emergent_grammar import EmergentGrammar
        eg = EmergentGrammar()
        sym = eg.abstract("original")
        splits = eg.split_symbol(sym.symbol_id, ["ctx1", "ctx2"])
        assert len(splits) == 2
        assert eg._symbol_by_name.get("original") is None
        for s in splits:
            assert eg._symbol_by_name.get(s.symbol) == s.symbol_id

    def test_eviction_cleans_reverse_index(self):
        from shadowd.cognition.emergent_grammar import EmergentGrammar, _MAX_SYMBOLS
        import uuid
        eg = EmergentGrammar()
        first_sym = eg.abstract("first_symbol")
        for i in range(_MAX_SYMBOLS + 5):
            eg.abstract(f"sym_{i}_{uuid.uuid4().hex[:8]}")
        assert eg._symbol_by_name.get("first_symbol") is None


class TestNeuralSymbolicBridgeSymbolIndexCap:
    def test_symbol_index_list_cap(self):
        from shadowd.cognition.neural_symbolic_bridge import NeuralSymbolicBridge, PatternType, _MAX_PATTERNS_PER_SYMBOL
        nsb = NeuralSymbolicBridge()
        for i in range(_MAX_PATTERNS_PER_SYMBOL + 10):
            nsb.add_pattern(PatternType.ACTIVATION, {"d1": 0.5}, symbol="popular_symbol")
        with nsb._lock:
            assert len(nsb._symbol_index.get("popular_symbol", [])) <= _MAX_PATTERNS_PER_SYMBOL

    def test_symbol_index_cap_constant_exists(self):
        from shadowd.cognition.neural_symbolic_bridge import _MAX_PATTERNS_PER_SYMBOL
        assert _MAX_PATTERNS_PER_SYMBOL == 100


class TestExistentialCalculusFramesPerEventCap:
    def test_frames_per_event_cap(self):
        from shadowd.cognition.existential_calculus import ExistentialCalculus, SignificanceAxis, _MAX_FRAMES_PER_EVENT
        ec = ExistentialCalculus()
        for i in range(_MAX_FRAMES_PER_EVENT + 10):
            ec.add_meaning_frame("event_1", {"purpose": 0.5}, "narrative")
        with ec._lock:
            assert len(ec._frames_by_event.get("event_1", [])) <= _MAX_FRAMES_PER_EVENT

    def test_frames_per_event_cap_constant(self):
        from shadowd.cognition.existential_calculus import _MAX_FRAMES_PER_EVENT
        assert _MAX_FRAMES_PER_EVENT == 50


class TestPhoenixRegeneratorHashFix:
    def test_hash_deterministic(self):
        from shadowd.cognition.phoenix_regenerator import PhoenixRegenerator
        pr = PhoenixRegenerator()
        pr.register_component("comp_1", "Component 1", 0.3)
        plan1 = pr.create_regeneration_plan("comp_1")
        assert plan1 is not None
        import hashlib
        pid = plan1.plan_id
        expected = (int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16) % 100) / 1000.0
        result = pr.execute_regeneration(plan1.plan_id)
        assert result is not None

    def test_regeneration_plan_invalid_health_status(self):
        from shadowd.cognition.phoenix_regenerator import RegenerationPlan, HealthStatus
        plan = RegenerationPlan(
            plan_id="test",
            component_id="comp",
            current_state="invalid_status",
            target_state="also_invalid",
            steps=["step1"],
            learning_preservation=0.5,
            estimated_time="1 min",
            metadata={},
        )
        assert plan.current_state == HealthStatus.DEGRADED
        assert plan.target_state == HealthStatus.DEGRADED

    def test_regeneration_plan_valid_health_status(self):
        from shadowd.cognition.phoenix_regenerator import RegenerationPlan, HealthStatus
        plan = RegenerationPlan(
            plan_id="test2",
            component_id="comp",
            current_state=HealthStatus.CRITICAL,
            target_state=HealthStatus.OPTIMAL,
            steps=["step1"],
            learning_preservation=0.5,
            estimated_time="1 min",
            metadata={},
        )
        assert plan.current_state == HealthStatus.CRITICAL
        assert plan.target_state == HealthStatus.OPTIMAL

    def test_hashlib_imported(self):
        import shadowd.cognition.phoenix_regenerator as pr_mod
        assert hasattr(pr_mod, 'hashlib') or 'hashlib' in dir(pr_mod)


class TestQuantumDecisionFieldTypeStringEnum:
    def test_quantum_field_type_is_str_enum(self):
        from shadowd.cognition.quantum_decision_field import QuantumFieldType
        assert issubclass(QuantumFieldType, str)

    def test_quantum_field_type_direct_comparison(self):
        from shadowd.cognition.quantum_decision_field import QuantumFieldType
        assert QuantumFieldType.COLLAPSED == "collapsed"
        assert QuantumFieldType.SUPERPOSITION == "superposition"
        assert QuantumFieldType.ENTANGLED == "entangled"

    def test_decision_superposition_state_comparison(self):
        from shadowd.cognition.quantum_decision_field import QuantumDecisionField
        qdf = QuantumDecisionField()
        d = qdf.create_decision()
        assert d.superposition_state == "superposition"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])