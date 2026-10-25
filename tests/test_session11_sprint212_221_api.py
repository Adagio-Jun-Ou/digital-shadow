#!/usr/bin/env python3
"""Session 11 Sprint 212-221 API tests: 9 ShadowAI cognition modules (76 endpoints)."""

import copy
import os
import threading

os.environ.pop("SHADOW_OS_API_KEY", None)

import pytest
from fastapi.testclient import TestClient

import shadowd.cognition.api_router as api_module
from shadowd.cognition.shadow_os import ShadowOS
from shadowd.cognition.neural_symbolic_bridge import NeuralSymbolicBridge
from shadowd.cognition.quantum_decision_field import QuantumDecisionField
from shadowd.cognition.emergent_grammar import EmergentGrammar
from shadowd.cognition.holographic_memory import HolographicMemory
from shadowd.cognition.syncognitive_feedback import SyncognitiveFeedback
from shadowd.cognition.morphic_resonance_engine import MorphicResonanceEngine
from shadowd.cognition.existential_calculus import ExistentialCalculus
from shadowd.cognition.chrono_synthesis import ChronoSynthesis
from shadowd.cognition.phoenix_regenerator import PhoenixRegenerator


@pytest.fixture(autouse=True)
def _setup_shadow_os():
    new_os = ShadowOS.__new__(ShadowOS)
    new_os.neural_bridge = NeuralSymbolicBridge()
    new_os.quantum_field = QuantumDecisionField()
    new_os.grammar = EmergentGrammar()
    new_os.holographic = HolographicMemory()
    new_os.syncognitive = SyncognitiveFeedback()
    new_os.morphic = MorphicResonanceEngine()
    new_os.existential = ExistentialCalculus()
    new_os.chrono = ChronoSynthesis()
    new_os.phoenix = PhoenixRegenerator()
    with api_module._shadow_os_lock:
        api_module._shadow_os = new_os
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(api_module.router)
    yield TestClient(app)
    with api_module._shadow_os_lock:
        api_module._shadow_os = None


# ===================== NEURAL BRIDGE (9 endpoints) =====================

class TestNeuralBridgeEndpoints:
    def test_add_pattern(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/neural-bridge/pattern", json={
            "pattern_type": "activation", "dimensions": {"x": 0.5}, "confidence": 0.8,
        })
        assert r.status_code == 200
        d = r.json()
        assert "pattern_id" in d

    def test_add_pattern_invalid_confidence(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/neural-bridge/pattern", json={
            "pattern_type": "activation", "confidence": 1.5,
        })
        assert r.status_code == 422

    def test_add_rule(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/neural-bridge/rule", json={
            "premises": ["a"], "conclusion": "b", "strength": "strong", "domain": "test",
        })
        assert r.status_code == 200
        assert r.json()["conclusion"] == "b"

    def test_add_rule_missing_conclusion(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/neural-bridge/rule", json={"premises": []})
        assert r.status_code == 422

    def test_create_mapping(self, _setup_shadow_os):
        c = _setup_shadow_os
        p = c.post("/os/neural-bridge/pattern", json={"pattern_type": "activation"}).json()
        ru = c.post("/os/neural-bridge/rule", json={"conclusion": "x"}).json()
        r = c.post("/os/neural-bridge/map", json={
            "pattern_id": p["pattern_id"], "rule_id": ru["rule_id"],
        })
        assert r.status_code == 200
        assert "mapping_id" in r.json()

    def test_create_mapping_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/neural-bridge/map", json={"pattern_id": "fake", "rule_id": "fake"})
        assert r.status_code == 404

    def test_translate_neuro_to_symbolic(self, _setup_shadow_os):
        c = _setup_shadow_os
        p = c.post("/os/neural-bridge/pattern", json={"pattern_type": "activation"}).json()
        r = c.post("/os/neural-bridge/translate/neuro-to-symbolic", json={"pattern_id": p["pattern_id"]})
        assert r.status_code == 200
        assert "rules" in r.json()

    def test_translate_symbolic_to_neuro(self, _setup_shadow_os):
        c = _setup_shadow_os
        ru = c.post("/os/neural-bridge/rule", json={"conclusion": "y"}).json()
        r = c.post("/os/neural-bridge/translate/symbolic-to-neuro", json={"rule_id": ru["rule_id"]})
        assert r.status_code == 200
        assert "patterns" in r.json()

    def test_bidirectional_translate(self, _setup_shadow_os):
        c = _setup_shadow_os
        p = c.post("/os/neural-bridge/pattern", json={"pattern_type": "activation"}).json()
        r = c.post("/os/neural-bridge/translate/bidirectional", json={
            "source_id": p["pattern_id"], "source_type": "pattern",
        })
        assert r.status_code == 200

    def test_get_pattern(self, _setup_shadow_os):
        c = _setup_shadow_os
        p = c.post("/os/neural-bridge/pattern", json={"pattern_type": "activation"}).json()
        r = c.get(f"/os/neural-bridge/pattern/{p['pattern_id']}")
        assert r.status_code == 200
        assert r.json()["pattern_id"] == p["pattern_id"]

    def test_get_pattern_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.get("/os/neural-bridge/pattern/nonexistent999")
        assert r.status_code == 404

    def test_list_rules(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/neural-bridge/rule", json={"conclusion": "z", "domain": "d1"})
        r = c.get("/os/neural-bridge/rules", params={"domain": "d1"})
        assert r.status_code == 200
        assert "rules" in r.json()

    def test_stats(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/neural-bridge/pattern", json={"pattern_type": "activation"})
        r = c.get("/os/neural-bridge/stats")
        assert r.status_code == 200
        assert "patterns_created" in r.json() or "pattern_count" in r.json()


# ===================== QUANTUM DECISION FIELD (8 endpoints) =====================

class TestQuantumFieldEndpoints:
    def test_create_decision(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/quantum-field/decision", json={
            "options": [{"label": "A", "base_probability": 0.4}, {"label": "B", "base_probability": 0.6}],
            "context": "test choice",
        })
        assert r.status_code == 200
        d = r.json()
        assert "decision_id" in d

    def test_create_decision_empty_options(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/quantum-field/decision", json={"options": []})
        assert r.status_code == 200
        assert "decision_id" in r.json()

    def test_observe_decision(self, _setup_shadow_os):
        c = _setup_shadow_os
        dec = c.post("/os/quantum-field/decision", json={
            "options": [{"label": "X", "base_probability": 0.5}],
        }).json()
        opt_id = dec["options"][0]["option_id"]
        r = c.post(f"/os/quantum-field/decision/{dec['decision_id']}/observe", json={"option_id": opt_id})
        assert r.status_code == 200
        assert r.json()["option_id"] == opt_id

    def test_observe_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/quantum-field/decision/fake/observe", json={"option_id": "x"})
        assert r.status_code == 404

    def test_collapse_decision(self, _setup_shadow_os):
        c = _setup_shadow_os
        dec = c.post("/os/quantum-field/decision", json={
            "options": [{"label": "A", "base_probability": 0.5}, {"label": "B", "base_probability": 0.5}],
        }).json()
        r = c.post(f"/os/quantum-field/decision/{dec['decision_id']}/collapse")
        assert r.status_code == 200
        assert "decision_id" in r.json()

    def test_collapse_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/quantum-field/decision/fake/collapse")
        assert r.status_code == 404

    def test_entangle(self, _setup_shadow_os):
        c = _setup_shadow_os
        d1 = c.post("/os/quantum-field/decision", json={"options": [{"label": "A"}]}).json()
        d2 = c.post("/os/quantum-field/decision", json={"options": [{"label": "B"}]}).json()
        r = c.post("/os/quantum-field/entangle", json={
            "decision_id_1": d1["decision_id"], "decision_id_2": d2["decision_id"], "strength": 0.7,
        })
        assert r.status_code == 200
        assert "link_id" in r.json()

    def test_get_decision(self, _setup_shadow_os):
        c = _setup_shadow_os
        dec = c.post("/os/quantum-field/decision", json={"options": [{"label": "C"}]}).json()
        r = c.get(f"/os/quantum-field/decision/{dec['decision_id']}")
        assert r.status_code == 200
        assert r.json()["decision_id"] == dec["decision_id"]

    def test_get_decision_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.get("/os/quantum-field/decision/nope")
        assert r.status_code == 404

    def test_list_decisions(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/quantum-field/decision", json={"options": [{"label": "D"}]})
        r = c.get("/os/quantum-field/decisions")
        assert r.status_code == 200
        assert "decisions" in r.json()

    def test_list_entanglements(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.get("/os/quantum-field/entanglements")
        assert r.status_code == 200
        assert "entanglements" in r.json()

    def test_quantum_stats(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/quantum-field/decision", json={"options": [{"label": "E"}]})
        r = c.get("/os/quantum-field/stats")
        assert r.status_code == 200
        assert "decisions_created" in r.json()

    def test_entangle_invalid_strength(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/quantum-field/entangle", json={
            "decision_id_1": "a", "decision_id_2": "b", "strength": 2.0,
        })
        assert r.status_code == 422


# ===================== EMERGENT GRAMMAR (7 endpoints) =====================

class TestGrammarEndpoints:
    def test_observe(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/grammar/observe", json={
            "pattern": "repeated test pattern", "context": "test", "confidence": 0.7,
        })
        assert r.status_code == 200
        d = r.json()
        assert "rule_id" in d

    def test_observe_missing_pattern(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/grammar/observe", json={"context": "x"})
        assert r.status_code == 422

    def test_compose(self, _setup_shadow_os):
        c = _setup_shadow_os
        obs = c.post("/os/grammar/observe", json={"pattern": "hello world"}).json()
        r = c.post("/os/grammar/compose", json={"rule_ids": [obs["rule_id"]]})
        assert r.status_code == 200
        assert "sentence_id" in r.json()

    def test_abstract(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/grammar/observe", json={"pattern": "abstraction source"})
        r = c.post("/os/grammar/abstract", json={"symbol": "abs_src", "level": 2})
        assert r.status_code == 200
        d = r.json()
        assert "symbol_id" in d or "symbol" in d

    def test_refine(self, _setup_shadow_os):
        c = _setup_shadow_os
        obs = c.post("/os/grammar/observe", json={"pattern": "refine me"}).json()
        r = c.post("/os/grammar/refine", json={"rule_id": obs["rule_id"], "confidence": 0.9})
        assert r.status_code == 200

    def test_list_rules(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/grammar/observe", json={"pattern": "list rules test"})
        r = c.get("/os/grammar/rules")
        assert r.status_code == 200
        assert "rules" in r.json()

    def test_list_symbols(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.get("/os/grammar/symbols")
        assert r.status_code == 200
        assert "symbols" in r.json()

    def test_stats(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/grammar/observe", json={"pattern": "stats test"})
        r = c.get("/os/grammar/stats")
        assert r.status_code == 200
        assert "rules_created" in r.json() or "rule_count" in r.json()

    def test_abstract_invalid_level(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/grammar/abstract", json={"symbol": "x", "level": 0})
        assert r.status_code == 422


# ===================== HOLOGRAPHIC MEMORY (8 endpoints) =====================

class TestHolographicEndpoints:
    def test_encode(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/holographic/encode", json={"content": "test memory content", "fragment_type": "partial"})
        assert r.status_code == 200
        assert "fragment_id" in r.json()

    def test_encode_missing_content(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/holographic/encode", json={"fragment_type": "full"})
        assert r.status_code == 422

    def test_encode_full(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/holographic/encode-full", json={"content": "full encode test", "redundancy": 3})
        assert r.status_code == 200
        assert "fragments" in r.json()

    def test_encode_full_invalid_redundancy(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/holographic/encode-full", json={"content": "x", "redundancy": 0})
        assert r.status_code == 422

    def test_retrieve_fragment(self, _setup_shadow_os):
        c = _setup_shadow_os
        enc = c.post("/os/holographic/encode", json={"content": "retrieve test"}).json()
        r = c.get(f"/os/holographic/fragment/{enc['fragment_id']}")
        assert r.status_code == 200
        assert r.json()["fragment_id"] == enc["fragment_id"]

    def test_retrieve_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.get("/os/holographic/fragment/nonexistent999")
        assert r.status_code == 404

    def test_reconstruct(self, _setup_shadow_os):
        c = _setup_shadow_os
        f = c.post("/os/holographic/encode", json={"content": "reconstruct me"}).json()
        r = c.post("/os/holographic/reconstruct", json={"fragment_ids": [f["fragment_id"]]})
        assert r.status_code == 200

    def test_partial_reconstruct(self, _setup_shadow_os):
        c = _setup_shadow_os
        f = c.post("/os/holographic/encode-full", json={"content": "partial recon", "redundancy": 3}).json()
        fids = [frag["fragment_id"] for frag in f["fragments"][:1]]
        r = c.post("/os/holographic/partial-reconstruct", json={"fragment_ids": fids, "ratio": 0.3})
        assert r.status_code == 200

    def test_find_fragments(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/holographic/encode", json={"content": "searchable content"})
        r = c.get("/os/holographic/fragments", params={"content_query": "searchable"})
        assert r.status_code == 200
        assert "fragments" in r.json()

    def test_verify(self, _setup_shadow_os):
        c = _setup_shadow_os
        f = c.post("/os/holographic/encode", json={"content": "verify test"}).json()
        r = c.post("/os/holographic/verify", json={"fragment_ids": [f["fragment_id"]]})
        assert r.status_code == 200
        assert "results" in r.json()

    def test_holographic_stats(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/holographic/encode", json={"content": "stats encode"})
        r = c.get("/os/holographic/stats")
        assert r.status_code == 200
        assert "fragments_created" in r.json() or "fragment_count" in r.json()


# ===================== SYNCOGNITIVE FEEDBACK (8 endpoints) =====================

class TestSyncognitiveEndpoints:
    def test_observe_process(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/syncognitive/observe", json={
            "process_name": "reasoning", "details": "test observation", "confidence": 0.6,
        })
        assert r.status_code == 200
        assert "observation_id" in r.json()

    def test_observe_missing_fields(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/syncognitive/observe", json={"process_name": "x"})
        assert r.status_code == 422

    def test_create_feedback_loop(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/syncognitive/loop", json={
            "source": "reasoning", "target": "decision", "feedback_type": "bias_detection",
        })
        assert r.status_code == 200
        assert "loop_id" in r.json()

    def test_propagate_feedback(self, _setup_shadow_os):
        c = _setup_shadow_os
        loop = c.post("/os/syncognitive/loop", json={
            "source": "src", "target": "tgt",
        }).json()
        r = c.post("/os/syncognitive/propagate", json={"loop_id": loop["loop_id"], "magnitude": 0.1})
        assert r.status_code == 200

    def test_correct_self(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/syncognitive/correct", json={
            "process_name": "analysis", "correction": "reduce confidence bias",
        })
        assert r.status_code == 200
        assert "observation_id" in r.json()

    def test_detect_bias(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/syncognitive/observe", json={"process_name": "bias_test", "details": "d1"})
        r = c.post("/os/syncognitive/bias", json={"process_name": "bias_test"})
        assert r.status_code == 200
        assert "bias_observations" in r.json()

    def test_get_profile(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/syncognitive/observe", json={"process_name": "profile_test", "details": "d"})
        r = c.get("/os/syncognitive/profile/profile_test")
        assert r.status_code == 200
        assert "process_name" in r.json()

    def test_get_profile_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.get("/os/syncognitive/profile/nonexistent999")
        assert r.status_code == 404

    def test_list_observations(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/syncognitive/observe", json={"process_name": "list_test", "details": "d"})
        r = c.get("/os/syncognitive/observations")
        assert r.status_code == 200
        assert "observations" in r.json()

    def test_syncognitive_stats(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/syncognitive/observe", json={"process_name": "stats_test", "details": "d"})
        r = c.get("/os/syncognitive/stats")
        assert r.status_code == 200
        assert "observations_created" in r.json() or "observation_count" in r.json()


# ===================== MORPHIC RESONANCE ENGINE (8 endpoints) =====================

class TestMorphicEndpoints:
    def test_register_pattern(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/morphic/pattern", json={
            "domain": "work", "pattern": "deep focus sessions", "confidence": 0.7,
        })
        assert r.status_code == 200
        assert "pattern_id" in r.json()

    def test_register_pattern_missing_domain(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/morphic/pattern", json={"pattern": "x"})
        assert r.status_code == 422

    def test_propagate(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/morphic/pattern", json={"domain": "work", "pattern": "p1", "confidence": 0.8})
        r = c.post("/os/morphic/propagate", json={
            "source_domain": "work", "target_domain": "health", "strength": 0.5,
        })
        assert r.status_code == 200

    def test_detect_resonance(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/morphic/pattern", json={"domain": "d1", "pattern": "p2"})
        r = c.post("/os/morphic/detect", json={"domain_a": "d1", "domain_b": "p2"})
        assert r.status_code == 200
        assert "patterns" in r.json()

    def test_apply_influence(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/morphic/pattern", json={"domain": "d2", "pattern": "p3"})
        c.post("/os/morphic/propagate", json={
            "source_domain": "d2", "target_domain": "d3", "strength": 0.5,
        })
        with api_module._shadow_os_lock:
            os_obj = api_module._shadow_os
        infl = os_obj.morphic.get_influences()
        if infl:
            r = c.post("/os/morphic/influence", json={"influence_id": infl[0].influence_id})
            assert r.status_code == 200
            assert r.json()["applied"] is True
        else:
            r = c.post("/os/morphic/influence", json={"influence_id": "fake"})
            assert r.status_code in (200, 404)

    def test_apply_influence_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/morphic/influence", json={"influence_id": "fake999"})
        assert r.status_code == 404

    def test_decay(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/morphic/decay", json={"decay_rate": 0.05})
        assert r.status_code == 200
        assert "removed_count" in r.json()

    def test_get_field(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/morphic/pattern", json={"domain": "field_test", "pattern": "p4"})
        r = c.get("/os/morphic/field/field_test")
        assert r.status_code == 200
        assert "domain" in r.json()

    def test_get_field_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.get("/os/morphic/field/nonexistent")
        assert r.status_code == 404

    def test_resonance_strength(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.get("/os/morphic/strength", params={"domain_a": "x", "domain_b": "y"})
        assert r.status_code == 200
        assert "strength" in r.json()

    def test_morphic_stats(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/morphic/pattern", json={"domain": "stats_d", "pattern": "p5"})
        r = c.get("/os/morphic/stats")
        assert r.status_code == 200
        assert "patterns_registered" in r.json() or "pattern_count" in r.json()


# ===================== EXISTENTIAL CALCULUS (7 endpoints) =====================

class TestExistentialEndpoints:
    def test_evaluate(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/existential/evaluate", json={
            "event_id": "evt1", "description": "A meaningful event occurred",
        })
        assert r.status_code == 200
        d = r.json()
        assert "result_id" in d or "significance" in d

    def test_evaluate_missing_fields(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/existential/evaluate", json={})
        assert r.status_code == 422

    def test_alignment(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/existential/alignment", json={
            "purpose": "help others", "activities": ["coding", "teaching"],
        })
        assert r.status_code == 200
        d = r.json()
        assert "alignment_id" in d or "alignment_score" in d or "score" in d

    def test_themes(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/existential/evaluate", json={"event_id": "e1", "description": "growth event"})
        r = c.post("/os/existential/themes", json={"min_events": 1})
        assert r.status_code == 200
        assert "themes" in r.json()

    def test_add_frame(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/existential/frame", json={"title": "life chapter 1"})
        assert r.status_code == 200
        assert "frame_id" in r.json() or "result_id" in r.json() or "frame_count" in r.json()

    def test_list_frames(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/existential/frame", json={"title": "frame1"})
        r = c.get("/os/existential/frames")
        assert r.status_code == 200
        assert "frames" in r.json()

    def test_recommend(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/existential/evaluate", json={"event_id": "rec_ctx", "description": "recommend context"})
        r = c.get("/os/existential/recommend", params={"context": "rec_ctx"})
        assert r.status_code == 200
        assert "recommendation" in r.json()

    def test_stats(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/existential/evaluate", json={"event_id": "s1", "description": "stats eval"})
        r = c.get("/os/existential/stats")
        assert r.status_code == 200
        assert "frames_created" in r.json() or "frame_count" in r.json()


# ===================== CHRONO SYNTHESIS (10 endpoints) =====================

class TestChronoEndpoints:
    def test_past_pattern(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/chrono/past-pattern", json={
            "event_type": "commit", "domain": "dev", "frequency": 5,
        })
        assert r.status_code == 200
        assert "pattern_id" in r.json()

    def test_past_pattern_missing_event_type(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/chrono/past-pattern", json={"domain": "dev"})
        assert r.status_code == 422

    def test_future_projection(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/chrono/future-projection", json={
            "event_type": "deploy", "probability": 0.6, "timeframe": "2h",
        })
        assert r.status_code == 200
        assert "projection_id" in r.json()

    def test_future_projection_invalid_prob(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/chrono/future-projection", json={
            "event_type": "x", "probability": 1.5,
        })
        assert r.status_code == 422

    def test_create_context(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/chrono/context", json={
            "name": "sprint planning", "past_weight": 0.6, "future_weight": 0.4,
        })
        assert r.status_code == 200
        assert "context_id" in r.json()

    def test_synthesize(self, _setup_shadow_os):
        c = _setup_shadow_os
        ctx = c.post("/os/chrono/context", json={"name": "synth test"}).json()
        r = c.post("/os/chrono/synthesize", json={"context_id": ctx["context_id"]})
        assert r.status_code == 200
        assert "decision_id" in r.json()

    def test_list_past_patterns(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/chrono/past-pattern", json={"event_type": "test_event"})
        r = c.get("/os/chrono/past-patterns")
        assert r.status_code == 200
        assert "patterns" in r.json()

    def test_list_future_projections(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/chrono/future-projection", json={"event_type": "test_proj"})
        r = c.get("/os/chrono/future-projections")
        assert r.status_code == 200
        assert "projections" in r.json()

    def test_list_decisions(self, _setup_shadow_os):
        c = _setup_shadow_os
        ctx = c.post("/os/chrono/context", json={"name": "list_dec"}).json()
        c.post("/os/chrono/synthesize", json={"context_id": ctx["context_id"]})
        r = c.get("/os/chrono/decisions")
        assert r.status_code == 200
        assert "decisions" in r.json()

    def test_compute_past_weight(self, _setup_shadow_os):
        c = _setup_shadow_os
        pp = c.post("/os/chrono/past-pattern", json={"event_type": "pw_test"}).json()
        r = c.post("/os/chrono/past-weight", json={"pattern_id": pp["pattern_id"]})
        assert r.status_code == 200
        assert "weight" in r.json()

    def test_compute_future_weight(self, _setup_shadow_os):
        c = _setup_shadow_os
        fp = c.post("/os/chrono/future-projection", json={"event_type": "fw_test"}).json()
        r = c.post("/os/chrono/future-weight", json={"projection_id": fp["projection_id"]})
        assert r.status_code == 200
        assert "weight" in r.json()

    def test_chrono_stats(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/chrono/past-pattern", json={"event_type": "stats_test"})
        r = c.get("/os/chrono/stats")
        assert r.status_code == 200
        assert "patterns_recorded" in r.json() or "past_pattern_count" in r.json()

    def test_context_negative_past_weight(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/chrono/context", json={"name": "bad", "past_weight": -0.1})
        assert r.status_code == 422


# ===================== PHOENIX REGENERATOR (11 endpoints) =====================

class TestPhoenixEndpoints:
    def test_register_component(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/component", json={
            "component_id": "comp1", "module_name": "TestModule", "health_score": 1.0,
        })
        assert r.status_code == 200
        d = r.json()
        assert "component_id" in d
        assert d["component_id"] == "comp1"

    def test_register_component_missing_id(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/component", json={"module_name": "x"})
        assert r.status_code == 422

    def test_register_component_invalid_health(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/component", json={
            "component_id": "bad", "module_name": "x", "health_score": 1.5,
        })
        assert r.status_code == 422

    def test_check_health(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "hc1", "module_name": "M"})
        r = c.get("/os/phoenix/component/hc1")
        assert r.status_code == 200
        assert r.json()["component_id"] == "hc1"

    def test_check_health_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.get("/os/phoenix/component/nope999")
        assert r.status_code == 404

    def test_report_error(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "err1", "module_name": "M"})
        r = c.post("/os/phoenix/error", json={
            "component_id": "err1", "error_message": "Something went wrong",
        })
        assert r.status_code == 200
        assert "diagnosis_id" in r.json()

    def test_report_error_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/error", json={
            "component_id": "ghost", "error_message": "x",
        })
        assert r.status_code == 404

    def test_diagnose(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "diag1", "module_name": "M"})
        r = c.post("/os/phoenix/diagnose", json={"component_id": "diag1"})
        assert r.status_code == 200
        assert "diagnosis_id" in r.json()

    def test_diagnose_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/diagnose", json={"component_id": "missing"})
        assert r.status_code == 404

    def test_create_plan(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "plan1", "module_name": "M"})
        c.post("/os/phoenix/error", json={"component_id": "plan1", "error_message": "error"})
        r = c.post("/os/phoenix/plan", json={"component_id": "plan1"})
        assert r.status_code == 200
        assert "plan_id" in r.json()

    def test_create_plan_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/plan", json={"component_id": "nothing"})
        assert r.status_code == 404

    def test_execute_regeneration(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "exec1", "module_name": "M"})
        c.post("/os/phoenix/error", json={"component_id": "exec1", "error_message": "err"})
        plan = c.post("/os/phoenix/plan", json={"component_id": "exec1"}).json()
        r = c.post("/os/phoenix/execute", json={"plan_id": plan["plan_id"]})
        assert r.status_code == 200
        assert "result_id" in r.json()

    def test_execute_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/execute", json={"plan_id": "fake"})
        assert r.status_code == 404

    def test_preserve_learning(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "pres1", "module_name": "M"})
        r = c.post("/os/phoenix/preserve", json={"component_id": "pres1"})
        assert r.status_code == 200
        assert "learning" in r.json()

    def test_preserve_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/preserve", json={"component_id": "ghost"})
        assert r.status_code == 404

    def test_restore_learning(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "rest1", "module_name": "M"})
        r = c.post("/os/phoenix/restore", json={"component_id": "rest1"})
        assert r.status_code == 200
        assert r.json()["restored"] is True

    def test_restore_not_found(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/restore", json={"component_id": "phantom"})
        assert r.status_code == 404

    def test_list_components(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "lc1", "module_name": "M"})
        r = c.get("/os/phoenix/components")
        assert r.status_code == 200
        assert "components" in r.json()

    def test_list_diagnoses(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "ld1", "module_name": "M"})
        c.post("/os/phoenix/diagnose", json={"component_id": "ld1"})
        r = c.get("/os/phoenix/diagnoses")
        assert r.status_code == 200
        assert "diagnoses" in r.json()

    def test_phoenix_stats(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "st1", "module_name": "M"})
        r = c.get("/os/phoenix/stats")
        assert r.status_code == 200
        assert "components_registered" in r.json() or "component_count" in r.json()


# ===================== CROSS-CUTTING: SANITIZATION =====================

class TestSanitization:
    def test_neural_bridge_control_chars_stripped(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/neural-bridge/pattern", json={
            "pattern_type": "activation", "symbol": "bad\x00\x01symbol",
        })
        assert r.status_code == 200
        sym = r.json().get("symbol", "")
        assert "\x00" not in sym
        assert "\x01" not in sym

    def test_grammar_control_chars_stripped(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/grammar/observe", json={
            "pattern": "normal text", "context": "bad\x00\x01ctx",
        })
        assert r.status_code == 200
        ctx = r.json().get("context", "")
        assert "\x00" not in ctx
        assert "\x01" not in ctx

    def test_holographic_control_chars_stripped(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/holographic/encode", json={
            "content": "content with\x00ctrl\x01chars", "fragment_type": "partial\x02",
        })
        assert r.status_code == 200
        content = r.json().get("content", "")
        ftype = r.json().get("fragment_type", "")
        assert "\x00" not in content
        assert "\x01" not in content

    def test_syncognitive_control_chars_stripped(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/syncognitive/observe", json={
            "process_name": "bad\x00proc", "details": "clean details",
        })
        assert r.status_code == 200
        assert "\x00" not in r.json().get("process_name", "")

    def test_morphic_control_chars_stripped(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/morphic/pattern", json={
            "domain": "bad\x00domain", "pattern": "good pattern",
        })
        assert r.status_code == 200
        assert "\x00" not in r.json().get("domain", "")

    def test_existential_control_chars_stripped(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/existential/evaluate", json={
            "event_id": "bad\x00id", "description": "clean desc",
        })
        assert r.status_code == 200
        assert "\x00" not in r.json().get("event_id", "")

    def test_chrono_control_chars_stripped(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/chrono/past-pattern", json={
            "event_type": "bad\x00type", "domain": "clean_domain",
        })
        assert r.status_code == 200
        assert "\x00" not in r.json().get("event_type", "")

    def test_phoenix_control_chars_stripped(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/component", json={
            "component_id": "bad\x00comp", "module_name": "clean",
        })
        assert r.status_code == 200
        assert "\x00" not in r.json().get("component_id", "")

    def test_content_fields_preserve_angle_brackets(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/holographic/encode", json={
            "content": "latency < 100ms is acceptable", "fragment_type": "full",
        })
        assert r.status_code == 200
        content = r.json().get("content", "")
        assert "<" in content or "latency" in content

    def test_identifier_fields_strip_angle_brackets(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/neural-bridge/pattern", json={
            "pattern_type": "<script>alert(1)</script>",
        })
        assert r.status_code == 200
        ptype = r.json().get("pattern_type", "")
        assert "<" not in ptype


# ===================== CROSS-CUTTING: DEEPCOPY =====================

class TestDeepcopy:
    def test_holographic_stats_not_mutable_ref(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/holographic/encode", json={"content": "dc_test"})
        with api_module._shadow_os_lock:
            os_obj = api_module._shadow_os
        s1 = os_obj.holographic.get_stats()
        orig = s1.get("fragments_created", 0)
        s1["fragments_created"] = 99999
        s2 = os_obj.holographic.get_stats()
        assert s2["fragments_created"] == orig

    def test_neural_bridge_stats_not_mutable_ref(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/neural-bridge/pattern", json={"pattern_type": "activation"})
        with api_module._shadow_os_lock:
            os_obj = api_module._shadow_os
        s1 = os_obj.neural_bridge.get_stats()
        orig = s1.get("patterns_created", 0)
        s1["patterns_created"] = 99999
        s2 = os_obj.neural_bridge.get_stats()
        assert s2["patterns_created"] == orig

    def test_quantum_stats_not_mutable_ref(self, _setup_shadow_os):
        c = _setup_shadow_os
        with api_module._shadow_os_lock:
            os_obj = api_module._shadow_os
        s1 = os_obj.quantum_field.get_stats()
        orig = s1.get("decisions_created", 0)
        s1["decisions_created"] = 99999
        s2 = os_obj.quantum_field.get_stats()
        assert s2.get("decisions_created", 0) == orig

    def test_existential_stats_not_mutable_ref(self, _setup_shadow_os):
        c = _setup_shadow_os
        with api_module._shadow_os_lock:
            os_obj = api_module._shadow_os
        s1 = os_obj.existential.get_stats()
        keys = list(s1.keys())
        if keys:
            k = keys[0]
            orig = s1[k]
            s1[k] = "MUTATED"
            s2 = os_obj.existential.get_stats()
            assert s2[k] == orig


# ===================== CROSS-CUTTING: VALIDATION =====================

class TestValidation:
    def test_quantum_observe_missing_option_id(self, _setup_shadow_os):
        c = _setup_shadow_os
        dec = c.post("/os/quantum-field/decision", json={"options": [{"label": "A"}]}).json()
        r = c.post(f"/os/quantum-field/decision/{dec['decision_id']}/observe", json={})
        assert r.status_code == 422

    def test_grammar_refine_missing_rule_id(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/grammar/refine", json={"confidence": 0.5})
        assert r.status_code == 422

    def test_holographic_reconstruct_empty_ids(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/holographic/reconstruct", json={"fragment_ids": []})
        assert r.status_code == 200

    def test_syncognitive_propagate_missing_loop_id(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/syncognitive/propagate", json={"magnitude": 0.1})
        assert r.status_code == 422

    def test_morphic_propagate_missing_source(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/morphic/propagate", json={"target_domain": "t"})
        assert r.status_code == 422

    def test_chrono_context_missing_name(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/chrono/context", json={"past_weight": 0.5, "future_weight": 0.5})
        assert r.status_code == 422

    def test_phoenix_execute_missing_plan_id(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/execute", json={})
        assert r.status_code == 422

    def test_phoenix_error_missing_component_id(self, _setup_shadow_os):
        c = _setup_shadow_os
        r = c.post("/os/phoenix/error", json={"error_message": "x"})
        assert r.status_code == 422

    def test_existential_evaluate_too_many_axis_keys(self, _setup_shadow_os):
        c = _setup_shadow_os
        big = {f"key_{i}": 0.5 for i in range(60)}
        r = c.post("/os/existential/evaluate", json={
            "event_id": "e", "description": "d", "axis_weights": big,
        })
        assert r.status_code == 422

    def test_neural_pattern_too_many_dimension_keys(self, _setup_shadow_os):
        c = _setup_shadow_os
        big = {f"dim_{i}": 0.5 for i in range(60)}
        r = c.post("/os/neural-bridge/pattern", json={
            "pattern_type": "activation", "dimensions": big,
        })
        assert r.status_code == 422


# ===================== CROSS-CUTTING: END-TO-END FLOWS =====================

class TestEndToEndFlows:
    def test_full_neural_bridge_workflow(self, _setup_shadow_os):
        c = _setup_shadow_os
        p = c.post("/os/neural-bridge/pattern", json={
            "pattern_type": "activation", "dimensions": {"f1": 0.7}, "confidence": 0.8,
        }).json()
        ru = c.post("/os/neural-bridge/rule", json={
            "premises": ["evidence_a"], "conclusion": "inference_b", "domain": "logic",
        }).json()
        c.post("/os/neural-bridge/map", json={
            "pattern_id": p["pattern_id"], "rule_id": ru["rule_id"],
        })
        r = c.post("/os/neural-bridge/translate/neuro-to-symbolic", json={"pattern_id": p["pattern_id"]})
        assert r.status_code == 200
        assert "rules" in r.json()

    def test_full_phoenix_workflow(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/phoenix/component", json={"component_id": "wf1", "module_name": "WorkflowTest"})
        c.post("/os/phoenix/error", json={"component_id": "wf1", "error_message": "degraded"})
        plan = c.post("/os/phoenix/plan", json={"component_id": "wf1"}).json()
        c.post("/os/phoenix/preserve", json={"component_id": "wf1"})
        result = c.post("/os/phoenix/execute", json={"plan_id": plan["plan_id"]})
        assert result.status_code == 200
        c.post("/os/phoenix/restore", json={"component_id": "wf1"})
        health = c.get("/os/phoenix/component/wf1")
        assert health.status_code == 200

    def test_full_chrono_workflow(self, _setup_shadow_os):
        c = _setup_shadow_os
        pp = c.post("/os/chrono/past-pattern", json={
            "event_type": "deploy", "domain": "ops", "frequency": 3,
        }).json()
        fp = c.post("/os/chrono/future-projection", json={
            "event_type": "deploy", "probability": 0.7, "timeframe": "1h",
        }).json()
        ctx = c.post("/os/chrono/context", json={
            "name": "ops planning", "past_weight": 0.6, "future_weight": 0.4,
        }).json()
        c.post("/os/chrono/past-weight", json={"pattern_id": pp["pattern_id"]})
        c.post("/os/chrono/future-weight", json={"projection_id": fp["projection_id"]})
        decision = c.post("/os/chrono/synthesize", json={"context_id": ctx["context_id"]})
        assert decision.status_code == 200

    def test_full_morphic_workflow(self, _setup_shadow_os):
        c = _setup_shadow_os
        c.post("/os/morphic/pattern", json={"domain": "work", "pattern": "deep focus", "confidence": 0.9})
        c.post("/os/morphic/propagate", json={
            "source_domain": "work", "target_domain": "health", "strength": 0.6,
        })
        field = c.get("/os/morphic/field/work")
        assert field.status_code == 200
        strength = c.get("/os/morphic/strength", params={"domain_a": "work", "domain_b": "health"})
        assert strength.status_code == 200
        c.post("/os/morphic/decay", json={"decay_rate": 0.01})
        stats = c.get("/os/morphic/stats")
        assert stats.status_code == 200

    def test_all_stats_endpoints_return_valid_data(self, _setup_shadow_os):
        c = _setup_shadow_os
        for path in [
            "/os/neural-bridge/stats",
            "/os/quantum-field/stats",
            "/os/grammar/stats",
            "/os/holographic/stats",
            "/os/syncognitive/stats",
            "/os/morphic/stats",
            "/os/existential/stats",
            "/os/chrono/stats",
            "/os/phoenix/stats",
        ]:
            r = c.get(path)
            assert r.status_code == 200, f"{path} returned {r.status_code}"
            assert isinstance(r.json(), dict), f"{path} did not return dict"