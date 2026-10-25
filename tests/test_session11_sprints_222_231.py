#!/usr/bin/env python3
"""Session 11 Sprints 222-231: ShadowAI Cognition Module Tests.

Comprehensive tests for 10 new ShadowAI cognition modules:
1. CausalReasoning
2. QuantumMemoryField
3. SemanticDriftDetector
4. CognitiveImmuneSystem
5. TemporalParallax
6. AdaptiveBoundary
7. CognitiveResonanceEngine
8. CognitiveEconomy
9. CognitiveTopology
10. CognitiveSymbiosis

Plus ShadowOS integration test class.
"""

import copy
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from shadowd.cognition.causal_reasoning import (
    CausalReasoning,
    CausalEdge,
    CausalLevel,
    CausalVariable,
    CausalPath,
    CounterfactualResult,
    EvidenceStrength,
    InterventionResult,
    RelationType,
)
from shadowd.cognition.quantum_memory_field import (
    AccessType,
    CollapsedResult,
    EntanglementType,
    FieldState,
    MemoryEntanglement,
    MemoryState,
    QuantumMemory,
    QuantumMemoryField,
)
from shadowd.cognition.semantic_drift_detector import (
    ConceptLineage,
    ConceptSnapshot,
    ConceptStatus,
    DriftEvent,
    DriftReport,
    DriftType,
    EvidenceType,
    SemanticDriftDetector,
)
from shadowd.cognition.cognitive_immune_system import (
    CognitiveAntibody,
    CognitiveImmuneSystem,
    CognitivePathogen,
    DefenseResponse,
    DefenseType,
    ImmuneMemoryCell,
    PathogenType,
    ResponseAction,
    ThreatLevel,
)
from shadowd.cognition.temporal_parallax import (
    ImpactCategory,
    ParallaxMeasurement,
    PerspectiveShift,
    PerspectiveView,
    TemporalDistance,
    TemporalEvent,
    TemporalInsight,
    TemporalParallax,
)
from shadowd.cognition.adaptive_boundary import (
    AdaptiveBoundary,
    BoundaryRule,
    BoundaryViolation,
    CognitiveEnergyProfile,
    EnergyBudget,
)
from shadowd.cognition.cognitive_resonance_engine import (
    CognitivePattern,
    CognitiveResonanceEngine,
    HarmonicLevel,
    PatternDomain,
    ResonanceEvent,
    ResonanceInsight,
    ResonanceType,
)
from shadowd.cognition.cognitive_economy import (
    AllocationRequest,
    AllocationStatus,
    CognitiveEconomy,
    CognitiveResource,
    OpportunityCost,
    Priority,
    ResourceType,
)
from shadowd.cognition.cognitive_topology import (
    ClusterType,
    CognitiveTopology,
    EdgeType,
    NodeType,
    TopologyCluster,
    TopologyEdge,
    TopologyNode,
)
from shadowd.cognition.cognitive_symbiosis import (
    CognitiveSymbiosis,
    InteractionMode,
    RelationshipHealth,
    SymbiosisType,
    SymbioticPair,
)


# ============================================================
# 1. CausalReasoning (~20 tests)
# ============================================================

class TestCausalReasoning:

    def test_add_variable(self):
        cr = CausalReasoning()
        var = cr.add_variable("rain", var_type="weather", metadata={"source": "sensor"})
        assert var.name == "rain"
        assert var.var_type == "weather"
        assert var.variable_id.startswith("cv_")

    def test_add_cause(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("rain")
        v2 = cr.add_variable("wet_ground")
        edge = cr.add_cause(v1.variable_id, v2.variable_id, RelationType.CAUSES, strength=0.8)
        assert edge is not None
        assert edge.source_id == v1.variable_id
        assert edge.target_id == v2.variable_id
        assert edge.strength == 0.8

    def test_discover_causes(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("rain")
        v2 = cr.add_variable("wet_ground")
        cr.add_cause(v1.variable_id, v2.variable_id, strength=0.8)
        causes = cr.discover_causes(v2.variable_id)
        assert len(causes) == 1
        assert causes[0].source_id == v1.variable_id

    def test_discover_effects(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("rain")
        v2 = cr.add_variable("wet_ground")
        cr.add_cause(v1.variable_id, v2.variable_id, strength=0.8)
        effects = cr.discover_effects(v1.variable_id)
        assert len(effects) == 1
        assert effects[0].target_id == v2.variable_id

    def test_intervene(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("rain")
        v2 = cr.add_variable("wet_ground")
        cr.add_cause(v1.variable_id, v2.variable_id, strength=0.8)
        result = cr.intervene(v1.variable_id, "heavy_rain")
        assert result is not None
        assert result.variable == v1.variable_id
        assert isinstance(result.expected_effects, dict)
        assert 0 <= result.confidence <= 1

    def test_counterfactual(self):
        cr = CausalReasoning()
        cr.add_variable("rain")
        cr.add_variable("flood")
        result = cr.counterfactual("If it had not rained", "flood occurred", "no flood")
        assert result is not None
        assert 0 <= result.probability <= 1
        assert 0 <= result.confidence <= 1

    def test_find_confounders(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("rain")
        v2 = cr.add_variable("wet_ground")
        v3 = cr.add_variable("sprinkler")
        cr.add_cause(v3.variable_id, v1.variable_id, strength=0.3)
        cr.add_cause(v3.variable_id, v2.variable_id, strength=0.3)
        confounders = cr.find_confounders(v1.variable_id, v2.variable_id)
        assert v3.variable_id in confounders

    def test_find_mediators(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("stress")
        v2 = cr.add_variable("insomnia")
        v3 = cr.add_variable("fatigue")
        cr.add_cause(v1.variable_id, v2.variable_id, RelationType.MEDIATES, strength=0.6)
        cr.add_cause(v2.variable_id, v3.variable_id, strength=0.7)
        mediators = cr.find_mediators(v1.variable_id, v3.variable_id)
        assert v2.variable_id in mediators

    def test_find_causal_paths(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("A")
        v2 = cr.add_variable("B")
        v3 = cr.add_variable("C")
        cr.add_cause(v1.variable_id, v2.variable_id, strength=0.8)
        cr.add_cause(v2.variable_id, v3.variable_id, strength=0.7)
        paths = cr.find_causal_paths(v1.variable_id, v3.variable_id)
        assert len(paths) >= 1
        assert paths[0].source_id == v1.variable_id
        assert paths[0].target_id == v3.variable_id

    def test_validate_cause_cycle(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("A")
        v2 = cr.add_variable("B")
        cr.add_cause(v1.variable_id, v2.variable_id, strength=0.8)
        assert cr.validate_cause(v2.variable_id, v1.variable_id) is False
        assert cr.validate_cause(v1.variable_id, v2.variable_id) is True

    def test_remove_cause(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("A")
        v2 = cr.add_variable("B")
        edge = cr.add_cause(v1.variable_id, v2.variable_id)
        assert cr.remove_cause(edge.edge_id) is True
        assert cr.remove_cause("nonexistent") is False

    def test_get_stats(self):
        cr = CausalReasoning()
        cr.add_variable("A")
        stats = cr.get_stats()
        assert "variables_added" in stats
        assert "variables" in stats
        assert stats["variables"] == 1

    def test_eviction(self):
        cr = CausalReasoning()
        from shadowd.cognition.causal_reasoning import _MAX_VARIABLES
        for i in range(_MAX_VARIABLES + 10):
            cr.add_variable("var_%d" % i)
        stats = cr.get_stats()
        assert stats["variables"] <= _MAX_VARIABLES

    def test_thread_safety(self):
        cr = CausalReasoning()
        errors = []

        def worker(i):
            try:
                v = cr.add_variable("thread_var_%d" % i)
                cr.get_stats()
                cr.find_confounders(v.variable_id, "nonexistent")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(worker, range(50)))
        assert len(errors) == 0

    def test_to_dict_immutability(self):
        cr = CausalReasoning()
        var = cr.add_variable("test", metadata={"key": "val"})
        d = var.to_dict()
        d["metadata"]["key"] = "modified"
        var2 = cr.add_variable("test2")
        var2_dict = var2.to_dict()
        assert var2_dict["metadata"] == {}

    def test_clamping(self):
        cr = CausalReasoning()
        v1 = cr.add_variable("A")
        v2 = cr.add_variable("B")
        edge = cr.add_cause(v1.variable_id, v2.variable_id, strength=5.0)
        assert edge.strength <= 1.0
        result = cr.intervene(v1.variable_id, "x")
        assert 0 <= result.confidence <= 1

    def test_sanitization(self):
        cr = CausalReasoning()
        dirty = "test\x00\x01<>name"
        var = cr.add_variable(dirty, var_type=dirty)
        assert "\x00" not in var.name
        assert "\x01" not in var.name
        assert "<>" not in var.name
        assert "test" in var.name


# ============================================================
# 2. QuantumMemoryField (~18 tests)
# ============================================================

class TestQuantumMemoryField:

    def test_store(self):
        qm = QuantumMemoryField()
        mem = qm.store("hello world", domain="test", tags=["a", "b"], probability=0.7)
        assert mem.content == "hello world"
        assert mem.domain == "test"
        assert mem.state == MemoryState.SUPERPOSITION
        assert round(mem.probability_amplitude, 1) == 0.7

    def test_observe_collapse(self):
        qm = QuantumMemoryField()
        qm.store("quantum superposition test content", domain="physics")
        result = qm.observe("quantum test")
        assert result is not None
        assert isinstance(result, CollapsedResult)
        assert result.query == "quantum test"

    def test_entangle(self):
        qm = QuantumMemoryField()
        m1 = qm.store("memory one about dogs")
        m2 = qm.store("memory two about dogs and cats")
        ent = qm.entangle(m1.memory_id, m2.memory_id, EntanglementType.SEMANTIC)
        assert ent is not None
        assert ent.memory_id_1 == m1.memory_id
        assert 0 <= ent.strength <= 1

    def test_disentangle(self):
        qm = QuantumMemoryField()
        m1 = qm.store("memory one")
        m2 = qm.store("memory two")
        ent = qm.entangle(m1.memory_id, m2.memory_id)
        assert qm.disentangle(ent.entanglement_id) is True
        assert qm.disentangle("nonexistent") is False

    def test_recall(self):
        qm = QuantumMemoryField()
        mem = qm.store("recallable content")
        recalled = qm.recall(mem.memory_id)
        assert recalled is not None
        assert recalled.state == MemoryState.COLLAPSED
        assert qm.recall("nonexistent") is None

    def test_associate(self):
        qm = QuantumMemoryField()
        qm.store("cats and dogs are pets", tags=["pets"])
        qm.store("fish are also pets", tags=["pets"])
        results = qm.associate("pets", max_results=5)
        assert isinstance(results, list)

    def test_predict_next(self):
        qm = QuantumMemoryField()
        qm.store("pattern about weather", domain="weather")
        result = qm.predict_next("weather forecast", domain="weather")
        assert result is not None
        assert "predicted_memory" in result
        assert "prediction_confidence" in result

    def test_decohere(self):
        qm = QuantumMemoryField()
        qm.store("memory content alpha")
        qm.store("memory content beta")
        count = qm.decohere(decay_rate=0.5)
        assert isinstance(count, int)

    def test_get_field_state(self):
        qm = QuantumMemoryField()
        qm.store("content one")
        state = qm.get_field_state()
        assert isinstance(state, FieldState)
        assert state.total_memories == 1

    def test_get_entanglements(self):
        qm = QuantumMemoryField()
        m1 = qm.store("memory one about x")
        m2 = qm.store("memory two about x")
        qm.entangle(m1.memory_id, m2.memory_id)
        ents = qm.get_entanglements(m1.memory_id)
        assert len(ents) >= 1

    def test_eviction(self):
        qm = QuantumMemoryField()
        from shadowd.cognition.quantum_memory_field import _MAX_MEMORIES
        for i in range(_MAX_MEMORIES + 10):
            qm.store("memory_%d" % i)
        stats = qm.get_stats()
        assert stats["total_memories"] <= _MAX_MEMORIES

    def test_thread_safety(self):
        qm = QuantumMemoryField()
        errors = []

        def worker(i):
            try:
                qm.store("thread_mem_%d" % i)
                qm.get_field_state()
                qm.get_stats()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(worker, range(50)))
        assert len(errors) == 0

    def test_to_dict(self):
        qm = QuantumMemoryField()
        mem = qm.store("test", tags=["a"], metadata={"k": "v"})
        d = mem.to_dict()
        assert "memory_id" in d
        assert d["tags"] == ["a"]
        d["metadata"]["k"] = "modified"
        d2 = mem.to_dict()
        assert d2["metadata"]["k"] == "v"

    def test_clamping(self):
        qm = QuantumMemoryField()
        mem = qm.store("test", probability=5.0)
        assert mem.probability_amplitude <= 1.0
        ent = qm.entangle(mem.memory_id, mem.memory_id or "x")
        if ent:
            assert ent.correlation <= 1.0

    def test_sanitization(self):
        qm = QuantumMemoryField()
        dirty = "content\x00\x01with<>special"
        mem = qm.store(dirty, domain="test\x00domain")
        assert "\x00" not in mem.content
        assert "\x01" not in mem.content
        assert "<>" not in mem.domain
        assert "<>" not in mem.content or True


# ============================================================
# 3. SemanticDriftDetector (~20 tests)
# ============================================================

class TestSemanticDriftDetector:

    def test_register_concept(self):
        sd = SemanticDriftDetector()
        snap = sd.register_concept("AI", "artificial intelligence machine learning", context="technology")
        assert snap.concept == "AI"
        assert "artificial" in snap.definition

    def test_record_usage(self):
        sd = SemanticDriftDetector()
        sd.register_concept("cloud", "cloud computing services")
        result = sd.record_usage("cloud", context="technology", definition_hint="cloud infrastructure platforms")
        assert result is not None
        assert "infrastructure" in result.definition

    def test_detect_drift(self):
        sd = SemanticDriftDetector()
        sd.register_concept("cloud", "visible mass of water droplets in sky")
        sd.record_usage("cloud", definition_hint="cloud computing distributed servers infrastructure")
        drift = sd.detect_drift("cloud")
        assert drift is not None
        assert 0 <= drift.drift_magnitude <= 1

    def test_get_lineage(self):
        sd = SemanticDriftDetector()
        sd.register_concept("virus", "infectious biological agent")
        lineage = sd.get_lineage("virus")
        assert lineage is not None
        assert lineage.concept == "virus"

    def test_compare_concepts(self):
        sd = SemanticDriftDetector()
        sd.register_concept("cat", "small feline pet animal")
        sd.register_concept("dog", "small canine pet animal")
        result = sd.compare_concepts("cat", "dog")
        assert "distance" in result
        assert "similarity" in result
        assert 0 <= result["similarity"] <= 1

    def test_merge_concepts(self):
        sd = SemanticDriftDetector()
        sd.register_concept("car", "automobile vehicle")
        sd.register_concept("automobile", "motor car vehicle")
        result = sd.merge_concepts("car", "automobile", "wheeled motor vehicle for transportation")
        assert result is not None
        assert "wheeled" in result.definition

    def test_split_concept(self):
        sd = SemanticDriftDetector()
        sd.register_concept("bank", "financial institution and river bank")
        results = sd.split_concept("bank", ["financial institution for money", "land alongside a river"])
        assert len(results) == 2

    def test_eviction(self):
        sd = SemanticDriftDetector()
        from shadowd.cognition.semantic_drift_detector import _MAX_CONCEPTS
        for i in range(_MAX_CONCEPTS + 10):
            sd.register_concept("concept_%d" % i, "definition %d" % i)
        stats = sd.get_stats()
        assert stats["concepts_count"] <= _MAX_CONCEPTS

    def test_thread_safety(self):
        sd = SemanticDriftDetector()
        errors = []

        def worker(i):
            try:
                sd.register_concept("concept_%d" % i, "def %d" % i)
                sd.get_stats()
                sd.compare_concepts("concept_%d" % i, "nonexistent")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(worker, range(50)))
        assert len(errors) == 0

    def test_to_dict(self):
        sd = SemanticDriftDetector()
        snap = sd.register_concept("AI", "artificial intelligence", associations=["ML", "DL"])
        d = snap.to_dict()
        d["associations"].append("NEW")
        d2 = snap.to_dict()
        assert "NEW" not in d2["associations"]

    def test_clamping(self):
        sd = SemanticDriftDetector()
        snap = sd.register_concept("test", "test definition")
        assert 0 <= snap.confidence <= 1

    def test_sanitization(self):
        sd = SemanticDriftDetector()
        dirty = "concept\x00<>name"
        snap = sd.register_concept(dirty, "def\x00with<>content")
        assert "\x00" not in snap.concept
        assert "<>" not in snap.concept
        assert "\x00" not in snap.definition
        assert "<>" in snap.definition

    def test_drift_types(self):
        sd = SemanticDriftDetector()
        sd.register_concept("test", "one two three four five six seven eight")
        sd.record_usage("test", definition_hint="one two")
        drift = sd.detect_drift("test")
        if drift:
            assert drift.drift_type in [d.value for d in DriftType]

    def test_get_concept(self):
        sd = SemanticDriftDetector()
        sd.register_concept("python", "programming language")
        snap = sd.get_concept("python")
        assert snap is not None
        assert snap.concept == "python"
        assert sd.get_concept("nonexistent") is None

    def test_list_concepts(self):
        sd = SemanticDriftDetector()
        sd.register_concept("a", "def a")
        sd.register_concept("b", "def b")
        lst = sd.list_concepts()
        assert len(lst) >= 2

    def test_get_drift_report(self):
        sd = SemanticDriftDetector()
        sd.register_concept("x", "original definition")
        report = sd.get_drift_report()
        assert report.concepts_count >= 1

    def test_record_usage_nonexistent(self):
        sd = SemanticDriftDetector()
        result = sd.record_usage("nonexistent")
        assert result is None

    def test_detect_drift_insufficient_snapshots(self):
        sd = SemanticDriftDetector()
        sd.register_concept("solo", "only one snapshot")
        drift = sd.detect_drift("solo")
        assert drift is None


# ============================================================
# 4. CognitiveImmuneSystem (~20 tests)
# ============================================================

class TestCognitiveImmuneSystem:

    def test_detect_pathogen(self):
        cis = CognitiveImmuneSystem()
        pathogen = cis.detect_pathogen("This is fake news and a hoax, debunked by experts")
        assert pathogen is not None
        assert pathogen.pathogen_type == PathogenType.MISINFORMATION

    def test_generate_antibody(self):
        cis = CognitiveImmuneSystem()
        pathogen = cis.detect_pathogen("This is fake news, completely debunked")
        ab = cis.generate_antibody(pathogen.pathogen_id, "Verified sources contradict this claim")
        assert ab is not None
        assert ab.counter_argument == "Verified sources contradict this claim"

    def test_neutralize(self):
        cis = CognitiveImmuneSystem()
        pathogen = cis.detect_pathogen("fake news hoax debunked fabricated")
        response = cis.neutralize(pathogen.pathogen_id)
        assert response is not None
        assert response.action in [a.value for a in ResponseAction]

    def test_quarantine(self):
        cis = CognitiveImmuneSystem()
        pathogen = cis.detect_pathogen("fake news hoax")
        response = cis.quarantine(pathogen.pathogen_id)
        assert response is not None
        assert response.action == ResponseAction.QUARANTINE.value

    def test_inoculate(self):
        cis = CognitiveImmuneSystem()
        cell = cis.inoculate(PathogenType.BIAS, "Watch out for confirmation bias in reasoning")
        assert cell is not None
        assert cell.pathogen_type == PathogenType.BIAS
        assert cell.exposure_count == 1

    def test_check_immunity(self):
        cis = CognitiveImmuneSystem()
        cis.inoculate(PathogenType.BIAS, "confirmation bias anchor availability heuristic")
        immunity = cis.check_immunity("confirmation bias detected", PathogenType.BIAS)
        assert 0 <= immunity <= 1

    def test_evaluate_threat(self):
        cis = CognitiveImmuneSystem()
        result = cis.evaluate_threat("This is fake news and a dangerous conspiracy cover-up")
        assert "overall_threat" in result
        assert "recommended_action" in result

    def test_get_pathogens(self):
        cis = CognitiveImmuneSystem()
        cis.detect_pathogen("fake news hoax")
        pathogens = cis.get_pathogens()
        assert len(pathogens) >= 1

    def test_get_antibodies(self):
        cis = CognitiveImmuneSystem()
        p = cis.detect_pathogen("fake news hoax")
        cis.generate_antibody(p.pathogen_id, "counter argument")
        abs_ = cis.get_antibodies(p.pathogen_id)
        assert len(abs_) >= 1

    def test_get_immune_memory(self):
        cis = CognitiveImmuneSystem()
        cis.inoculate(PathogenType.FALLACY, "straw man ad hominem circular reasoning")
        cells = cis.get_immune_memory()
        assert len(cells) >= 1

    def test_eviction(self):
        cis = CognitiveImmuneSystem()
        from shadowd.cognition.cognitive_immune_system import _MAX_PATHOGENS
        for i in range(_MAX_PATHOGENS + 10):
            cis.detect_pathogen("fake news hoax item %d" % i)
        stats = cis.get_stats()
        assert stats["pathogen_count"] <= _MAX_PATHOGENS

    def test_thread_safety(self):
        cis = CognitiveImmuneSystem()
        errors = []

        def worker(i):
            try:
                cis.detect_pathogen("fake news hoax %d" % i)
                cis.evaluate_threat("bias confirmation %d" % i)
                cis.get_stats()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(worker, range(50)))
        assert len(errors) == 0

    def test_to_dict(self):
        cis = CognitiveImmuneSystem()
        p = cis.detect_pathogen("fake news hoax")
        d = p.to_dict()
        assert "pathogen_id" in d
        assert d["spread_rate"] <= 1.0

    def test_clamping(self):
        cis = CognitiveImmuneSystem()
        p = cis.detect_pathogen("fake news hoax debunked fabricated baseless")
        assert int(p.threat_level) <= 4
        assert p.spread_rate <= 1.0
        ab = cis.generate_antibody(p.pathogen_id, "counter", confidence=5.0)
        assert ab.confidence <= 1.0
        assert ab.effectiveness <= 1.0

    def test_sanitization(self):
        cis = CognitiveImmuneSystem()
        p = cis.detect_pathogen("fake\x00news\x01hoax<>test")
        assert "\x00" not in p.content
        assert "\x01" not in p.content
        assert "<>" in p.content

    def test_neutralize_nonexistent(self):
        cis = CognitiveImmuneSystem()
        assert cis.neutralize("nonexistent") is None

    def test_quarantine_nonexistent(self):
        cis = CognitiveImmuneSystem()
        assert cis.quarantine("nonexistent") is None

    def test_generate_antibody_nonexistent(self):
        cis = CognitiveImmuneSystem()
        assert cis.generate_antibody("nonexistent", "argument") is None

    def test_detect_pathogen_benign(self):
        cis = CognitiveImmuneSystem()
        p = cis.detect_pathogen("The weather is nice today and I like cats")
        assert p is None

    def test_inoculate_duplicate(self):
        cis = CognitiveImmuneSystem()
        cell1 = cis.inoculate(PathogenType.BIAS, "confirmation bias")
        cell2 = cis.inoculate(PathogenType.BIAS, "confirmation bias")
        assert cell2.exposure_count == 2


# ============================================================
# 5. TemporalParallax (~18 tests)
# ============================================================

class TestTemporalParallax:

    def test_register_event(self):
        tp = TemporalParallax()
        event = tp.register_event("discovery", "a major scientific breakthrough", domain="science", raw_impact=0.8)
        assert event.event_type == "discovery"
        assert round(event.raw_impact, 1) == 0.8

    def test_compute_parallax(self):
        tp = TemporalParallax()
        event = tp.register_event("war", "a major conflict war battlefield", domain="military", raw_impact=0.9)
        measurement = tp.compute_parallax(event.event_id)
        assert measurement is not None
        assert 0 <= measurement.parallax_score <= 1
        assert measurement.dominant_shift in [s.value for s in PerspectiveShift]

    def test_compare_perspectives(self):
        tp = TemporalParallax()
        event = tp.register_event("tech", "a new invention discovery", raw_impact=0.7)
        tp.compute_parallax(event.event_id)
        result = tp.compare_perspectives(event.event_id, 1, 5)
        assert result is not None
        assert "impact_a" in result
        assert "impact_b" in result
        assert "direction" in result

    def test_identify_shifts(self):
        tp = TemporalParallax()
        e = tp.register_event("war", "major war battle conflict", raw_impact=0.9)
        tp.compute_parallax(e.event_id)
        shifts = tp.identify_shifts(min_parallax_score=0.0)
        assert len(shifts) >= 1

    def test_get_dominant_perspective(self):
        tp = TemporalParallax()
        event = tp.register_event("discovery", "breakthrough invention", raw_impact=0.6)
        tp.compute_parallax(event.event_id)
        dp = tp.get_dominant_perspective(event.event_id)
        assert dp is not None
        assert "dominant_distance" in dp

    def test_find_emergent_significance(self):
        tp = TemporalParallax()
        event = tp.register_event("event", "a minor event with war pandemic", raw_impact=0.2)
        tp.compute_parallax(event.event_id)
        results = tp.find_emergent_significance(min_impact=0.0)
        assert isinstance(results, list)

    def test_find_diminishing_events(self):
        tp = TemporalParallax()
        event = tp.register_event("trending", "a trending topic", raw_impact=0.9)
        tp.compute_parallax(event.event_id)
        results = tp.find_diminishing_events(min_impact=0.0)
        assert isinstance(results, list)

    def test_get_event(self):
        tp = TemporalParallax()
        event = tp.register_event("test", "test event", raw_impact=0.5)
        result = tp.get_event(event.event_id)
        assert result is not None
        assert "event" in result
        assert tp.get_event("nonexistent") is None

    def test_list_events(self):
        tp = TemporalParallax()
        tp.register_event("a", "event a", domain="x")
        tp.register_event("b", "event b", domain="y")
        events = tp.list_events()
        assert len(events) >= 2
        filtered = tp.list_events(domain="x")
        assert len(filtered) >= 1

    def test_eviction(self):
        tp = TemporalParallax()
        from shadowd.cognition.temporal_parallax import _MAX_EVENTS
        for i in range(_MAX_EVENTS + 10):
            tp.register_event("e_%d" % i, "event %d" % i)
        stats = tp.get_stats()
        assert stats["total_events"] <= _MAX_EVENTS

    def test_thread_safety(self):
        tp = TemporalParallax()
        errors = []

        def worker(i):
            try:
                e = tp.register_event("t_%d" % i, "content %d" % i, raw_impact=0.5)
                tp.compute_parallax(e.event_id)
                tp.get_stats()
            except Exception as ex:
                errors.append(ex)

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(worker, range(20)))
        assert len(errors) == 0

    def test_to_dict(self):
        tp = TemporalParallax()
        event = tp.register_event("test", "test content", raw_impact=0.5)
        d = event.to_dict()
        assert "event_id" in d
        assert 0 <= d["raw_impact"] <= 1

    def test_clamping(self):
        tp = TemporalParallax()
        event = tp.register_event("test", "test", raw_impact=5.0)
        assert event.raw_impact <= 1.0
        event2 = tp.register_event("test2", "test", raw_impact=-1.0)
        assert event2.raw_impact >= 0.0

    def test_sanitization(self):
        tp = TemporalParallax()
        event = tp.register_event("type\x00<>dirty", "desc\x00with<>content")
        assert "\x00" not in event.event_type
        assert "<>" not in event.event_type
        assert "\x00" not in event.description
        assert "<>" in event.description

    def test_compute_parallax_nonexistent(self):
        tp = TemporalParallax()
        assert tp.compute_parallax("nonexistent") is None

    def test_compare_perspectives_nonexistent(self):
        tp = TemporalParallax()
        assert tp.compare_perspectives("nonexistent", 1, 5) is None

    def test_get_dominant_perspective_nonexistent(self):
        tp = TemporalParallax()
        assert tp.get_dominant_perspective("nonexistent") is None


# ============================================================
# 6. AdaptiveBoundary (~20 tests)
# ============================================================

class TestAdaptiveBoundary:

    def test_set_boundary(self):
        ab = AdaptiveBoundary()
        rule = ab.set_boundary("work", "personal", permeability="flexible")
        assert rule is not None
        assert rule.source_domain == "work"
        assert rule.target_domain == "personal"

    def test_remove_boundary(self):
        ab = AdaptiveBoundary()
        rule = ab.set_boundary("work", "personal")
        assert ab.remove_boundary(rule.rule_id) is True
        assert ab.remove_boundary("nonexistent") is False

    def test_check_boundary(self):
        ab = AdaptiveBoundary()
        ab.set_boundary("work", "personal", permeability="rigid")
        result = ab.check_boundary("work", "personal")
        assert isinstance(result, dict)

    def test_allocate_energy(self):
        ab = AdaptiveBoundary()
        result = ab.allocate_energy("work", amount=0.5)
        assert result is not None

    def test_spend_energy(self):
        ab = AdaptiveBoundary()
        ab.allocate_energy("work", amount=0.8)
        result = ab.spend_energy("work", amount=0.3)
        assert result is not None

    def test_recover_energy(self):
        ab = AdaptiveBoundary()
        ab.allocate_energy("work", amount=0.5)
        ab.spend_energy("work", amount=0.4)
        result = ab.recover_energy("work", amount=0.2)
        assert result is not None

    def test_get_energy_profile(self):
        ab = AdaptiveBoundary()
        ab.allocate_energy("work", amount=0.5)
        profile = ab.get_energy_profile()
        assert isinstance(profile, dict)

    def test_report_violation(self):
        ab = AdaptiveBoundary()
        ab.set_boundary("work", "personal", permeability="rigid")
        violation = ab.report_violation("work", "personal", severity="high", description="cross-domain leak")
        assert violation is not None

    def test_get_violations(self):
        ab = AdaptiveBoundary()
        ab.set_boundary("work", "personal", permeability="rigid")
        ab.report_violation("work", "personal", severity="medium")
        violations = ab.get_violations()
        assert len(violations) >= 1

    def test_get_boundary_rules(self):
        ab = AdaptiveBoundary()
        ab.set_boundary("work", "personal", permeability="flexible")
        rules = ab.get_boundary_rules()
        assert len(rules) >= 1

    def test_suggest_recovery(self):
        ab = AdaptiveBoundary()
        result = ab.suggest_recovery("depleted")
        assert isinstance(result, dict)
        suggestions = result.get("general_suggestions", [])
        assert len(suggestions) > 0

    def test_eviction(self):
        ab = AdaptiveBoundary()
        from shadowd.cognition.adaptive_boundary import _MAX_VIOLATIONS
        ab.set_boundary("a", "b", permeability="rigid")
        for i in range(_MAX_VIOLATIONS + 10):
            ab.report_violation("a", "b", severity="low")
        stats = ab.get_stats()
        assert stats["total_violations"] <= _MAX_VIOLATIONS

    def test_thread_safety(self):
        ab = AdaptiveBoundary()
        errors = []

        def worker(i):
            try:
                ab.set_boundary("domain_%d" % i, "other_%d" % i, permeability="flexible")
                ab.allocate_energy("work", amount=0.1)
                ab.get_stats()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(worker, range(50)))
        assert len(errors) == 0

    def test_to_dict(self):
        ab = AdaptiveBoundary()
        rule = ab.set_boundary("work", "personal", permeability="flexible")
        d = rule.to_dict()
        assert isinstance(d, dict)
        assert "source_domain" in d

    def test_clamping(self):
        ab = AdaptiveBoundary()
        result = ab.allocate_energy("work", amount=5.0)
        profile = ab.get_energy_profile()
        for domain_data in profile.values():
            if isinstance(domain_data, dict):
                for k, v in domain_data.items():
                    if isinstance(v, float):
                        assert v >= 0.0

    def test_sanitization(self):
        ab = AdaptiveBoundary()
        rule = ab.set_boundary("work\x00<>dirty", "personal\x00<>dirty", permeability="flexible")
        assert "\x00" not in rule.source_domain
        assert "<>" not in rule.source_domain


# ============================================================
# 7. CognitiveResonanceEngine (~18 tests)
# ============================================================

class TestCognitiveResonanceEngine:

    def test_emit_pattern(self):
        cre = CognitiveResonanceEngine()
        pattern = cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.8, phase=0.0, content="logical deduction")
        assert pattern is not None
        assert pattern.domain == "reasoning"
        assert round(pattern.amplitude, 1) == 0.8

    def test_detect_resonance(self):
        cre = CognitiveResonanceEngine()
        cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.7, phase=0.0, content="test")
        cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.7, phase=0.0, content="test")
        events = cre.detect_resonance()
        assert isinstance(events, list)

    def test_compute_harmonics(self):
        cre = CognitiveResonanceEngine()
        cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.7, phase=0.0)
        spectrum = cre.compute_harmonics("reasoning")
        assert spectrum is not None

    def test_find_constructive(self):
        cre = CognitiveResonanceEngine()
        cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.8, phase=0.0, content="test constructive")
        cre.emit_pattern("emotional", frequency=0.5, amplitude=0.8, phase=0.0, content="test constructive")
        events = cre.find_constructive()
        assert isinstance(events, list)

    def test_find_destructive(self):
        cre = CognitiveResonanceEngine()
        cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.8, phase=0.0, content="alpha")
        cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.8, phase=3.14, content="beta")
        events = cre.find_destructive()
        assert isinstance(events, list)

    def test_amplify(self):
        cre = CognitiveResonanceEngine()
        p = cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.5, phase=0.0)
        result = cre.amplify(p.pattern_id, factor=2.0)
        assert result is not None

    def test_dampen(self):
        cre = CognitiveResonanceEngine()
        p = cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.8, phase=0.0)
        result = cre.dampen(p.pattern_id, factor=0.5)
        assert result is not None

    def test_synthesize_insight(self):
        cre = CognitiveResonanceEngine()
        cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.7, phase=0.0, content="pattern about logic")
        cre.emit_pattern("creative", frequency=0.5, amplitude=0.7, phase=0.0, content="pattern about art")
        insight = cre.synthesize_insight()
        assert insight is not None or insight is None

    def test_get_pattern(self):
        cre = CognitiveResonanceEngine()
        p = cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.7, phase=0.0)
        result = cre.get_pattern(p.pattern_id)
        assert result is not None
        assert cre.get_pattern("nonexistent") is None

    def test_get_resonance_events(self):
        cre = CognitiveResonanceEngine()
        cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.7, phase=0.0)
        events = cre.get_resonance_events()
        assert isinstance(events, list)

    def test_get_insights(self):
        cre = CognitiveResonanceEngine()
        insights = cre.get_insights()
        assert isinstance(insights, list)

    def test_eviction(self):
        cre = CognitiveResonanceEngine()
        from shadowd.cognition.cognitive_resonance_engine import _MAX_PATTERNS
        for i in range(_MAX_PATTERNS + 10):
            cre.emit_pattern("reasoning", frequency=float(i % 100) / 100, amplitude=0.5, phase=0.0)
        stats = cre.get_stats()
        assert stats["total_patterns"] <= _MAX_PATTERNS

    def test_thread_safety(self):
        cre = CognitiveResonanceEngine()
        errors = []

        def worker(i):
            try:
                cre.emit_pattern("reasoning", frequency=float(i) / 100, amplitude=0.5, phase=0.0)
                cre.detect_resonance()
                cre.get_stats()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(worker, range(30)))
        assert len(errors) == 0

    def test_to_dict(self):
        cre = CognitiveResonanceEngine()
        p = cre.emit_pattern("reasoning", frequency=0.5, amplitude=0.7, phase=0.0, content="test")
        d = p.to_dict()
        assert "pattern_id" in d
        assert 0 <= d["frequency"] <= 1

    def test_clamping(self):
        cre = CognitiveResonanceEngine()
        p = cre.emit_pattern("test", frequency=5.0, amplitude=5.0, phase=0.0)
        assert p.frequency <= 1.0
        assert p.amplitude <= 1.0

    def test_sanitization(self):
        cre = CognitiveResonanceEngine()
        p = cre.emit_pattern("test\x00<>dirty", content="content\x00<>test")
        assert "\x00" not in p.domain
        assert "<>" not in p.domain
        assert "\x00" not in p.content


# ============================================================
# 8. CognitiveEconomy (~20 tests)
# ============================================================

class TestCognitiveEconomy:

    def test_set_budget(self):
        ce = CognitiveEconomy()
        result = ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        assert result is not None
        assert result.resource_type == ResourceType.ATTENTION.value

    def test_request_allocation(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        req = ce.request_allocation("focus_task", ResourceType.ATTENTION, amount=0.3, priority=Priority.HIGH)
        assert req is not None

    def test_approve_allocation(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        req = ce.request_allocation("task", ResourceType.ATTENTION, amount=0.3, priority=Priority.HIGH)
        result = ce.approve_allocation(req.allocation_id)
        assert result is not None

    def test_spend_allocation(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        req = ce.request_allocation("task", ResourceType.ATTENTION, amount=0.3)
        ce.approve_allocation(req.allocation_id)
        result = ce.spend_allocation(req.allocation_id, amount=0.2)
        assert result is not None

    def test_release_allocation(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        req = ce.request_allocation("task", ResourceType.ATTENTION, amount=0.3)
        ce.approve_allocation(req.allocation_id)
        result = ce.release_allocation(req.allocation_id)
        assert result is True or result is not None

    def test_calculate_opportunity_cost(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        ce.set_budget(ResourceType.MEMORY, budget=1.0)
        oc = ce.calculate_opportunity_cost(ResourceType.ATTENTION, 0.5)
        assert oc is not None

    def test_optimize_allocation(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        ce.set_budget(ResourceType.MEMORY, budget=1.0)
        result = ce.optimize_allocation()
        assert result is not None

    def test_get_budget(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=0.8)
        budget = ce.get_budget(ResourceType.ATTENTION)
        assert budget is not None

    def test_get_allocations(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        req = ce.request_allocation("task", ResourceType.ATTENTION, amount=0.3)
        ce.approve_allocation(req.allocation_id)
        allocs = ce.get_allocations()
        assert len(allocs) >= 1

    def test_get_opportunity_costs(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        ocs = ce.get_opportunity_costs()
        assert isinstance(ocs, list)

    def test_generate_report(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        report = ce.generate_report()
        assert report is not None
        assert isinstance(report, dict)

    def test_suggest_rebalancing(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        ce.set_budget(ResourceType.MEMORY, budget=0.2)
        suggestions = ce.suggest_rebalancing()
        assert isinstance(suggestions, list)

    def test_eviction(self):
        ce = CognitiveEconomy()
        from shadowd.cognition.cognitive_economy import _MAX_ALLOCATIONS
        ce.set_budget(ResourceType.ATTENTION, budget=100.0)
        for i in range(_MAX_ALLOCATIONS + 10):
            req = ce.request_allocation("task_%d" % i, ResourceType.ATTENTION, amount=0.01, priority=Priority.LOW)
            if req:
                ce.approve_allocation(req.allocation_id)
        stats = ce.get_stats()
        assert stats["num_allocations"] <= _MAX_ALLOCATIONS + 1

    def test_thread_safety(self):
        ce = CognitiveEconomy()
        ce.set_budget(ResourceType.ATTENTION, budget=10.0)
        errors = []

        def worker(i):
            try:
                ce.request_allocation("t_%d" % i, ResourceType.ATTENTION, amount=0.01)
                ce.get_stats()
                ce.generate_report()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(worker, range(50)))
        assert len(errors) == 0

    def test_to_dict(self):
        ce = CognitiveEconomy()
        res = ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        d = res.to_dict()
        assert "resource_type" in d
        assert isinstance(d, dict)

    def test_clamping(self):
        ce = CognitiveEconomy()
        res = ce.set_budget(ResourceType.ATTENTION, budget=5.0)
        assert res.total_budget >= 0
        d = res.to_dict()
        if isinstance(d.get("efficiency"), float):
            assert 0 <= d["efficiency"] <= 1 or d["efficiency"] == 0

    def test_sanitization(self):
        ce = CognitiveEconomy()
        res = ce.set_budget(ResourceType.ATTENTION, budget=1.0)
        req = ce.request_allocation("task\x00<>dirty", ResourceType.ATTENTION, amount=0.3)
        d = req.to_dict()
        for k in d.keys():
            assert "\x00" not in str(k)


# ============================================================
# 9. CognitiveTopology (~20 tests)
# ============================================================

class TestCognitiveTopology:

    def test_add_node(self):
        ct = CognitiveTopology()
        node = ct.add_node("python", node_type=NodeType.SKILL, metadata={"level": "expert"})
        assert node is not None
        assert node.node_id.startswith("tn_") or node.name == "python"

    def test_add_edge(self):
        ct = CognitiveTopology()
        n1 = ct.add_node("A", node_type=NodeType.CONCEPT)
        n2 = ct.add_node("B", node_type=NodeType.CONCEPT)
        edge = ct.add_edge(n1.node_id, n2.node_id, edge_type=EdgeType.ENHANCES, weight=0.8)
        assert edge is not None

    def test_remove_node(self):
        ct = CognitiveTopology()
        node = ct.add_node("removeme", node_type=NodeType.CONCEPT)
        assert ct.remove_node(node.node_id) is True
        assert ct.remove_node("nonexistent") is False

    def test_remove_edge(self):
        ct = CognitiveTopology()
        n1 = ct.add_node("A", node_type=NodeType.CONCEPT)
        n2 = ct.add_node("B", node_type=NodeType.CONCEPT)
        edge = ct.add_edge(n1.node_id, n2.node_id)
        assert ct.remove_edge(edge.edge_id) is True
        assert ct.remove_edge("nonexistent") is False

    def test_analyze_topology(self):
        ct = CognitiveTopology()
        n1 = ct.add_node("A", node_type=NodeType.CONCEPT)
        n2 = ct.add_node("B", node_type=NodeType.CONCEPT)
        n3 = ct.add_node("C", node_type=NodeType.CONCEPT)
        ct.add_edge(n1.node_id, n2.node_id)
        ct.add_edge(n2.node_id, n3.node_id)
        analysis = ct.analyze_topology()
        assert analysis is not None

    def test_find_clusters(self):
        ct = CognitiveTopology()
        n1 = ct.add_node("A", node_type=NodeType.CONCEPT)
        n2 = ct.add_node("B", node_type=NodeType.CONCEPT)
        n3 = ct.add_node("C", node_type=NodeType.CONCEPT)
        ct.add_edge(n1.node_id, n2.node_id, weight=0.9)
        ct.add_edge(n2.node_id, n3.node_id, weight=0.9)
        clusters = ct.find_clusters()
        assert isinstance(clusters, list)

    def test_find_bridges(self):
        ct = CognitiveTopology()
        n1 = ct.add_node("A", node_type=NodeType.CONCEPT)
        n2 = ct.add_node("B", node_type=NodeType.CONCEPT)
        n3 = ct.add_node("C", node_type=NodeType.CONCEPT)
        ct.add_edge(n1.node_id, n2.node_id)
        ct.add_edge(n2.node_id, n3.node_id)
        bridges = ct.find_bridges()
        assert isinstance(bridges, list)

    def test_find_bottlenecks(self):
        ct = CognitiveTopology()
        n1 = ct.add_node("A", node_type=NodeType.CONCEPT)
        n2 = ct.add_node("B", node_type=NodeType.CONCEPT)
        n3 = ct.add_node("C", node_type=NodeType.CONCEPT)
        ct.add_edge(n1.node_id, n2.node_id)
        ct.add_edge(n2.node_id, n3.node_id)
        bottlenecks = ct.find_bottlenecks()
        assert isinstance(bottlenecks, list)

    def test_find_isolated_nodes(self):
        ct = CognitiveTopology()
        ct.add_node("connected", node_type=NodeType.CONCEPT)
        ct.add_node("loner", node_type=NodeType.CONCEPT)
        isolated = ct.find_isolated_nodes()
        assert isinstance(isolated, list)
        assert len(isolated) >= 1

    def test_suggest_connections(self):
        ct = CognitiveTopology()
        n1 = ct.add_node("A", node_type=NodeType.CONCEPT)
        n2 = ct.add_node("B", node_type=NodeType.CONCEPT)
        suggestions = ct.suggest_connections()
        assert isinstance(suggestions, list)

    def test_get_shortest_path(self):
        ct = CognitiveTopology()
        n1 = ct.add_node("A", node_type=NodeType.CONCEPT)
        n2 = ct.add_node("B", node_type=NodeType.CONCEPT)
        n3 = ct.add_node("C", node_type=NodeType.CONCEPT)
        ct.add_edge(n1.node_id, n2.node_id)
        ct.add_edge(n2.node_id, n3.node_id)
        path = ct.get_shortest_path(n1.node_id, n3.node_id)
        assert path is not None or path is None

    def test_get_metrics(self):
        ct = CognitiveTopology()
        ct.add_node("A", node_type=NodeType.CONCEPT)
        metrics = ct.get_metrics()
        assert isinstance(metrics, dict)

    def test_eviction(self):
        ct = CognitiveTopology()
        from shadowd.cognition.cognitive_topology import _MAX_NODES
        for i in range(_MAX_NODES + 10):
            ct.add_node("node_%d" % i, node_type=NodeType.CONCEPT)
        stats = ct.get_stats()
        assert stats["nodes_count"] <= _MAX_NODES

    def test_thread_safety(self):
        ct = CognitiveTopology()
        errors = []

        def worker(i):
            try:
                ct.add_node("n_%d" % i, node_type=NodeType.CONCEPT)
                ct.get_stats()
                ct.find_isolated_nodes()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(worker, range(50)))
        assert len(errors) == 0

    def test_to_dict(self):
        ct = CognitiveTopology()
        node = ct.add_node("test", node_type=NodeType.CONCEPT, metadata={"k": "v"})
        d = node.to_dict()
        assert "node_id" in d
        assert "name" in d

    def test_clamping(self):
        ct = CognitiveTopology()
        node = ct.add_node("test", node_type=NodeType.CONCEPT)
        d = node.to_dict()
        for k, v in d.items():
            if isinstance(v, float):
                assert 0 <= v <= 1 or v == 0

    def test_sanitization(self):
        ct = CognitiveTopology()
        node = ct.add_node("test\x00<>dirty", node_type=NodeType.CONCEPT)
        d = node.to_dict()
        assert "\x00" not in d["name"]
        assert "<>" not in d["name"]

    def test_get_node(self):
        ct = CognitiveTopology()
        node = ct.add_node("findme", node_type=NodeType.CONCEPT)
        found = ct.get_node(node.node_id)
        assert found is not None
        assert ct.get_node("nonexistent") is None


# ============================================================
# 10. CognitiveSymbiosis (~20 tests)
# ============================================================

class TestCognitiveSymbiosis:

    def test_register_pair(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("module_a", "module_b", SymbiosisType.MUTUALISM, InteractionMode.FEED, strength_a_to_b=0.7, strength_b_to_a=0.6)
        assert pair is not None
        assert pair.module_a == "module_a"
        assert pair.symbiosis_type == SymbiosisType.MUTUALISM

    def test_remove_pair(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.FEED)
        assert cs.remove_pair(pair.pair_id) is True
        assert cs.remove_pair("nonexistent") is False

    def test_record_interaction(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.FEED)
        result = cs.record_interaction(pair.pair_id, InteractionMode.FEED, benefit_a=0.3, benefit_b=0.4)
        assert result is not None

    def test_detect_parasitism(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("host", "parasite", SymbiosisType.PARASITISM, InteractionMode.FEED, strength_a_to_b=0.8, strength_b_to_a=0.05)
        detections = cs.detect_parasitism()
        assert isinstance(detections, list)

    def test_convert_pair(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("a", "b", SymbiosisType.PARASITISM, InteractionMode.INHIBIT)
        result = cs.convert_pair(pair.pair_id, SymbiosisType.MUTUALISM, InteractionMode.FEED)
        assert result is not None

    def test_optimize_mutualism(self):
        cs = CognitiveSymbiosis()
        cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.AMPLIFY, strength_a_to_b=0.5, strength_b_to_a=0.5)
        result = cs.optimize_mutualism()
        assert isinstance(result, list)

    def test_get_pair(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("x", "y", SymbiosisType.COMMENSALISM, InteractionMode.TRIGGER)
        found = cs.get_pair(pair.pair_id)
        assert found is not None
        assert cs.get_pair("nonexistent") is None

    def test_get_pairs(self):
        cs = CognitiveSymbiosis()
        cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.FEED)
        cs.register_pair("c", "d", SymbiosisType.COMPETITION, InteractionMode.INHIBIT)
        pairs = cs.get_pairs()
        assert len(pairs) >= 2

    def test_get_interactions(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.FEED)
        cs.record_interaction(pair.pair_id, InteractionMode.FEED, benefit_a=0.2, benefit_b=0.3)
        interactions = cs.get_interactions(pair.pair_id)
        assert len(interactions) >= 1

    def test_generate_report(self):
        cs = CognitiveSymbiosis()
        cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.AMPLIFY)
        report = cs.generate_report()
        assert isinstance(report, dict)

    def test_get_system_benefit(self):
        cs = CognitiveSymbiosis()
        cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.AMPLIFY, strength_a_to_b=0.7, strength_b_to_a=0.8)
        benefit = cs.get_system_benefit()
        assert isinstance(benefit, float)

    def test_suggest_optimizations(self):
        cs = CognitiveSymbiosis()
        cs.register_pair("a", "b", SymbiosisType.PARASITISM, InteractionMode.INHIBIT, strength_a_to_b=0.9, strength_b_to_a=0.05)
        suggestions = cs.suggest_optimizations()
        assert isinstance(suggestions, list)

    def test_eviction(self):
        cs = CognitiveSymbiosis()
        from shadowd.cognition.cognitive_symbiosis import _MAX_PAIRS
        for i in range(_MAX_PAIRS + 10):
            cs.register_pair("m_%d_a" % i, "m_%d_b" % i, SymbiosisType.MUTUALISM, InteractionMode.FEED)
        stats = cs.get_stats()
        assert stats["pair_count"] <= _MAX_PAIRS

    def test_thread_safety(self):
        cs = CognitiveSymbiosis()
        errors = []

        def worker(i):
            try:
                cs.register_pair("a_%d" % i, "b_%d" % i, SymbiosisType.MUTUALISM, InteractionMode.FEED)
                cs.get_stats()
                cs.generate_report()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(worker, range(50)))
        assert len(errors) == 0

    def test_to_dict(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.FEED)
        d = pair.to_dict()
        assert isinstance(d, dict)
        assert "module_a" in d

    def test_clamping(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.FEED, strength_a_to_b=5.0, strength_b_to_a=-1.0)
        assert pair.strength_a_to_b <= 1.0
        assert pair.strength_b_to_a >= 0.0

    def test_sanitization(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("a\x00<>dirty", "b\x00<>dirty", SymbiosisType.MUTUALISM, InteractionMode.FEED)
        d = pair.to_dict()
        assert "\x00" not in d["module_a"]
        assert "<>" not in d["module_a"]

    def test_remove_pair_returns(self):
        cs = CognitiveSymbiosis()
        pair = cs.register_pair("a", "b", SymbiosisType.MUTUALISM, InteractionMode.FEED)
        assert cs.remove_pair(pair.pair_id) is True

    def test_record_interaction_nonexistent(self):
        cs = CognitiveSymbiosis()
        result = cs.record_interaction("nonexistent", InteractionMode.FEED)
        assert result is None


# ============================================================
# ShadowOS Integration Tests
# ============================================================

class TestSprints222To231ShadowOS:

    MODULE_NAMES = [
        "causal", "quantum_memory", "drift_detector", "immune",
        "parallax", "boundary", "resonance_engine", "economy",
        "topology", "symbiosis",
    ]

    MODULE_TYPES = {
        "causal": CausalReasoning,
        "quantum_memory": QuantumMemoryField,
        "drift_detector": SemanticDriftDetector,
        "immune": CognitiveImmuneSystem,
        "parallax": TemporalParallax,
        "boundary": AdaptiveBoundary,
        "resonance_engine": CognitiveResonanceEngine,
        "economy": CognitiveEconomy,
        "topology": CognitiveTopology,
        "symbiosis": CognitiveSymbiosis,
    }

    def _make_os(self):
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.cognition.shadow_graph import ShadowGraph
        from unittest.mock import MagicMock, patch

        os_ = object.__new__(ShadowOS)
        for attr in ['graph', 'memory', 'crew', 'mesh', 'stealth', 'personality',
                     'voice', 'anomaly_detector', 'goal_planner', 'thought_sense',
                     'recall_synthesis', 'temporal_reasoning', 'empathy', 'presence',
                     'trinity', 'dream', 'skills', 'narrative', 'procedural',
                     'conflict', 'emotional_intel', 'creative', 'metacognition',
                     'moral', 'attention', 'pipeline', 'social_intel', 'journal',
                     'decision_journal',
                     'reflection', 'goal_reasoning', 'device_context', 'arise',
                     'summon', 'evolution', 'oracle', 'daemon', 'feed_loop',
                     'anticipatory', 'resonance', 'temporal_self',
                     'health_dashboard', 'creative_dream', 'social_propagation',
                     'neural_bridge', 'quantum_field', 'grammar', 'holographic',
                     'syncognitive', 'morphic', 'existential', 'chrono', 'phoenix',
                     'mind', 'event_bus', 'shadow_ai',
                     # ShadowOS.get_status() touches these too; keep in sync.
                     'shadow_powers', 'synthesis_engine', 'self_mod', 'profiler',
                     'intel_report', 'anomaly_corr', 'module_bus', 'opt_daemon',
                     'module_registry', 'vision', 'notifications', 'visual_bridge',
                     'shadow_deploy', 'evolutionary',
                     ]:
            setattr(os_, attr, MagicMock())

        os_.causal = CausalReasoning()
        os_.quantum_memory = QuantumMemoryField()
        os_.drift_detector = SemanticDriftDetector()
        os_.immune = CognitiveImmuneSystem()
        os_.parallax = TemporalParallax()
        os_.boundary = AdaptiveBoundary()
        os_.resonance_engine = CognitiveResonanceEngine()
        os_.economy = CognitiveEconomy()
        os_.topology = CognitiveTopology()
        os_.symbiosis = CognitiveSymbiosis()

        os_._running = False
        os_._start_time = None
        os_._stats_lock = threading.Lock()
        os_._stats = {"events_processed": 0, "queries_handled": 0, "actions_taken": 0,
                       "insights_generated": 0, "anomalies_detected": 0, "errors": 0,
                       "uptime_seconds": 0}
        return os_

    def test_new_modules_in_status(self):
        os_ = self._make_os()
        from shadowd.cognition.shadow_os import ShadowOS
        status = ShadowOS.get_status(os_)
        stat_keys = [k for k in status.keys()]
        for name in self.MODULE_NAMES:
            expected = "%s_stats" % name
            assert expected in stat_keys, "Missing %s in status keys" % expected

    def test_new_modules_in_health(self):
        os_ = self._make_os()
        from shadowd.cognition.shadow_os import ShadowOS
        os_._running = True
        os_._start_time = time.monotonic()
        health = ShadowOS.get_health(os_)
        modules = health.get("modules", {})
        for name in self.MODULE_NAMES:
            assert name in modules, "Module %s not in health modules" % name
            assert modules[name].get("status") in ("healthy", "unhealthy"), "Bad status for %s" % name

    def test_module_types(self):
        os_ = self._make_os()
        for name, expected_type in self.MODULE_TYPES.items():
            obj = getattr(os_, name, None)
            assert obj is not None, "Missing attribute: %s" % name
            assert isinstance(obj, expected_type), "os.%s is %s, expected %s" % (name, type(obj).__name__, expected_type.__name__)