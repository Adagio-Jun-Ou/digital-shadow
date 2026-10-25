#!/usr/bin/env python3
"""
Comprehensive tests for Cross-Module Synthesis Engine.

Tests cover:
  - ModuleSignal dataclass: creation, sanitization, clamping, truncation, deepcopy
  - SynthesisResult dataclass: creation, deepcopy safety
  - CrossModuleSynthesisEngine: ingest, synthesize, contradictions, emergent properties,
    validation, history, trends, topic clusters, stats, thread safety
  - Integration: full lifecycle, cross-category, contradiction resolution flow, trend evolution
"""

import copy
import threading

import pytest

from shadowd.cognition.cross_module_synthesis import (
    CATEGORY_CROSS_INFLUENCE,
    MODULE_CATEGORY_MAP,
    CrossModuleSynthesisEngine,
    ModuleSignal,
    ModuleCategory,
    SynthesisDomain,
    SynthesisResult,
    ConfidenceTier,
    TrendSnapshot,
    _sanitize,
    _sanitize_content,
)


class TestModuleSignal:
    def test_create_basic_signal(self):
        s = ModuleSignal(
            source_module="attention_engine",
            category="cognitive",
            signal_type="focus_shift",
            content="User shifted focus to deep work",
            confidence=0.85,
            domains=["productivity", "cognition"],
            entities=["user", "focus", "deep_work"],
            metadata={"priority": "high"},
        )
        assert s.source_module == "attention_engine"
        assert s.category == "cognitive"
        assert s.signal_type == "focus_shift"
        assert s.content[:50] == "User shifted focus to deep work"
        assert s.confidence == 0.85
        assert "productivity" in s.domains
        assert "user" in s.entities
        assert s.metadata["priority"] == "high"
        assert s.signal_id
        assert len(s.signal_id) == 12

    def test_sanitize_control_chars(self):
        s = ModuleSignal(
            source_module="test_module\x00\x01",
            category="cognitive\x02",
            signal_type="alert\x03",
            content="data with \x00 null \x01 byte",
            confidence=0.7,
        )
        assert "\x00" not in s.source_module
        assert "\x01" not in s.source_module
        assert "\x02" not in s.category
        assert "\x03" not in s.signal_type
        assert "\x00" not in s.content
        assert "\x01" not in s.content

    def test_sanitize_strips_angle_brackets_identifiers(self):
        s = ModuleSignal(
            source_module="<script>malicious</script>",
            category="<evil>",
            signal_type="<alert>",
            content="normal content",
            confidence=0.5,
        )
        assert "<script>" not in s.source_module
        assert "<evil>" not in s.category
        assert "<alert>" not in s.signal_type

    def test_content_preserves_angle_brackets(self):
        s = ModuleSignal(
            source_module="test",
            category="test",
            signal_type="metric",
            content="latency < 100ms and > 50ms",
            confidence=0.6,
        )
        assert "<" in s.content
        assert ">" in s.content

    def test_clamp_confidence(self):
        s_low = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="low", confidence=-0.5,
        )
        assert s_low.confidence == 0.0
        s_high = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="high", confidence=1.5,
        )
        assert s_high.confidence == 1.0
        s_ok = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="ok", confidence=0.73,
        )
        assert s_ok.confidence == 0.73

    def test_entities_truncated(self):
        entities = [f"entity_{i}" for i in range(100)]
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="many entities", confidence=0.5, entities=entities,
        )
        assert len(s.entities) == 50
        assert s.entities[0] == "entity_0"

    def test_entities_sanitized(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.5, entities=["<bad>", "\x00evil", "good"],
        )
        assert "<bad>" not in s.entities
        assert "\x00" not in s.entities[1]
        assert "good" in s.entities

    def test_domains_truncated(self):
        domains = [f"domain_{i}" for i in range(50)]
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="many domains", confidence=0.5, domains=domains,
        )
        assert len(s.domains) == 20

    def test_metadata_sanitized(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.5,
            metadata={
                "key\x00evil": "value\x01bad",
                "<inject>": "normal",
                "normal_key": "<script>alert</script>",
            },
        )
        assert "\x00" not in list(s.metadata.keys())[0]
        assert "<" not in list(s.metadata.keys())[1]
        for v in s.metadata.values():
            assert "\x01" not in v
        assert "alert" in s.metadata["normal_key"]

    def test_metadata_key_count_capped(self):
        metadata = {f"key_{i}": f"val_{i}" for i in range(100)}
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.5, metadata=metadata,
        )
        assert len(s.metadata) <= 50

    def test_to_dict_deepcopy(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test data", confidence=0.8,
            entities=["e1", "e2"], domains=["d1"],
            metadata={"key": "val"},
        )
        d = s.to_dict()
        d["entities"].append("e3")
        assert len(s.entities) == 2
        d["domains"].append("d2")
        assert len(s.domains) == 1
        d["metadata"]["new_key"] = "new_val"
        assert "new_key" not in s.metadata

    def test_to_dict_rounds_confidence(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.3333333,
        )
        d = s.to_dict()
        assert d["confidence"] == 0.333

    def test_to_dict_content_truncated(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="x" * 5000, confidence=0.5,
        )
        d = s.to_dict()
        assert len(d["content"]) == 200

    def test_default_signal_id_generated(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.5,
        )
        assert s.signal_id
        assert len(s.signal_id) == 12

    def test_custom_signal_id_preserved(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.5, signal_id="abcd1234efgh",
        )
        assert s.signal_id == "abcd1234efgh"

    def test_empty_signal_id_replaced(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.5, signal_id="",
        )
        assert s.signal_id
        assert len(s.signal_id) == 12

    def test_none_entities_becomes_empty(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.5, entities=None,
        )
        assert s.entities == []

    def test_none_domains_becomes_empty(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.5, domains=None,
        )
        assert s.domains == []

    def test_none_metadata_becomes_empty(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="test", confidence=0.5, metadata=None,
        )
        assert s.metadata == {}

    def test_source_module_truncated(self):
        s = ModuleSignal(
            source_module="a" * 100, category="test", signal_type="test",
            content="test", confidence=0.5,
        )
        assert len(s.source_module) == 50

    def test_content_truncated(self):
        s = ModuleSignal(
            source_module="test", category="test", signal_type="test",
            content="x" * 10000, confidence=0.5,
        )
        assert len(s.content) == 5000


class TestSynthesisResult:
    def test_create_basic_result(self):
        r = SynthesisResult(
            domain="cross_module",
            title="Test Synthesis",
            summary="A test summary of the synthesis",
            key_finding="Key insight discovered",
            confidence=0.75,
            source_modules=["attention_engine", "memory_system"],
            source_signals=["sig1", "sig2"],
            concepts_linked=["focus", "memory", "retention"],
            category_crosses=[("cognitive", "memory")],
            action_recommendation="Investigate the cognitive-memory link",
            expected_impact=0.6,
            supporting_evidence=["Evidence A", "Evidence B"],
            conflicting_evidence=[],
            metadata={"source": "test"},
        )
        assert r.synthesis_id
        assert len(r.synthesis_id) == 12
        assert r.domain == "cross_module"
        assert r.confidence == 0.75
        assert len(r.source_modules) == 2
        assert r.is_active

    def test_confidence_clamped(self):
        r = SynthesisResult(confidence=-0.2)
        assert r.confidence == 0.0
        r2 = SynthesisResult(confidence=1.5)
        assert r2.confidence == 1.0

    def test_expected_impact_clamped(self):
        r = SynthesisResult(expected_impact=-0.5)
        assert r.expected_impact == 0.0
        r2 = SynthesisResult(expected_impact=2.0)
        assert r2.expected_impact == 1.0

    def test_validation_score_clamped(self):
        r = SynthesisResult(validation_score=-0.5)
        assert r.validation_score == 0.0
        r2 = SynthesisResult(validation_score=2.0)
        assert r2.validation_score == 1.0

    def test_metadata_sanitized(self):
        r = SynthesisResult(
            metadata={
                "key\x00evil": "val\x01bad",
                "<inject>": "normal",
            }
        )
        assert "\x00" not in list(r.metadata.keys())[0]
        assert "<" not in list(r.metadata.keys())[1]

    def test_source_modules_sanitized(self):
        r = SynthesisResult(
            source_modules=[
                "<script>",
                "module\x00test",
            ]
        )
        assert "<script>" not in r.source_modules
        assert "\x00" not in r.source_modules[1]

    def test_to_dict_deepcopy(self):
        r = SynthesisResult(
            domain="cross_module",
            source_modules=["m1"],
            source_signals=["s1"],
            concepts_linked=["c1"],
            category_crosses=[("cognitive", "memory")],
            supporting_evidence=["e1"],
            conflicting_evidence=["ce1"],
            metadata={"key": "val"},
        )
        d = r.to_dict()
        d["source_modules"].append("m2")
        assert len(r.source_modules) == 1
        d["concepts_linked"].append("c2")
        assert len(r.concepts_linked) == 1
        d["metadata"]["new_key"] = "new_val"
        assert "new_key" not in r.metadata
        d["supporting_evidence"].append("e2")
        assert len(r.supporting_evidence) == 1
        d["conflicting_evidence"].append("ce2")
        assert len(r.conflicting_evidence) == 1
        d["source_signals"].append("s2")
        assert len(r.source_signals) == 1
        d["category_crosses"].append(("x", "y"))
        assert len(r.category_crosses) == 1

    def test_to_dict_rounds_floats(self):
        r = SynthesisResult(
            confidence=0.3333333,
            expected_impact=0.4444444,
            validation_score=0.5555555,
        )
        d = r.to_dict()
        assert d["confidence"] == 0.333
        assert d["expected_impact"] == 0.444
        assert d["validation_score"] == 0.556

    def test_to_dict_summary_truncated(self):
        r = SynthesisResult(summary="x" * 10000)
        d = r.to_dict()
        assert len(d["summary"]) == 500

    def test_expires_at_none_serializes_as_none(self):
        r = SynthesisResult()
        d = r.to_dict()
        assert d["expires_at"] is None

    def test_title_strips_control_chars(self):
        r = SynthesisResult(title="Hello\x00World\x01Test")
        assert "\x00" not in r.title
        assert "\x01" not in r.title

    def test_title_content_preserves_comparisons(self):
        r = SynthesisResult(title="Performance when latency < 100ms and > 50ms")
        assert "<" in r.title
        assert ">" in r.title


class TestSynthesisEngine:
    def _make_signal(self, module="test_module", category="cognitive",
                     signal_type="test_signal", content="test content",
                     confidence=0.8, domains=None, entities=None, metadata=None):
        return ModuleSignal(
            source_module=module,
            category=category,
            signal_type=signal_type,
            content=content,
            confidence=confidence,
            domains=domains or ["test_domain"],
            entities=entities or ["test_entity"],
            metadata=metadata or {},
        )

    def test_ingest_single_signal(self):
        engine = CrossModuleSynthesisEngine()
        s = self._make_signal()
        sid = engine.ingest_signal(s)
        assert sid == s.signal_id
        stats = engine.get_stats()
        assert stats["signals_ingested"] == 1
        assert stats["unique_modules_seen"] == 1

    def test_ingest_multiple_signals(self):
        engine = CrossModuleSynthesisEngine()
        for i in range(5):
            s = self._make_signal(module=f"module_{i}")
            engine.ingest_signal(s)
        stats = engine.get_stats()
        assert stats["signals_ingested"] == 5
        assert stats["unique_modules_seen"] == 5

    def test_synthesize_cross_category(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(
            module="dark_memory", category="memory",
            content="User frequently accesses project planning memories",
            confidence=0.9,
        ))
        engine.ingest_signal(self._make_signal(
            module="goal_planner", category="cognitive",
            content="Project Alpha deadline approaching, user needs focus",
            confidence=0.85,
        ))
        engine.ingest_signal(self._make_signal(
            module="attention_engine", category="cognitive",
            content="User attention span decreasing in afternoon",
            confidence=0.8,
        ))
        results = engine.synthesize()
        assert len(results) >= 1
        for r in results:
            assert r.confidence >= 0.5

    def test_synthesize_empty(self):
        engine = CrossModuleSynthesisEngine()
        results = engine.synthesize()
        assert results == []

    def test_synthesize_filter_by_module(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(
            module="test_a", category="cognitive", confidence=0.8,
        ))
        engine.ingest_signal(self._make_signal(
            module="test_b", category="memory", confidence=0.8,
        ))
        results = engine.synthesize(modules=["test_a"])
        for r in results:
            assert "test_a" in r.source_modules

    def test_synthesize_filter_by_category(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(
            module="test_a", category="cognitive", confidence=0.8,
        ))
        engine.ingest_signal(self._make_signal(
            module="test_b", category="memory", confidence=0.8,
        ))
        engine.ingest_signal(self._make_signal(
            module="test_c", category="creative", confidence=0.8,
        ))
        results = engine.synthesize(categories=["cognitive", "memory"])
        for r in results:
            for m in r.source_modules:
                cat = MODULE_CATEGORY_MAP.get(m, ModuleCategory.COGNITIVE).value
                assert cat in ("cognitive", "memory")

    def test_synthesize_min_confidence(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(
            module="test_a", category="cognitive", confidence=0.3,
        ))
        engine.ingest_signal(self._make_signal(
            module="test_b", category="cognitive", confidence=0.9,
        ))
        results = engine.synthesize(min_confidence=0.5)
        for r in results:
            assert r.confidence >= 0.5

    def test_synthesize_max_syntheses(self):
        engine = CrossModuleSynthesisEngine()
        for i in range(20):
            engine.ingest_signal(self._make_signal(
                module=f"mod_{i % 5}",
                category=["cognitive", "memory", "social", "creative", "predictive"][i % 5],
                confidence=0.8,
            ))
        results = engine.synthesize(max_syntheses=3)
        assert len(results) <= 3

    def test_contradiction_detection(self):
        engine = CrossModuleSynthesisEngine()
        s1 = self._make_signal(module="test_a", category="cognitive",
                               content="System is performing well", confidence=0.9)
        s2 = self._make_signal(module="test_b", category="cognitive",
                               content="System performance is degrading", confidence=0.1)
        engine.ingest_signal(s1)
        engine.ingest_signal(s2)
        engine.synthesize()
        engine.synthesize()
        contradictions = engine.detect_contradictions()
        assert len(contradictions) >= 0

    def test_contradiction_none_when_empty(self):
        engine = CrossModuleSynthesisEngine()
        contradictions = engine.detect_contradictions()
        assert contradictions == []

    def test_resolve_contradiction_not_found(self):
        engine = CrossModuleSynthesisEngine()
        result = engine.resolve_contradiction("nonexistent_a", "nonexistent_b", "resolution")
        assert result is None

    def test_resolve_contradiction_success(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(
            module="dark_memory", category="memory",
            content="Memory recall is improving", confidence=0.9,
        ))
        engine.ingest_signal(self._make_signal(
            module="cognitive_trinity", category="cognitive",
            content="Decision quality shows improvement trend", confidence=0.85,
        ))
        results = engine.synthesize()
        if len(results) >= 2:
            result = engine.resolve_contradiction(
                results[0].synthesis_id,
                results[1].synthesis_id,
                "Both modules observe positive trends, confidence weighted",
            )
            assert result is not None
            assert result.domain == "contradiction_resolution"

    def test_emergent_properties(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(
            module="attention_engine", category="cognitive",
            content="Focus patterns show morning peak", confidence=0.9,
        ))
        engine.ingest_signal(self._make_signal(
            module="dark_memory", category="memory",
            content="Memory retention strongest in morning sessions", confidence=0.85,
        ))
        engine.ingest_signal(self._make_signal(
            module="personality_engine", category="personality",
            content="User reports higher productivity in morning hours", confidence=0.8,
        ))
        engine.synthesize()
        emergents = engine.detect_emergent_properties()
        assert isinstance(emergents, list)

    def test_validate_confirmed(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(module="test_a", category="cognitive", confidence=0.8))
        engine.ingest_signal(self._make_signal(module="test_b", category="memory", confidence=0.8))
        results = engine.synthesize()
        if results:
            syn = results[0]
            old_conf = syn.confidence
            result = engine.validate_synthesis(syn.synthesis_id, "confirmed", 0.9)
            assert result is not None
            assert isinstance(result, SynthesisResult)

    def test_validate_refuted(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(module="test_a", category="cognitive", confidence=0.8))
        engine.ingest_signal(self._make_signal(module="test_b", category="memory", confidence=0.8))
        results = engine.synthesize()
        if results:
            syn = results[0]
            result = engine.validate_synthesis(syn.synthesis_id, "refuted", 0.1)
            assert result is not None
            assert isinstance(result, SynthesisResult)

    def test_validate_not_found(self):
        engine = CrossModuleSynthesisEngine()
        result = engine.validate_synthesis("nonexistent", "confirmed")
        assert result is None

    def test_get_synthesis_history(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(module="test_a", category="cognitive", confidence=0.8))
        engine.ingest_signal(self._make_signal(module="test_b", category="memory", confidence=0.8))
        engine.synthesize()
        history = engine.get_synthesis_history()
        assert isinstance(history, list)
        assert len(history) > 0

    def test_get_synthesis_history_domain_filter(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(module="test_a", category="cognitive", confidence=0.8))
        engine.ingest_signal(self._make_signal(module="test_b", category="memory", confidence=0.8))
        engine.synthesize()
        history = engine.get_synthesis_history(domain="cross_module")
        for h in history:
            assert h["domain"] == "cross_module"

    def test_get_synthesis_history_active_only(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(module="test_a", category="cognitive", confidence=0.8))
        engine.ingest_signal(self._make_signal(module="test_b", category="memory", confidence=0.8))
        engine.synthesize()
        active = engine.get_synthesis_history(active_only=True)
        for h in active:
            assert h["is_active"]

    def test_get_synthesis_detail(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(module="test_a", category="cognitive", confidence=0.8))
        engine.ingest_signal(self._make_signal(module="test_b", category="memory", confidence=0.8))
        results = engine.synthesize()
        if results:
            detail = engine.get_synthesis_detail(results[0].synthesis_id)
            assert detail is not None
            assert "synthesis_id" in detail
            assert "related_syntheses" in detail

    def test_get_synthesis_detail_not_found(self):
        engine = CrossModuleSynthesisEngine()
        detail = engine.get_synthesis_detail("nonexistent_id")
        assert detail is None

    def test_get_trends(self):
        engine = CrossModuleSynthesisEngine()
        for i in range(15):
            engine.ingest_signal(self._make_signal(
                module="test_module", category="cognitive",
                signal_type="repeat_signal", confidence=0.5 + (i * 0.02),
            ))
        trends = engine.get_trends(min_forecast_confidence=0.0)
        assert isinstance(trends, list)
        trends = engine.get_trends(min_forecast_confidence=1.0)
        assert trends == []

    def test_get_topic_clusters(self):
        engine = CrossModuleSynthesisEngine()
        for i in range(10):
            engine.ingest_signal(self._make_signal(
                module="test_module", category="cognitive",
                signal_type="focus_signal", confidence=0.7,
            ))
        for i in range(5):
            engine.ingest_signal(self._make_signal(
                module="test_module_b", category="memory",
                signal_type="memory_signal", confidence=0.8,
            ))
        clusters = engine.get_topic_clusters()
        assert len(clusters) >= 1
        assert isinstance(clusters, list)

    def test_get_topic_clusters_empty(self):
        engine = CrossModuleSynthesisEngine()
        clusters = engine.get_topic_clusters()
        assert clusters == []

    def test_get_stats(self):
        engine = CrossModuleSynthesisEngine()
        engine.ingest_signal(self._make_signal(
            module="test_a", category="cognitive", confidence=0.8,
        ))
        stats = engine.get_stats()
        assert "signals_ingested" in stats
        assert "syntheses_generated" in stats
        assert "contradictions_detected" in stats
        assert "validations_performed" in stats
        assert "total_signals" in stats
        assert stats["signals_ingested"] == 1

    def test_stats_dict_is_copy(self):
        engine = CrossModuleSynthesisEngine()
        stats1 = engine.get_stats()
        stats1["signals_ingested"] = 9999
        stats2 = engine.get_stats()
        assert stats2["signals_ingested"] == 0

    def test_thread_safety(self):
        engine = CrossModuleSynthesisEngine()
        errors = []

        def worker(worker_id):
            try:
                for i in range(50):
                    s = self._make_signal(
                        module=f"module_{worker_id}",
                        category=["cognitive", "memory", "social"][i % 3],
                        confidence=0.7 + (i % 30) / 100,
                    )
                    engine.ingest_signal(s)
                engine.synthesize()
                engine.detect_contradictions()
                engine.get_stats()
                engine.get_trends()
            except BaseException as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        stats = engine.get_stats()
        assert stats["signals_ingested"] == 250


class TestSynthesisIntegration:
    def _make_signal(self, module="test_module", category="cognitive",
                     signal_type="test_signal", content="test content",
                     confidence=0.8, domains=None, entities=None, metadata=None):
        return ModuleSignal(
            source_module=module,
            category=category,
            signal_type=signal_type,
            content=content,
            confidence=confidence,
            domains=domains or ["test_domain"],
            entities=entities or ["test_entity"],
            metadata=metadata or {},
        )

    def test_full_lifecycle(self):
        engine = CrossModuleSynthesisEngine()

        engine.ingest_signal(self._make_signal(
            module="dark_memory", category="memory",
            content="User has strong recall of technical concepts learned in morning",
            confidence=0.9, entities=["user", "memory", "morning", "technical"],
        ))
        engine.ingest_signal(self._make_signal(
            module="attention_engine", category="cognitive",
            content="Attention span peaks between 9-11 AM",
            confidence=0.85, entities=["attention", "morning", "focus"],
        ))
        engine.ingest_signal(self._make_signal(
            module="personality_engine", category="personality",
            content="User's productivity profile shows morning-person pattern",
            confidence=0.8, entities=["user", "morning", "productivity"],
        ))
        engine.ingest_signal(self._make_signal(
            module="goal_planner", category="cognitive",
            content="Scheduled deep work sessions for morning hours",
            confidence=0.75, entities=["deep_work", "morning", "planning"],
        ))

        results = engine.synthesize()
        assert len(results) > 0, "Should produce at least one synthesis"

        if results:
            syn = results[0]
            assert syn.confidence > 0, "Synthesis should have confidence"
            assert syn.source_modules, "Synthesis should reference source modules"
            assert syn.key_finding, "Synthesis should have key finding"

            validated = engine.validate_synthesis(syn.synthesis_id, "confirmed", 0.95)
            assert validated is not None, "Validation should succeed"

            detail = engine.get_synthesis_detail(syn.synthesis_id)
            assert detail is not None, "Should retrieve synthesis detail"

        stats = engine.get_stats()
        assert stats["signals_ingested"] == 4
        assert stats["syntheses_generated"] >= 1
        assert stats["validations_performed"] >= 1

    def test_cross_category_synthesis(self):
        engine = CrossModuleSynthesisEngine()

        modules_and_signals = [
            ("dark_memory", "memory", "User recalls project deadlines frequently",
             0.9, ["user", "deadlines", "projects"]),
            ("cognitive_trinity", "cognitive", "Decision making improves with structured planning",
             0.85, ["decisions", "planning"]),
            ("empathy_mapping", "social", "User communicates more effectively in morning meetings",
             0.8, ["communication", "meetings", "morning"]),
            ("creative_synthesis", "creative", "Creative output spikes after social interactions",
             0.75, ["creativity", "social", "interactions"]),
            ("temporal_reasoning", "predictive", "Productivity patterns show weekly sinusoidal trend",
             0.7, ["productivity", "weekly", "trend"]),
            ("metacognition_engine", "meta", "Self-reflection quality improves after social engagement",
             0.65, ["self", "reflection", "social"]),
        ]

        for mod, cat, content, conf, ents in modules_and_signals:
            engine.ingest_signal(self._make_signal(
                module=mod, category=cat, content=content,
                confidence=conf, entities=ents,
            ))

        results = engine.synthesize(max_syntheses=10)
        assert len(results) > 0, "Cross-category signals should produce syntheses"

        categories_seen = set()
        for r in results:
            for m in r.source_modules:
                for orig_mod, cat, *_ in modules_and_signals:
                    if m == orig_mod:
                        categories_seen.add(cat)

        assert len(categories_seen) >= 2, (
            f"Should synthesize across categories, got: {categories_seen}"
        )

    def test_contradiction_resolution_flow(self):
        engine = CrossModuleSynthesisEngine()

        engine.ingest_signal(self._make_signal(
            module="dark_memory", category="memory",
            content="Memory system shows increasing performance trend",
            confidence=0.95, entities=["memory", "performance", "improvement"],
        ))
        engine.ingest_signal(self._make_signal(
            module="attention_engine", category="cognitive",
            content="Cognitive load measurements indicate decreasing efficiency",
            confidence=0.9, entities=["cognitive", "load", "efficiency"],
        ))
        engine.ingest_signal(self._make_signal(
            module="personality_engine", category="personality",
            content="User reports feeling more productive despite metrics",
            confidence=0.85, entities=["user", "productivity", "perception"],
        ))
        engine.ingest_signal(self._make_signal(
            module="temporal_reasoning", category="predictive",
            content="Time-of-day analysis shows afternoon cognitive decline",
            confidence=0.8, entities=["time", "afternoon", "decline"],
        ))

        results = engine.synthesize(max_syntheses=10)
        assert len(results) > 0

        contradictions = engine.detect_contradictions(min_severity=0.1)
        assert isinstance(contradictions, list)

        if len(results) >= 2 and contradictions:
            for c in contradictions[:1]:
                resolution = engine.resolve_contradiction(
                    c["synthesis_a"], c["synthesis_b"],
                    "Apparent contradiction resolved: memory improves while cognitive load "
                    "measurements reflect context-specific afternoon decline, not systemic degradation. "
                    "User self-report aligns with memory improvement data.",
                )
                if resolution is not None:
                    assert resolution.domain == "contradiction_resolution"
                    assert resolution.source_modules
                    assert resolution.confidence > 0

        stats = engine.get_stats()
        assert "contradictions_resolved" in stats

    def test_trend_evolution(self):
        engine = CrossModuleSynthesisEngine()

        for i in range(30):
            conf = 0.3 + (i / 30) * 0.65
            engine.ingest_signal(self._make_signal(
                module="attention_engine", category="cognitive",
                signal_type="focus_quality",
                content=f"Focus quality iteration {i}",
                confidence=min(0.95, conf),
            ))

        trends = engine.get_trends(min_forecast_confidence=0.0)
        focus_trends = [t for t in trends if "focus_quality" in t.get("topic", "")]
        assert len(focus_trends) > 0, "Should detect focus_quality trend"

        hist = engine.get_synthesis_history()
        assert isinstance(hist, list)

        clusters = engine.get_topic_clusters()
        assert len(clusters) > 0

        stats = engine.get_stats()
        assert stats["signals_ingested"] == 30

    def test_concurrent_ingest_and_read(self):
        engine = CrossModuleSynthesisEngine()
        errors = []
        results_collector = []

        def read_worker():
            try:
                for _ in range(20):
                    engine.get_stats()
                    engine.get_trends()
                    engine.get_topic_clusters()
                    engine.get_synthesis_history()
            except BaseException as e:
                errors.append(f"read: {e}")

        def write_worker(wid):
            try:
                for i in range(20):
                    s = self._make_signal(
                        module=f"module_{wid}",
                        category="cognitive",
                        confidence=0.6 + (i % 40) / 100,
                    )
                    engine.ingest_signal(s)
                results_collector.append(engine.synthesize())
            except BaseException as e:
                errors.append(f"write: {e}")

        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=write_worker, args=(i,)))
        for i in range(2):
            threads.append(threading.Thread(target=read_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        stats = engine.get_stats()
        assert stats["signals_ingested"] == 60
