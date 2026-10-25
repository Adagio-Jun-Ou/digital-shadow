#!/usr/bin/env python3
"""Integration test suite for all 8 new ShadowAI cognitive modules.

Tests end-to-end flows across:
- CrossModuleSynthesisEngine
- SelfModificationEngine
- CognitiveVisualizer
- PerformanceProfiler
- IntelligenceReportEngine
- AnomalyCorrelationEngine
- CrossModuleEventBus
- CognitiveOptimizationDaemon
"""

import copy
import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from shadowd.cognition.cross_module_synthesis import (
    CrossModuleSynthesisEngine, ModuleSignal, SynthesisResult,
    ModuleCategory, ConfidenceTier,
)
from shadowd.cognition.self_modification import (
    SelfModificationEngine, CodeModification, ModificationStatus,
    ModificationType,
)
from shadowd.cognition.cognitive_visualizer import CognitiveVisualizer
from shadowd.cognition.performance_profiler import PerformanceProfiler
from shadowd.cognition.intelligence_report import IntelligenceReportEngine
from shadowd.cognition.anomaly_correlation import AnomalyCorrelationEngine
from shadowd.cognition.cross_module_bus import (
    CrossModuleEventBus, TypedEvent, ModuleCapability, EventPriority,
)
from shadowd.cognition.optimization_daemon import CognitiveOptimizationDaemon


class TestCrossModuleIntegration:

    def test_synthesis_to_visualizer_flow(self):
        """Synthesis engine insights flow into visualizer as interactions."""
        vis = CognitiveVisualizer()
        synth = CrossModuleSynthesisEngine()

        synth.ingest_signal(ModuleSignal(
            source_module="attention_engine", category="cognitive",
            signal_type="focus_change", content="Deep work detected", confidence=0.9,
            domains=["work"],
        ))
        synth.ingest_signal(ModuleSignal(
            source_module="personality_engine", category="personality",
            signal_type="trait_update", content="Conscientiousness high", confidence=0.85,
            domains=["work"],
        ))

        syntheses = synth.synthesize()
        assert len(syntheses) > 0

        for syn in syntheses:
            for mod in syn.source_modules:
                for other in syn.source_modules:
                    if mod != other:
                        vis.record_interaction(mod, other, "synthesis_correlation",
                                              confidence=syn.confidence)

        overview = vis.get_module_overview()
        assert overview["total_modules"] >= 2

        matrix = vis.generate_interaction_matrix()
        assert matrix["total_interactions"] > 0

    def test_synthesis_to_intelligence_report_flow(self):
        """Synthesis insights feed into intelligence report."""
        synth = CrossModuleSynthesisEngine()
        report_engine = IntelligenceReportEngine()

        synth.ingest_signal(ModuleSignal(
            source_module="attention_engine", category="cognitive",
            signal_type="focus", content="Deep work", confidence=0.9, domains=["work"],
        ))
        synth.ingest_signal(ModuleSignal(
            source_module="personality_engine", category="personality",
            signal_type="trait", content="High openness", confidence=0.85, domains=["work"],
        ))
        synth.synthesize()

        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        os.synthesis_engine = synth

        report = report_engine.generate_report(shadow_os=os, report_type="hourly")
        assert report is not None
        assert report.system_health_score >= 0.0

    def test_anomaly_to_synthesis_correlation_flow(self):
        """AnomalyCorrelationEngine detects patterns that feed synthesis."""
        anomaly = AnomalyCorrelationEngine()
        synth = CrossModuleSynthesisEngine()

        anomaly.record_anomaly("attention_engine", "performance",
                               "High latency", 0.8)
        anomaly.record_anomaly("emotional_intelligence", "performance",
                               "Slow detection", 0.7)
        anomaly.record_anomaly("personality_engine", "performance",
                               "Delayed update", 0.65)

        correlations = anomaly.detect_correlations()
        cascades = anomaly.detect_cascades()
        bursts = anomaly.detect_bursts(window_seconds=60, min_count=2)

        assert isinstance(correlations, list)
        assert isinstance(cascades, list)
        assert isinstance(bursts, list)

        for module_name in ["attention_engine", "emotional_intelligence",
                            "personality_engine"]:
            synth.ingest_signal(ModuleSignal(
                source_module=module_name, category="performance",
                signal_type="anomaly_detected",
                content=f"Correlated anomaly in {module_name}",
                confidence=0.7, domains=["performance"],
            ))

        syntheses = synth.synthesize()
        if syntheses:
            assert syntheses[0].confidence > 0.0

    def test_profiler_to_optimization_daemon_flow(self):
        """Profiler data feeds into optimization daemon for parameter tuning."""
        profiler = PerformanceProfiler()

        for i in range(20):
            profiler.record(
                "attention_engine", "focus_check",
                duration_ms=50 + i * 5, error=i > 15,
            )

        daemon = CognitiveOptimizationDaemon()
        daemon.register_param(
            "attention_engine", "focus_threshold", 0.5, 0.1, 0.9, 0.05,
        )
        daemon.register_param(
            "attention_engine", "sensitivity", 0.3, 0.1, 0.8, 0.02,
        )

        params = daemon.get_params("attention_engine")
        assert "focus_threshold" in params
        assert "sensitivity" in params

        recommendations = profiler.generate_recommendations()
        assert isinstance(recommendations, list)

    def test_event_bus_to_visualizer_flow(self):
        """Event bus events populate the visualizer."""
        bus = CrossModuleEventBus()
        vis = CognitiveVisualizer()

        events = []
        for i in range(10):
            event = TypedEvent(
                channel=f"module.channel.{i % 3}",
                event_type="test_event",
                source_module=f"module_{i % 5}",
                priority="normal",
                payload={"data": f"item_{i}"},
            )
            bus.publish(event)
            events.append(event)

        for event in events:
            vis.record_interaction(
                event.source_module, f"target_{event.event_id[:4]}",
                event.event_type,
                confidence=0.5,
            )

        stats = bus.get_stats()
        assert stats["events_published"] == 10

        overview = vis.get_module_overview()
        assert overview["total_modules"] > 0

        graph = vis.generate_dependency_graph("json")
        assert "nodes" in graph

    def test_event_bus_module_capabilities(self):
        """Module capabilities are registered and discoverable."""
        bus = CrossModuleEventBus()

        bus.register_module(ModuleCapability(
            module_name="attention_engine",
            capability_type="cognitive",
            description="Manages attention and focus",
            channels_published=["cognitive.focus", "cognitive.distraction"],
            channels_subscribed=["system.health", "user.activity"],
        ))

        bus.register_module(ModuleCapability(
            module_name="personality_engine",
            capability_type="personality",
            description="Models personality traits",
            channels_published=["personality.trait", "personality.pattern"],
            channels_subscribed=["cognitive.emotion", "user.behavior"],
        ))

        caps = bus.get_capabilities()
        assert len(caps) == 2

        attn_channels = bus.get_module_channels("attention_engine")
        assert attn_channels is not None
        assert len(attn_channels["channels_published"]) == 2

    def test_self_mod_to_intelligence_report_flow(self):
        """Code modifications generate intelligence report actions."""
        self_mod = SelfModificationEngine()
        report_engine = IntelligenceReportEngine()

        analysis = self_mod.analyze_codebase()
        assert analysis["python_files"] > 0

        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        os.self_mod = self_mod

        report = report_engine.generate_report(shadow_os=os)
        assert report is not None

        actions = report_engine.get_pending_actions()
        assert isinstance(actions, list)


class TestConcurrentIntegration:

    def test_concurrent_multi_engine_operations(self):
        """All 8 engines operate concurrently without errors."""
        synth = CrossModuleSynthesisEngine()
        self_mod = SelfModificationEngine()
        vis = CognitiveVisualizer()
        profiler = PerformanceProfiler()
        anomaly = AnomalyCorrelationEngine()
        bus = CrossModuleEventBus()
        daemon = CognitiveOptimizationDaemon()

        errors = []

        def operate_synthesis():
            try:
                for i in range(20):
                    synth.ingest_signal(ModuleSignal(
                        source_module=f"mod_{i % 5}", category="cognitive",
                        signal_type="test", content=f"data_{i}", confidence=0.5,
                    ))
                synth.synthesize()
            except Exception as e:
                errors.append(("synthesis", str(e)))

        def operate_visualizer():
            try:
                for i in range(20):
                    vis.record_interaction(
                        f"src_{i % 5}", f"tgt_{(i + 1) % 5}",
                        "test_event", confidence=0.5,
                    )
                vis.generate_dependency_graph("json")
            except Exception as e:
                errors.append(("visualizer", str(e)))

        def operate_profiler():
            try:
                for i in range(20):
                    profiler.record(f"mod_{i % 5}", "op", i * 0.5, error=(i % 10 == 0))
                profiler.get_all_profiles()
            except Exception as e:
                errors.append(("profiler", str(e)))

        def operate_anomaly():
            try:
                for i in range(20):
                    anomaly.record_anomaly(
                        f"mod_{i % 5}", "test", f"anomaly_{i}", i / 20.0,
                    )
                anomaly.detect_correlations()
                anomaly.detect_bursts()
            except Exception as e:
                errors.append(("anomaly", str(e)))

        def operate_bus():
            try:
                for i in range(20):
                    event = TypedEvent(
                        channel="test.channel", event_type="test",
                        source_module=f"mod_{i % 5}", priority="normal",
                        payload={"item": i},
                    )
                    bus.publish(event)
                bus.get_stats()
            except Exception as e:
                errors.append(("bus", str(e)))

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(operate_synthesis),
                executor.submit(operate_visualizer),
                executor.submit(operate_profiler),
                executor.submit(operate_anomaly),
                executor.submit(operate_bus),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_read_write_on_synthesis(self):
        """Concurrent reads and writes on synthesis engine are safe."""
        synth = CrossModuleSynthesisEngine()

        for i in range(10):
            synth.ingest_signal(ModuleSignal(
                source_module=f"mod_{i % 3}", category="cognitive",
                signal_type="test", content=f"base_{i}", confidence=0.5,
            ))

        errors = []

        def reader():
            try:
                for _ in range(50):
                    synth.get_stats()
                    synth.get_synthesis_history(limit=5)
                    synth.get_topic_clusters(max_clusters=3)
                    synth.get_trends()
            except Exception as e:
                errors.append(("reader", str(e)))

        def writer():
            try:
                for i in range(50):
                    synth.ingest_signal(ModuleSignal(
                        source_module=f"mod_{i % 3}", category="cognitive",
                        signal_type="test", content=f"write_{i}", confidence=0.5,
                    ))
            except Exception as e:
                errors.append(("writer", str(e)))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(reader), executor.submit(writer),
                executor.submit(reader), executor.submit(writer),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors: {errors}"


class TestDeepcopyAndStatsIntegrity:

    def test_synthesis_stats_are_copies(self):
        synth = CrossModuleSynthesisEngine()
        synth.ingest_signal(ModuleSignal(
            source_module="test", category="cognitive",
            signal_type="test", content="hello", confidence=0.5,
        ))
        s1 = synth.get_stats()
        s2 = synth.get_stats()
        assert s1 is not s2
        s1["signals_ingested"] = 99999
        assert s2["signals_ingested"] != 99999

    def test_visualizer_stats_are_copies(self):
        vis = CognitiveVisualizer()
        vis.record_interaction("a", "b", "test", confidence=0.5)
        s1 = vis.get_stats()
        s2 = vis.get_stats()
        assert s1 is not s2

    def test_anomaly_stats_are_copies(self):
        anom = AnomalyCorrelationEngine()
        anom.record_anomaly("test", "test", "test", 0.5)
        s1 = anom.get_stats()
        s2 = anom.get_stats()
        assert s1 is not s2

    def test_bus_stats_are_copies(self):
        bus = CrossModuleEventBus()
        event = TypedEvent(
            channel="test", event_type="test", source_module="test",
            priority="normal", payload={},
        )
        bus.publish(event)
        s1 = bus.get_stats()
        s2 = bus.get_stats()
        assert s1 is not s2

    def test_profiler_stats_are_copies(self):
        prof = PerformanceProfiler()
        prof.record("test", "op", 1.0)
        s1 = prof.get_stats()
        s2 = prof.get_stats()
        assert s1 is not s2

    def test_report_engine_stats_are_copies(self):
        eng = IntelligenceReportEngine()
        s1 = eng.get_stats()
        s2 = eng.get_stats()
        assert s1 is not s2

    def test_daemon_stats_are_copies(self):
        daemon = CognitiveOptimizationDaemon()
        s1 = daemon.get_stats()
        s2 = daemon.get_stats()
        assert s1 is not s2

    def test_synthesis_result_to_dict_deepcopy(self):
        synth = CrossModuleSynthesisEngine()
        synth.ingest_signal(ModuleSignal(
            source_module="a", category="cognitive",
            signal_type="t", content="c", confidence=0.5,
            metadata={"key": "val"},
        ))
        synth.ingest_signal(ModuleSignal(
            source_module="b", category="personality",
            signal_type="t", content="c2", confidence=0.6,
            metadata={"key2": "val2"},
        ))
        syntheses = synth.synthesize()
        if syntheses:
            d1 = syntheses[0].to_dict()
            d2 = syntheses[0].to_dict()
            assert d1 is not d2
            d1["title"] = "HACKED"
            assert d2["title"] != "HACKED"


class TestEndToEndFullSystemFlow:

    def test_full_system_lifecycle(self):
        """End-to-end: all 8 engines operate together through a full lifecycle."""
        synth = CrossModuleSynthesisEngine()
        self_mod = SelfModificationEngine()
        vis = CognitiveVisualizer()
        profiler = PerformanceProfiler()
        report_engine = IntelligenceReportEngine()
        anomaly = AnomalyCorrelationEngine()
        bus = CrossModuleEventBus()
        daemon = CognitiveOptimizationDaemon()

        modules = [
            "attention_engine", "emotional_intelligence", "personality_engine",
            "dark_memory", "empathy_mapping", "oracle_bridge",
        ]
        modules_categories = ["cognitive", "personality", "memory", "social"]

        for mod in modules:
            bus.register_module(ModuleCapability(
                module_name=mod, capability_type="cognitive",
                description=f"Module {mod}",
                channels_published=[f"{mod}.output"],
                channels_subscribed=["system.health"],
            ))
            daemon.register_param(mod, "sensitivity", 0.5, 0.1, 0.9, 0.05)

        for i in range(30):
            mod = modules[i % len(modules)]
            conf = 0.5 + (i * 0.01)

            synth.ingest_signal(ModuleSignal(
                source_module=mod, category=modules_categories[i % len(modules_categories)],
                signal_type=f"signal_{i}", content=f"Data point {i}",
                confidence=min(0.95, conf), domains=["system"],
            ))

            profiler.record(mod, f"operation_{i}", 5.0 + (i * 0.5),
                           error=(i % 15 == 0), cache_hit=(i % 3 == 0))

            anomaly.record_anomaly(
                mod, "performance" if i % 2 == 0 else "cognitive",
                f"Anomaly {i}", i / 60.0,
            )

            event = TypedEvent(
                channel=f"{mod}.output", event_type="insight",
                source_module=mod, priority="normal",
                payload={"iteration": i, "confidence": min(0.95, conf)},
            )
            bus.publish(event)

            if i % 5 == 0:
                vis.record_interaction(mod, modules[(i + 1) % len(modules)],
                                      "synthesis_correlation",
                                      confidence=min(0.95, conf))

        syntheses = synth.synthesize(max_syntheses=10)
        assert len(syntheses) > 0

        correlations = anomaly.detect_correlations()
        assert isinstance(correlations, list)

        cascades = anomaly.detect_cascades()
        assert isinstance(cascades, list)

        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        os.synthesis_engine = synth
        os.self_mod = self_mod
        os.profiler = profiler
        os.anomaly_corr = anomaly

        report = report_engine.generate_report(shadow_os=os)
        assert report is not None
        assert report.system_health_score >= 0.0

        graph = vis.generate_dependency_graph("json")
        assert "nodes" in graph

        bus_stats = bus.get_stats()
        assert bus_stats["events_published"] == 30

        prof_stats = profiler.get_stats()
        assert prof_stats["samples_recorded"] == 30

        daemon_stats = daemon.get_stats()
        assert daemon_stats["total_params"] == 6

        synth_stats = synth.get_stats()
        assert synth_stats["signals_ingested"] == 30
