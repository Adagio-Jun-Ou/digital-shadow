#!/usr/bin/env python3
"""Sprint 257 — Cross-Module Stress Tests + Deep Integration"""

import pytest
import sys
import os
import asyncio
import copy
import threading
import time
import random
import string

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SHADOW_OS_API_KEY", "")

from shadowd.cognition.dark_memory import DarkMemory, AtomType, MemoryTier
from shadowd.cognition.personality_engine import PersonalityEngine
from shadowd.cognition.stealth_layer import StealthLayer, DataCategory
from shadowd.cognition.daily_briefing import DailyBriefingEngine
from shadowd.cognition.thought_sense import ThoughtSense
from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
from shadowd.cognition.metacognition_engine import MetacognitionEngine
from shadowd.cognition.moral_reasoning import MoralReasoning
from shadowd.cognition.cognitive_pipeline import CognitivePipeline
from shadowd.cognition.device_context import DeviceContextEngine, DeviceType
from shadowd.cognition.attention_engine import AttentionEngine
from shadowd.cognition.shadow_arise import ShadowArise
from shadowd.cognition.shadow_summon import ShadowSummon
from shadowd.cognition.evolution_guide import EvolutionGuide


def _run_async(coro):
    return asyncio.run(coro)


class TestConcurrentMemoryWrites:
    def test_concurrent_store(self):
        dm = DarkMemory.__new__(DarkMemory)
        dm.__init__()
        errors = []

        def writer(i):
            try:
                for j in range(30):
                    _run_async(dm.store(
                        f"observation_{i}_{j}",
                        atom_type=AtomType.OBSERVATION,
                        tier=MemoryTier.WORKING,
                        domain="test",
                        importance=0.5,
                    ))
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_sequential_store_and_retrieve(self):
        dm = DarkMemory.__new__(DarkMemory)
        dm.__init__()
        atom_ids = []

        for i in range(20):
            atom = _run_async(dm.store(
                f"content_{i}",
                atom_type=AtomType.OBSERVATION,
                tier=MemoryTier.WORKING,
                domain="stress",
                importance=0.7,
            ))
            atom_ids.append(atom.atom_id)

        for aid in atom_ids:
            atom = _run_async(dm.retrieve(aid))
            assert atom is not None, f"Atom {aid} not retrieved"


class TestConcurrentPersonalityUpdates:
    def test_concurrent_behavior_recordings(self):
        pe = PersonalityEngine()
        errors = []

        def recorder(i):
            try:
                for j in range(30):
                    pe.record_behavior(
                        f"action_{i}_{j}",
                        {"detail": f"val_{i}_{j}"},
                    )
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=recorder, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        stats = pe.get_stats()
        assert stats["behaviors_recorded"] >= 100


class TestConcurrentStealthAccess:
    def test_concurrent_audit_reads(self):
        sl = StealthLayer()
        all_stats = []
        errors = []

        def reader(_):
            try:
                s = sl.guard.get_stats()
                all_stats.append(s)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(all_stats) == 10
        for s in all_stats:
            assert isinstance(s, dict)


class TestConcurrentDeviceContextIngest:
    def test_concurrent_iphone_ingest(self):
        dc = DeviceContextEngine()
        dc.register_device("phone_1", DeviceType.IPHONE, name="Test Phone")
        errors = []

        def ingest(i):
            try:
                dc.ingest_iphone("phone_1", {
                    "heart_rate": 60 + (i % 40),
                    "steps": 1000 + i * 100,
                    "battery_level": 0.8,
                })
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=ingest, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"


class TestConcurrentEvolutionGuide:
    def test_concurrent_add_observation(self):
        eg = EvolutionGuide()
        errors = []

        def observer(i):
            try:
                eg.add_observation(
                    description=f"User behavior pattern {i}",
                    domain="work",
                    observation_type="behavior",
                    confidence=0.6,
                )
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=observer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_generate_suggestions(self):
        eg = EvolutionGuide()
        eg.add_observation("User avoids code reviews", "work", "friction")
        eg.add_observation("Consistent late-night work", "work", "pattern")

        suggestions = eg.generate_suggestions(domain="work")
        assert isinstance(suggestions, list)


class TestPipelineCrossModule:
    def test_pipeline_emotion_metacognition(self):
        ts = ThoughtSense()
        ei = EmotionalIntelligence()
        me = MetacognitionEngine()
        mr = MoralReasoning()
        pipe = CognitivePipeline(
            thought_sense=ts,
            emotional_intelligence=ei,
            metacognition=me,
            moral_reasoning=mr,
        )

        result = pipe.process(
            "I'm frustrated about this unfair decision.",
            context="work",
        )
        assert result is not None
        assert isinstance(result, dict)
        assert "overall_confidence" in result

    def test_pipeline_neutral_text(self):
        ts = ThoughtSense()
        ei = EmotionalIntelligence()
        me = MetacognitionEngine()
        mr = MoralReasoning()
        pipe = CognitivePipeline(
            thought_sense=ts,
            emotional_intelligence=ei,
            metacognition=me,
            moral_reasoning=mr,
        )

        result = pipe.process(
            "What's the weather like today?",
            context="general",
        )
        assert result is not None
        assert isinstance(result, dict)


class TestDeviceContextBriefing:
    def test_briefing_with_device_context(self):
        dc = DeviceContextEngine()
        dc.register_device("phone_1", DeviceType.IPHONE, name="iPhone")
        dc.ingest_iphone("phone_1", {
            "heart_rate": 85,
            "steps": 5000,
            "battery_level": 0.6,
        })

        be = DailyBriefingEngine()
        briefing = _run_async(be.generate_briefing(device_context=dc))
        assert briefing is not None
        d = briefing.to_dict()
        assert "wellness_score" in d


class TestDeviceContextAttention:
    def test_attention_with_device_context(self):
        dc = DeviceContextEngine()
        dc.register_device("watch_1", DeviceType.APPLE_WATCH, name="Watch")
        dc.ingest_watch("watch_1", {
            "heart_rate": 110,
            "hrv": 25,
            "steps": 200,
            "sleep_hours": 3.5,
        })

        ae = AttentionEngine()
        state = ae.detect_focus_state(device_context=dc)
        assert state is not None


class TestShadowAriseSummon:
    def test_create_agent(self):
        arise = ShadowArise()
        agent = arise.arise(
            name="ResearchAgent",
            objective="Analyze data patterns",
            priority="high",
        )
        assert agent is not None
        assert agent.name == "ResearchAgent"

    def test_summon_abilities(self):
        summon = ShadowSummon()
        abilities = summon.get_abilities()
        assert isinstance(abilities, list)
        assert len(abilities) == 10

    def test_summon_voice(self):
        summon = ShadowSummon()
        result = summon.process_voice("what is the status")
        assert result is not None


class TestDeepCopyUnderConcurrency:
    def test_personality_deepcopy_safe(self):
        pe = PersonalityEngine()
        for i in range(20):
            pe.record_behavior(f"action_{i}", {"detail": f"val_{i}"})

        copies = []
        lock = threading.Lock()

        def copier(_):
            c = pe.get_profile()
            with lock:
                copies.append(c)

        threads = [threading.Thread(target=copier, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(copies) == 10
        for c in copies:
            assert isinstance(c, dict)

    def test_attention_get_priorities_independent(self):
        ae = AttentionEngine()
        ae.add_attention_item(topic="urgent task", level="high", domain="work")
        ae.add_attention_item(topic="side project", level="low", domain="personal")

        p1 = ae.get_priorities()
        p2 = ae.get_priorities()
        assert p1 is not p2


class TestStressLargeData:
    def test_large_memory_store(self):
        dm = DarkMemory.__new__(DarkMemory)
        dm.__init__()
        ids = []
        for i in range(100):
            atom = _run_async(dm.store(
                f"content_{i}" * 5,
                atom_type=AtomType.OBSERVATION,
                tier=MemoryTier.WORKING,
                domain="stress_test",
                importance=random.uniform(0.1, 1.0),
            ))
            ids.append(atom.atom_id)

        assert len(ids) == 100
        stats = dm.get_stats()
        assert stats["total_atoms"] >= 100

    def test_large_behavior_history(self):
        pe = PersonalityEngine()
        for i in range(500):
            pe.record_behavior(f"action_{i}", {"detail": f"val_{i}"})

        stats = pe.get_stats()
        assert stats["behaviors_recorded"] == 500


class TestStatsLockUnderConcurrency:
    def test_personality_stats_safe_read(self):
        pe = PersonalityEngine()
        for i in range(20):
            pe.record_behavior(f"act_{i}", {"detail": "d"})

        all_stats = []

        def reader(_):
            s = pe.get_stats()
            all_stats.append(s)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(all_stats) == 20
        for s in all_stats:
            assert isinstance(s, dict)
            assert "behaviors_recorded" in s

    def test_evolution_stats_safe_read(self):
        eg = EvolutionGuide()
        eg.add_observation("Test pattern", "work", "behavior")

        all_stats = []
        errors = []

        def reader(_):
            try:
                s = eg.get_stats()
                all_stats.append(s)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(all_stats) == 20
        for s in all_stats:
            assert isinstance(s, dict)


class TestMultiModuleChain:
    def test_thought_to_emotion_to_metacognition(self):
        ts = ThoughtSense()
        ei = EmotionalIntelligence()
        me = MetacognitionEngine()

        thought = ts.analyze("This deadline is stressing me out")
        assert thought is not None

        emotion = ei.detect_emotion(
            "This deadline is stressing me out",
            context="work",
        )
        assert emotion is not None

        biases = me.detect_biases(
            text="This deadline is stressing me out",
        )
        assert isinstance(biases, list)

    def test_personality_predict(self):
        pe = PersonalityEngine()
        pe.record_behavior("decision_made", {"speed": "fast"})

        options = [
            {"name": "Option A", "risk_level": 0.3},
            {"name": "Option B", "risk_level": 0.7},
        ]
        prediction = pe.predict_decision(options=options)
        assert prediction is not None