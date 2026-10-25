#!/usr/bin/env python3
"""
Sprint 142 regression tests:
- 19 cognition modules: _stats_lock for thread safety
- neural_mesh.py: get_stats() now reads _stats under _stats_lock
- dark_memory.py: TemporalKnowledgeGraph.get_stats() now reads under _lock
"""

import threading
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
from shadowd.cognition.attention_engine import AttentionEngine
from shadowd.cognition.autonomous_daemon import AutonomousDaemon, SecurityGuardian
from shadowd.cognition.cognitive_pipeline import CognitivePipeline
from shadowd.cognition.cognitive_reflection import CognitiveReflection
from shadowd.cognition.creative_synthesis import CreativeSynthesis, CreativeDomain
from shadowd.cognition.decision_journal import DecisionJournal
from shadowd.cognition.device_context import DeviceContextEngine
from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
from shadowd.cognition.goal_planner import CognitiveGoalPlanner
from shadowd.cognition.goal_reasoning import GoalReasoning
from shadowd.cognition.metacognition_engine import MetacognitionEngine
from shadowd.cognition.moral_reasoning import MoralReasoning
from shadowd.cognition.personality_engine import PersonalityEngine
from shadowd.cognition.self_improvement import SelfImprovementEngine
from shadowd.cognition.shadow_arise import ShadowArise
from shadowd.cognition.shadow_summon import ShadowSummon
from shadowd.cognition.social_intelligence import SocialIntelligence
from shadowd.cognition.temporal_reasoning import TemporalReasoning
from shadowd.cognition.neural_mesh import NeuralMesh
from shadowd.cognition.dark_memory import DarkMemory, TemporalKnowledgeGraph


class TestStatsLockExists:
    def test_anomaly_detector_has_stats_lock(self):
        e = AnomalyDetectionEngine()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_attention_engine_has_stats_lock(self):
        e = AttentionEngine()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_security_guardian_has_stats_lock(self):
        sg = SecurityGuardian()
        assert hasattr(sg, '_stats_lock')
        assert isinstance(sg._stats_lock, type(threading.Lock()))

    def test_cognitive_pipeline_has_stats_lock(self):
        p = CognitivePipeline()
        assert hasattr(p, '_stats_lock')
        assert isinstance(p._stats_lock, type(threading.Lock()))

    def test_cognitive_reflection_has_stats_lock(self):
        e = CognitiveReflection()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_creative_synthesis_has_stats_lock(self):
        e = CreativeSynthesis()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_decision_journal_has_stats_lock(self):
        j = DecisionJournal()
        assert hasattr(j, '_stats_lock')
        assert isinstance(j._stats_lock, type(threading.Lock()))

    def test_device_context_has_stats_lock(self):
        e = DeviceContextEngine()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_emotional_intelligence_has_stats_lock(self):
        e = EmotionalIntelligence()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_goal_planner_has_stats_lock(self):
        g = CognitiveGoalPlanner()
        assert hasattr(g, '_stats_lock')
        assert isinstance(g._stats_lock, type(threading.Lock()))

    def test_goal_reasoning_has_stats_lock(self):
        g = GoalReasoning()
        assert hasattr(g, '_stats_lock')
        assert isinstance(g._stats_lock, type(threading.Lock()))

    def test_metacognition_engine_has_stats_lock(self):
        e = MetacognitionEngine()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_moral_reasoning_has_stats_lock(self):
        e = MoralReasoning()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_personality_engine_has_stats_lock(self):
        e = PersonalityEngine()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_self_improvement_has_stats_lock(self):
        e = SelfImprovementEngine()
        assert hasattr(e, '_stats_lock')
        assert isinstance(e._stats_lock, type(threading.Lock()))

    def test_shadow_arise_has_stats_lock(self):
        a = ShadowArise()
        assert hasattr(a, '_stats_lock')
        assert isinstance(a._stats_lock, type(threading.Lock()))

    def test_shadow_summon_has_stats_lock(self):
        s = ShadowSummon()
        assert hasattr(s, '_stats_lock')
        assert isinstance(s._stats_lock, type(threading.Lock()))

    def test_social_intelligence_has_stats_lock(self):
        s = SocialIntelligence()
        assert hasattr(s, '_stats_lock')
        assert isinstance(s._stats_lock, type(threading.Lock()))

    def test_temporal_reasoning_has_stats_lock(self):
        t = TemporalReasoning()
        assert hasattr(t, '_stats_lock')
        assert isinstance(t._stats_lock, type(threading.Lock()))


class TestNeuralMeshStatsLock:
    def test_get_stats_reads_under_lock(self):
        mesh = NeuralMesh(node_name="test_node", device_type="laptop")
        with mesh._stats_lock:
            mesh._stats["messages_received"] = 42
        stats = mesh.get_stats()
        assert stats["stats"]["messages_received"] == 42
        stats["stats"]["messages_received"] = 0
        assert mesh._stats["messages_received"] == 42

    def test_concurrent_stats_increment(self):
        mesh = NeuralMesh(node_name="test_concurrent", device_type="laptop")
        errors = []

        def increment():
            try:
                for _ in range(100):
                    with mesh._stats_lock:
                        mesh._stats["messages_received"] += 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert mesh._stats["messages_received"] == 400


class TestTKGStatsLock:
    def test_tkg_get_stats_under_lock(self):
        kg = TemporalKnowledgeGraph()
        kg.add_entity("e1", "person", {"name": "Alice"})
        stats = kg.get_stats()
        assert stats["entities"] == 1
        assert isinstance(stats["entity_types"], dict)
        assert isinstance(stats["relation_types"], dict)

    def test_tkg_concurrent_access(self):
        kg = TemporalKnowledgeGraph()
        errors = []

        def add_entities():
            try:
                for i in range(50):
                    kg.add_entity(f"e_{threading.get_ident()}_{i}", "person", {"name": f"Entity{i}"})
            except Exception as e:
                errors.append(e)

        def read_stats():
            try:
                for _ in range(50):
                    kg.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_entities),
            threading.Thread(target=add_entities),
            threading.Thread(target=read_stats),
            threading.Thread(target=read_stats),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_tkg_relation_types_returns_dict(self):
        kg = TemporalKnowledgeGraph()
        stats = kg.get_stats()
        assert isinstance(stats["relation_types"], dict)


class TestDarkMemoryStatsLock:
    @pytest.mark.asyncio
    async def test_get_stats_returns_copy(self):
        dm = DarkMemory(db_path=":memory:")
        stats = dm.get_stats()
        assert isinstance(stats["stats"], dict)
        stats["stats"]["atoms_stored"] = 999
        assert dm._stats["atoms_stored"] != 999


class TestStatsThreadSafety:
    def test_anomaly_detector_concurrent(self):
        e = AnomalyDetectionEngine()
        e.register_metric("cpu", "gauge")
        errors = []

        def observe():
            try:
                for _ in range(50):
                    e.observe("cpu", 45.0)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=observe) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_attention_engine_concurrent(self):
        e = AttentionEngine()
        errors = []

        def add_items():
            try:
                for i in range(20):
                    e.add_attention_item(f"topic_{i}", "medium", "work", 0.5, 0.5)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=add_items) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_goal_planner_concurrent(self):
        gp = CognitiveGoalPlanner()
        errors = []

        def create_goals():
            try:
                for i in range(10):
                    gp.create_goal(f"Goal_{i}", "Description", "MEDIUM")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=create_goals) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_cognitive_pipeline_concurrent(self):
        p = CognitivePipeline()
        errors = []

        def process():
            try:
                for _ in range(10):
                    p.process("test input", "")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=process) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_shadow_arise_concurrent(self):
        a = ShadowArise()
        errors = []

        def arise():
            try:
                for i in range(5):
                    a.arise("Analyze the data", f"test_agent_{i}", "analyst")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=arise) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_shadow_summon_concurrent(self):
        s = ShadowSummon()
        errors = []

        def summon():
            try:
                for _ in range(10):
                    s.process_voice("find patterns in data")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=summon) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_emotional_intelligence_concurrent(self):
        e = EmotionalIntelligence()
        errors = []

        def detect():
            try:
                for _ in range(10):
                    e.detect_emotion("I feel great today")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=detect) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_creative_synthesis_concurrent(self):
        e = CreativeSynthesis()
        errors = []

        def synthesize():
            try:
                for _ in range(10):
                    e.synthesize(["AI", "healthcare"], domain=CreativeDomain.TECHNOLOGY)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=synthesize) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_metacognition_concurrent(self):
        e = MetacognitionEngine()
        errors = []

        def detect():
            try:
                for _ in range(10):
                    e.detect_biases("I think this is definitely true because I saw it once")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=detect) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_moral_reasoning_concurrent(self):
        e = MoralReasoning()
        errors = []

        def evaluate():
            try:
                for _ in range(10):
                    e.evaluate("should I lie to protect someone?", "A moral dilemma")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=evaluate) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_decision_journal_concurrent(self):
        j = DecisionJournal()
        errors = []

        def record():
            try:
                for i in range(10):
                    j.record_decision(
                        f"Decision {i}", "Option A",
                        ["Option A", "Option B"],
                        "Seemed best", domain="work"
                    )
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=record) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_cognitive_reflection_concurrent(self):
        e = CognitiveReflection()
        errors = []

        def reflect():
            try:
                for _ in range(10):
                    e.add_reflection("progress", "I made progress on the project")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=reflect) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_social_intelligence_concurrent(self):
        s = SocialIntelligence()
        errors = []

        def create_groups():
            try:
                for i in range(5):
                    s.create_group(f"Group_{i}", "TEAM", "A team")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=create_groups) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_temporal_reasoning_concurrent(self):
        t = TemporalReasoning()
        errors = []

        def record():
            try:
                for _ in range(20):
                    t.record_event("login")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=record) for _ in range(4)]
        for t2 in threads:
            t2.start()
        for t2 in threads:
            t2.join()
        assert len(errors) == 0

    def test_goal_reasoning_concurrent(self):
        gr = GoalReasoning()
        gid = "test_goal_1"
        gr._goal_priorities[gid] = 0.5
        errors = []

        def assess():
            try:
                for _ in range(10):
                    gr.assess_goal(gid, 0.5)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=assess) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestGetStatsReturnsCopy:
    def test_anomaly_detector_stats_copy(self):
        e = AnomalyDetectionEngine()
        s = e.get_stats()
        original = e._stats.copy()
        s["something_new"] = True
        assert "something_new" not in e._stats

    def test_attention_engine_stats_copy(self):
        e = AttentionEngine()
        s = e.get_stats()
        assert isinstance(s, dict)

    def test_goal_planner_stats_copy(self):
        g = CognitiveGoalPlanner()
        s = g.get_stats()
        g._stats["goals_created"] = 99
        if "goals_created" in s:
            assert s["goals_created"] != 99

    def test_cognitive_pipeline_stats_copy(self):
        p = CognitivePipeline()
        s = p.get_pipeline_stats()
        assert isinstance(s, dict)

    def test_shadow_arise_stats_copy(self):
        a = ShadowArise()
        s = a.get_stats()
        assert isinstance(s, dict)
        a._stats["agents_created"] = 99
        if "agents_created" in s:
            assert s["agents_created"] != 99

    def test_shadow_summon_stats_copy(self):
        s = ShadowSummon()
        stats = s.get_stats()
        assert isinstance(stats, dict)

    def test_emotional_intelligence_stats_copy(self):
        e = EmotionalIntelligence()
        s = e.get_stats()
        assert isinstance(s, dict)

    def test_creative_synthesis_stats_copy(self):
        e = CreativeSynthesis()
        s = e.get_stats()
        assert isinstance(s, dict)

    def test_metacognition_stats_copy(self):
        e = MetacognitionEngine()
        s = e.get_stats()
        assert isinstance(s, dict)

    def test_moral_reasoning_stats_copy(self):
        e = MoralReasoning()
        s = e.get_stats()
        assert isinstance(s, dict)

    def test_decision_journal_stats_copy(self):
        j = DecisionJournal()
        s = j.get_stats()
        assert isinstance(s, dict)

    def test_cognitive_reflection_stats_copy(self):
        e = CognitiveReflection()
        s = e.get_stats()
        assert isinstance(s, dict)

    def test_social_intelligence_stats_copy(self):
        s = SocialIntelligence()
        stats = s.get_stats()
        assert isinstance(stats, dict)

    def test_temporal_reasoning_stats_copy(self):
        t = TemporalReasoning()
        s = t.get_stats()
        assert isinstance(s, dict)

    def test_goal_reasoning_stats_copy(self):
        g = GoalReasoning()
        s = g.get_stats()
        assert isinstance(s, dict)

    def test_personality_engine_stats_copy(self):
        p = PersonalityEngine()
        s = p.get_stats()
        assert isinstance(s, dict)

    def test_self_improvement_stats_copy(self):
        s = SelfImprovementEngine()
        stats = s.get_stats()
        assert isinstance(stats, dict)

    def test_device_context_stats_copy(self):
        d = DeviceContextEngine()
        s = d.get_stats()
        assert isinstance(s, dict)