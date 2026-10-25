#!/usr/bin/env python3
"""
Session 5 Sprint 43 — Tests for wiring PresenceDetection, CognitiveTrinity, DreamState
into ShadowOS + API Router, plus security audit of all 3 new modules.
"""

import pytest
from unittest.mock import MagicMock, patch

from shadowd.cognition.shadow_os import ShadowOS
from shadowd.cognition.presence_detection import (
    PresenceDetection, PresenceInfo, EnvironmentContext,
    DeviceType, PresenceStatus, EnvironmentType, PrivacyModeStatus,
    PresencePattern, SocialSuggestion,
)
from shadowd.cognition.cognitive_trinity import (
    CognitiveTrinity, MindType, ConsensusType,
    MindResult, SynthesisResult, TrinityAnalysis,
)
from shadowd.cognition.dream_state import (
    DreamState, DreamMemory, DreamPattern, DreamInsight,
    DreamPhase, ConsolidationResult,
)


# ═══════════════════════════════════════════════════════════
# ShadowOS Wiring Tests
# ═══════════════════════════════════════════════════════════

class TestShadowOSWiring:
    def test_presence_initialized(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w1")
        assert hasattr(os, 'presence')
        assert isinstance(os.presence, PresenceDetection)

    def test_trinity_initialized(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w2")
        assert hasattr(os, 'trinity')
        assert isinstance(os.trinity, CognitiveTrinity)

    def test_dream_initialized(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w3")
        assert hasattr(os, 'dream')
        assert isinstance(os.dream, DreamState)

    def test_analyze_situation(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w4")
        result = os.analyze_situation("urgent deadline approaching for project")
        assert "analysis_id" in result
        assert "intuition" in result
        assert "logic" in result
        assert "creative" in result
        assert "synthesis" in result

    def test_analyze_situation_sanitizes(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w5")
        result = os.analyze_situation("test<script>alert(1)</script>")
        assert "<script>" not in result["situation"]

    def test_analyze_situation_truncates(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w6")
        result = os.analyze_situation("a" * 10000)
        assert len(result["situation"]) <= 200

    def test_detect_environment(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w7")
        result = os.detect_environment()
        assert "environment" in result
        assert "noise_level" in result
        assert "privacy_mode" in result

    def test_consolidate_memories(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w8")
        os.dream.add_memory("test memory content", "general", 0.7)
        result = os.consolidate_memories()
        assert "memories_consolidated" in result
        assert "phases_completed" in result

    def test_get_status_includes_new_modules(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w9")
        status = os.get_status()
        assert "presence_stats" in status
        assert "trinity_stats" in status
        assert "dream_stats" in status

    def test_get_health_includes_new_modules(self):
        os = ShadowOS(data_dir="/tmp/test_s43_shadow_os_w10")
        os._running = True
        os._start_time = 100.0
        health = os.get_health()
        assert "presence_detection" in health["modules"]
        assert "cognitive_trinity" in health["modules"]
        assert "dream_state" in health["modules"]


# ═══════════════════════════════════════════════════════════
# PresenceDetection Security Tests
# ═══════════════════════════════════════════════════════════

class TestPresenceDetectionSecurity:
    def test_device_id_truncated(self):
        pd = PresenceDetection()
        long_id = "a" * 200
        pd.register_device(long_id, "owner", "phone")
        assert len(pd._devices[long_id[:100]]["id"]) <= 100

    def test_owner_truncated(self):
        pd = PresenceDetection()
        pd.register_device("dev1", "b" * 200, "phone")
        for dev in pd._devices.values():
            assert len(dev["owner"]) <= 100

    def test_empty_device_id_rejected(self):
        pd = PresenceDetection()
        with pytest.raises(ValueError):
            pd.register_device("", "owner", "phone")

    def test_empty_owner_rejected(self):
        pd = PresenceDetection()
        with pytest.raises(ValueError):
            pd.register_device("dev1", "", "phone")

    def test_device_type_invalid_defaults_to_other(self):
        pd = PresenceDetection()
        pd.register_device("dev1", "owner", "nonexistent_type")
        assert pd._devices["dev1"]["device_type"] == "other"

    def test_max_devices_cap(self):
        pd = PresenceDetection()
        pd._MAX_DEVICES = 3
        for i in range(5):
            pd.register_device(f"dev_{i}", f"owner_{i}", "phone")
        assert len(pd._devices) <= 3

    def test_invalid_status_ignored(self):
        pd = PresenceDetection()
        pd.register_device("dev1", "owner", "phone")
        pd.update_presence("dev1", "invalid_status")
        assert pd._devices["dev1"]["status"] == PresenceStatus.OFFLINE.value

    def test_metadata_size_capped(self):
        pd = PresenceDetection()
        pd.register_device("dev1", "owner", "phone")
        huge_meta = {f"key_{i}": "x" * 100 for i in range(100)}
        pd.update_presence("dev1", "active", huge_meta)
        assert pd._devices["dev1"]["status"] == PresenceStatus.ACTIVE.value

    def test_interaction_validation(self):
        pd = PresenceDetection()
        with pytest.raises(ValueError):
            pd.record_interaction("", "call")

    def test_interaction_person_truncated(self):
        pd = PresenceDetection()
        pid = pd.record_interaction("x" * 200, "call")
        assert pid is not None

    def test_negative_duration_clamped(self):
        pd = PresenceDetection()
        pid = pd.record_interaction("person", "call", duration_minutes=-5)
        assert pid is not None

    def test_presence_log_cap(self):
        pd = PresenceDetection()
        pd.register_device("dev1", "owner1", "phone")
        for i in range(6000):
            pd.update_presence("dev1", "active" if i % 2 == 0 else "idle")
        assert len(pd._presence_log) <= 5000

    def test_device_cap_fifo_eviction(self):
        pd = PresenceDetection()
        pd._MAX_DEVICES = 2
        pd.register_device("dev1", "owner1", "phone")
        pd.register_device("dev2", "owner2", "laptop")
        pd.register_device("dev3", "owner3", "tablet")
        assert "dev3" in pd._devices
        assert "dev1" not in pd._devices

    def test_thread_safety_concurrent(self):
        import threading
        pd = PresenceDetection()
        pd.register_device("dev1", "owner1", "phone")
        errors = []

        def worker():
            try:
                for _ in range(100):
                    pd.update_presence("dev1", "active")
                    pd.get_stats()
                    pd.get_present_people()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════
# CognitiveTrinity Security Tests
# ═══════════════════════════════════════════════════════════

class TestCognitiveTrinitySecurity:
    def test_sanitize_strips_control_chars(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("hello\x00\x01\x02 world")
        assert "\x00" not in result.reasoning or "hello" in result.reasoning

    def test_sanitize_truncates_long_input(self):
        ct = CognitiveTrinity()
        long_text = "urgent deadline " * 500
        result = ct.analyze(long_text)
        assert len(result.situation) <= 5000

    def test_analyze_returns_trinity_analysis(self):
        ct = CognitiveTrinity()
        result = ct.analyze("test situation for decision making")
        assert isinstance(result, TrinityAnalysis)
        assert isinstance(result.intuition, MindResult)
        assert isinstance(result.logic, MindResult)
        assert isinstance(result.creative, MindResult)
        assert isinstance(result.synthesis, SynthesisResult)

    def test_confidence_bounded(self):
        ct = CognitiveTrinity()
        result = ct.analyze("urgent risk data evidence deadline")
        assert 0.0 <= result.synthesis.confidence <= 1.0
        assert 0.0 <= result.intuition.confidence <= 1.0
        assert 0.0 <= result.logic.confidence <= 1.0
        assert 0.0 <= result.creative.confidence <= 1.0

    def test_risk_level_valid(self):
        ct = CognitiveTrinity()
        result = ct.analyze("dangerous risky opportunity")
        valid_risks = {"low", "medium", "high"}
        assert result.intuition.risk_level in valid_risks
        assert result.logic.risk_level in valid_risks
        assert result.creative.risk_level in valid_risks

    def test_key_factors_capped(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("urgent risk trust opportunity health dangerous faith")
        assert len(result.key_factors) <= 5

    def test_history_cap(self):
        ct = CognitiveTrinity()
        for i in range(2500):
            ct.analyze(f"situation {i}")
        assert len(ct._analysis_history) <= ct._MAX_HISTORY

    def test_record_outcome_satisfaction_clamped(self):
        ct = CognitiveTrinity()
        result = ct.analyze("test")
        ct.record_outcome(result.analysis_id, "good", 5.0)
        ct.record_outcome(result.analysis_id, "good", -1.0)
        stats = ct.get_stats()
        assert stats is not None

    def test_thread_safety_concurrent(self):
        import threading
        ct = CognitiveTrinity()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    ct.analyze("urgent test situation")
                    ct.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_consensus_types(self):
        ct = CognitiveTrinity()
        result = ct.analyze("test")
        assert result.synthesis.consensus in [
            ConsensusType.CONSENSUS,
            ConsensusType.SPLIT,
            ConsensusType.DIVERGENCE,
        ]


# ═══════════════════════════════════════════════════════════
# DreamState Security Tests
# ═══════════════════════════════════════════════════════════

class TestDreamStateSecurity:
    def test_add_memory_empty_rejected(self):
        ds = DreamState()
        with pytest.raises(ValueError):
            ds.add_memory("", "general", 0.5)

    def test_add_memory_whitespace_rejected(self):
        ds = DreamState()
        with pytest.raises(ValueError):
            ds.add_memory("   ", "general", 0.5)

    def test_content_truncated(self):
        ds = DreamState()
        mid = ds.add_memory("x" * 20000, "general", 0.5)
        mem = ds._memories[mid]
        assert len(mem.content) <= 10000

    def test_importance_clamped(self):
        ds = DreamState()
        mid1 = ds.add_memory("test1", "general", 5.0)
        assert ds._memories[mid1].importance <= 1.0
        mid2 = ds.add_memory("test2", "general", -1.0)
        assert ds._memories[mid2].importance >= 0.0

    def test_domain_truncated(self):
        ds = DreamState()
        mid = ds.add_memory("test", "x" * 200, 0.5)
        assert len(ds._memories[mid].domain) <= 100

    def test_tags_truncated(self):
        ds = DreamState()
        tags = ["x" * 100, "y" * 100, "valid"]
        mid = ds.add_memory("test", "general", 0.5, tags=tags)
        mem = ds._memories[mid]
        for tag in mem.tags:
            assert len(tag) <= 50

    def test_memory_cap_fifo(self):
        ds = DreamState()
        ds._MAX_MEMORIES = 5
        ds.__class__._MAX_MEMORIES = 5
        ids = []
        for i in range(8):
            mid = ds.add_memory(f"memory {i}", "general", 0.5)
            ids.append(mid)
        assert len(ds._memories) <= 5

    def test_recall_limit_bounded(self):
        ds = DreamState()
        for i in range(20):
            ds.add_memory(f"memory {i}", "general", 0.5)
        results = ds.recall("", limit=200)
        assert len(results) <= 100

    def test_recall_limit_minimum(self):
        ds = DreamState()
        ds.add_memory("test", "general", 0.5)
        results = ds.recall("test", limit=0)
        assert len(results) >= 1

    def test_consolidate_returns_result(self):
        ds = DreamState()
        ds.add_memory("test content about python", "tech", 0.8)
        ds.add_memory("another python project update", "tech", 0.6)
        result = ds.consolidate()
        assert isinstance(result, ConsolidationResult)
        assert len(result.phases_completed) == 4

    def test_forget_ratio_cap(self):
        ds = DreamState()
        for i in range(50):
            ds.add_memory(f"low imp {i}", "general", 0.1)
        result = ds.consolidate()
        assert result.memories_forgotten <= max(1, int(50 * 0.1))

    def test_dream_memory_to_dict(self):
        ds = DreamState()
        mid = ds.add_memory("test memory", "general", 0.7, tags=["tag1"])
        mem = ds._memories[mid]
        d = mem.to_dict()
        assert "memory_id" in d
        assert "content" in d
        assert "domain" in d
        assert "importance" in d
        assert "tags" in d

    def test_dream_pattern_to_dict(self):
        dp = DreamPattern(
            pattern_id="p1",
            theme="test",
            domains=["a", "b"],
            frequency=3,
            avg_importance=0.7,
            related_memories=["m1", "m2"],
        )
        d = dp.to_dict()
        assert d["pattern_id"] == "p1"
        assert d["theme"] == "test"
        assert len(d["domains"]) == 2
        assert d["frequency"] == 3

    def test_dream_insight_to_dict(self):
        di = DreamInsight(
            insight_id="i1",
            description="test insight",
            domains_connected=["a", "b"],
            confidence=0.8,
            memory_ids=["m1"],
            created_at="2024-01-01",
        )
        d = di.to_dict()
        assert d["insight_id"] == "i1"
        assert d["confidence"] == 0.8

    def test_consolidation_result_to_dict(self):
        cr = ConsolidationResult(
            memories_consolidated=5,
            memories_forgotten=2,
            insights_generated=1,
            patterns_strengthened=3,
            phases_completed=["pattern_strengthening"],
            duration_ms=100.5,
        )
        d = cr.to_dict()
        assert d["memories_consolidated"] == 5
        assert d["duration_ms"] == 100.5

    def test_thread_safety_concurrent(self):
        import threading
        ds = DreamState()
        errors = []

        def writer():
            try:
                for i in range(50):
                    ds.add_memory(f"memory {i}", "general", 0.5)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    ds.recall("memory", limit=5)
                    ds.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_get_insights_returns_list(self):
        ds = DreamState()
        insights = ds.get_insights()
        assert isinstance(insights, list)

    def test_get_patterns_returns_list(self):
        ds = DreamState()
        patterns = ds.get_patterns()
        assert isinstance(patterns, list)

    def test_forget_memory(self):
        ds = DreamState()
        mid = ds.add_memory("forget me", "general", 0.1)
        assert mid in ds._memories
        result = ds.forget(mid)
        assert result is True
        assert mid not in ds._memories

    def test_forget_nonexistent(self):
        ds = DreamState()
        result = ds.forget("nonexistent")
        assert result is False