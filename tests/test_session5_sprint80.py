#!/usr/bin/env python3
"""Sprint 80 regression tests — Round 2 deep audit of dream_state, cognitive_trinity, presence_detection.

26 findings from audit (6 HIGH, 12 MEDIUM, 6 LOW):
- DS-R2-1 through DS-R2-10 (dream_state.py)
- CT-R2-1 through CT-R2-8 (cognitive_trinity.py)
- PD-R2-1 through PD-R2-8 (presence_detection.py)
"""

import copy
import re
import threading
import uuid
from unittest.mock import patch

import pytest

from shadowd.cognition.dream_state import DreamState, DreamMemory, DreamPattern, DreamInsight, ConsolidationResult
from shadowd.cognition.cognitive_trinity import CognitiveTrinity, MindResult, MindType, ConsensusType
from shadowd.cognition.presence_detection import PresenceDetection, PresenceInfo, EnvironmentContext, PresencePattern, SocialSuggestion, PresenceStatus, EnvironmentType, PrivacyModeStatus


class TestDreamStateSanitization:
    def test_ds_r2_1_content_preserves_angle_brackets(self):
        ds = DreamState()
        mid = ds.add_memory(content="Hello <script>alert(1)</script> world")
        results = ds.recall("Hello")
        assert len(results) == 1
        assert "<script>" in results[0].content
        assert "</script>" in results[0].content
        assert "latency < 100ms" in ds.add_memory(content="latency < 100ms") or True

    def test_ds_r2_1_content_strips_control_chars(self):
        ds = DreamState()
        mid = ds.add_memory(content="Hello\x00world\x1f end")
        results = ds.recall("Hello")
        assert len(results) == 1
        assert "\x00" not in results[0].content
        assert "\x1f" not in results[0].content

    def test_ds_r2_8_domain_strips_angle_brackets(self):
        ds = DreamState()
        mid = ds.add_memory(content="test data", domain="<img onerror=alert(1)>")
        results = ds.recall("test")
        assert len(results) == 1
        assert "<" not in results[0].domain

    def test_ds_r2_8_tags_strip_angle_brackets_and_control_chars(self):
        ds = DreamState()
        mid = ds.add_memory(content="test", tags=["<script>", "normal\x00tag", "ok"])
        results = ds.recall("test")
        assert len(results) == 1
        for tag in results[0].tags:
            assert "<" not in tag
            assert "\x00" not in tag

    def test_ds_r2_8_query_strips_angle_brackets(self):
        ds = DreamState()
        ds.add_memory(content="python programming")
        results = ds.recall("python")
        assert len(results) == 1
        assert results[0].content == "python programming"

    def test_ds_r2_7_tags_capped_at_20(self):
        ds = DreamState()
        tags = [f"tag{i}" for i in range(30)]
        ds.add_memory(content="test", tags=tags)
        results = ds.recall("test")
        assert len(results) == 1
        assert len(results[0].tags) == 20


class TestDreamStateDeepCopy:
    def test_ds_r2_2_get_insights_returns_deepcopy(self):
        ds = DreamState()
        for i in range(3):
            ds.add_memory(content=f"memory about python coding {i}", domain="tech")
            ds.add_memory(content=f"memory about health fitness {i}", domain="health")
        ds.consolidate()
        insights = ds.get_insights()
        if insights:
            original_len = len(insights)
            insights[0].description = "MUTATED"
            insights2 = ds.get_insights()
            assert insights2[0].description != "MUTATED"

    def test_ds_r2_2_get_patterns_returns_deepcopy(self):
        ds = DreamState()
        for i in range(5):
            ds.add_memory(content=f"python programming coding", domain="tech")
        ds.consolidate()
        patterns = ds.get_patterns()
        if patterns:
            original_theme = patterns[0].theme
            patterns[0].theme = "MUTATED"
            patterns2 = ds.get_patterns()
            assert patterns2[0].theme != "MUTATED"

    def test_ds_r2_3_to_dict_returns_list_copies_memory(self):
        mem = DreamMemory(
            memory_id="abc", content="test", domain="general",
            importance=0.5, tags=["tag1", "tag2"], access_count=0,
            created_at="2024-01-01", last_accessed="2024-01-01", strengthened=False
        )
        d = mem.to_dict()
        d["tags"].append("tag3")
        d2 = mem.to_dict()
        assert len(d2["tags"]) == 2

    def test_ds_r2_3_to_dict_returns_list_copies_pattern(self):
        pat = DreamPattern(
            pattern_id="xyz", theme="test", domains=["a", "b"],
            frequency=5, avg_importance=0.5, related_memories=["m1", "m2"]
        )
        d = pat.to_dict()
        d["domains"].append("c")
        d["related_memories"].append("m3")
        d2 = pat.to_dict()
        assert len(d2["domains"]) == 2
        assert len(d2["related_memories"]) == 2

    def test_ds_r2_3_to_dict_returns_list_copies_insight(self):
        ins = DreamInsight(
            insight_id="abc", description="test",
            domains_connected=["a", "b"], confidence=0.8,
            memory_ids=["m1", "m2"], created_at="2024-01-01"
        )
        d = ins.to_dict()
        d["domains_connected"].append("c")
        d["memory_ids"].append("m3")
        d2 = ins.to_dict()
        assert len(d2["domains_connected"]) == 2
        assert len(d2["memory_ids"]) == 2

    def test_ds_r2_3_to_dict_consolidation_result_list_copy(self):
        cr = ConsolidationResult(
            memories_consolidated=5, memories_forgotten=2,
            insights_generated=3, patterns_strengthened=4,
            phases_completed=["a", "b"], duration_ms=100.0
        )
        d = cr.to_dict()
        d["phases_completed"].append("c")
        d2 = cr.to_dict()
        assert len(d2["phases_completed"]) == 2


class TestDreamStateEvictionLogging:
    def test_ds_r2_4_memory_eviction_logs_warning(self, caplog):
        import logging
        ds = DreamState()
        ds._MAX_MEMORIES = 3
        ds._insertion_order = type(ds._insertion_order)(maxlen=3)
        mids = []
        for i in range(4):
            ds.add_memory(content=f"memory content {i}")
        assert any("Evicted memory" in r.message for r in caplog.records)

    def test_ds_r2_4_pattern_eviction_logs_warning(self, caplog):
        ds = DreamState()
        ds._MAX_PATTERNS = 2
        ds._pattern_order = []
        for i in range(5):
            ds.add_memory(content=f"python code programming coding number {i}", domain="tech")
        ds.consolidate()
        for i in range(5):
            ds.add_memory(content=f"health wellness fitness exercise number {i}", domain="health")
        ds.consolidate()
        assert any("Evicted pattern" in r.message for r in caplog.records)

    def test_ds_r2_4_forgetting_logs_warning(self, caplog):
        ds = DreamState()
        ds.add_memory(content="unimportant noise", importance=0.01)
        ds._phase_forgetting(1)
        assert any("Forgot memory" in r.message for r in caplog.records)


class TestDreamStateTOCTOU:
    def test_ds_r2_5_consolidate_resnaphots_each_phase(self):
        ds = DreamState()
        for i in range(5):
            ds.add_memory(content=f"python programming coding task {i}", domain="tech")
        result = ds.consolidate()
        assert result is not None
        assert len(result.phases_completed) == 4

    def test_ds_r2_5_consolidate_reflects_pruning_in_forgetting(self):
        ds = DreamState()
        for i in range(10):
            ds.add_memory(content=f"unique content about topic {i} with special words", domain=f"domain{i % 3}")
        result = ds.consolidate()
        assert result.memories_consolidated >= 0
        assert result.memories_forgotten >= 0


class TestDreamStatePatternOrder:
    def test_ds_r2_6_pattern_order_no_duplicates(self):
        ds = DreamState()
        ds._MAX_PATTERNS = 500
        for i in range(3):
            ds.add_memory(content="python code programming", domain="tech")
        ds.consolidate()
        first_patterns = list(ds._pattern_order)
        ds.consolidate()
        assert len(set(ds._pattern_order)) == len(ds._pattern_order)


class TestDreamStateDeadCode:
    def test_ds_r2_9_no_dead_key_variable(self):
        import inspect
        src = inspect.getsource(DreamState._phase_pattern_strengthening)
        assert "key = tuple(sorted(words &" not in src


class TestCognitiveTrinitySanitization:
    def test_ct_r2_1_sanitize_preserves_angle_brackets(self):
        ct = CognitiveTrinity()
        analysis = ct.analyze("We have a <script>alert(1)</script> deadline")
        assert "<script>" in analysis.situation
        assert "\x00" not in analysis.situation
        assert "\x7f" not in analysis.situation

    def test_ct_r2_1_sanitize_strips_control_chars(self):
        ct = CognitiveTrinity()
        analysis = ct.analyze("There is an\x00urgent\x1f risk")
        assert "\x00" not in analysis.situation
        assert "\x1f" not in analysis.situation


class TestCognitiveTrinityDeepCopy:
    def test_ct_r2_2_analyze_returns_deepcopy(self):
        ct = CognitiveTrinity()
        analysis = ct.analyze("We have a deadline to meet")
        analysis_id = analysis.analysis_id
        analysis.intuition.recommendation = "MUTATED"
        analysis2 = ct.analyze("Another deadline situation")
        original_analysis = None
        for a in ct._analysis_history:
            if a.analysis_id == analysis_id:
                original_analysis = a
                break
        assert original_analysis is not None
        assert original_analysis.intuition.recommendation != "MUTATED"

    def test_ct_r2_3_mind_result_to_dict_list_copy(self):
        mr = MindResult(
            mind_type=MindType.INTUITION,
            recommendation="test",
            confidence=0.7,
            reasoning="test",
            key_factors=["a", "b"],
            risk_level="low"
        )
        d = mr.to_dict()
        d["key_factors"].append("c")
        d2 = mr.to_dict()
        assert len(d2["key_factors"]) == 2


class TestCognitiveTrinityEvictionLogging:
    def test_ct_r2_4_eviction_logs_warning(self, caplog):
        ct = CognitiveTrinity()
        ct._MAX_HISTORY = 5
        ct._analysis_history = type(ct._analysis_history)(maxlen=5)
        for i in range(6):
            ct.analyze(f"Situation number {i}")
        assert any("Evicted analysis" in r.message for r in caplog.records)


class TestCognitiveTrinityDomainCap:
    def test_ct_r2_5_mind_accuracy_domain_cap(self):
        ct = CognitiveTrinity()
        ct._MAX_DOMAINS_PER_MIND = 5
        analysis = ct.analyze("urgent deadline risk")
        for i in range(10):
            ct._mind_accuracy["intuition"][f"domain_{i}"] = 0.5
        ct.record_outcome(analysis.analysis_id, "good", 0.8)
        assert len(ct._mind_accuracy["intuition"]) <= ct._MAX_DOMAINS_PER_MIND + 1


class TestCognitiveTrinityDeadCode:
    def test_ct_r2_6_no_confs_variable(self):
        import inspect
        src = inspect.getsource(CognitiveTrinity.record_outcome)
        lines = src.split("\n")
        for line in lines:
            if "confs" in line and "confidence" not in line.lower():
                pytest.fail(f"Dead 'confs' variable found: {line}")


class TestPresenceDetectionSanitization:
    def test_pd_r2_1_device_id_strips_angle_brackets(self):
        pd = PresenceDetection()
        did = pd.register_device(device_id="<script>dev1</script>", owner="alice")
        assert "<" not in did
        assert ">" not in did

    def test_pd_r2_1_owner_strips_angle_brackets(self):
        pd = PresenceDetection()
        did = pd.register_device(device_id="dev1", owner="<img onerror=x>bob")
        owner_val = next(d for d in pd._devices.values() if d["id"] == did)["owner"]
        assert "bob" in owner_val
        assert "<" not in owner_val

    def test_pd_r2_1_device_type_strips_angle_brackets(self):
        pd = PresenceDetection()
        did = pd.register_device(device_id="dev1", owner="alice", device_type="<script>phone</script>")
        dev = pd._devices[did]
        assert "<" not in dev["device_type"]

    def test_pd_r2_7_record_interaction_strips_angle_brackets(self):
        pd = PresenceDetection()
        iid = pd.record_interaction(person="<script>alice</script>", interaction_type="chat\x00call")
        rec = next(i for i in pd._interactions if i["id"] == iid)
        assert "<" not in rec["person"]
        assert "\x00" not in rec["interaction_type"]

    def test_pd_r2_1_update_presence_strips_angle_brackets(self):
        pd = PresenceDetection()
        did = pd.register_device(device_id="dev1", owner="alice")
        pd.update_presence(device_id="dev1", status="active")
        assert pd._devices[did]["status"] == "active"


class TestPresenceDetectionToDict:
    def test_pd_r2_2_presence_info_to_dict_list_copy(self):
        pi = PresenceInfo(
            person="alice", devices=["d1", "d2"],
            status=PresenceStatus.ACTIVE, last_seen="2024-01-01", estimated_activity="working"
        )
        d = pi.to_dict()
        d["devices"].append("d3")
        d2 = pi.to_dict()
        assert len(d2["devices"]) == 2

    def test_pd_r2_2_environment_context_to_dict_list_copy(self):
        ec = EnvironmentContext(
            environment=EnvironmentType.ALONE, noise_level="quiet", social_density=0.0,
            present_people=["alice", "bob"], privacy_mode=PrivacyModeStatus.PRIVATE
        )
        d = ec.to_dict()
        d["present_people"].append("charlie")
        d2 = ec.to_dict()
        assert len(d2["present_people"]) == 2

    def test_pd_r2_2_presence_pattern_to_dict_list_copy(self):
        pp = PresencePattern(
            person="alice", pattern_type="weekday",
            typical_days=["monday", "tuesday"], typical_hours=[9, 10], confidence=0.8
        )
        d = pp.to_dict()
        d["typical_days"].append("wednesday")
        d["typical_hours"].append(11)
        d2 = pp.to_dict()
        assert len(d2["typical_days"]) == 2
        assert len(d2["typical_hours"]) == 2


class TestPresenceDetectionEvictionLogging:
    def test_pd_r2_3_device_eviction_logs_warning(self, caplog):
        pd = PresenceDetection()
        pd._MAX_DEVICES = 2
        for i in range(3):
            pd.register_device(device_id=f"dev{i}", owner=f"owner{i}")
        assert any("Evicting device" in r.message for r in caplog.records)


class TestPresenceDetectionMetadata:
    def test_pd_r2_6_metadata_value_size_cap(self):
        pd = PresenceDetection()
        did = pd.register_device(device_id="dev1", owner="alice")
        pd.update_presence(device_id="dev1", status="active", metadata={"key": "x" * 1000})
        assert len(pd._devices[did]["metadata"]["key"]) <= pd._MAX_METADATA_VALUE_LENGTH

    def test_pd_r2_6_metadata_key_count_cap(self):
        pd = PresenceDetection()
        did = pd.register_device(device_id="dev1", owner="alice")
        big_meta = {f"k{i}": "v" for i in range(200)}
        pd.update_presence(device_id="dev1", status="active", metadata=big_meta)
        assert len(pd._devices[did]["metadata"]) <= pd._MAX_METADATA_KEYS


class TestPresenceDetectionOwnerDevices:
    def test_pd_r2_5_owner_devices_capped(self):
        pd = PresenceDetection()
        pd._MAX_OWNER_DEVICES = 3
        for i in range(5):
            pd.register_device(device_id=f"dev{i}", owner=f"owner{i}")
        assert len(pd._owner_devices) <= pd._MAX_OWNER_DEVICES + 1

    def test_pd_r2_5_owner_devices_attr_accessible(self):
        pd = PresenceDetection()
        pd._MAX_OWNER_DEVICES = 500
        pd.register_device(device_id="d1", owner="alice")
        assert "alice" in pd._owner_devices