#!/usr/bin/env python3
"""Session 8 Sprint 57 — Remaining MEDIUM/LOW audit fixes for 5 cognitive modules."""

import threading
import uuid
from datetime import datetime, timezone, timedelta


class TestTemporalReasoningSprint57:
    """TR-T8..T11: window_days/within_hours validation, predictions_made counter, pattern IDs."""

    def test_analyze_trend_negative_window(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        result = tr.analyze_trend("test", window_days=-10)
        assert result.window_days == 1

    def test_analyze_trend_zero_window(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        result = tr.analyze_trend("test", window_days=0)
        assert result.window_days == 1

    def test_analyze_trend_huge_window(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        result = tr.analyze_trend("test", window_days=99999)
        assert result.window_days == 3650

    def test_predict_next_negative_within_hours(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        tr.record_event("test", timestamp=datetime.now(timezone.utc).isoformat())
        result = tr.predict_next("test", within_hours=-5)
        assert isinstance(result, list)

    def test_predict_next_zero_within_hours(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        tr.record_event("test", timestamp=datetime.now(timezone.utc).isoformat())
        result = tr.predict_next("test", within_hours=0)
        assert isinstance(result, list)

    def test_predictions_made_counter_consistent(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        ts = datetime.now(timezone.utc).isoformat()
        for i in range(10):
            tr.record_event("meeting", timestamp=ts)
        tr.detect_patterns("meeting")
        preds1 = tr.predict_next("meeting")
        stats1 = tr.get_stats()
        assert stats1["predictions_made"] == len(preds1)
        preds2 = tr.predict_next("meeting")
        stats2 = tr.get_stats()
        assert stats2["predictions_made"] == len(preds2)

    def test_pattern_id_12_char_hex(self):
        from shadowd.cognition.temporal_reasoning import TemporalReasoning
        tr = TemporalReasoning()
        base = datetime.now(timezone.utc) - timedelta(days=30)
        for i in range(10):
            ts = (base + timedelta(days=i*3)).isoformat()
            tr.record_event("standup", timestamp=ts)
        patterns = tr.detect_patterns("standup")
        for p in patterns:
            assert len(p.pattern_id.split("_")[1]) == 12


class TestEmpathyMappingSprint57:
    """EM-T6..T8: interaction ID 12-char, no 'cautious' dead state, approach validation."""

    def test_interaction_id_12_char(self):
        from shadowd.cognition.empathy_mapping import EmpathyMapping
        em = EmpathyMapping()
        pid = em.add_person("Alice", "friend")
        iid = em.record_interaction(pid, "text", 0.5)
        assert len(iid) == 12

    def test_no_cautious_dead_state(self):
        from shadowd.cognition.empathy_mapping import EmpathyMapping
        em = EmpathyMapping()
        pid = em.add_person("Bob", "friend")
        em.record_interaction(pid, "text", -0.6)
        em.record_interaction(pid, "text", -0.7)
        em.record_interaction(pid, "text", -0.8)
        profile = em.get_person_profile(pid)
        assert profile.formality_preference != "cautious"
        assert profile.formality_preference in ("mixed", "casual", "formal")

    def test_communication_style_no_cautious(self):
        from shadowd.cognition.empathy_mapping import EmpathyMapping
        em = EmpathyMapping()
        pid = em.add_person("Carol", "friend")
        for _ in range(3):
            em.record_interaction(pid, "text", -0.5)
        style = em.get_communication_style(pid)
        assert style.formality_level != "cautious"
        assert style.formality_level in ("mixed", "casual", "formal")


class TestPresenceDetectionSprint57:
    """PD-T8..T9: ALONE not UNKNOWN, metadata validation."""

    def test_zero_active_people_is_alone(self):
        from shadowd.cognition.presence_detection import PresenceDetection, EnvironmentType
        pd = PresenceDetection()
        pd.register_device("dev1", "owner1", "phone")
        pd.update_presence("dev1", "offline")
        env = pd.detect_environment()
        assert env.environment == EnvironmentType.ALONE

    def test_no_devices_alone(self):
        from shadowd.cognition.presence_detection import PresenceDetection, EnvironmentType
        pd = PresenceDetection()
        env = pd.detect_environment()
        assert env.environment == EnvironmentType.ALONE

    def test_metadata_dict_size_capped(self):
        from shadowd.cognition.presence_detection import PresenceDetection
        pd = PresenceDetection()
        did = pd.register_device("dev1", "owner1", "phone")
        big_meta = {f"key_{i}": f"val_{i}" for i in range(200)}
        pd.update_presence(did, "active", metadata=big_meta)
        dev = pd._devices.get(did)
        assert len(dev["metadata"]) <= 100


class TestCognitiveTrinitySprint57:
    """CT-T7..T8: record_outcome validates analysis_id, logs warning if not found."""

    def test_record_outcome_nonexistent_analysis(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        # Should not raise, just log warning
        ct.record_outcome("nonexistent_id", "test", 0.8)

    def test_record_outcome_found(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        analysis = ct.analyze("test situation")
        stats_before = ct.get_stats()
        ct.record_outcome(analysis.analysis_id, "test", 0.8)
        stats_after = ct.get_stats()
        assert stats_after["mind_accuracy"]["intuition"]["general"] != 0.5 or stats_after["mind_accuracy"]["logic"]["general"] != 0.5


class TestDreamStateSprint57:
    """DS-T8..T9: insight IDs 12-char, consolidation releases lock between phases."""

    def test_insight_id_12_char(self):
        from shadowd.cognition.dream_state import DreamState
        ds = DreamState()
        ds.add_memory("python programming and data science", "tech", 0.8)
        ds.add_memory("python machine learning model training", "career", 0.7)
        ds.consolidate()
        insights = ds.get_insights()
        for ins in insights:
            assert len(ins.insight_id) == 12

    def test_consolidation_releases_lock_between_phases(self):
        from shadowd.cognition.dream_state import DreamState
        ds = DreamState()
        for i in range(5):
            ds.add_memory(f"memory content about topic {i}", "tech", 0.6)
        # If lock wasn't released, this would deadlock
        result = ds.consolidate()
        assert len(result.phases_completed) == 4

    def test_pattern_not_created_for_every_word(self):
        from shadowd.cognition.dream_state import DreamState
        ds = DreamState()
        ds.add_memory("unique uncommon xyzabc test", "tech", 0.5)
        ds.add_memory("totally different content here", "tech", 0.5)
        ds.consolidate()
        stats = ds.get_stats()
        # With only 2 memories sharing no 3+ words, patterns should be minimal
        assert stats["total_patterns"] <= 2