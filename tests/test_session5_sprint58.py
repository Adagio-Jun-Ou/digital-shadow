#!/usr/bin/env python3
"""Session 8 Sprint 58 — AttentionEngine module + ShadowOS wiring tests."""

import threading
from datetime import datetime, timezone, timedelta


class TestAttentionEngineBasics:
    def test_init(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        stats = ae.get_stats()
        assert stats["items_tracked"] == 0
        assert stats["focus_sessions"] == 0

    def test_add_attention_item(self):
        from shadowd.cognition.attention_engine import AttentionEngine, AttentionLevel
        ae = AttentionEngine()
        item_id = ae.add_attention_item("Review pull request", "high", "career", 0.8, 0.7)
        assert len(item_id) == 12
        priorities = ae.get_priorities()
        assert len(priorities) == 1
        assert priorities[0].level == AttentionLevel.HIGH

    def test_add_attention_item_empty_topic(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        import pytest
        with pytest.raises(ValueError):
            ae.add_attention_item("", "medium")

    def test_add_attention_item_invalid_level(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        import pytest
        with pytest.raises(ValueError):
            ae.add_attention_item("test", "ultra")

    def test_urgency_importance_clamped(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.add_attention_item("test", "medium", urgency=2.0, importance=-1.0)
        priorities = ae.get_priorities()
        assert priorities[0].urgency_score == 1.0
        assert priorities[0].importance_score == 0.0

    def test_add_many_items_eviction(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        for i in range(ae._MAX_ATTENTION_ITEMS + 10):
            ae.add_attention_item(f"item {i}", "low", urgency=0.1, importance=0.1)
        stats = ae.get_stats()
        assert stats["items_tracked"] <= ae._MAX_ATTENTION_ITEMS

    def test_priority_ordering(self):
        from shadowd.cognition.attention_engine import AttentionEngine, AttentionLevel
        ae = AttentionEngine()
        ae.add_attention_item("low task", "low", urgency=0.2, importance=0.2)
        ae.add_attention_item("critical task", "critical", urgency=0.9, importance=0.9)
        ae.add_attention_item("medium task", "medium", urgency=0.5, importance=0.5)
        priorities = ae.get_priorities()
        assert priorities[0].level == AttentionLevel.CRITICAL


class TestFocusSessions:
    def test_start_end_focus(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        sid = ae.start_focus("writing code")
        assert len(sid) == 12
        session = ae.end_focus(sid)
        assert session is not None
        assert session.ended_at is not None
        assert session.duration_minutes >= 0

    def test_start_focus_empty_topic(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        import pytest
        with pytest.raises(ValueError):
            ae.start_focus("")

    def test_end_focus_no_session(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        result = ae.end_focus()
        assert result is None

    def test_end_focus_wrong_id(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.start_focus("task1")
        result = ae.end_focus("wrong_id")
        assert result is None

    def test_start_focus_ends_previous(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.start_focus("task1")
        ae.start_focus("task2")
        stats = ae.get_stats()
        assert stats["focus_sessions"] == 1


class TestDistractions:
    def test_record_distraction(self):
        from shadowd.cognition.attention_engine import AttentionEngine, DistractionType
        ae = AttentionEngine()
        ae.start_focus("coding")
        eid = ae.record_distraction("notification", "Slack message", 30)
        assert len(eid) == 12
        profile = ae.get_attention_profile()
        assert profile.total_distractions == 1

    def test_record_distraction_invalid_type(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        import pytest
        with pytest.raises(ValueError):
            ae.record_distraction("alien_signal")

    def test_distraction_increments_interruptions(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.start_focus("deep work")
        ae.record_distraction("interruption")
        ae.record_distraction("notification")
        ae.end_focus()
        sessions = list(ae._focus_sessions)
        assert sessions[-1].interruptions == 2

    def test_recovery_time_clamped(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        eid = ae.record_distraction("fatigue", recovery_time_seconds=-10)
        events = list(ae._distractions)
        assert events[-1].recovery_time_seconds == 0.0


class TestFocusDetection:
    def test_idle_state(self):
        from shadowd.cognition.attention_engine import AttentionEngine, FocusState
        ae = AttentionEngine()
        state = ae.detect_focus_state()
        assert state == FocusState.IDLE

    def test_active_state(self):
        from shadowd.cognition.attention_engine import AttentionEngine, FocusState
        ae = AttentionEngine()
        ae.start_focus("important task")
        state = ae.detect_focus_state()
        assert state in (FocusState.ACTIVE, FocusState.DEEP_WORK, FocusState.SCANNING)

    def test_fatigued_state(self):
        from shadowd.cognition.attention_engine import AttentionEngine, FocusState
        ae = AttentionEngine()
        ae._fatigue_level = 0.9
        state = ae.detect_focus_state()
        assert state == FocusState.FATIGUED

    def test_distracted_state(self):
        from shadowd.cognition.attention_engine import AttentionEngine, FocusState
        ae = AttentionEngine()
        for _ in range(6):
            ae.record_distraction("notification")


class TestAttentionProfile:
    def test_profile_fields(self):
        from shadowd.cognition.attention_engine import AttentionEngine, FocusState
        ae = AttentionEngine()
        profile = ae.get_attention_profile()
        assert isinstance(profile.current_state, FocusState)
        assert 0.0 <= profile.cognitive_load <= 1.0
        assert 0.0 <= profile.fatigue_level <= 1.0

    def test_profile_after_session(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.start_focus("task")
        ae.end_focus()
        profile = ae.get_attention_profile()
        assert profile.avg_focus_span_minutes >= 0


class TestRecoverFocus:
    def test_recover_reduces_fatigue(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae._fatigue_level = 0.7
        ae.recover_focus(10.0)
        assert ae._fatigue_level < 0.7

    def test_recover_reduces_cognitive_load(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae._cognitive_load = 0.8
        ae.recover_focus(5.0)
        assert ae._cognitive_load < 0.8

    def test_recover_minutes_clamped(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.recover_focus(-5.0)
        ae.recover_focus(9999.0)


class TestSuggestAction:
    def test_suggest_idle(self):
        from shadowd.cognition.attention_engine import AttentionEngine, FocusState
        ae = AttentionEngine()
        result = ae.suggest_attention_action()
        assert result["current_state"] == FocusState.IDLE.value
        assert len(result["suggestions"]) > 0

    def test_suggest_with_priorities(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.add_attention_item("urgent bug", "critical", urgency=0.9, importance=0.9)
        result = ae.suggest_attention_action()
        assert result["top_priority"] is not None


class TestThreadSafety:
    def test_concurrent_adds(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        errors = []

        def add_items():
            try:
                for i in range(50):
                    ae.add_attention_item(f"item {i}", "medium")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_items) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestInputSanitization:
    def test_control_chars_stripped(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        iid = ae.add_attention_item("task\x00with\x01ctrl", "medium")
        priorities = ae.get_priorities()
        assert "\x00" not in priorities[0].topic

    def test_long_topic_truncated(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.add_attention_item("x" * 1000, "medium")
        priorities = ae.get_priorities()
        assert len(priorities[0].topic) <= ae._MAX_TOPIC_LENGTH

    def test_metadata_capped(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        big_meta = {f"k{i}": f"v{i}" for i in range(200)}
        iid = ae.add_attention_item("test", "medium", metadata=big_meta)
        assert len(ae._attention_items[iid].metadata) <= 100


class TestDeepWork:
    def test_deep_work_counted(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.start_focus("deep analysis")
        ae._current_session.started_at = (
            datetime.now(timezone.utc) - timedelta(minutes=30)
        ).isoformat()
        ae.end_focus()
        stats = ae.get_stats()
        assert stats["deep_work_sessions"] == 1

    def test_short_session_not_deep_work(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        ae.start_focus("quick check")
        ae.end_focus()
        stats = ae.get_stats()
        assert stats["deep_work_sessions"] == 0