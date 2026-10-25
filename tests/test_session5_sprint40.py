#!/usr/bin/env python3
"""Tests for PresenceDetection module (Session 5, Sprint 40)"""

import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from shadowd.cognition.presence_detection import (
    PresenceDetection,
    DeviceType,
    PresenceStatus,
    EnvironmentType,
    PrivacyModeStatus,
    PresenceInfo,
    EnvironmentContext,
    PresencePattern,
    SocialSuggestion,
)


@pytest.fixture
def pd():
    return PresenceDetection()


@pytest.fixture
def pd_with_devices(pd):
    pd.register_device("phone-alice", "Alice", "phone")
    pd.register_device("laptop-bob", "Bob", "laptop")
    pd.register_device("tablet-carol", "Carol", "tablet")
    return pd


# --- Registration ---

class TestRegisterDevice:
    def test_register_returns_device_id(self, pd):
        result = pd.register_device("dev1", "Alice", "phone")
        assert result == "dev1"

    def test_register_default_type(self, pd):
        did = pd.register_device("dev1", "Alice")
        assert pd._devices["dev1"]["device_type"] == "unknown"

    def test_register_invalid_type_becomes_other(self, pd):
        did = pd.register_device("dev1", "Alice", "toaster")
        assert pd._devices["dev1"]["device_type"] == "other"

    def test_register_all_valid_types(self, pd):
        for dt in DeviceType:
            did = pd.register_device(f"dev-{dt.value}", "Owner", dt.value)
            assert pd._devices[did]["device_type"] == dt.value

    def test_register_sets_offline_status(self, pd):
        pd.register_device("dev1", "Alice", "phone")
        assert pd._devices["dev1"]["status"] == PresenceStatus.OFFLINE.value

    def test_register_sets_last_seen(self, pd):
        pd.register_device("dev1", "Alice", "phone")
        assert pd._devices["dev1"]["last_seen"] is not None

    def test_register_tracks_owner_devices(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.register_device("d2", "Alice", "laptop")
        assert "d1" in pd._owner_devices["Alice"]
        assert "d2" in pd._owner_devices["Alice"]

    def test_register_duplicate_device_overwrites(self, pd):
        pd.register_device("dev1", "Alice", "phone")
        pd.register_device("dev1", "Bob", "laptop")
        assert pd._devices["dev1"]["owner"] == "Bob"

    def test_register_max_devices_evicts_oldest(self, pd):
        pd._MAX_DEVICES = 3
        pd.register_device("d1", "A", "phone")
        pd.register_device("d2", "B", "laptop")
        pd.register_device("d3", "C", "tablet")
        pd.register_device("d4", "D", "desktop")
        assert "d1" not in pd._devices
        assert "d4" in pd._devices
        assert len(pd._devices) == 3


# --- Update Presence ---

class TestUpdatePresence:
    def test_update_changes_status(self, pd_with_devices):
        pd_with_devices.update_presence("phone-alice", "active")
        assert pd_with_devices._devices["phone-alice"]["status"] == "active"

    def test_update_all_valid_statuses(self, pd_with_devices):
        for status in PresenceStatus:
            pd_with_devices.update_presence("phone-alice", status.value)
            assert pd_with_devices._devices["phone-alice"]["status"] == status.value

    def test_update_invalid_status_noop(self, pd_with_devices):
        pd_with_devices.update_presence("phone-alice", "active")
        pd_with_devices.update_presence("phone-alice", "flying")
        assert pd_with_devices._devices["phone-alice"]["status"] == "active"

    def test_update_unknown_device_noop(self, pd):
        pd.update_presence("nonexistent", "active")
        assert "nonexistent" not in pd._devices

    def test_update_updates_last_seen(self, pd_with_devices):
        old_ts = pd_with_devices._devices["phone-alice"]["last_seen"]
        time.sleep(0.01)
        pd_with_devices.update_presence("phone-alice", "active")
        assert pd_with_devices._devices["phone-alice"]["last_seen"] >= old_ts

    def test_update_stores_metadata(self, pd_with_devices):
        pd_with_devices.update_presence("phone-alice", "active", {"battery": 85})
        assert pd_with_devices._devices["phone-alice"]["metadata"]["battery"] == 85

    def test_update_appends_to_presence_log(self, pd_with_devices):
        pd_with_devices.update_presence("phone-alice", "active")
        assert len(pd_with_devices._presence_log) == 1
        assert pd_with_devices._presence_log[-1]["device_id"] == "phone-alice"


# --- Get Present People ---

class TestGetPresentPeople:
    def test_empty_when_no_devices(self, pd):
        assert pd.get_present_people() == []

    def test_returns_people_with_devices(self, pd_with_devices):
        pd_with_devices.update_presence("phone-alice", "active")
        people = pd_with_devices.get_present_people()
        names = [p.person for p in people]
        assert "Alice" in names
        assert "Bob" in names
        assert "Carol" in names

    def test_active_person_has_working_activity(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.update_presence("d1", "active")
        people = pd.get_present_people()
        alice = [p for p in people if p.person == "Alice"][0]
        assert alice.estimated_activity == "working"
        assert alice.status == PresenceStatus.ACTIVE

    def test_idle_person_has_leisure_activity(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.update_presence("d1", "idle")
        people = pd.get_present_people()
        alice = [p for p in people if p.person == "Alice"][0]
        assert alice.estimated_activity == "leisure"

    def test_away_person_has_unknown_activity(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.update_presence("d1", "away")
        people = pd.get_present_people()
        alice = [p for p in people if p.person == "Alice"][0]
        assert alice.estimated_activity == "unknown"

    def test_multi_device_owner_gets_best_status(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.register_device("d2", "Alice", "laptop")
        pd.update_presence("d1", "idle")
        pd.update_presence("d2", "active")
        people = pd.get_present_people()
        alice = [p for p in people if p.person == "Alice"][0]
        assert alice.status == PresenceStatus.ACTIVE
        assert len(alice.devices) == 2

    def test_presence_info_to_dict(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.update_presence("d1", "active")
        people = pd.get_present_people()
        d = people[0].to_dict()
        assert "person" in d
        assert "status" in d
        assert d["status"] == "active"


# --- Detect Environment ---

class TestDetectEnvironment:
    def test_no_people_alone(self, pd):
        ctx = pd.detect_environment()
        assert ctx.environment == EnvironmentType.ALONE
        assert ctx.noise_level == "quiet"
        assert ctx.social_density == 0.0

    def test_one_person_alone(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.update_presence("d1", "active")
        ctx = pd.detect_environment()
        assert ctx.environment == EnvironmentType.ALONE

    def test_two_to_three_family(self, pd):
        for i, name in enumerate(["A", "B", "C"]):
            pd.register_device(f"d{i}", name, "phone")
            pd.update_presence(f"d{i}", "active")
        ctx = pd.detect_environment()
        assert ctx.environment == EnvironmentType.WITH_FAMILY

    def test_four_to_six_colleagues(self, pd):
        for i in range(5):
            pd.register_device(f"d{i}", f"Person{i}", "laptop")
            pd.update_presence(f"d{i}", "active")
        ctx = pd.detect_environment()
        assert ctx.environment == EnvironmentType.WITH_COLLEAGUES

    def test_seven_to_ten_social(self, pd):
        for i in range(8):
            pd.register_device(f"d{i}", f"Person{i}", "phone")
            pd.update_presence(f"d{i}", "active")
        ctx = pd.detect_environment()
        assert ctx.environment == EnvironmentType.SOCIAL

    def test_noise_levels(self, pd):
        for i in range(7):
            pd.register_device(f"d{i}", f"P{i}", "phone")
            pd.update_presence(f"d{i}", "active")
        ctx = pd.detect_environment()
        assert ctx.noise_level == "loud"

    def test_social_density_capped_at_10(self, pd):
        for i in range(15):
            pd.register_device(f"d{i}", f"P{i}", "phone")
            pd.update_presence(f"d{i}", "active")
        ctx = pd.detect_environment()
        assert ctx.social_density == 10.0

    def test_idle_people_counted(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.update_presence("d1", "idle")
        ctx = pd.detect_environment()
        assert "Alice" in ctx.present_people

    def test_away_people_not_counted(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.update_presence("d1", "away")
        ctx = pd.detect_environment()
        assert "Alice" not in ctx.present_people

    def test_environment_context_to_dict(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd.update_presence("d1", "active")
        ctx = pd.detect_environment()
        d = ctx.to_dict()
        assert "environment" in d
        assert "noise_level" in d
        assert "social_density" in d
        assert "privacy_mode" in d


# --- Privacy Mode ---

class TestPrivacyMode:
    def test_alone_is_private(self, pd):
        pd.register_device("d1", "Me", "phone")
        pd.update_presence("d1", "active")
        with patch("shadowd.cognition.presence_detection.datetime") as mock_dt:
            mock_now = datetime(2026, 4, 11, 20, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mode = pd.get_privacy_mode()
            assert mode == PrivacyModeStatus.PRIVATE

    def test_family_is_social(self, pd):
        for i, name in enumerate(["A", "B"]):
            pd.register_device(f"d{i}", name, "phone")
            pd.update_presence(f"d{i}", "active")
        mode = pd.get_privacy_mode()
        assert mode == PrivacyModeStatus.SOCIAL

    def test_focus_mode_weekday_work_hours(self, pd):
        pd.register_device("d1", "Me", "phone")
        pd.update_presence("d1", "active")
        with patch("shadowd.cognition.presence_detection.datetime") as mock_dt:
            mock_now = datetime(2026, 4, 13, 10, 0, tzinfo=timezone.utc)  # Monday 10am
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mode = pd.get_privacy_mode()
            assert mode == PrivacyModeStatus.FOCUS

    def test_no_focus_mode_weekend(self, pd):
        pd.register_device("d1", "Me", "phone")
        pd.update_presence("d1", "active")
        with patch("shadowd.cognition.presence_detection.datetime") as mock_dt:
            mock_now = datetime(2026, 4, 11, 10, 0, tzinfo=timezone.utc)  # Saturday
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mode = pd.get_privacy_mode()
            assert mode != PrivacyModeStatus.FOCUS

    def test_no_focus_mode_evening(self, pd):
        pd.register_device("d1", "Me", "phone")
        pd.update_presence("d1", "active")
        with patch("shadowd.cognition.presence_detection.datetime") as mock_dt:
            mock_now = datetime(2026, 4, 13, 19, 0, tzinfo=timezone.utc)  # Monday 7pm
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mode = pd.get_privacy_mode()
            assert mode == PrivacyModeStatus.PRIVATE


# --- Record Interaction ---

class TestRecordInteraction:
    def test_returns_interaction_id(self, pd):
        iid = pd.record_interaction("Alice", "conversation", 15)
        assert isinstance(iid, str)
        assert len(iid) == 12

    def test_interaction_stored(self, pd):
        pd.record_interaction("Alice", "call", 30)
        assert len(pd._interactions) == 1
        assert pd._interactions[0]["person"] == "Alice"
        assert pd._interactions[0]["interaction_type"] == "call"
        assert pd._interactions[0]["duration_minutes"] == 30

    def test_multiple_interactions(self, pd):
        for i in range(5):
            pd.record_interaction("Alice", "chat", i)
        assert len(pd._interactions) == 5

    def test_timestamp_set(self, pd):
        pd.record_interaction("Alice", "chat")
        assert pd._interactions[0]["timestamp"] is not None


# --- Presence Patterns ---

class TestPresencePatterns:
    def test_no_patterns_without_data(self, pd):
        assert pd.get_presence_patterns() == []

    def test_pattern_for_specific_person(self, pd):
        pd.record_interaction("Alice", "chat", 10)
        pd.record_interaction("Alice", "chat", 10)
        patterns = pd.get_presence_patterns("Alice")
        assert len(patterns) > 0
        assert patterns[0].person == "Alice"

    def test_patterns_for_all(self, pd):
        pd.record_interaction("Alice", "chat", 10)
        pd.record_interaction("Bob", "call", 5)
        pd.record_interaction("Alice", "chat", 10)
        pd.record_interaction("Bob", "call", 5)
        patterns = pd.get_presence_patterns()
        persons = {p.person for p in patterns}
        assert "Alice" in persons
        assert "Bob" in persons

    def test_pattern_has_required_fields(self, pd):
        for _ in range(5):
            pd.record_interaction("Alice", "chat", 10)
        patterns = pd.get_presence_patterns("Alice")
        p = patterns[0]
        assert hasattr(p, "pattern_type")
        assert hasattr(p, "typical_days")
        assert hasattr(p, "typical_hours")
        assert hasattr(p, "confidence")

    def test_pattern_to_dict(self, pd):
        for _ in range(3):
            pd.record_interaction("Alice", "chat", 10)
        patterns = pd.get_presence_patterns("Alice")
        d = patterns[0].to_dict()
        assert "person" in d
        assert "pattern_type" in d
        assert "confidence" in d

    def test_no_patterns_for_unknown_person(self, pd):
        pd.record_interaction("Alice", "chat", 10)
        patterns = pd.get_presence_patterns("Ghost")
        assert patterns == []


# --- Social Suggestions ---

class TestSocialSuggestions:
    def test_suggestions_when_alone(self, pd):
        pd.register_device("d1", "Me", "phone")
        pd.update_presence("d1", "active")
        suggestions = pd.suggest_social_actions()
        actions = [s.action for s in suggestions]
        assert any("social interaction" in a.lower() for a in actions)

    def test_suggestion_for_absent_person(self, pd):
        pd.register_device("d1", "Alice", "phone")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
        pd._devices["d1"]["status"] = "away"
        pd._devices["d1"]["last_seen"] = old_ts
        suggestions = pd.suggest_social_actions()
        people = [s.person for s in suggestions if s.person == "Alice"]
        assert len(people) > 0

    def test_suggestion_urgency_levels(self, pd):
        pd.register_device("d1", "Alice", "phone")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
        pd._devices["d1"]["status"] = "away"
        pd._devices["d1"]["last_seen"] = old_ts
        suggestions = pd.suggest_social_actions()
        urgencies = [s.urgency for s in suggestions]
        assert "high" in urgencies

    def test_suggestion_to_dict(self, pd):
        pd.register_device("d1", "Me", "phone")
        pd.update_presence("d1", "active")
        suggestions = pd.suggest_social_actions()
        if suggestions:
            d = suggestions[0].to_dict()
            assert "action" in d
            assert "urgency" in d


# --- Stats ---

class TestStats:
    def test_empty_stats(self, pd):
        stats = pd.get_stats()
        assert stats["total_devices"] == 0
        assert stats["active_devices"] == 0
        assert stats["total_people"] == 0

    def test_stats_with_devices(self, pd_with_devices):
        pd_with_devices.update_presence("phone-alice", "active")
        stats = pd_with_devices.get_stats()
        assert stats["total_devices"] == 3
        assert stats["active_devices"] == 1
        assert stats["total_people"] == 3

    def test_stats_interaction_count(self, pd):
        pd.record_interaction("Alice", "chat", 10)
        pd.record_interaction("Bob", "call", 5)
        assert pd.get_stats()["interaction_count"] == 2

    def test_stats_max_devices(self, pd):
        assert pd.get_stats()["max_devices"] == 100

    def test_presence_log_size(self, pd_with_devices):
        pd_with_devices.update_presence("phone-alice", "active")
        pd_with_devices.update_presence("laptop-bob", "idle")
        assert pd_with_devices.get_stats()["presence_log_size"] == 2


# --- Thread Safety ---

class TestThreadSafety:
    def test_concurrent_registrations(self, pd):
        import threading
        errors = []

        def register(idx):
            try:
                pd.register_device(f"dev-{idx}", f"Owner-{idx}", "phone")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(pd._devices) == 50

    def test_concurrent_updates(self, pd):
        pd.register_device("d1", "Alice", "phone")
        errors = []

        def update(status):
            try:
                pd.update_presence("d1", status)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update, args=(s.value,))
            for s in PresenceStatus
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# --- Deque Eviction ---

class TestDequeEviction:
    def test_presence_log_evicts_at_max(self, pd):
        pd.register_device("d1", "Alice", "phone")
        for i in range(5001):
            pd.update_presence("d1", "active")
        assert len(pd._presence_log) == 5000

    def test_interaction_evicts_at_max(self, pd):
        for i in range(10001):
            pd.record_interaction("Alice", "chat", 1)
        assert len(pd._interactions) == 10000


# --- Offline Activity Estimation ---

class TestOfflineActivity:
    def test_offline_long_time_sleeping(self, pd):
        pd.register_device("d1", "Alice", "phone")
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        pd._devices["d1"]["last_seen"] = old_ts
        pd._devices["d1"]["status"] = "offline"
        people = pd.get_present_people()
        alice = [p for p in people if p.person == "Alice"][0]
        assert alice.estimated_activity == "sleeping"

    def test_offline_recent_unknown(self, pd):
        pd.register_device("d1", "Alice", "phone")
        pd._devices["d1"]["status"] = "offline"
        people = pd.get_present_people()
        alice = [p for p in people if p.person == "Alice"][0]
        assert alice.estimated_activity == "unknown"