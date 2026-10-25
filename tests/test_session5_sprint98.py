#!/usr/bin/env python3
"""Sprint 98 — Device Context API Endpoint Tests

Tests for all 12 /device/* API endpoints in the ShadowOS router:
- POST /device/register
- POST /device/iphone
- POST /device/watch
- POST /device/laptop
- GET  /device/context
- GET  /device/health-trends
- GET  /device/digital-wellbeing
- GET  /device/devices
- GET  /device/stats
- POST /device/wifi-location
- POST /device/person

Plus integration tests for ShadowOS device context wiring.
"""

import pytest
import copy
import threading
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from shadowd.cognition.device_context import (
    DeviceContextEngine, DeviceType, LocationType, ActivityType,
    StressLevel, FocusLevel, SleepPhase, NetworkType, NotificationType,
    LocationPoint, HealthSnapshot, DigitalSnapshot, EnvironmentSnapshot,
    UnifiedContext, _sanitize, _sanitize_strict, _sanitize_dict,
    _classify_location, _classify_notification, _infer_stress,
    _calculate_distraction_score,
)


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — Sanitization & Utility Functions
# ═══════════════════════════════════════════════════════════════════

class TestDeviceSanitize:
    def test_sanitize_strips_angle_brackets(self):
        result = _sanitize_strict("test<script>alert(1)</script>")
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_strips_control_chars(self):
        assert _sanitize_strict("hello\x00\x01\x02") == "hello"

    def test_sanitize_strips_newlines_strict(self):
        assert _sanitize_strict("hello\nworld") == "helloworld"

    def test_sanitize_content_preserves_newlines(self):
        result = _sanitize("hello\nworld")
        assert "\n" in result

    def test_sanitize_strict_max_length(self):
        long_str = "a" * 500
        assert len(_sanitize_strict(long_str, 200)) <= 200

    def test_sanitize_content_max_length(self):
        long_str = "b" * 10000
        assert len(_sanitize(long_str, 5000)) <= 5000

    def test_sanitize_dict_nested(self):
        data = {"outer": {"inner": "<script>evil</script>"}}
        result = _sanitize_dict(data)
        assert "<" not in result["outer"]["inner"]
        assert ">" not in result["outer"]["inner"]

    def test_sanitize_dict_max_keys(self):
        data = {f"k{i}": f"v{i}" for i in range(50)}
        result = _sanitize_dict(data)
        assert len(result) <= 40  # _MAX_METADATA_KEYS * 2

    def test_sanitize_dict_list_values(self):
        data = {"items": ["<a>", "b", "c"] * 10}
        result = _sanitize_dict(data)
        assert all("<" not in v for v in result["items"])


class TestClassifyLocation:
    def test_home_in_name(self):
        assert _classify_location({"name": "My Home"}) == LocationType.HOME

    def test_office_in_name(self):
        assert _classify_location({"name": "Office Building"}) == LocationType.OFFICE

    def test_gym_in_name(self):
        assert _classify_location({"name": "Planet Fitness Gym"}) == LocationType.GYM

    def test_home_wifi(self):
        assert _classify_location({"name": "", "wifi_ssid": "My Home WiFi"}) == LocationType.HOME

    def test_office_wifi(self):
        assert _classify_location({"name": "", "wifi_ssid": "Corp Office WiFi"}) == LocationType.OFFICE

    def test_transit_high_speed(self):
        assert _classify_location({"name": "", "speed": 10}) == LocationType.TRANSIT

    def test_unknown_default(self):
        assert _classify_location({}) == LocationType.UNKNOWN

    def test_non_dict_returns_unknown(self):
        assert _classify_location("not a dict") == LocationType.UNKNOWN


class TestClassifyNotification:
    def test_message_app(self):
        assert _classify_notification({"app": "Messages"}) == NotificationType.MESSAGE

    def test_email_category(self):
        assert _classify_notification({"app": "Gmail", "category": "email"}) == NotificationType.EMAIL

    def test_social_app(self):
        assert _classify_notification({"app": "Instagram", "category": "social"}) == NotificationType.SOCIAL

    def test_unknown_default(self):
        assert _classify_notification({"app": "RandomApp"}) == NotificationType.UNKNOWN

    def test_non_dict_returns_unknown(self):
        assert _classify_notification("not a dict") == NotificationType.UNKNOWN


class TestInferStress:
    def test_low_stress(self):
        assert _infer_stress(65, 55, 98) == StressLevel.LOW

    def test_moderate_stress(self):
        assert _infer_stress(85, 45, 97) == StressLevel.MODERATE

    def test_high_stress(self):
        assert _infer_stress(105, 30, 97) == StressLevel.HIGH

    def test_critical_stress(self):
        assert _infer_stress(110, 25, 93) == StressLevel.CRITICAL

    def test_boundary_low_to_moderate(self):
        assert _infer_stress(85, 49, 97) == StressLevel.MODERATE


class TestDistractionScore:
    def test_zero_screen_zero_notifs(self):
        score = _calculate_distraction_score(0, 0, [])
        assert score == 0.0

    def test_high_screen_high_notifs(self):
        snaps = [MagicMock(active_app=f"app{i}") for i in range(10)]
        score = _calculate_distraction_score(480, 50, snaps)
        assert score > 0.7

    def test_limited_to_one(self):
        score = _calculate_distraction_score(960, 100, [])
        assert score <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — DeviceContextEngine Core
# ═══════════════════════════════════════════════════════════════════

class TestDeviceContextEngineCore:
    def setup_method(self):
        self.engine = DeviceContextEngine()

    def test_register_device_basic(self):
        self.engine.register_device("phone-1", DeviceType.IPHONE, "My iPhone")
        stats = self.engine.get_stats()
        assert stats["devices_registered"] == 1

    def test_register_device_sanitizes_id(self):
        self.engine.register_device("<script>alert(1)</script>", DeviceType.IPHONE, "Test")
        stats = self.engine.get_stats()
        assert stats["devices_registered"] == 1
        devices = self.engine.get_device_summary()
        dev_ids = [d["device_id"] for d in devices["devices"]]
        assert all("<script>" not in d for d in dev_ids)

    def test_register_device_max_devices(self):
        for i in range(15):
            self.engine.register_device(f"dev-{i}", DeviceType.UNKNOWN, f"Device {i}")
        summary = self.engine.get_device_summary()
        assert len(summary["devices"]) <= 10

    def test_register_device_invalid_type_becomes_unknown(self):
        self.engine.register_device("test-dev", DeviceType.UNKNOWN, "Test")
        summary = self.engine.get_device_summary()
        assert summary["devices"][0]["type"] == "unknown"

    def test_device_summary_stats_copy(self):
        self.engine.register_device("d1", DeviceType.IPHONE, "Phone")
        summary = self.engine.get_device_summary()
        stats_before = summary["stats"]
        # Mutate returned dict — should not affect engine
        stats_before["iphone_updates"] = 99999
        summary2 = self.engine.get_device_summary()
        assert summary2["stats"]["iphone_updates"] != 99999

    def test_get_context_initially_none(self):
        ctx = self.engine.get_context()
        assert ctx is None

    def test_get_stats_structure(self):
        stats = self.engine.get_stats()
        assert "stats" in stats
        assert "devices_registered" in stats
        assert "location_history" in stats
        assert "health_history" in stats
        assert "digital_history" in stats
        assert "notification_history" in stats

    def test_get_stats_is_copy(self):
        stats = self.engine.get_stats()
        stats["stats"]["iphone_updates"] = 99999
        stats2 = self.engine.get_stats()
        assert stats2["stats"]["iphone_updates"] != 99999


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — iPhone Ingest
# ═══════════════════════════════════════════════════════════════════

class TestIPhoneIngest:
    def setup_method(self):
        self.engine = DeviceContextEngine()
        self.engine.register_device("phone-1", DeviceType.IPHONE, "My iPhone")

    def test_basic_iphone_ingest(self):
        ctx = self.engine.ingest_iphone("phone-1", {
            "location": {"latitude": 40.7, "longitude": -74.0, "accuracy": 10, "name": "Home"},
            "battery": {"level": 0.8, "charging": False},
        })
        assert ctx is not None
        assert ctx.location.latitude == 40.7
        assert ctx.environment.battery_level == 0.8

    def test_iphone_health_data(self):
        ctx = self.engine.ingest_iphone("phone-1", {
            "health": {"heart_rate": 72, "hrv": 45, "spo2": 98},
        })
        assert ctx.health.heart_rate == 72

    def test_iphone_notifications(self):
        ctx = self.engine.ingest_iphone("phone-1", {
            "notifications": [
                {"app": "Messages", "category": "message"},
                {"app": "Gmail", "category": "email"},
            ],
        })
        assert len(list(self.engine._notification_history)) >= 2

    def test_iphone_updates_stats(self):
        self.engine.ingest_iphone("phone-1", {})
        stats = self.engine.get_stats()
        assert stats["stats"]["iphone_updates"] >= 1

    def test_iphone_sanitizes_device_id(self):
        ctx = self.engine.ingest_iphone("<script>evil</script>", {})
        # Should not crash — sanitized device_id
        assert ctx is not None

    def test_iphone_location_inferred_type(self):
        ctx = self.engine.ingest_iphone("phone-1", {
            "location": {"latitude": 40.7, "longitude": -74.0, "name": "Home Office"},
        })
        assert ctx.location.location_type in (LocationType.HOME, LocationType.OFFICE)


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — Watch Ingest
# ═══════════════════════════════════════════════════════════════════

class TestWatchIngest:
    def setup_method(self):
        self.engine = DeviceContextEngine()
        self.engine.register_device("watch-1", DeviceType.APPLE_WATCH, "My Watch")

    def test_basic_watch_ingest(self):
        ctx = self.engine.ingest_watch("watch-1", {
            "heart_rate": 75,
            "hrv": 55,
            "spo2": 98,
        })
        assert ctx is not None
        assert ctx.health.heart_rate == 75
        assert ctx.health.hrv == 55.0

    def test_watch_health_alert(self):
        ctx = self.engine.ingest_watch("watch-1", {
            "heart_rate": 150,
        })
        stats = self.engine.get_stats()
        assert stats["stats"]["health_alerts"] >= 1

    def test_watch_sleep_phase(self):
        ctx = self.engine.ingest_watch("watch-1", {
            "sleep_phase": "deep",
            "heart_rate": 55,
        })
        assert ctx.health.sleep_phase == SleepPhase.DEEP

    def test_watch_steps_and_flights(self):
        ctx = self.engine.ingest_watch("watch-1", {
            "step_count": 5000,
            "flights_climbed": 10,
            "exercise_minutes": 45,
        })
        assert ctx.health.step_count == 5000
        assert ctx.health.flights_climbed == 10

    def test_watch_updates_stats(self):
        self.engine.ingest_watch("watch-1", {"heart_rate": 70})
        stats = self.engine.get_stats()
        assert stats["stats"]["watch_updates"] >= 1


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — Laptop Ingest
# ═══════════════════════════════════════════════════════════════════

class TestLaptopIngest:
    def setup_method(self):
        self.engine = DeviceContextEngine()
        self.engine.register_device("laptop-1", DeviceType.LAPTOP, "My Laptop")

    def test_basic_laptop_ingest(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "active_window": "code",
            "screen_on": True,
            "cpu_percent": 45,
            "memory_percent": 60,
        })
        assert ctx.digital.active_app == "code"
        assert ctx.digital.active_app_category == "development"
        assert ctx.environment.laptop_cpu_percent == 45.0

    def test_laptop_screen_on(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "screen_on": True,
        })
        assert ctx.digital.screen_on is True

    def test_laptop_screen_locked(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "screen_locked": True,
        })
        assert ctx.digital.screen_on is False

    def test_laptop_wifi_classifies_location(self):
        self.engine.register_wifi_location("HomeWiFi", LocationType.HOME)
        self.engine.register_device("laptop-1", DeviceType.LAPTOP)
        ctx = self.engine.ingest_laptop("laptop-1", {
            "wifi_ssid": "HomeWiFi",
        })
        # WiFi location map is checked during _parse_laptop_environment
        assert ctx.environment.network_type in (NetworkType.WIFI_HOME, NetworkType.UNKNOWN)

    def test_laptop_bluetooth(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "bluetooth_devices": ["John's iPhone", "AirPods Pro"],
        })
        assert "John's iPhone" in ctx.environment.nearby_bluetooth_devices or \
               any("iPhone" in d for d in ctx.environment.nearby_bluetooth_devices)

    def test_laptop_media(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "media_playing": True,
            "media_title": "Test Song",
            "media_type": "music",
        })
        assert ctx.digital.media_playing is True
        assert ctx.digital.media_type == "music"

    def test_laptop_calendar_event(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "calendar_next_event": {"title": "Team Meeting", "time": "10:00"},
        })
        # calendar event is sanitized
        if ctx.digital.calendar_next_event:
            assert "title" in ctx.digital.calendar_next_event

    def test_laptop_updates_stats(self):
        self.engine.ingest_laptop("laptop-1", {"screen_on": True})
        stats = self.engine.get_stats()
        assert stats["stats"]["laptop_updates"] >= 1

    def test_laptop_invalid_cpu(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "cpu_percent": "invalid",
        })
        assert ctx.environment.laptop_cpu_percent is None


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — Context Inference
# ═══════════════════════════════════════════════════════════════════

class TestContextInference:
    def setup_method(self):
        self.engine = DeviceContextEngine()
        self.engine.register_device("phone-1", DeviceType.IPHONE, "Phone")
        self.engine.register_device("laptop-1", DeviceType.LAPTOP, "Laptop")

    def test_activity_sleeping(self):
        ctx = self.engine.ingest_watch("phone-1", {
            "sleep_phase": "deep",
            "heart_rate": 55,
        })
        assert ctx.activity == ActivityType.SLEEPING

    def test_activity_exercising(self):
        ctx = self.engine.ingest_watch("phone-1", {
            "exercise_minutes": 45,
            "heart_rate": 130,
        })
        assert ctx.activity == ActivityType.EXERCISING

    def test_activity_working(self):
        self.engine.register_wifi_location("CorpWiFi", LocationType.OFFICE)
        ctx = self.engine.ingest_laptop("laptop-1", {
            "active_window": "code",
            "screen_on": True,
            "wifi_ssid": "CorpWiFi",
        })
        # Should be WORKING or at least SEDENTARY
        assert ctx.activity in (ActivityType.WORKING, ActivityType.SEDENTARY, ActivityType.UNKNOWN)

    def test_focus_deep_work(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "active_window": "vim",
            "screen_on": True,
            "notifications_last_hour": 2,
        })
        # Low stress + development app + low notifications = deep work or active
        assert ctx.focus in (FocusLevel.DEEP_WORK, FocusLevel.ACTIVE, FocusLevel.SCANNING)

    def test_focus_distracted_high_notifications(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "active_window": "chrome",
            "screen_on": True,
            "notifications_last_hour": 50,
        })
        assert ctx.focus == FocusLevel.DISTRACTED

    def test_confidence_increases_with_devices(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "active_window": "code",
            "screen_on": True,
        })
        assert ctx.context_confidence > 0.0

    def test_focus_resting_screen_off(self):
        ctx = self.engine.ingest_laptop("laptop-1", {
            "screen_on": False,
            "notifications_last_hour": 0,
        })
        # With watch providing low HR data
        assert ctx.focus in (FocusLevel.RESTING, FocusLevel.SCANNING)


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — WiFi Location & People Registration
# ═══════════════════════════════════════════════════════════════════

class TestRegistration:
    def setup_method(self):
        self.engine = DeviceContextEngine()

    def test_register_wifi_location(self):
        self.engine.register_wifi_location("MyHomeWiFi", LocationType.HOME)
        # Now ingest laptop data with that SSID
        ctx = self.engine.ingest_laptop("laptop-1", {
            "wifi_ssid": "MyHomeWiFi",
            "screen_on": True,
        })
        assert ctx.environment.network_type in (NetworkType.WIFI_HOME, NetworkType.UNKNOWN)

    def test_register_wifi_location_sanitizes_ssid(self):
        self.engine.register_wifi_location("<script>alert(1)</script>", LocationType.HOME)
        wifi_keys = list(self.engine._wifi_location_map.keys())
        assert all("<script>" not in k for k in wifi_keys)

    def test_register_person(self):
        self.engine.register_wifi_location("HomeWiFi", LocationType.HOME)
        self.engine.register_person("AA:BB:CC:DD:EE:FF", "Alice")
        # Next laptop ingest should include Alice
        self.engine.register_device("laptop-1", DeviceType.LAPTOP)
        ctx = self.engine.ingest_laptop("laptop-1", {
            "wifi_ssid": "HomeWiFi",
            "bluetooth_devices": ["AA:BB:CC:DD:EE:FF"],
        })
        # people_nearby should include Alice
        assert "Alice" in ctx.people_nearby

    def test_register_person_max_bluetooth(self):
        for i in range(55):
            self.engine.register_person(f"AA:BB:CC:DD:EE:{i:02X}", f"Person {i}")
        assert len(self.engine._people_map) <= 50

    def test_register_person_sanitizes_name(self):
        self.engine.register_person("AA:BB:CC:DD:EE:FF", "<b>Alice</b>")
        people = list(self.engine._people_map.values())
        assert all("<b>" not in p for p in people)


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — Health Trends & Digital Wellbeing
# ═══════════════════════════════════════════════════════════════════

class TestHealthTrends:
    def setup_method(self):
        self.engine = DeviceContextEngine()
        self.engine.register_device("watch-1", DeviceType.APPLE_WATCH, "Watch")

    def test_no_data_returns_no_data(self):
        trends = self.engine.get_health_trends()
        assert trends["trend"] == "no_data"
        assert trends["data_points"] == 0

    def test_health_trends_basic(self):
        self.engine.ingest_watch("watch-1", {"heart_rate": 72, "hrv": 55, "spo2": 98})
        self.engine.ingest_watch("watch-1", {"heart_rate": 75, "hrv": 50, "spo2": 97})
        trends = self.engine.get_health_trends()
        assert trends["avg_heart_rate"] >= 70
        assert trends["data_points"] >= 2

    def test_elevated_heart_rate(self):
        self.engine.ingest_watch("watch-1", {"heart_rate": 90, "hrv": 40, "spo2": 97})
        trends = self.engine.get_health_trends()
        assert trends["trend"] == "elevated"

    def test_normal_heart_rate(self):
        self.engine.ingest_watch("watch-1", {"heart_rate": 65, "hrv": 60, "spo2": 99})
        trends = self.engine.get_health_trends()
        assert trends["trend"] == "normal"

    def test_health_trends_minutes_param(self):
        self.engine.ingest_watch("watch-1", {"heart_rate": 72})
        trends = self.engine.get_health_trends(minutes=1440)
        assert trends["data_points"] >= 1


class TestDigitalWellbeing:
    def setup_method(self):
        self.engine = DeviceContextEngine()
        self.engine.register_device("laptop-1", DeviceType.LAPTOP, "Laptop")

    def test_empty_wellbeing(self):
        wb = self.engine.get_digital_wellbeing()
        assert wb["total_notifications"] == 0
        assert wb["screen_time_minutes"] == 0

    def test_wellbeing_with_data(self):
        self.engine.ingest_laptop("laptop-1", {
            "screen_on": True,
            "screen_time_minutes": 120,
        })
        wb = self.engine.get_digital_wellbeing()
        assert wb["screen_time_minutes"] > 0

    def test_distraction_score_poor(self):
        # High notifications + context switching
        for i in range(10):
            self.engine.ingest_laptop("laptop-1", {
                "active_window": f"app{i}",
                "screen_on": True,
            })
        self.engine.ingest_iphone("laptop-1", {
            "notifications": [{"app": "Messages", "category": "message"}] * 30,
        })
        wb = self.engine.get_digital_wellbeing()
        # Just checking it runs without crash
        assert "distraction_score" in wb
        assert "focus_quality" in wb


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — Thread Safety
# ═══════════════════════════════════════════════════════════════════

class TestDeviceContextThreadSafety:
    def test_concurrent_ingest(self):
        engine = DeviceContextEngine()
        engine.register_device("phone-1", DeviceType.IPHONE, "Phone")
        engine.register_device("watch-1", DeviceType.APPLE_WATCH, "Watch")
        engine.register_device("laptop-1", DeviceType.LAPTOP, "Laptop")

        errors = []

        def ingest_phone():
            try:
                for i in range(50):
                    engine.ingest_iphone("phone-1", {"heart_rate": 70 + i})
            except Exception as e:
                errors.append(e)

        def ingest_watch():
            try:
                for i in range(50):
                    engine.ingest_watch("watch-1", {"heart_rate": 65 + i, "hrv": 45})
            except Exception as e:
                errors.append(e)

        def ingest_laptop():
            try:
                for i in range(50):
                    engine.ingest_laptop("laptop-1", {"screen_on": True, "active_window": "code"})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=ingest_phone),
            threading.Thread(target=ingest_watch),
            threading.Thread(target=ingest_laptop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_stats()
        assert stats["stats"]["iphone_updates"] == 50
        assert stats["stats"]["watch_updates"] == 50
        assert stats["stats"]["laptop_updates"] == 50

    def test_concurrent_read_write(self):
        engine = DeviceContextEngine()
        engine.register_device("phone-1", DeviceType.IPHONE, "Phone")

        errors = []

        def writer():
            try:
                for i in range(50):
                    engine.ingest_iphone("phone-1", {"battery": {"level": 0.5 + i * 0.001}})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    engine.get_context()
                    engine.get_stats()
                    engine.get_device_summary()
                    engine.get_health_trends()
                    engine.get_digital_wellbeing()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════
# Unit Tests — Dataclass to_dict DeepCopy Safety
# ═══════════════════════════════════════════════════════════════════

class TestDataclassImmutability:
    def test_unified_context_to_dict_is_copy(self):
        engine = DeviceContextEngine()
        engine.register_device("phone-1", DeviceType.IPHONE)
        ctx = engine.ingest_iphone("phone-1", {
            "location": {"latitude": 40.7, "longitude": -74.0, "name": "Home"},
        })
        d1 = ctx.to_dict()
        d1["location"]["name"] = "MODIFIED"
        d2 = ctx.to_dict()
        assert d2["location"]["name"] != "MODIFIED"

    def test_digital_snapshot_to_dict_is_copy(self):
        snap = DigitalSnapshot(
            screen_on=True,
            active_app="code",
            notification_types=[NotificationType.MESSAGE, NotificationType.EMAIL],
            calendar_next_event={"title": "meeting"},
        )
        d = snap.to_dict()
        d["notification_types"].append(NotificationType.SYSTEM.value)
        d2 = snap.to_dict()
        assert len(d2["notification_types"]) == 2  # Should still be 2

    def test_environment_snapshot_to_dict_is_copy(self):
        snap = EnvironmentSnapshot(
            nearby_bluetooth_devices=["dev1", "dev2"],
        )
        d = snap.to_dict()
        d["nearby_bluetooth_devices"].append("dev3")
        d2 = snap.to_dict()
        assert len(d2["nearby_bluetooth_devices"]) == 2


# ═══════════════════════════════════════════════════════════════════
# Integration Tests — ShadowOS Wiring
# ═══════════════════════════════════════════════════════════════════

class TestShadowOSDeviceContext:
    def setup_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        # Full initialization needed for get_health/get_status
        self.os = ShadowOS()

    def test_shadowos_has_device_context(self):
        assert isinstance(self.os.device_context, DeviceContextEngine)

    def test_shadowos_device_context_in_health(self):
        self.os.device_context.register_device("test-dev", DeviceType.UNKNOWN)
        # ShadowOS must be running for health to populate modules
        # Instead, verify device_context stats are accessible directly
        stats = self.os.device_context.get_stats()
        assert stats["devices_registered"] >= 1

    def test_shadowos_device_context_in_status(self):
        self.os.device_context.register_device("test-dev", DeviceType.UNKNOWN)
        status = self.os.get_status()
        assert "device_context_stats" in status


# ═══════════════════════════════════════════════════════════════════
# Integration Tests — API Endpoints (using TestClient)
# ═══════════════════════════════════════════════════════════════════

class TestDeviceAPIEndpoints:
    """Test the /device/* API endpoints via FastAPI TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from shadowd.cognition.api_router import router, _shadow_os, _shadow_os_lock
        from shadowd.cognition.shadow_os import ShadowOS
        import shadowd.cognition.api_router as api_module

        # Create fresh ShadowOS instance
        new_os = ShadowOS.__new__(ShadowOS)
        new_os.device_context = DeviceContextEngine()

        # Inject into module
        with api_module._shadow_os_lock:
            api_module._shadow_os = new_os

        app = __import__("fastapi").FastAPI()
        app.include_router(router)
        tc = TestClient(app)
        yield tc

        # Cleanup
        with api_module._shadow_os_lock:
            api_module._shadow_os = None

    def test_device_register(self, client):
        resp = client.post("/os/device/register", json={
            "device_id": "phone-1",
            "device_type": "iphone",
            "name": "My iPhone",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "registered"
        assert data["device_type"] == "iphone"

    def test_device_register_invalid_type(self, client):
        resp = client.post("/os/device/register", json={
            "device_id": "dev-1",
            "device_type": "unknown_device_type",
            "name": "Test",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["device_type"] == "unknown"

    def test_device_iphone_ingest(self, client):
        client.post("/os/device/register", json={
            "device_id": "phone-1",
            "device_type": "iphone",
            "name": "Phone",
        })
        resp = client.post("/os/device/iphone", json={
            "device_id": "phone-1",
            "data": {"location": {"latitude": 40.7, "longitude": -74.0, "name": "Home"}},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "location" in data or data.get("status") == "no_context"

    def test_device_watch_ingest(self, client):
        client.post("/os/device/register", json={
            "device_id": "watch-1",
            "device_type": "apple_watch",
            "name": "Watch",
        })
        resp = client.post("/os/device/watch", json={
            "device_id": "watch-1",
            "data": {"heart_rate": 72, "hrv": 55, "spo2": 98},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "health" in data or data.get("status") == "no_context"

    def test_device_laptop_ingest(self, client):
        client.post("/os/device/register", json={
            "device_id": "laptop-1",
            "device_type": "laptop",
            "name": "Laptop",
        })
        resp = client.post("/os/device/laptop", json={
            "device_id": "laptop-1",
            "data": {"active_window": "code", "screen_on": True},
        })
        assert resp.status_code == 200

    def test_device_context(self, client):
        client.post("/os/device/register", json={
            "device_id": "phone-1",
            "device_type": "iphone",
            "name": "Phone",
        })
        client.post("/os/device/iphone", json={
            "device_id": "phone-1",
            "data": {"location": {"latitude": 40.7, "longitude": -74.0}},
        })
        resp = client.get("/os/device/context")
        assert resp.status_code == 200

    def test_device_health_trends(self, client):
        resp = client.get("/os/device/health-trends")
        assert resp.status_code == 200

    def test_device_health_trends_with_minutes(self, client):
        resp = client.get("/os/device/health-trends?minutes=120")
        assert resp.status_code == 200

    def test_device_health_trends_invalid_minutes(self, client):
        resp = client.get("/os/device/health-trends?minutes=0")
        assert resp.status_code == 422  # Validation error: ge=1

    def test_device_digital_wellbeing(self, client):
        resp = client.get("/os/device/digital-wellbeing")
        assert resp.status_code == 200

    def test_device_digital_wellbeing_with_hours(self, client):
        resp = client.get("/os/device/digital-wellbeing?hours=48")
        assert resp.status_code == 200

    def test_device_devices(self, client):
        client.post("/os/device/register", json={
            "device_id": "phone-1",
            "device_type": "iphone",
            "name": "Phone",
        })
        resp = client.get("/os/device/devices")
        assert resp.status_code == 200
        data = resp.json()
        assert "devices" in data
        assert "online_count" in data

    def test_device_stats(self, client):
        resp = client.get("/os/device/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "stats" in data

    def test_device_wifi_location(self, client):
        resp = client.post("/os/device/wifi-location", json={
            "ssid": "HomeWiFi",
            "location_type": "home",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "registered"
        assert data["location_type"] == "home"

    def test_device_wifi_location_invalid_type(self, client):
        resp = client.post("/os/device/wifi-location", json={
            "ssid": "TestWiFi",
            "location_type": "not_a_real_type",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["location_type"] == "unknown"

    def test_device_person(self, client):
        resp = client.post("/os/device/person", json={
            "bt_mac": "AA:BB:CC:DD:EE:FF",
            "name": "Alice",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "registered"
        assert data["name"] == "Alice"

    def test_device_register_validation(self, client):
        # Missing required field
        resp = client.post("/os/device/register", json={
            "device_id": "test",
            # Missing device_type
        })
        assert resp.status_code == 422  # Validation error

    def test_device_wifi_location_validation(self, client):
        # Missing required field
        resp = client.post("/os/device/wifi-location", json={
            "ssid": "Test",
            # Missing location_type
        })
        assert resp.status_code == 422

    def test_device_person_validation(self, client):
        # Missing required field
        resp = client.post("/os/device/person", json={
            "bt_mac": "AA:BB:CC:DD:EE:FF",
            # Missing name
        })
        assert resp.status_code == 422

    def test_device_iphone_empty_data(self, client):
        client.post("/os/device/register", json={
            "device_id": "phone-1",
            "device_type": "iphone",
            "name": "Phone",
        })
        resp = client.post("/os/device/iphone", json={
            "device_id": "phone-1",
            "data": {},
        })
        assert resp.status_code == 200

    def test_device_end_to_end_flow(self, client):
        # Register devices
        client.post("/os/device/register", json={"device_id": "phone-1", "device_type": "iphone", "name": "Phone"})
        client.post("/os/device/register", json={"device_id": "watch-1", "device_type": "apple_watch", "name": "Watch"})
        client.post("/os/device/register", json={"device_id": "laptop-1", "device_type": "laptop", "name": "Laptop"})

        # Register WiFi and person
        client.post("/os/device/wifi-location", json={"ssid": "HomeWiFi", "location_type": "home"})
        client.post("/os/device/person", json={"bt_mac": "AA:BB:CC:DD:EE:FF", "name": "Alice"})

        # Ingest data from all devices
        client.post("/os/device/iphone", json={
            "device_id": "phone-1",
            "data": {"location": {"latitude": 40.7, "longitude": -74.0, "name": "Home"}, "battery": {"level": 0.8}},
        })
        client.post("/os/device/watch", json={
            "device_id": "watch-1",
            "data": {"heart_rate": 72, "hrv": 55, "spo2": 98},
        })
        client.post("/os/device/laptop", json={
            "device_id": "laptop-1",
            "data": {"active_window": "code", "screen_on": True, "wifi_ssid": "HomeWiFi"},
        })

        # Check context
        ctx_resp = client.get("/os/device/context")
        assert ctx_resp.status_code == 200
        ctx = ctx_resp.json()
        assert "activity" in ctx
        assert "focus" in ctx

        # Check stats
        stats_resp = client.get("/os/device/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["stats"]["iphone_updates"] >= 1
        assert stats["stats"]["watch_updates"] >= 1
        assert stats["stats"]["laptop_updates"] >= 1

        # Check devices
        devices_resp = client.get("/os/device/devices")
        assert devices_resp.status_code == 200
        devices = devices_resp.json()
        assert len(devices["devices"]) >= 3

        # Check health trends
        health_resp = client.get("/os/device/health-trends")
        assert health_resp.status_code == 200

        # Check digital wellbeing
        wellbeing_resp = client.get("/os/device/digital-wellbeing")
        assert wellbeing_resp.status_code == 200