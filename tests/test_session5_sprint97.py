import copy
import threading
from datetime import datetime, timedelta, timezone

import pytest

from shadowd.cognition.device_context import (
    DeviceContextEngine, DeviceType, LocationType, ActivityType,
    StressLevel, FocusLevel, SleepPhase, NetworkType, NotificationType,
    UnifiedContext, HealthSnapshot, DigitalSnapshot, EnvironmentSnapshot,
    LocationPoint, _classify_location, _classify_notification,
    _infer_stress, _calculate_distraction_score, _sanitize_dict,
)


class TestClassifyLocation:
    def test_home_in_name(self):
        assert _classify_location({"name": "Home"}) == LocationType.HOME

    def test_office_in_name(self):
        assert _classify_location({"name": "Office Building"}) == LocationType.OFFICE

    def test_gym_in_name(self):
        assert _classify_location({"name": "Planet Fitness Gym"}) == LocationType.GYM

    def test_wifi_ssid_home(self):
        assert _classify_location({"name": "", "wifi_ssid": "Home_WiFi"}) == LocationType.HOME

    def test_wifi_ssid_office(self):
        assert _classify_location({"name": "", "wifi_ssid": "Corp_Office"}) == LocationType.OFFICE

    def test_high_speed_transit(self):
        assert _classify_location({"name": "", "speed": 15}) == LocationType.TRANSIT

    def test_unknown_default(self):
        assert _classify_location({"name": "Cafe"}) == LocationType.UNKNOWN

    def test_empty_dict(self):
        assert _classify_location({}) == LocationType.UNKNOWN

    def test_non_dict(self):
        assert _classify_location("bad") == LocationType.UNKNOWN


class TestClassifyNotification:
    def test_message_app(self):
        assert _classify_notification({"app": "Messages"}) == NotificationType.MESSAGE

    def test_email_app(self):
        assert _classify_notification({"app": "Gmail"}) == NotificationType.EMAIL

    def test_social_category(self):
        assert _classify_notification({"app": "Insta", "category": "social"}) == NotificationType.SOCIAL

    def test_reminder(self):
        assert _classify_notification({"app": "Reminders"}) == NotificationType.REMINDER

    def test_unknown(self):
        assert _classify_notification({"app": "SomeApp"}) == NotificationType.UNKNOWN

    def test_non_dict(self):
        assert _classify_notification("bad") == NotificationType.UNKNOWN


class TestInferStress:
    def test_low_stress(self):
        assert _infer_stress(65, 60, 99) == StressLevel.LOW

    def test_moderate_stress(self):
        assert _infer_stress(85, 45, 98) == StressLevel.MODERATE

    def test_high_stress(self):
        result = _infer_stress(105, 25, 93)
        assert result in (StressLevel.HIGH, StressLevel.CRITICAL)

    def test_critical_stress(self):
        result = _infer_stress(110, 25, 92)
        assert result in (StressLevel.CRITICAL, StressLevel.HIGH)

    def test_low_hr_moderate_hrv(self):
        assert _infer_stress(70, 40, 98) == StressLevel.LOW

    def test_low_spo2(self):
        assert _infer_stress(75, 55, 93) in (StressLevel.HIGH, StressLevel.CRITICAL)


class TestCalculateDistractionScore:
    def test_no_data(self):
        assert _calculate_distraction_score(0, 0, []) == 0.0

    def test_high_screen_high_notifs(self):
        score = _calculate_distraction_score(480, 50, [])
        assert score >= 0.5

    def test_low_screen_low_notifs(self):
        score = _calculate_distraction_score(30, 2, [])
        assert score < 0.2

    def test_context_switching(self):
        snaps = [DigitalSnapshot(active_app=f"app_{i % 3}") for i in range(10)]
        score = _calculate_distraction_score(100, 10, snaps)
        assert score > 0


class TestDeviceContextEngine:
    def get_engine(self):
        return DeviceContextEngine()

    def test_register_device(self):
        engine = self.get_engine()
        engine.register_device("iphone_1", DeviceType.IPHONE, "Demo iPhone")
        summary = engine.get_device_summary()
        assert summary["devices"][0]["type"] == "iphone"

    def test_register_max_devices(self):
        engine = self.get_engine()
        for i in range(15):
            engine.register_device(f"dev_{i}", DeviceType.IPHONE if i % 2 == 0 else DeviceType.APPLE_WATCH)
        assert len(engine._devices) <= DeviceContextEngine._MAX_DEVICES

    def test_ingest_iphone_location(self):
        engine = self.get_engine()
        engine.register_device("iphone_1", DeviceType.IPHONE)
        ctx = engine.ingest_iphone("iphone_1", {
            "location": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "accuracy": 10.0,
                "name": "Home",
            }
        })
        assert ctx.location.location_type == LocationType.HOME
        assert ctx.location.latitude == 37.7749

    def test_ingest_iphone_battery(self):
        engine = self.get_engine()
        engine.register_device("iphone_1", DeviceType.IPHONE)
        ctx = engine.ingest_iphone("iphone_1", {
            "battery": {"level": 0.85, "charging": True}
        })
        assert ctx.environment.battery_level == 0.85
        assert ctx.environment.battery_charging is True

    def test_ingest_watch_health(self):
        engine = self.get_engine()
        engine.register_device("watch_1", DeviceType.APPLE_WATCH)
        ctx = engine.ingest_watch("watch_1", {
            "health": {
                "heart_rate": 72,
                "hrv": 55.0,
                "spo2": 98.5,
                "step_count": 5000,
                "exercise_minutes": 30,
                "sleep_phase": "awake",
            }
        })
        assert ctx.health.heart_rate == 72
        assert ctx.health.step_count == 5000
        assert ctx.health.spo2 == 98.5

    def test_ingest_watch_high_hr_alert(self):
        engine = self.get_engine()
        engine.register_device("watch_1", DeviceType.APPLE_WATCH)
        engine.ingest_watch("watch_1", {
            "health": {"heart_rate": 120, "spo2": 95}
        })
        assert engine._stats["health_alerts"] >= 1

    def test_ingest_laptop(self):
        engine = self.get_engine()
        engine.register_device("laptop_1", DeviceType.LAPTOP)
        ctx = engine.ingest_laptop("laptop_1", {
            "active_window": "Code",
            "screen_on": True,
            "wifi_ssid": "Home_WiFi",
            "cpu_percent": 45.0,
            "memory_percent": 62.0,
        })
        assert ctx.digital.active_app == "Code"
        assert ctx.digital.screen_on is True
        assert ctx.environment.laptop_cpu_percent == 45.0

    def test_ingest_laptop_app_category(self):
        engine = self.get_engine()
        engine.register_device("lap_1", DeviceType.LAPTOP)
        ctx = engine.ingest_laptop("lap_1", {
            "active_window": "Code",
        })
        assert ctx.digital.active_app_category == "development"

    def test_ingest_laptop_browsing_category(self):
        engine = self.get_engine()
        engine.register_device("lap_1", DeviceType.LAPTOP)
        ctx = engine.ingest_laptop("lap_1", {
            "active_window": "Chrome",
        })
        assert ctx.digital.active_app_category == "browsing"

    def test_get_context_returns_copy(self):
        engine = self.get_engine()
        engine.register_device("iphone_1", DeviceType.IPHONE)
        engine.ingest_iphone("iphone_1", {
            "location": {"latitude": 37.0, "longitude": -122.0, "name": "Home"}
        })
        ctx1 = engine.get_context()
        ctx2 = engine.get_context()
        assert ctx1 is not ctx2
        assert ctx1.location.latitude == ctx2.location.latitude

    def test_get_health_trends(self):
        engine = self.get_engine()
        engine.register_device("watch_1", DeviceType.APPLE_WATCH)
        for hr in [70, 72, 68, 75]:
            engine.ingest_watch("watch_1", {"health": {"heart_rate": hr, "hrv": 55.0, "spo2": 98.0}})
        trends = engine.get_health_trends(minutes=60)
        assert trends["avg_heart_rate"] > 0
        assert trends["data_points"] >= 4

    def test_get_health_trends_no_data(self):
        engine = self.get_engine()
        trends = engine.get_health_trends()
        assert trends["trend"] == "no_data"

    def test_get_digital_wellbeing(self):
        engine = self.get_engine()
        engine.register_device("lap_1", DeviceType.LAPTOP)
        engine.ingest_laptop("lap_1", {"screen_time_minutes": 120})
        wb = engine.get_digital_wellbeing()
        assert "screen_time_minutes" in wb
        assert "distraction_score" in wb

    def test_get_stats(self):
        engine = self.get_engine()
        stats = engine.get_stats()
        assert "stats" in stats
        assert "devices_registered" in stats

    def test_activity_inference_sleeping(self):
        engine = self.get_engine()
        engine.register_device("watch_1", DeviceType.APPLE_WATCH)
        ctx = engine.ingest_watch("watch_1", {
            "health": {"sleep_phase": "deep", "heart_rate": 55}
        })
        assert ctx.activity == ActivityType.SLEEPING

    def test_activity_inference_exercising(self):
        engine = self.get_engine()
        engine.register_device("watch_1", DeviceType.APPLE_WATCH)
        ctx = engine.ingest_watch("watch_1", {
            "health": {"exercise_minutes": 45, "heart_rate": 130, "sleep_phase": "awake"}
        })
        assert ctx.activity == ActivityType.EXERCISING

    def test_focus_inference_deep_work(self):
        engine = self.get_engine()
        engine.register_device("lap_1", DeviceType.LAPTOP)
        ctx = engine.ingest_laptop("lap_1", {
            "active_window": "Code",
            "screen_on": True,
            "notifications_last_hour": 2,
        })
        engine.ingest_watch("watch_1", {"health": {"heart_rate": 65, "hrv": 60, "spo2": 99}})
        engine.register_device("watch_1", DeviceType.APPLE_WATCH)
        ctx = engine.ingest_watch("watch_1", {"health": {"heart_rate": 65, "hrv": 60, "spo2": 99}})
        assert ctx.focus in (FocusLevel.DEEP_WORK, FocusLevel.ACTIVE)

    def test_focus_inference_distracted(self):
        engine = self.get_engine()
        engine.register_device("lap_1", DeviceType.LAPTOP)
        ctx = engine.ingest_laptop("lap_1", {
            "active_window": "Safari",
            "screen_on": True,
            "notifications_last_hour": 30,
        })
        assert ctx.focus == FocusLevel.DISTRACTED

    def test_register_wifi_location(self):
        engine = self.get_engine()
        engine.register_wifi_location("Demo_Home", LocationType.HOME)
        engine.register_wifi_location("Corp_Office", LocationType.OFFICE)
        assert engine._wifi_location_map["Demo_Home"] == LocationType.HOME

    def test_register_person(self):
        engine = self.get_engine()
        engine.register_person("AA:BB:CC:DD:EE:FF", "Sarah")
        ctx = engine.ingest_iphone("iphone_1", {})
        assert "Sarah" in ctx.people_nearby

    def test_sanitize_dict(self):
        raw = {"key": "<script>bad</script>", "num": 42, "nested": {"x": "y"}}
        result = _sanitize_dict(raw)
        assert "<script>" not in result["key"]
        assert result["num"] == 42
        assert result["nested"]["x"] == "y"

    def test_sanitize_dict_max_keys(self):
        raw = {f"k{i}": f"v{i}" for i in range(50)}
        result = _sanitize_dict(raw)
        assert len(result) <= 40  # _MAX_METADATA_KEYS * 2

    def test_context_confidence_increases_with_devices(self):
        engine = self.get_engine()
        ctx_empty = engine.get_context()
        assert ctx_empty is None or ctx_empty.context_confidence >= 0
        engine.register_device("iphone_1", DeviceType.IPHONE)
        engine.register_device("watch_1", DeviceType.APPLE_WATCH)
        engine.ingest_iphone("iphone_1", {
            "location": {"latitude": 37.0, "longitude": -122.0, "name": "Home"},
            "health": {"heart_rate": 72, "spo2": 98},
        })
        ctx = engine.get_context()
        assert ctx.context_confidence > 0

    def test_notification_history_tracked(self):
        engine = self.get_engine()
        engine.register_device("iphone_1", DeviceType.IPHONE)
        engine.ingest_iphone("iphone_1", {
            "notifications": [
                {"app": "Messages", "category": "message"},
                {"app": "Gmail", "category": "email"},
            ]
        })
        assert len(engine._notification_history) == 2

    def test_device_stale_detection(self):
        engine = self.get_engine()
        engine.register_device("iphone_1", DeviceType.IPHONE)
        with engine._lock:
            engine._devices["iphone_1"]["last_seen"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        summary = engine.get_device_summary()
        assert summary["online_count"] == 0


class TestSanitizeStrict:
    def test_removes_angle_brackets(self):
        from shadowd.cognition.device_context import _sanitize_strict
        result = _sanitize_strict("<script>alert(1)</script>")
        assert "<" not in result
        assert ">" not in result

    def test_removes_control_chars(self):
        from shadowd.cognition.device_context import _sanitize_strict
        result = _sanitize_strict("hello\x00world\x01test")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_max_length(self):
        from shadowd.cognition.device_context import _sanitize_strict
        result = _sanitize_strict("a" * 500, max_len=50)
        assert len(result) <= 50


class TestDataclassToDict:
    def test_location_point_to_dict(self):
        loc = LocationPoint(latitude=37.77, longitude=-122.4, accuracy=10.0, location_type=LocationType.HOME, name="My Home")
        d = loc.to_dict()
        assert d["latitude"] == 37.77
        assert d["location_type"] == "home"
        assert d["name"] == "My Home"

    def test_health_snapshot_to_dict(self):
        h = HealthSnapshot(heart_rate=72, hrv=55.0, spo2=98.5, stress_level=StressLevel.LOW)
        d = h.to_dict()
        assert d["heart_rate"] == 72
        assert d["stress_level"] == "low"

    def test_digital_snapshot_to_dict_calendar(self):
        d = DigitalSnapshot(active_app="Code", calendar_next_event={"title": "Meeting", "time": "3pm"})
        result = d.to_dict()
        assert result["active_app"] == "Code"
        assert result["calendar_next_event"]["title"] == "Meeting"

    def test_digital_snapshot_to_dict_no_calendar(self):
        d = DigitalSnapshot(active_app="Code")
        result = d.to_dict()
        assert result["calendar_next_event"] is None

    def test_unified_context_to_dict(self):
        ctx = UnifiedContext(
            location=LocationPoint(location_type=LocationType.OFFICE),
            health=HealthSnapshot(heart_rate=80, stress_level=StressLevel.MODERATE),
            activity=ActivityType.WORKING,
            focus=FocusLevel.ACTIVE,
        )
        d = ctx.to_dict()
        assert d["activity"] == "working"
        assert d["focus"] == "active"
        assert d["location"]["location_type"] == "office"

    def test_unified_context_people_nearby_sanitize(self):
        ctx = UnifiedContext(people_nearby=["<script>evil</script>", "Alice"])
        d = ctx.to_dict()
        assert "<script>" not in d["people_nearby"][0]
        assert "Alice" in d["people_nearby"][1]