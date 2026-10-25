"""Tests for the Hearth location-pattern ability."""

from datetime import datetime, timezone, timedelta

import pytest

from shikigami.hearth import (
    classify, compute_hearth, _extract_location_hint,
    _LOCATION_PATTERNS,
)


@pytest.fixture
def now():
    return datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


class TestClassify:
    def test_empty(self):
        assert classify("") == "unknown"

    @pytest.mark.parametrize("raw,expected", [
        ("HomeWiFi", "home"),
        ("My_Apartment_5G", "home"),
        ("Corp Guest", "work"),
        ("OfficeNetwork", "work"),
        ("Starbucks_Free", "cafe"),
        ("Library WiFi", "public"),
        ("MIT Campus", "school"),
        ("Crossfit Box", "gym"),
        ("Random nonsense", "other"),
    ])
    def test_basic_labels(self, raw, expected):
        assert classify(raw) == expected


class TestExtractHint:
    def test_wifi_ssid(self):
        o = {"type": "wifi", "content": {"ssid": "HomeNet"}}
        assert _extract_location_hint(o) == "HomeNet"

    def test_explicit_location(self):
        o = {"type": "note", "content": {"location": "Coffee shop"}}
        assert _extract_location_hint(o) == "Coffee shop"

    def test_gps_city(self):
        o = {"type": "gps", "content": {"city": "Brooklyn"}}
        assert _extract_location_hint(o) == "Brooklyn"

    def test_calendar_location(self):
        o = {"type": "calendar_event", "content": {"location": "Zoom"}}
        assert _extract_location_hint(o) == "Zoom"

    def test_returns_none_when_absent(self):
        o = {"type": "git_commit", "content": {"message": "fix"}}
        assert _extract_location_hint(o) is None

    def test_nested_networks_list(self):
        o = {"type": "wifi_scan", "content": {"networks": [{"ssid": "HomeWiFi"}, {"ssid": "Other"}]}}
        assert _extract_location_hint(o) == "HomeWiFi"


class TestCompute:
    def test_empty(self, now):
        report = compute_hearth([], now=now)
        assert report.total_observations == 0
        assert report.labeled_observations == 0
        assert report.time_per_label_min == {}

    def test_unlabeled_observations_skipped(self, now):
        obs = [{"type": "terminal_command", "content": {"command": "ls"},
                "timestamp": now.isoformat()}]
        report = compute_hearth(obs, now=now)
        assert report.total_observations == 1
        assert report.labeled_observations == 0

    def test_visits_and_durations(self, now):
        obs = []
        for h in range(8, 12):
            obs.append({
                "type": "wifi", "content": {"ssid": "HomeWiFi"},
                "timestamp": now.replace(hour=h, minute=0).isoformat(),
            })
        for h in range(13, 17):
            obs.append({
                "type": "wifi", "content": {"ssid": "CorpWiFi"},
                "timestamp": now.replace(hour=h, minute=0).isoformat(),
            })
        report = compute_hearth(obs, now=now)
        assert "home" in report.time_per_label_min
        assert "work" in report.time_per_label_min

    def test_hour_mode(self, now):
        obs = []
        for d in range(5):
            day = now - timedelta(days=d)
            obs.append({
                "type": "wifi", "content": {"ssid": "HomeWiFi"},
                "timestamp": day.replace(hour=8).isoformat(),
            })
            obs.append({
                "type": "wifi", "content": {"ssid": "CorpWiFi"},
                "timestamp": day.replace(hour=14).isoformat(),
            })
        report = compute_hearth(obs, now=now)
        assert report.hour_label_mode.get(8) == "home"
        assert report.hour_label_mode.get(14) == "work"

    def test_transitions(self, now):
        obs = []
        for d in range(3):
            day = now - timedelta(days=d)
            # Long gap forces visit boundary.
            obs.append({"type": "wifi", "content": {"ssid": "HomeWiFi"},
                        "timestamp": day.replace(hour=8).isoformat()})
            obs.append({"type": "wifi", "content": {"ssid": "CorpWiFi"},
                        "timestamp": day.replace(hour=14).isoformat()})
        report = compute_hearth(obs, now=now)
        transitions = {(a, b): n for a, b, n in report.top_transitions}
        assert ("home", "work") in transitions or ("work", "home") in transitions

    def test_anomaly_detection(self, now):
        # Establish "usually home at 12" by piling up home observations at hour 12.
        obs = []
        for d in range(1, 10):
            day = now - timedelta(days=d)
            obs.append({"type": "wifi", "content": {"ssid": "HomeWiFi"},
                        "timestamp": day.replace(hour=now.hour).isoformat()})
        # Today's observation at the same hour is at a different place.
        obs.append({"type": "wifi", "content": {"ssid": "CorpWiFi"},
                    "timestamp": now.isoformat()})
        report = compute_hearth(obs, now=now)
        # Anomaly should be flagged.
        assert len(report.anomalies) >= 1
