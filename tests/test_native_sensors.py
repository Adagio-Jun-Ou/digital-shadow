"""Tests for the native Linux sensor probes used by Omni Sense."""

import asyncio
import pytest

from sensors.native_sensors import (
    load_state, uptime_state, battery_state, audio_state,
    infer_activity_level, native_snapshot,
)
from phone.wifi_scanner import _signal_pct_to_dbm


class TestLoad:
    def test_returns_floats(self):
        load = load_state()
        if not load:
            pytest.skip("/proc/loadavg unavailable on this platform")
        assert "load_1m" in load
        assert isinstance(load["load_1m"], float)
        assert load["ncpu"] >= 1


class TestUptime:
    def test_returns_uptime_seconds(self):
        u = uptime_state()
        if not u:
            pytest.skip("/proc/uptime unavailable")
        assert u["uptime_s"] > 0
        assert u["uptime_h"] > 0


class TestBattery:
    def test_returns_dict_or_empty(self):
        b = battery_state()
        assert isinstance(b, dict)
        if b:  # battery present
            assert "name" in b


class TestAudio:
    def test_returns_backend_field(self):
        result = asyncio.run(audio_state())
        assert "backend" in result
        assert result["backend"] in ("pactl", "unavailable")


class TestActivityInference:
    def test_heavy_load_classified(self):
        load = {"load_1m": 8.0, "ncpu": 4}
        result = infer_activity_level(load, {}, {})
        assert result in ("heavy", "moderate", "active_media")

    def test_idle_default(self):
        result = infer_activity_level({"load_1m": 0.1, "ncpu": 8}, {}, {})
        assert result == "idle"

    def test_charging_idle(self):
        result = infer_activity_level(
            {"load_1m": 0.0, "ncpu": 8},
            {},
            {"status": "Charging"},
        )
        assert result == "idle_charging"

    def test_listening_when_recording(self):
        result = infer_activity_level(
            {"load_1m": 0.1, "ncpu": 8},
            {"state": "recording"},
            {},
        )
        assert result == "listening"

    def test_active_media_with_playback_and_load(self):
        result = infer_activity_level(
            {"load_1m": 4.0, "ncpu": 4},  # 1.0 normalized load
            {"state": "playing"},
            {},
        )
        assert result == "active_media"


class TestSignalPctToDbm:
    def test_zero_maps_to_minus_100(self):
        assert _signal_pct_to_dbm(0) == -100

    def test_hundred_maps_to_minus_50(self):
        assert _signal_pct_to_dbm(100) == -50

    def test_mid_range(self):
        # 50% should be around -75 dBm
        assert -80 <= _signal_pct_to_dbm(50) <= -70

    def test_clamps_negative(self):
        # nmcli should never emit negative percent, but be defensive
        assert _signal_pct_to_dbm(-5) == -100


class TestNativeSnapshot:
    def test_returns_all_keys(self):
        snap = asyncio.run(native_snapshot())
        assert "audio" in snap
        assert "battery" in snap
        assert "load" in snap
        assert "uptime" in snap
        assert "inferred_activity" in snap
        assert "captured_at" in snap

    def test_captured_at_recent(self):
        import time
        snap = asyncio.run(native_snapshot())
        # Within 5 seconds of now
        assert abs(snap["captured_at"] - int(time.time())) < 5
