"""
Sprint 126 — AutonomousDaemon Thread Safety + Sanitization + Security Audit

Tests for:
- AD-T1: scan_processes thread safety (lock around baseline + stats)
- AD-T2: get_recent_transitions returns deep copy
- AD-T4: engage/disengage_kill_switch thread safety
- AD-T5: _tick pulse_data fallback (no NameError on failed metrics)
- AD-T6: asyncio.get_running_loop() instead of get_event_loop()
- AD-T8: observe_filesystem acquires lock for watched_paths + file_hashes
- AD-T9: scan_network increments under lock
- AD-T10: daemon/vitals/baselines and /state GET endpoints have auth
- Content-safe vs strict sanitize: descriptions preserve <>, identifiers strip <>
- UserRhythmModel: get_recent_transitions deep copy, edge cases
"""

import os
import sys
import time
import threading
import asyncio
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shadowd.cognition.autonomous_daemon import (
    AutonomousDaemon, UserRhythmModel, RhythmSample, UserActivityState,
    MultiModalObserver, SecurityGuardian, SecurityEventType, ThreatLevel,
    ObservationPulse, ObservationType, AutonomousActionType, FileIntegrityEntry,
    _sanitize, _sanitize_content, _sanitize_strict, _SANITIZE_RE,
    _SANITIZE_CONTENT_RE, _SANITIZE_STRICT_RE,
)


class TestSprint126SanitizationPatterns:
    """AD-T10: Verify _sanitize is content-safe, _sanitize_strict strips <>."""

    def test_sanitize_preserves_angle_brackets(self):
        assert _sanitize("latency < 100ms") == "latency < 100ms"

    def test_sanitize_preserves_newlines_tabs(self):
        assert _sanitize("hello\nworld\ttab") == "hello\nworld\ttab"

    def test_sanitize_strips_control_chars(self):
        assert _sanitize("hello\x00world\x01test") == "helloworldtest"

    def test_sanitize_strips_del(self):
        assert _sanitize("hello\x7fworld") == "helloworld"

    def test_sanitize_content_preserves_angle_brackets(self):
        assert _sanitize_content("latency < 100ms") == "latency < 100ms"

    def test_sanitize_content_preserves_newlines(self):
        assert _sanitize_content("hello\nworld") == "hello\nworld"

    def test_sanitize_content_strips_control_chars(self):
        assert _sanitize_content("hello\x00world") == "helloworld"

    def test_sanitize_strict_strips_angle_brackets(self):
        assert _sanitize_strict("latency < 100ms") == "latency  100ms"

    def test_sanitize_strict_strips_newlines(self):
        assert _sanitize_strict("hello\nworld") == "helloworld"

    def test_sanitize_strict_strips_tabs(self):
        assert _sanitize_strict("hello\tworld") == "helloworld"

    def test_sanitize_strict_strips_all_control_chars(self):
        assert _sanitize_strict("a\x00b\x01c\x7fd") == "abcd"

    def test_regex_patterns_correct(self):
        import re
        assert _SANITIZE_CONTENT_RE.pattern == r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'
        assert _SANITIZE_STRICT_RE.pattern == r'[<>\x00-\x1f\x7f]'
        assert _SANITIZE_RE.pattern == r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'


class TestSprint126SecurityDescriptionSanitization:
    """AutonomousDecision and SecurityEvent descriptions use content-safe sanitize."""

    def test_decision_reason_preserves_angle_brackets(self):
        from shadowd.cognition.autonomous_daemon import AutonomousDecision
        dec = AutonomousDecision(reason="latency < 100ms is acceptable", confidence=0.8, risk_level=0.1)
        assert "<" in dec.reason

    def test_security_event_description_preserves_angle_brackets(self):
        from shadowd.cognition.autonomous_daemon import SecurityEvent
        evt = SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_BEHAVIOR,
            description="Detected <script>alert(1)</script> in process",
            source="test",
        )
        assert "<" in evt.description

    def test_security_event_source_strips_angle_brackets(self):
        from shadowd.cognition.autonomous_daemon import SecurityEvent
        evt = SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_BEHAVIOR,
            description="test",
            source="src<script>",
        )
        assert "<" not in evt.source


class TestSprint126KillSwitchThreadSafety:
    """AD-T4: engage/disengage kill switch acquire lock."""

    def test_engage_kill_switch_thread_safe(self):
        g = SecurityGuardian()
        g.engage_kill_switch()
        assert g.is_kill_switch_engaged()

    def test_disengage_kill_switch_thread_safe(self):
        g = SecurityGuardian()
        g.engage_kill_switch()
        g.disengage_kill_switch()
        assert not g.is_kill_switch_engaged()

    def test_concurrent_kill_switch(self):
        g = SecurityGuardian()
        errors = []
        def engage_loop():
            for _ in range(100):
                g.engage_kill_switch()
        def disengage_loop():
            for _ in range(100):
                g.disengage_kill_switch()
        t1 = threading.Thread(target=engage_loop)
        t2 = threading.Thread(target=disengage_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert g.get_threat_level() in (ThreatLevel.NONE, ThreatLevel.CRITICAL)


class TestSprint126ScanProcessesThreadSafety:
    """AD-T1: scan_processes fully under lock."""

    def test_scan_processes_baseline_empty(self):
        g = SecurityGuardian()
        procs = [{"command": "bash", "pid": "1", "cpu_pct": "0.1", "mem_pct": "0.2", "user": "root"}]
        events = g.scan_processes(procs)
        assert len(events) == 0

    def test_scan_processes_suspicious(self):
        g = SecurityGuardian()
        g._process_baseline = {"bash", "ls"}
        procs = [{"command": "xmrig", "pid": "1234", "cpu_pct": "99", "mem_pct": "50", "user": "nobody"}]
        events = g.scan_processes(procs)
        assert len(events) == 1
        assert events[0].event_type == SecurityEventType.PROCESS_ANOMALY
        assert events[0].threat_level == ThreatLevel.CRITICAL

    def test_scan_processes_stats_under_lock(self):
        g = SecurityGuardian()
        procs = [{"command": "bash", "pid": "1"}]
        g.scan_processes(procs)
        stats = g.get_stats()
        assert stats["scans_run"] == 1


class TestSprint126ScanNetworkUnderLock:
    """AD-T9: scan_network increments stats under lock."""

    def test_scan_network_stats(self):
        g = SecurityGuardian()
        g.scan_network(["normal_connection"])
        stats = g.get_stats()
        assert stats["scans_run"] == 1

    def test_scan_network_suspicious(self):
        g = SecurityGuardian()
        events = g.scan_network(["192.168.1.1:4444->10.0.0.1:80"])
        assert len(events) >= 1

    def test_scan_network_concurrent(self):
        g = SecurityGuardian()
        def scan_loop():
            for _ in range(50):
                g.scan_network(["normal"])
        threads = [threading.Thread(target=scan_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stats = g.get_stats()
        assert stats["scans_run"] == 200


class TestSprint126GetRecentTransitionsDeepCopy:
    """AD-T2: get_recent_transitions returns deep copy."""

    def test_transitions_are_deep_copy(self):
        rm = UserRhythmModel()
        rm.ingest(RhythmSample(activity_state=UserActivityState.PRESENT))
        rm.ingest(RhythmSample(activity_state=UserActivityState.IDLE))
        t1 = rm.get_recent_transitions()
        t2 = rm.get_recent_transitions()
        assert t1 == t2
        if t1:
            t1[0]["from"] = "MUTATED"
            assert t2[0]["from"] != "MUTATED"


class TestSprint126ObserverLockSafety:
    """AD-T8: observe_filesystem, add_watch_path, remove_watch_path under lock."""

    def test_add_watch_path_thread_safe(self):
        obs = MultiModalObserver()
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.write(b"test")
            f.flush()
            f.close()
            result = obs.add_watch_path(f.name)
            assert result is True
            assert f.name in obs.get_watched_paths()
        finally:
            if not f.closed:
                f.close()
            os.unlink(f.name)

    def test_remove_watch_path_thread_safe(self):
        obs = MultiModalObserver()
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.write(b"test")
            f.flush()
            f.close()
            obs.add_watch_path(f.name)
            result = obs.remove_watch_path(f.name)
            assert result is True
            assert f.name not in obs.get_watched_paths()
        finally:
            if not f.closed:
                f.close()
            os.unlink(f.name)

    def test_watched_paths_returns_copy(self):
        obs = MultiModalObserver()
        paths = obs.get_watched_paths()
        paths.append("/fake/path")
        assert "/fake/path" not in obs.get_watched_paths()


class TestSprint126DaemonVitalsAuth:
    """AD-T10: daemon/vitals endpoints require auth."""

    def test_vitals_baselines_has_auth_dependency(self):
        from shadowd.cognition.api_router import router
        baselines_route = None
        state_route = None
        vitals_route = None
        for route in router.routes:
            if hasattr(route, 'path'):
                path = route.path
                if path.endswith("/daemon/vitals/baselines"):
                    baselines_route = route
                elif path.endswith("/daemon/vitals/state"):
                    state_route = route
                elif path.endswith("/daemon/vitals"):
                    vitals_route = route
        assert baselines_route is not None, "baselines route not found"
        assert state_route is not None, "state route not found"
        assert vitals_route is not None, "vitals route not found"
        assert len(baselines_route.dependencies) > 0, "baselines route missing auth"
        assert len(state_route.dependencies) > 0, "state route missing auth"
        assert len(vitals_route.dependencies) > 0, "vitals route missing auth"


class TestSprint126TickPulseDataFallback:
    """AD-T5: _tick uses local pulse_data variable instead of unreferenced 'pulse'."""

    def test_tick_creates_pulse_data_variable(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._tick)
        assert "pulse_data" in source
        assert "pulse.data if 'pulse' in dir()" not in source

    def test_tick_uses_get_running_loop(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._tick)
        assert "get_event_loop" not in source
        assert "get_running_loop" in source


class TestSprint126RhythmModelEdgeCases:
    """Additional edge cases for UserRhythmModel."""

    def test_ingest_empty_location_type(self):
        rm = UserRhythmModel()
        rm.ingest(RhythmSample(activity_state=UserActivityState.PRESENT, location_type=""))
        assert rm.get_stats()["locations_tracked"] == 0

    def test_ingest_location_type_sanitized(self):
        rm = UserRhythmModel()
        rm.ingest(RhythmSample(activity_state=UserActivityState.PRESENT, location_type="home<script>"))
        assert rm.get_stats()["locations_tracked"] == 1

    def test_ingest_clamps_heart_rate(self):
        rm = UserRhythmModel()
        rm.ingest(RhythmSample(activity_state=UserActivityState.PRESENT, heart_rate=500.0))
        baselines = rm.get_baselines()
        assert baselines["avg_heart_rate"] == 300.0

    def test_ingest_clamps_negative_stress(self):
        rm = UserRhythmModel()
        rm.ingest(RhythmSample(activity_state=UserActivityState.PRESENT, stress_level=-0.5))
        assert rm.get_current_stress() == 0.0

    def test_ingest_clamps_sleep_hours(self):
        rm = UserRhythmModel()
        rm.ingest(RhythmSample(activity_state=UserActivityState.AWAY, sleep_hours=30.0))
        baselines = rm.get_baselines()
        assert baselines["avg_sleep_hours"] == 24.0

    def test_activity_schedule_returns_all_24_hours(self):
        rm = UserRhythmModel()
        rm.ingest(RhythmSample(activity_state=UserActivityState.PRESENT))
        schedule = rm.get_activity_schedule()
        assert len(schedule) == 24
        assert all(0.0 <= v <= 1.0 for v in schedule.values())

    def test_rhythm_model_thread_safe_concurrent_ingest(self):
        rm = UserRhythmModel()
        errors = []
        def ingest_loop():
            try:
                for i in range(50):
                    rm.ingest(RhythmSample(
                        activity_state=UserActivityState.PRESENT,
                        heart_rate=60.0 + i,
                    ))
            except Exception as e:
                errors.append(type(e).__name__)
        threads = [threading.Thread(target=ingest_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert rm.get_stats()["samples_total"] == 200


class TestSprint126DaemonStatusDeepCopy:
    """Verify get_status returns deep copies of mutable data."""

    def test_status_stats_are_copy(self):
        daemon = AutonomousDaemon()
        s1 = daemon.get_status()
        s2 = daemon.get_status()
        assert s1["stats"] is not s2["stats"]

    def test_status_decisions_are_copy(self):
        daemon = AutonomousDaemon()
        s1 = daemon.get_status()
        s2 = daemon.get_status()
        assert s1["recent_decisions"] is not s2["recent_decisions"]


class TestSprint126SecurityEventDeepCopy:
    """Verify SecurityEvent.to_dict returns deep copy of details."""

    def test_security_event_details_deep_copy(self):
        from shadowd.cognition.autonomous_daemon import SecurityEvent
        evt = SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_BEHAVIOR,
            details={"key": "value", "nested": {"a": 1}},
        )
        d1 = evt.to_dict()
        d2 = evt.to_dict()
        assert d1["details"] is not d2["details"]
        assert d1["details"]["nested"] is not d2["details"]["nested"]


class TestSprint126ObservationPulseDeepCopy:
    """Verify ObservationPulse.to_dict returns deep copy of data."""

    def test_observation_pulse_data_deep_copy(self):
        pulse = ObservationPulse(
            observation_type=ObservationType.SYSTEM_METRICS,
            data={"key": "value", "nested": {"a": 1}},
            confidence=0.9,
        )
        d1 = pulse.to_dict()
        d2 = pulse.to_dict()
        assert d1["data"] is not d2["data"]
        assert d1["data"]["nested"] is not d2["data"]["nested"]


class TestSprint126ProcessBurstFix:
    """Bug fix: scan_processes burst event was created per new process, now only ONE."""

    def test_burst_creates_single_event(self):
        g = SecurityGuardian()
        g._process_baseline = {"bash", "ls", "ps", "cat"}
        procs = [{"command": f"newproc_{i}", "pid": str(100 + i)} for i in range(10)]
        events = g.scan_processes(procs)
        burst_events = [e for e in events if "Process burst" in e.description]
        assert len(burst_events) == 1  # was 10 (one per non-suspicious new process)

    def test_burst_and_suspicious_coexist(self):
        g = SecurityGuardian()
        g._process_baseline = {"bash", "ls"}
        procs = [
            {"command": "xmrig", "pid": "100"},
            {"command": "newproc_1", "pid": "101"},
            {"command": "newproc_2", "pid": "102"},
            {"command": "newproc_3", "pid": "103"},
            {"command": "newproc_4", "pid": "104"},
            {"command": "newproc_5", "pid": "105"},
            {"command": "newproc_6", "pid": "106"},
        ]
        events = g.scan_processes(procs)
        suspicious = [e for e in events if "SUSPICIOUS" in e.description]
        burst = [e for e in events if "burst" in e.description.lower()]
        assert len(suspicious) == 1
        assert len(burst) == 1

    def test_no_burst_under_threshold(self):
        g = SecurityGuardian()
        g._process_baseline = {"bash"}
        procs = [{"command": f"newproc_{i}", "pid": str(i)} for i in range(3)]
        events = g.scan_processes(procs)
        burst = [e for e in events if "burst" in e.description.lower()]
        assert len(burst) == 0


class TestSprint126RecordDecisionThreadSafety:
    """Verify _record_decision is thread-safe."""

    def test_concurrent_record_decision(self):
        daemon = AutonomousDaemon()
        errors = []
        def record_loop():
            try:
                for i in range(50):
                    from shadowd.cognition.autonomous_daemon import AutonomousDecision
                    d = AutonomousDecision(
                        action_type=AutonomousActionType.HEALTH_CHECK,
                        reason=f"test decision {i}",
                        confidence=0.8,
                        risk_level=0.1,
                    )
                    daemon._record_decision(d)
            except Exception as e:
                errors.append(type(e).__name__)
        threads = [threading.Thread(target=record_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(daemon._decisions) == 200


class TestSprint126DaemonStartStopIdempotent:
    """Verify start/stop are idempotent."""

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        daemon = AutonomousDaemon()
        await daemon.start()
        assert daemon.is_running()
        await daemon.start()
        assert daemon.is_running()
        await daemon.stop()
        assert not daemon.is_running()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        daemon = AutonomousDaemon()
        await daemon.start()
        await daemon.stop()
        assert not daemon.is_running()
        await daemon.stop()
        assert not daemon.is_running()


class TestSprint126DaemonTickClamping:
    """Verify tick interval clamping."""

    def test_tick_too_low_clamped(self):
        daemon = AutonomousDaemon(base_tick_seconds=1.0)
        assert daemon._base_tick == 5.0

    def test_tick_too_high_clamped(self):
        daemon = AutonomousDaemon(base_tick_seconds=120.0)
        assert daemon._base_tick == 60.0

    def test_tick_valid(self):
        daemon = AutonomousDaemon(base_tick_seconds=15.0)
        assert daemon._base_tick == 15.0


class TestSprint126DefensiveShadowOSAccess:
    """Verify daemon methods use getattr/hasattr for shadow_os attributes."""

    def test_curiosity_drive_uses_getattr(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._run_curiosity_drive)
        assert "getattr" in source

    def test_proactive_insight_uses_getattr(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._run_proactive_insight)
        assert "getattr" in source

    def test_pattern_mining_uses_getattr(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._run_pattern_mining)
        assert "getattr" in source

    def test_consolidation_uses_getattr(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._run_consolidation)
        assert "getattr" in source

    def test_dream_cycle_uses_getattr(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._run_dream_cycle)
        assert "getattr" in source

    def test_curiosity_drive_no_direct_attribute_access(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._run_curiosity_drive)
        assert "self._shadow_os.temporal_reasoning" not in source

    def test_proactive_insight_no_direct_attribute_access(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._run_proactive_insight)
        assert "self._shadow_os.anomaly_detector" not in source

    def test_pattern_mining_no_direct_attribute_access(self):
        import inspect
        source = inspect.getsource(AutonomousDaemon._run_pattern_mining)
        assert "self._shadow_os.recall_synthesis" not in source


class TestSprint126DangerousPathBlocking:
    """Verify add_watch_path blocks dangerous system paths."""

    def test_blocks_etc_shadow(self):
        obs = MultiModalObserver()
        result = obs.add_watch_path("/etc/shadow")
        assert result is False

    def test_blocks_root_ssh(self):
        obs = MultiModalObserver()
        result = obs.add_watch_path("/root/.ssh")
        assert result is False

    def test_allows_normal_path(self):
        import tempfile
        obs = MultiModalObserver()
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.write(b"test")
            f.flush()
            f.close()
            result = obs.add_watch_path(f.name)
            assert result is True
        finally:
            if not f.closed:
                f.close()
            os.unlink(f.name)

    def test_max_watched_paths(self):
        import tempfile
        obs = MultiModalObserver()
        temp_files = []
        try:
            for i in range(obs._MAX_WATCHED_PATHS):
                f = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}")
                f.write(b"test")
                f.flush()
                f.close()
                temp_files.append(f.name)
            results = [obs.add_watch_path(p) for p in temp_files]
            assert sum(1 for r in results if r) == obs._MAX_WATCHED_PATHS
            beyond = obs.add_watch_path(tempfile.mktemp())
            assert beyond is False
        finally:
            for p in temp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass


class TestSprint126HashFileSizeLimit:
    """Verify _hash_file has a 10MB read limit."""

    def test_hash_file_has_max_bytes(self):
        import inspect
        source = inspect.getsource(MultiModalObserver._hash_file)
        assert "max_bytes" in source

    def test_hash_file_small_file(self):
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.write(b"hello world")
            f.flush()
            f.close()
            result = MultiModalObserver._hash_file(f.name)
            assert len(result) == 64  # SHA-256 hex digest length
        finally:
            if not f.closed:
                f.close()
            os.unlink(f.name)


class TestSprint126DaemonStatusAuth:
    """Verify /daemon/status endpoint requires auth."""

    def test_daemon_status_has_auth(self):
        from shadowd.cognition.api_router import router
        status_route = None
        for route in router.routes:
            if hasattr(route, 'path') and route.path.endswith("/daemon/status"):
                status_route = route
        assert status_route is not None
        assert len(status_route.dependencies) > 0, "daemon/status missing auth"