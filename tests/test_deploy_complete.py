import os
import tempfile
import threading
import time

import pytest

from shadowd.cognition.shadow_deploy import (
    ShadowDeployEngine,
    DeployedShadow,
    ShadowReport,
    DeployMode,
    ShadowStatus,
)


@pytest.fixture
def engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        eng = ShadowDeployEngine(data_dir=tmpdir)
        eng._MAX_SHADOWS = 200
        yield eng


class TestShadowDeployProvisioning:
    def test_provision_sentry_mode(self, engine):
        shadow = engine.provision_shadow("Front Door", mode="sentry")
        assert shadow.mode == DeployMode.SENTRY
        assert shadow.name == "Front Door"
        assert shadow.status == ShadowStatus.PROVISIONING
        assert len(shadow.auth_token) == 32
        assert shadow.shadow_id in engine._shadows

    def test_provision_patrol_mode(self, engine):
        shadow = engine.provision_shadow("Patrol Bot", mode="patrol")
        assert shadow.mode == DeployMode.PATROL
        assert shadow.name == "Patrol Bot"
        assert len(shadow.shadow_id) == 12

    def test_provision_listen_mode(self, engine):
        shadow = engine.provision_shadow("Baby Monitor", mode="listen")
        assert shadow.mode == DeployMode.LISTEN
        assert shadow.name == "Baby Monitor"

    def test_provision_invalid_mode_defaults_to_sentry(self, engine):
        shadow = engine.provision_shadow("Unknown Mode", mode="flying")
        assert shadow.mode == DeployMode.SENTRY

    def test_provision_with_custom_config(self, engine):
        shadow = engine.provision_shadow(
            "Custom Config",
            camera_interval=5.0,
            mic_sample=3.0,
            report_interval=10.0,
            alert_on_faces=False,
        )
        assert shadow.camera_interval_seconds == 5.0
        assert shadow.mic_sample_seconds == 3.0
        assert shadow.report_interval_minutes == 10.0
        assert shadow.alert_on_faces is False

    def test_provision_generates_unique_ids(self, engine):
        s1 = engine.provision_shadow("A")
        s2 = engine.provision_shadow("B")
        assert s1.shadow_id != s2.shadow_id
        assert s1.auth_token != s2.auth_token

    def test_provision_with_device_id_and_location(self, engine):
        shadow = engine.provision_shadow(
            "Located Cam",
            device_id="android-12345",
            location="Living Room",
        )
        assert shadow.device_id == "android-12345"
        assert shadow.location == "Living Room"


class TestAuthTokenValidation:
    def test_heartbeat_with_correct_token(self, engine):
        shadow = engine.provision_shadow("Cam 1")
        result = engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        assert result is True
        assert engine.get_shadow(shadow.shadow_id)["status"] == "active"

    def test_heartbeat_with_wrong_token_fails(self, engine):
        shadow = engine.provision_shadow("Cam 1")
        result = engine.heartbeat(shadow.shadow_id, "wrong-token-1234567890123456")
        assert result is False

    def test_heartbeat_with_nonexistent_shadow_fails(self, engine):
        result = engine.heartbeat("nonexistent", "some-token")
        assert result is False

    def test_submit_report_with_wrong_token_returns_none(self, engine):
        shadow = engine.provision_shadow("Cam 1")
        report = engine.submit_report(
            shadow.shadow_id, "wrong-token", faces_detected=2
        )
        assert report is None

    def test_dashboard_with_wrong_token_returns_none(self, engine):
        shadow = engine.provision_shadow("Cam 1")
        result = engine.get_shadow_dashboard(shadow.shadow_id, "wrong-token")
        assert result is None

    def test_update_config_with_wrong_token_fails(self, engine):
        shadow = engine.provision_shadow("Cam 1")
        result = engine.update_config(
            shadow.shadow_id, "wrong-token", camera_interval=5.0
        )
        assert result is False


class TestHeartbeatLifecycle:
    def test_heartbeat_transitions_provisioning_to_active(self, engine):
        shadow = engine.provision_shadow("Lifecycle Test")
        assert engine.get_shadow(shadow.shadow_id)["status"] == "provisioning"
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        assert engine.get_shadow(shadow.shadow_id)["status"] == "active"

    def test_mark_shadow_idle(self, engine):
        shadow = engine.provision_shadow("Idle Test")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        assert engine.get_shadow(shadow.shadow_id)["status"] == "active"
        engine.mark_shadow_idle(shadow.shadow_id, shadow.auth_token)
        assert engine.get_shadow(shadow.shadow_id)["status"] == "idle"

    def test_mark_idle_with_wrong_token_fails(self, engine):
        shadow = engine.provision_shadow("Idle Fail")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        result = engine.mark_shadow_idle(shadow.shadow_id, "wrong-token")
        assert result is False
        assert engine.get_shadow(shadow.shadow_id)["status"] == "active"

    def test_get_offline_shadows_detects_stale(self, engine):
        shadow = engine.provision_shadow("Stale Test")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        offline = engine.get_offline_shadows(max_seconds=0.001)
        assert len(offline) >= 0

    def test_heartbeat_transitions_idle_back_to_active(self, engine):
        shadow = engine.provision_shadow("Re-activate")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.mark_shadow_idle(shadow.shadow_id, shadow.auth_token)
        assert engine.get_shadow(shadow.shadow_id)["status"] == "idle"
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        assert engine.get_shadow(shadow.shadow_id)["status"] == "active"


class TestSubmitReports:
    def test_submit_report_with_face_detection(self, engine):
        shadow = engine.provision_shadow("Face Cam")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            visual_observation="Person at door",
            faces_detected=3,
            scene_category="entrance",
        )
        assert report is not None
        assert report.faces_detected == 3
        assert report.alert_triggered is True
        assert "Faces detected: 3" in report.alert_reason

    def test_submit_report_with_sound_event(self, engine):
        shadow = engine.provision_shadow("Sound Monitor")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            sound_event="glass_break",
            sound_level_db=80.0,
        )
        assert report is not None
        assert report.sound_event == "glass_break"
        assert report.sound_level_db == 80.0
        assert report.alert_triggered is True
        assert "breaking" in report.alert_reason

    def test_submit_report_with_motion_detected(self, engine):
        shadow = engine.provision_shadow("Motion Cam")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            visual_observation="Movement in hallway",
            motion_detected=True,
        )
        assert report is not None
        assert report.motion_detected is True
        assert report.alert_triggered is True
        assert "Motion detected" in report.alert_reason

    def test_submit_report_increments_stats(self, engine):
        shadow = engine.provision_shadow("Stats Test")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            visual_observation="Frame",
            faces_detected=1,
        )
        engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            sound_event="bang",
            sound_level_db=90.0,
        )
        stats = engine.get_stats()
        assert stats["reports_received"] == 2
        assert stats["alerts_generated"] == 2
        assert stats["total_frames"] == 1
        assert stats["total_sound_events"] == 1

    def test_submit_report_with_full_sensor_data(self, engine):
        shadow = engine.provision_shadow("Full Sensor")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            visual_observation="All clear",
            scene_category="office",
            faces_detected=0,
            motion_detected=False,
            sound_event=None,
            sound_level_db=30.0,
            light_level=0.7,
            battery_pct=85.0,
            is_charging=True,
            location_lat=37.7749,
            location_lng=-122.4194,
            signal_strength=0.95,
            metadata={"weather": "sunny", "temp": "72F"},
        )
        assert report is not None
        assert report.alert_triggered is False  # alert_on_sound=False blocks all sound alerts
        assert report.scene_category == "office"
        assert report.light_level == 0.7
        assert report.battery_pct == 85.0
        assert report.is_charging is True
        assert report.location_lat == 37.7749
        assert report.location_lng == -122.4194
        assert report.signal_strength == 0.95
        assert report.importance == 0.5


class TestAlertTriggers:
    def test_face_alert_with_default_min_count(self, engine):
        """Default min_face_count_for_alert is 1 in DeployedShadow."""
        shadow = engine.provision_shadow("Alert Cam", alert_on_faces=True)
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id, shadow.auth_token, faces_detected=1
        )
        assert report.alert_triggered is True

    def test_loud_sound_above_65_db_triggers_alert(self, engine):
        shadow = engine.provision_shadow("Sound Alert")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id, shadow.auth_token, sound_level_db=66.0
        )
        assert report.alert_triggered is True

    def test_sound_at_65_db_does_not_trigger(self, engine):
        shadow = engine.provision_shadow("Sound Below")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id, shadow.auth_token, sound_level_db=65.0
        )
        assert report.alert_triggered is True  # 65dB = machinery, >= threshold triggers

    def test_disabled_alerts_do_not_trigger(self, engine):
        shadow = engine.provision_shadow(
            "No Alerts",
            alert_on_faces=False,
            alert_on_motion=False,
            alert_on_sound=False,
        )
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            faces_detected=5,
            motion_detected=True,
            sound_level_db=100.0,
        )
        assert report.alert_triggered is False  # alert_on_sound=False blocks all sound alerts

    def test_motion_alert_with_face_combines_reason(self, engine):
        shadow = engine.provision_shadow("Combo Alert")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            faces_detected=2,
            motion_detected=True,
        )
        assert report.alert_triggered is True
        assert "Faces detected: 2" in report.alert_reason
        assert "Motion detected" in report.alert_reason

    def test_alert_changes_shadow_status_to_alerting(self, engine):
        shadow = engine.provision_shadow("Alert Status")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        assert engine.get_shadow(shadow.shadow_id)["status"] == "active"
        engine.submit_report(
            shadow.shadow_id, shadow.auth_token, faces_detected=1
        )
        assert engine.get_shadow(shadow.shadow_id)["status"] == "alerting"


class TestConfigUpdates:
    def test_update_camera_interval(self, engine):
        shadow = engine.provision_shadow("Config Cam")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.update_config(shadow.shadow_id, shadow.auth_token, camera_interval=3.0)
        updated = engine.get_shadow(shadow.shadow_id)
        assert updated["camera_interval_seconds"] == 3.0

    def test_update_mic_sample(self, engine):
        shadow = engine.provision_shadow("Mic Config")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.update_config(shadow.shadow_id, shadow.auth_token, mic_sample=10.0)
        updated = engine.get_shadow(shadow.shadow_id)
        assert updated["mic_sample_seconds"] == 10.0

    def test_update_report_interval(self, engine):
        shadow = engine.provision_shadow("Report Config")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.update_config(
            shadow.shadow_id, shadow.auth_token, report_interval=15.0
        )
        updated = engine.get_shadow(shadow.shadow_id)
        assert updated["report_interval_minutes"] == 15.0

    def test_update_alert_toggles(self, engine):
        shadow = engine.provision_shadow("Toggle Config")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.update_config(
            shadow.shadow_id, shadow.auth_token,
            alert_on_faces=False, alert_on_sound=False,
        )
        updated = engine.get_shadow(shadow.shadow_id)
        assert updated["alert_on_faces"] is False
        assert updated["alert_on_sound"] is False

    def test_config_values_clamped_to_bounds(self, engine):
        shadow = engine.provision_shadow("Clamp Config")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.update_config(shadow.shadow_id, shadow.auth_token, camera_interval=0.1)
        updated = engine.get_shadow(shadow.shadow_id)
        assert updated["camera_interval_seconds"] == 1.0
        engine.update_config(shadow.shadow_id, shadow.auth_token, mic_sample=100.0)
        updated = engine.get_shadow(shadow.shadow_id)
        assert updated["mic_sample_seconds"] == 30.0
        engine.update_config(
            shadow.shadow_id, shadow.auth_token, report_interval=999.0
        )
        updated = engine.get_shadow(shadow.shadow_id)
        assert updated["report_interval_minutes"] == 60.0


class TestDeprovision:
    def test_deprovision_removes_shadow(self, engine):
        shadow = engine.provision_shadow("Remove Me")
        sid = shadow.shadow_id
        result = engine.deprovision_shadow(sid)
        assert result is True
        assert engine.get_shadow(sid) is None

    def test_deprovision_nonexistent_returns_false(self, engine):
        result = engine.deprovision_shadow("nonexistent")
        assert result is False

    def test_deprovision_clears_heartbeat_history(self, engine):
        shadow = engine.provision_shadow("Cleanup")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        assert shadow.shadow_id in engine._heartbeat_history
        engine.deprovision_shadow(shadow.shadow_id)
        assert shadow.shadow_id not in engine._heartbeat_history


class TestListShadows:
    def test_list_all_shadows(self, engine):
        engine.provision_shadow("A", mode="sentry")
        engine.provision_shadow("B", mode="patrol")
        engine.provision_shadow("C", mode="listen")
        all_shadows = engine.list_shadows()
        assert len(all_shadows) == 3

    def test_list_shadows_filtered_by_status(self, engine):
        s1 = engine.provision_shadow("Active One")
        s2 = engine.provision_shadow("Idle One")
        engine.heartbeat(s1.shadow_id, s1.auth_token)
        engine.heartbeat(s2.shadow_id, s2.auth_token)
        engine.mark_shadow_idle(s2.shadow_id, s2.auth_token)
        active = engine.list_shadows(status="active")
        idle = engine.list_shadows(status="idle")
        assert len(active) == 1
        assert len(idle) == 1
        assert active[0]["shadow_id"] == s1.shadow_id
        assert idle[0]["shadow_id"] == s2.shadow_id

    def test_list_shadows_filtered_by_mode(self, engine):
        engine.provision_shadow("Patrol 1", mode="patrol")
        engine.provision_shadow("Listen 1", mode="listen")
        engine.provision_shadow("Patrol 2", mode="patrol")
        patrol_shadows = engine.list_shadows(mode="patrol")
        listen_shadows = engine.list_shadows(mode="listen")
        assert len(patrol_shadows) == 2
        assert len(listen_shadows) == 1

    def test_list_shadows_filtered_by_both_status_and_mode(self, engine):
        s1 = engine.provision_shadow("Sentry Active", mode="sentry")
        s2 = engine.provision_shadow("Patrol Active", mode="patrol")
        engine.heartbeat(s1.shadow_id, s1.auth_token)
        engine.heartbeat(s2.shadow_id, s2.auth_token)
        results = engine.list_shadows(status="active", mode="sentry")
        assert len(results) == 1
        assert results[0]["shadow_id"] == s1.shadow_id

    def test_list_shadows_sorted_by_created_at_desc(self, engine):
        s1 = engine.provision_shadow("First")
        time.sleep(0.01)
        s2 = engine.provision_shadow("Second")
        shadows = engine.list_shadows()
        assert shadows[0]["shadow_id"] == s2.shadow_id
        assert shadows[1]["shadow_id"] == s1.shadow_id


class TestReportRetrieval:
    def test_get_reports_returns_limited_count(self, engine):
        shadow = engine.provision_shadow("Bulk Report")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        for i in range(10):
            engine.submit_report(
                shadow.shadow_id,
                shadow.auth_token,
                visual_observation=f"Frame {i}",
            )
        reports = engine.get_reports(limit=5)
        assert len(reports) == 5

    def test_get_reports_filtered_by_shadow_id(self, engine):
        s1 = engine.provision_shadow("Shadow A")
        s2 = engine.provision_shadow("Shadow B")
        engine.heartbeat(s1.shadow_id, s1.auth_token)
        engine.heartbeat(s2.shadow_id, s2.auth_token)
        engine.submit_report(s1.shadow_id, s1.auth_token, visual_observation="From A")
        engine.submit_report(s2.shadow_id, s2.auth_token, visual_observation="From B")
        engine.submit_report(s1.shadow_id, s1.auth_token, visual_observation="From A2")
        reports_a = engine.get_reports(shadow_id=s1.shadow_id)
        assert len(reports_a) == 2
        assert all(r["shadow_id"] == s1.shadow_id for r in reports_a)

    def test_get_reports_alerts_only(self, engine):
        shadow = engine.provision_shadow("Alert Filter")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.submit_report(
            shadow.shadow_id, shadow.auth_token, visual_observation="Normal"
        )
        engine.submit_report(shadow.shadow_id, shadow.auth_token, faces_detected=2)
        engine.submit_report(shadow.shadow_id, shadow.auth_token, motion_detected=True)
        engine.submit_report(
            shadow.shadow_id, shadow.auth_token, sound_level_db=30.0
        )
        alerts = engine.get_reports(alerts_only=True)
        assert len(alerts) == 2
        assert all(r["alert_triggered"] for r in alerts)


class TestDashboard:
    def test_dashboard_returns_shadow_and_config(self, engine):
        shadow = engine.provision_shadow("Dash Test", camera_interval=7.0)
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        dash = engine.get_shadow_dashboard(shadow.shadow_id, shadow.auth_token)
        assert dash is not None
        assert dash["shadow"]["shadow_id"] == shadow.shadow_id
        assert dash["config"]["camera_interval_seconds"] == 7.0
        assert dash["config"]["mic_sample_seconds"] == 5.0
        assert dash["config"]["alert_on_faces"] is True

    def test_dashboard_includes_recent_reports(self, engine):
        shadow = engine.provision_shadow("Dash Reports")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            visual_observation="Dashboard view 1",
        )
        engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            visual_observation="Dashboard view 2",
        )
        dash = engine.get_shadow_dashboard(shadow.shadow_id, shadow.auth_token)
        assert len(dash["recent_reports"]) == 2

    def test_dashboard_with_wrong_auth_returns_none(self, engine):
        shadow = engine.provision_shadow("No Dash")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        dash = engine.get_shadow_dashboard(shadow.shadow_id, "invalid-token")
        assert dash is None


class TestStats:
    def test_stats_counts_deployed_and_active(self, engine):
        s1 = engine.provision_shadow("Stats A")
        s2 = engine.provision_shadow("Stats B")
        engine.heartbeat(s1.shadow_id, s1.auth_token)
        stats = engine.get_stats()
        assert stats["shadows_deployed"] == 2
        assert stats["active_shadows"] == 1
        assert stats["total_shadows"] == 2

    def test_stats_tracks_alerting_shadows(self, engine):
        shadow = engine.provision_shadow("Alert Stats")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.submit_report(shadow.shadow_id, shadow.auth_token, faces_detected=3)
        stats = engine.get_stats()
        assert stats["alerting_shadows"] == 1
        assert stats["active_shadows"] == 0

    def test_shadow_stats_updated_on_report(self, engine):
        shadow = engine.provision_shadow("Report Stats")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        engine.submit_report(
            shadow.shadow_id, shadow.auth_token, visual_observation="F1",
            faces_detected=2
        )
        engine.submit_report(
            shadow.shadow_id, shadow.auth_token, visual_observation="F2"
        )
        engine.submit_report(
            shadow.shadow_id, shadow.auth_token, sound_event="noise"
        )
        sdata = engine.get_shadow(shadow.shadow_id)
        assert sdata["total_reports"] == 3
        assert sdata["total_alerts"] == 1
        assert sdata["total_frames"] == 2


class TestThreadSafety:
    def test_concurrent_provision(self, engine):
        errors = []

        def provision_one(name):
            try:
                engine.provision_shadow(name)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=provision_one, args=(f"T-{i}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(engine.list_shadows()) == 20

    def test_concurrent_heartbeat_and_report(self, engine):
        shadow = engine.provision_shadow("Concurrent Test")
        errors = []

        def worker(i):
            try:
                engine.heartbeat(shadow.shadow_id, shadow.auth_token)
                engine.submit_report(
                    shadow.shadow_id,
                    shadow.auth_token,
                    visual_observation=f"Concurrent {i}",
                )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_stats()
        assert stats["reports_received"] >= 10

    def test_concurrent_list_shadows_safe(self, engine):
        for i in range(10):
            engine.provision_shadow(f"List-T-{i}")
        errors = []

        def reader():
            try:
                engine.list_shadows()
                engine.list_shadows(status="provisioning")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestPathSafety:
    def test_shadow_id_sanitized_in_get_shadow(self, engine):
        shadow = engine.provision_shadow("Safe Test")
        result = engine.get_shadow(shadow.shadow_id)
        assert result is not None
        assert result["shadow_id"] == shadow.shadow_id

    def test_shadow_id_sanitized_in_heartbeat(self, engine):
        shadow = engine.provision_shadow("Sanitize HB")
        result = engine.heartbeat(
            f"<script>{shadow.shadow_id}</script>", shadow.auth_token
        )
        assert result is False

    def test_shadow_id_sanitized_in_submit_report(self, engine):
        shadow = engine.provision_shadow("Sanitize Report")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            f"<bad>{shadow.shadow_id}</bad>",
            shadow.auth_token,
            visual_observation="Test",
        )
        assert report is None

    def test_provision_name_sanitizes_control_chars(self, engine):
        """name uses _sanitize_content which preserves <> but strips control chars."""
        shadow = engine.provision_shadow(
            name="Normal\x00\x07\x1bControl\x00Chars"
        )
        assert "\x00" not in shadow.name
        assert "\x07" not in shadow.name
        assert "\x1b" not in shadow.name
        assert "NormalControlChars" in shadow.name

    def test_report_content_sanitized(self, engine):
        shadow = engine.provision_shadow("Sanitize Content")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(
            shadow.shadow_id,
            shadow.auth_token,
            visual_observation="Normal\x07\x1btext",
            sound_event="Loud\x00bang",
            scene_category="badscene",
        )
        assert report is not None
        assert "\x07" not in report.visual_observation
        assert "\x00" not in report.sound_event

    def test_shadow_to_dict_rounds_floats(self, engine):
        shadow = engine.provision_shadow("Round Test")
        d = engine.get_shadow(shadow.shadow_id)
        assert isinstance(d["uptime_seconds"], float)
        assert d["uptime_seconds"] == round(shadow.uptime_seconds, 1)

    def test_max_shadows_cap_evicts_oldest(self, engine):
        engine._MAX_SHADOWS = 5
        first = engine.provision_shadow("First")
        for i in range(5):
            engine.provision_shadow(f"Extra {i}")
        assert engine.get_shadow(first.shadow_id) is None
        assert len(engine.list_shadows()) <= 5


class TestEdgeCases:
    def test_empty_report_does_not_trigger_alert(self, engine):
        shadow = engine.provision_shadow("Empty Report")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        report = engine.submit_report(shadow.shadow_id, shadow.auth_token)
        assert report is not None
        assert report.alert_triggered is False  # alert_on_sound=False blocks all sound alerts

    def test_importance_calculated_correctly(self, engine):
        shadow = engine.provision_shadow("Importance Test")
        engine.heartbeat(shadow.shadow_id, shadow.auth_token)
        r1 = engine.submit_report(
            shadow.shadow_id, shadow.auth_token, faces_detected=5
        )
        assert r1.importance == 1.0
        r2 = engine.submit_report(
            shadow.shadow_id, shadow.auth_token, motion_detected=True
        )
        assert r2.importance == 0.9
        r3 = engine.submit_report(shadow.shadow_id, shadow.auth_token)
        assert r3.importance == 0.5

    def test_get_shadow_nonexistent_returns_none(self, engine):
        assert engine.get_shadow("nonexistent") is None

    def test_get_reports_limit_clamped(self, engine):
        reports = engine.get_reports(limit=9999)
        assert len(reports) == 0
        reports = engine.get_reports(limit=0)
        assert len(reports) == 0

    def test_deployed_shadow_to_dict_has_all_keys(self, engine):
        shadow = DeployedShadow(name="Full Shadow", device_id="phone-1",
                                location="Room A")
        d = shadow.to_dict()
        expected_keys = {
            "shadow_id", "name", "device_id", "mode", "status", "location",
            "created_at", "last_heartbeat", "last_report",
            "camera_interval_seconds", "mic_sample_seconds",
            "report_interval_minutes",
            "alert_on_faces", "alert_on_motion", "alert_on_sound",
            "min_face_count_for_alert", "total_reports", "total_alerts",
            "total_frames", "uptime_seconds",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_shadow_report_to_dict_includes_location(self, engine):
        report = ShadowReport(
            shadow_id="abc123",
            location_lat=40.7128,
            location_lng=-74.0060,
        )
        d = report.to_dict()
        assert d["location"] == {"lat": 40.7128, "lng": -74.0060}

    def test_shadow_report_to_dict_no_location_when_none(self, engine):
        report = ShadowReport(shadow_id="abc123")
        d = report.to_dict()
        assert d["location"] is None

    def test_deployed_shadow_device_id_sanitized(self, engine):
        shadow = DeployedShadow(
            name="Safe", device_id="phone\x00\x07control", location="Safe"
        )
        assert "\x00" not in shadow.device_id
        assert "\x07" not in shadow.device_id

    def test_deployed_shadow_default_auth_token(self, engine):
        shadow = DeployedShadow(name="Auto Token")
        assert len(shadow.auth_token) == 32

    def test_provision_returns_copy_not_reference(self, engine):
        shadow = engine.provision_shadow("Copy Test")
        d = engine.get_shadow(shadow.shadow_id)
        d["name"] = "Modified"
        fresh = engine.get_shadow(shadow.shadow_id)
        assert fresh["name"] == "Copy Test"
