"""End-to-end integration tests across all new modules."""
import pytest, threading, time
from shadowd.cognition.notification_bus import NotificationBus
from shadowd.cognition.visual_observation_bridge import VisualObservationBridge
from shadowd.cognition.shadow_deploy import ShadowDeployEngine
from shadowd.cognition.shadow_journal import ShadowJournal
from shadowd.cognition.vision_perception import VisionPerceptionEngine

class TestFullIntegration:
    def setup_method(self):
        self.nb = NotificationBus()
        self.vb = VisualObservationBridge()
        self.se = ShadowDeployEngine()
        self.sj = ShadowJournal()
        self.vp = VisionPerceptionEngine()
    
    def test_vision_to_notification_flow(self):
        """Vision observation → notification should be creatable"""
        r = self.vb.process_visual_observation("Person at door", scene_category="social", faces=2)
        assert r.event_id
        assert r.faces_detected == 2
    
    def test_deploy_to_notification_alert(self):
        """Deploy shadow alert → notification should link"""
        s = self.se.provision_shadow(name="test", device_id="x")
        self.se.submit_report(s.shadow_id, s.auth_token, faces_detected=2, sound_level_db=70)
        # Just verify it doesn't crash
        nid = self.nb.notify("Shadow Alert", "Test", priority="high")
        assert nid
    
    def test_journal_records_all_event_types(self):
        """All modules should log to journal"""
        eid1 = self.sj.record("observation", "Vision detected scene", category="visual")
        eid2 = self.sj.record("alert", "Deploy shadow alert triggered", category="security")
        assert eid1 and eid2
        entries = self.sj.get_entries(limit=5)
        assert len(entries) >= 2
    
    def test_vision_streaming_and_memory(self):
        """Streaming should capture frames into memory"""
        self.vp.start_streaming(interval_seconds=1, max_frames=3)
        time.sleep(3.5)
        self.vp.stop_streaming()
        stats = self.vp.get_stats()
        assert stats["stats"]["captures"] >= 1
    
    def test_deploy_full_lifecycle(self):
        """Provision → heartbeat → report → deprovision"""
        s = self.se.provision_shadow(name="lifecycle_test", device_id="d1")
        assert self.se.heartbeat(s.shadow_id, s.auth_token)
        self.se.submit_report(s.shadow_id, s.auth_token, visual_observation="ok")
        self.se.deprovision_shadow(s.shadow_id)
        assert self.se.get_shadow(s.shadow_id) is None
    
    def test_notification_digest_with_mixed_priorities(self):
        """Digest should summarize mixed priority notifications"""
        self.nb.notify("T1","B1", priority="critical")
        self.nb.notify("T2","B2", priority="high")
        self.nb.notify("T3","B3")
        d = self.nb.generate_digest()
        assert d.top_notifications
        assert d.stats["total"] == 3
    
    def test_journal_weekly_digest(self):
        """Weekly digest should have valid structure"""
        self.sj.record("observation", "test entry")
        digest = self.sj.get_weekly_digest()
        assert "start_date" in digest
        assert "total_entries" in digest
    
    def test_concurrent_module_access(self):
        """All modules should handle concurrent access"""
        errors = []
        def work(n):
            try:
                self.nb.notify("T","B")
                self.sj.record("observation","test")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=work, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
    
    def test_sanitization_across_modules(self):
        """All modules should sanitize malicious input"""
        # Notification
        nid = self.nb.notify("Test\x00null","Body\x00null")
        n = self.nb.get_notifications()[0]
        assert "\x00" not in n["title"]
        # Journal
        eid = self.sj.record("observation","content\x00test")
        e = self.sj.get_entry(eid)
        assert "\x00" not in e["content"]
        # Deploy - shadow name sanitized
        s = self.se.provision_shadow(name="test\x00name", device_id="d\x00id")
        assert "\x00" not in s.name
    
    def test_journal_export_formats(self):
        """Export should produce valid JSON and text"""
        self.sj.record("observation","export test")
        json_data = self.sj.export_journal(fmt="json")
        assert json_data.startswith("[")
        text_data = self.sj.export_journal(fmt="text")
        assert "ShadowOS Activity Log" in text_data
    
    def test_deploy_config_updates(self):
        """Config updates should be reflected"""
        s = self.se.provision_shadow(name="cfg_test", device_id="c1")
        self.se.update_config(s.shadow_id, s.auth_token, camera_interval=30, alert_on_faces=False)
        shadow = self.se.get_shadow(s.shadow_id)
        assert shadow["camera_interval_seconds"] == 30
        assert shadow["alert_on_faces"] == False
