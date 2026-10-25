import pytest, threading, time
from shadowd.cognition.notification_bus import (
    NotificationBus, Notification, NotificationDigest, NotificationPriority
)

class TestNotificationBusComprehensive:
    def setup_method(self):
        self.nb = NotificationBus()
    
    def test_basic_notify(self):
        nid = self.nb.notify("Hello", "World")
        assert nid and len(nid) == 12
        assert self.nb.get_unread_count() == 1
    
    def test_priority_levels(self):
        for p in ("low","normal","high","critical"):
            self.nb.notify("T", "B", priority=p)
        assert self.nb.get_unread_count() == 4
    
    def test_critical_alert(self):
        self.nb.notify_critical("SECURITY", "Breach detected")
        assert self.nb.get_unread_critical() == 1
    
    def test_anomaly_notification(self):
        self.nb.notify_from_anomaly("CPU at 95%", "HIGH")
        nfs = self.nb.get_notifications(priority="critical")
        assert len(nfs) >= 1
    
    def test_security_notification(self):
        self.nb.notify_from_security("Brute force", 3)
        self.nb.notify_from_security("Port scan", 1)
        crits = self.nb.get_notifications(priority="critical")
        norms = self.nb.get_notifications(priority="normal")
        assert len(crits) == 1 and len(norms) == 1
    
    def test_visual_notification(self):
        self.nb.notify_visual_observation("Person seen", faces=3)
        nfs = self.nb.get_notifications(priority="high")
        assert len(nfs) == 1
    
    def test_mark_read(self):
        nid = self.nb.notify("T","B")
        assert self.nb.mark_read(nid)
        assert self.nb.get_unread_count() == 0
    
    def test_mark_read_twice(self):
        nid = self.nb.notify("T","B")
        self.nb.mark_read(nid)
        assert not self.nb.mark_read(nid)
    
    def test_get_unread_only(self):
        nid1 = self.nb.notify("T1","B1")
        nid2 = self.nb.notify("T2","B2")
        self.nb.mark_read(nid1)
        unread = self.nb.get_notifications(unread_only=True)
        assert len(unread) == 1
    
    def test_clear_old(self):
        self.nb.notify("T","B")
        removed = self.nb.clear_old(max_age_hours=0.0)
        assert removed >= 0
    
    def test_digest_generation(self):
        for i in range(5):
            self.nb.notify(f"T{i}", f"B{i}", priority="high" if i%2 else "normal")
        d = self.nb.generate_digest()
        assert isinstance(d, NotificationDigest)
        assert d.title
        assert len(d.top_notifications) >= 1
    
    def test_digest_history(self):
        self.nb.notify("T","B")
        self.nb.generate_digest()
        digests = self.nb.get_digests(limit=5)
        assert len(digests) >= 1
    
    def test_stats_accuracy(self):
        self.nb.notify("T","B", priority="critical")
        self.nb.notify("T2","B2", priority="high")
        self.nb.notify("T3","B3")
        s = self.nb.get_stats()
        assert s["notifications_created"] == 3
        assert s["critical_priority"] == 1
        assert s["high_priority"] == 1
        assert s["unread_count"] == 3
    
    def test_thread_safety_stress(self):
        errors = []
        def push(n):
            try:
                self.nb.notify(f"T{n}", f"B{n}", priority="critical")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=push, args=(i,)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(errors) == 0
        assert self.nb.get_unread_count() == 50

class TestNotificationDataclass:
    def test_creation(self):
        n = Notification(title="Test", body="Body", priority=NotificationPriority.CRITICAL)
        assert n.priority == NotificationPriority.CRITICAL
    
    def test_sanitization(self):
        n = Notification(title="<script>xss</script>", body="test\x00nulls")
        assert "<script>" not in n.title
    
    def test_to_dict_safety(self):
        n = Notification(metadata={"key": "val"})
        d = n.to_dict()
        d["metadata"]["key"] = "mutated"
        assert n.metadata["key"] == "val"
    
    def test_expiration(self):
        n = Notification(title="T", body="B", expires_at="2027-01-01")
        assert n.expires_at == "2027-01-01"

class TestNotificationDigestDetails:
    def test_digest_creation(self):
        d = NotificationDigest(title="My Digest", summary="Summary")
        assert d.digest_id
        assert d.title == "My Digest"
    
    def test_digest_to_dict(self):
        d = NotificationDigest(title="D", summary="S", stats={"a": 1})
        result = d.to_dict()
        assert result["title"] == "D"
        assert result["stats"]["a"] == 1

print("NotificationBus comprehensive tests ready")
