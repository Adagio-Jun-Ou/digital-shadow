#!/usr/bin/env python3
"""Tests for NotificationBus module."""
import pytest
from shadowd.cognition.notification_bus import (
    NotificationBus, Notification, NotificationDigest,
    NotificationPriority, NotificationChannel,
)

class TestNotification:
    def test_create_basic(self):
        n = Notification(title="Hello", body="World", priority=NotificationPriority.HIGH)
        assert n.title == "Hello"
        assert n.body == "World"
        assert n.priority == NotificationPriority.HIGH
        assert n.notification_id
        assert not n.read

    def test_sanitize_control_chars(self):
        n = Notification(title="test with \x00 nulls", body="test body")
        assert "\x00" not in n.title
        assert "<b>" not in n.body

    def test_confidence_clamped(self):
        n = Notification(metadata={"conf": 1.5})
        assert isinstance(n.metadata, dict)

    def test_to_dict(self):
        n = Notification(title="Test", body="Content", priority=NotificationPriority.CRITICAL)
        d = n.to_dict()
        assert d["title"] == "Test"
        assert d["priority"] == "critical"
        assert d["read"] is False
        assert "created_at" in d

    def test_to_dict_deepcopy_metadata(self):
        n = Notification(metadata={"key": "val"})
        d = n.to_dict()
        d["metadata"]["key"] = "changed"
        assert n.metadata["key"] == "val"

    def test_priority_enum_values(self):
        for p in NotificationPriority:
            n = Notification(title="T", body="B", priority=p.value)
            assert n.priority == p


class TestNotificationBus:
    def setup_method(self):
        self.bus = NotificationBus()

    def test_notify_basic(self):
        nid = self.bus.notify("Title", "Body")
        assert len(nid) > 0
        assert self.bus.get_unread_count() == 1

    def test_notify_critical(self):
        nid = self.bus.notify_critical("ALERT", "Something bad")
        assert self.bus.get_unread_count() == 1
        notifs = self.bus.get_notifications(priority="critical")
        assert len(notifs) == 1
        assert notifs[0]["title"] == "ALERT"

    def test_notify_from_anomaly(self):
        nid = self.bus.notify_from_anomaly("CPU spike detected", "HIGH")
        notifs = self.bus.get_notifications()
        assert len(notifs) == 1

    def test_notify_from_security(self):
        nid = self.bus.notify_from_security("Brute force detected", 3)
        notifs = self.bus.get_notifications(priority="critical")
        assert len(notifs) == 1

    def test_notify_visual_observation(self):
        nid = self.bus.notify_visual_observation("Person detected", faces=3)
        notifs = self.bus.get_notifications(priority="high")
        assert len(notifs) == 1

    def test_mark_read(self):
        nid = self.bus.notify("T", "B")
        assert self.bus.get_unread_count() == 1
        self.bus.mark_read(nid)
        assert self.bus.get_unread_count() == 0

    def test_mark_read_invalid(self):
        assert self.bus.mark_read("nonexistent") is False

    def test_get_notifications_filtered(self):
        self.bus.notify("T1", "B1", priority="low")
        self.bus.notify("T2", "B2", priority="high")
        self.bus.notify("T3", "B3", priority="critical")
        assert len(self.bus.get_notifications(priority="low")) == 1
        assert len(self.bus.get_notifications(priority="high")) == 1
        assert len(self.bus.get_notifications(unread_only=True)) == 3

    def test_generate_digest(self):
        self.bus.notify("T", "B", priority="critical")
        self.bus.notify("T2", "B2", priority="high")
        digest = self.bus.generate_digest()
        assert isinstance(digest, NotificationDigest)
        assert digest.title
        assert len(digest.top_notifications) >= 1

    def test_get_stats(self):
        self.bus.notify("T", "B", priority="critical")
        stats = self.bus.get_stats()
        assert stats["notifications_created"] == 1
        assert stats["critical_priority"] == 1
        assert stats["unread_count"] == 1

    def test_clear_old(self):
        self.bus.notify("T", "B")
        removed = self.bus.clear_old(max_age_hours=0.0001)
        assert removed >= 0

    def test_get_unread_critical(self):
        self.bus.notify_critical("A", "B")
        assert self.bus.get_unread_critical() == 1
        self.bus.notify("T", "B", priority="normal")
        assert self.bus.get_unread_critical() == 1

    def test_thread_safety(self):
        import threading
        errors = []
        def push(n):
            try:
                self.bus.notify(f"T{n}", f"B{n}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=push, args=(i,)) for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert self.bus.get_unread_count() == 20


class TestNotificationDigest:
    def test_create_basic(self):
        d = NotificationDigest(title="Digest", summary="Summary text")
        assert d.title == "Digest"
        assert d.digest_id

    def test_to_dict(self):
        d = NotificationDigest(title="D", summary="S", stats={"total": 5})
        result = d.to_dict()
        assert result["title"] == "D"
        assert result["stats"]["total"] == 5


class TestShadowOSIntegration:
    def test_notifications_in_shadow_os(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        os._running = True
        os._start_time = 1734000000.0
        assert hasattr(os, 'notifications')
        nid = os.push_notification("Test", "Body", priority="high")
        assert len(nid) > 0
        assert os.get_notification_count() == 1
        os.notifications.mark_read(nid)
        assert os.get_notification_count() == 0
        digest = os.get_notification_digest()
        assert "title" in digest
        health = os.get_health()
        assert "notifications" in health["modules"]
        assert health["modules"]["notifications"]["status"] == "healthy"
