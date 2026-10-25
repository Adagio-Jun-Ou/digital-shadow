#!/usr/bin/env python3
"""
NotificationBus - Push notification system for ShadowOS.

Delivers alerts, summaries, and critical events to the user through
multiple channels: PWA push, web endpoint, and future SMS/email/webhook.
"""

import copy
import logging
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("shadowd.cognition.notification_bus")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_NOTIFICATIONS = 5000
_MAX_CHANNELS = 50
_MAX_BATCH_SIZE = 100


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 2000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    PWA = "pwa"
    API = "api"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SMS = "sms"


@dataclass
class Notification:
    notification_id: str = ""
    title: str = ""
    body: str = ""
    priority: NotificationPriority = NotificationPriority.NORMAL
    source_module: str = ""
    category: str = "general"
    action_url: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    read: bool = False
    delivered_channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.notification_id:
            self.notification_id = uuid.uuid4().hex[:12]
        # Title is user-facing and may end up rendered in HTML, so use the
        # strict sanitizer (drops `<>` too); body keeps richer text but still
        # strips control chars.
        self.title = _sanitize(self.title, 200)
        self.body = _sanitize_content(self.body, 5000)
        self.source_module = _sanitize(self.source_module, 100)
        self.category = _sanitize(self.category, 50)
        if self.action_url:
            self.action_url = _sanitize(self.action_url, 500)
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }
        self.delivered_channels = [_sanitize(c, 20) for c in self.delivered_channels[:20]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "title": self.title,
            "body": self.body,
            "priority": self.priority.value,
            "source_module": self.source_module,
            "category": self.category,
            "action_url": self.action_url,
            "expires_at": self.expires_at,
            "created_at": self.created_at.isoformat(),
            "read": self.read,
            "delivered_channels": list(self.delivered_channels),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class NotificationDigest:
    digest_id: str = ""
    title: str = "ShadowOS Digest"
    summary: str = ""
    top_notifications: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stats: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.digest_id:
            self.digest_id = uuid.uuid4().hex[:12]
        self.title = _sanitize_content(self.title, 200)
        self.summary = _sanitize_content(self.summary, 5000)
        self.stats = {_sanitize(str(k), 50): v for k, v in list((self.stats or {}).items())[:50]}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "digest_id": self.digest_id,
            "title": self.title,
            "summary": self.summary,
            "top_notifications": [copy.deepcopy(n) for n in self.top_notifications[:20]],
            "generated_at": self.generated_at.isoformat(),
            "stats": dict(self.stats),
        }


class NotificationBus:
    def __init__(self, shadow_os=None):
        self._shadow_os = shadow_os
        # Re-entrant lock: get_stats() acquires _lock, then calls
        # get_unread_count() which also acquires _lock → deadlock with a plain
        # Lock. RLock allows the same thread to reacquire safely.
        self._lock = threading.RLock()
        self._stats_lock = threading.RLock()

        self._notifications: Dict[str, Notification] = {}
        self._notif_order: deque = deque()
        self._channel_subscribers: Dict[str, Set[str]] = {}
        self._digests: List[NotificationDigest] = []

        self._stats: Dict[str, int] = {
            "notifications_created": 0,
            "notifications_read": 0,
            "digests_generated": 0,
            "high_priority": 0,
            "critical_priority": 0,
        }

    def notify(self, title: str, body: str, priority: str = "normal",
               source_module: str = "shadow_os", category: str = "general",
               action_url: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
               channels: Optional[List[str]] = None) -> str:
        try:
            prio = NotificationPriority(priority)
        except ValueError:
            prio = NotificationPriority.NORMAL

        notif = Notification(
            title=title,
            body=body,
            priority=prio,
            source_module=source_module,
            category=category,
            action_url=action_url,
            metadata=metadata or {},
            delivered_channels=channels or [],
        )

        with self._lock:
            self._notifications[notif.notification_id] = notif
            self._notif_order.append(notif.notification_id)
            while len(self._notifications) > _MAX_NOTIFICATIONS:
                old_id = self._notif_order.popleft()
                self._notifications.pop(old_id, None)

        with self._stats_lock:
            self._stats["notifications_created"] += 1
            if prio == NotificationPriority.HIGH:
                self._stats["high_priority"] += 1
            elif prio == NotificationPriority.CRITICAL:
                self._stats["critical_priority"] += 1

        return notif.notification_id

    def notify_critical(self, title: str, body: str, source_module: str = "shadow_os",
                        category: str = "security", metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.notify(
            title=title, body=body, priority="critical",
            source_module=source_module, category=category,
            metadata=metadata, channels=["pwa", "api"],
        )

    def notify_from_anomaly(self, anomaly_desc: str, severity: str,
                            source_module: str = "anomaly_detector") -> str:
        priority = "critical" if severity in ("CRITICAL", "HIGH") else "normal"
        return self.notify(
            title=f"Anomaly Detected: {severity}",
            body=anomaly_desc,
            priority=priority,
            source_module=source_module,
            category="anomaly",
        )

    def notify_from_security(self, event_desc: str, threat_level: int,
                             source_module: str = "security_guardian") -> str:
        priority = "critical" if threat_level >= 3 else "high" if threat_level >= 2 else "normal"
        return self.notify(
            title=f"Security Alert (Level {threat_level})",
            body=event_desc,
            priority=priority,
            source_module=source_module,
            category="security",
        )

    def notify_visual_observation(self, description: str, faces: int,
                                  category: str = "visual") -> str:
        if faces > 0:
            return self.notify(
                title=f"Person Detected ({faces} face(s))",
                body=description,
                priority="high" if faces >= 3 else "normal",
                source_module="vision",
                category=category,
            )
        return self.notify(
            title="Visual Observation",
            body=description,
            priority="low",
            source_module="vision",
            category=category,
        )

    def mark_read(self, notification_id: str) -> bool:
        notification_id = _sanitize(notification_id, 20)
        with self._lock:
            notif = self._notifications.get(notification_id)
            if notif and not notif.read:
                notif.read = True
                with self._stats_lock:
                    self._stats["notifications_read"] += 1
                return True
        return False

    def get_notifications(self, priority: Optional[str] = None,
                          category: Optional[str] = None,
                          unread_only: bool = False,
                          limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(200, limit))
        with self._lock:
            notifs = list(self._notifications.values())
        if priority:
            notifs = [n for n in notifs if n.priority.value == priority]
        if category:
            notifs = [n for n in notifs if n.category == category]
        if unread_only:
            notifs = [n for n in notifs if not n.read]
        notifs.sort(key=lambda n: n.created_at, reverse=True)
        return [n.to_dict() for n in notifs[:limit]]

    def get_unread_count(self) -> int:
        with self._lock:
            return sum(1 for n in self._notifications.values() if not n.read)

    def get_unread_critical(self) -> int:
        with self._lock:
            return sum(
                1 for n in self._notifications.values()
                if not n.read and n.priority == NotificationPriority.CRITICAL
            )

    def generate_digest(self) -> NotificationDigest:
        with self._lock:
            recent = list(self._notifications.values())
            recent.sort(key=lambda n: n.created_at, reverse=True)

        critical = [n for n in recent if n.priority == NotificationPriority.CRITICAL]
        high = [n for n in recent if n.priority == NotificationPriority.HIGH]
        top = (critical + high + recent)[:20]

        digest = NotificationDigest(
            title=f"ShadowOS Digest - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            summary=f"{len(critical)} critical, {len(high)} high-priority notifications in queue. "
                    f"Total unread: {self.get_unread_count()}.",
            top_notifications=[n.to_dict() for n in top],
            stats={
                "total": len(recent),
                "unread": self.get_unread_count(),
                "critical": len(critical),
                "high": len(high),
            },
        )

        with self._lock:
            self._digests.append(digest)
            if len(self._digests) > 100:
                self._digests = self._digests[-100:]

        with self._stats_lock:
            self._stats["digests_generated"] += 1

        return digest

    def get_digests(self, limit: int = 10) -> List[Dict[str, Any]]:
        limit = max(1, min(50, limit))
        with self._lock:
            digests = list(self._digests[-limit:])
        return [d.to_dict() for d in digests]

    def clear_old(self, max_age_hours: float = 24.0) -> int:
        cutoff = datetime.now(timezone.utc)
        removed = 0
        with self._lock:
            to_remove = [
                nid for nid, n in self._notifications.items()
                if (cutoff - n.created_at).total_seconds() > max_age_hours * 3600
            ]
            for nid in to_remove:
                self._notifications.pop(nid, None)
                removed += 1
        return removed

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._notifications)
            unread = self.get_unread_count()
        with self._stats_lock:
            stats = dict(self._stats)
        stats["total_notifications"] = total
        stats["unread_count"] = unread
        return stats
