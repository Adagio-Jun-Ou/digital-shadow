#!/usr/bin/env python3
"""
Unified Dashboard Aggregator for ShadowOS.

Pulls live data from all 70+ cognitive modules into a single
JSON response for the PWA dashboard. Optimized for fast polling:
- Parallel data collection with timeout per module
- Configurable refresh intervals
- Health summary, alerts, active modules, recent events
"""

import copy
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.dashboard_aggregator")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]

def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


@dataclass
class DashboardSnapshot:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    system_status: str = "degraded"
    uptime_seconds: float = 0.0
    module_count: int = 0
    healthy_count: int = 0
    degraded_count: int = 0

    unread_notifications: int = 0
    critical_alerts: int = 0
    active_shadows: int = 0
    alerting_shadows: int = 0

    journal_entries_today: int = 0
    memories_stored: int = 0
    vision_captures: int = 0
    visual_events_processed: int = 0

    cognitive_state: str = "listen"
    personality_traits: Dict[str, float] = field(default_factory=dict)

    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    top_alerts: List[Dict[str, Any]] = field(default_factory=list)

    module_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "system_status": self.system_status,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "module_count": self.module_count,
            "healthy_count": self.healthy_count,
            "degraded_count": self.degraded_count,
            "unread_notifications": self.unread_notifications,
            "critical_alerts": self.critical_alerts,
            "active_shadows": self.active_shadows,
            "alerting_shadows": self.alerting_shadows,
            "journal_entries_today": self.journal_entries_today,
            "memories_stored": self.memories_stored,
            "vision_captures": self.vision_captures,
            "visual_events_processed": self.visual_events_processed,
            "cognitive_state": self.cognitive_state,
            "personality_traits": dict(self.personality_traits),
            "recent_events": list(self.recent_events[:20]),
            "top_alerts": list(self.top_alerts[:10]),
            "module_details": copy.deepcopy(self.module_details),
            "metadata": copy.deepcopy(self.metadata),
        }


class DashboardAggregator:
    def __init__(self, shadow_os=None):
        self._shadow_os = shadow_os
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._last_snapshot: Optional[DashboardSnapshot] = None
        self._last_refresh: float = 0.0
        self._min_refresh_interval: float = 2.0

        self._stats = {"snapshots_generated": 0, "refresh_errors": 0}

    def get_snapshot(self, force: bool = False) -> DashboardSnapshot:
        now = time.monotonic()
        if not force and self._last_snapshot and (now - self._last_refresh) < self._min_refresh_interval:
            return self._last_snapshot

        os = self._shadow_os
        if os is None:
            return DashboardSnapshot()

        snap = DashboardSnapshot()
        os_stats = {}

        modules_to_collect = [
            ("health", self._collect_health),
            ("notifications", self._collect_notifications),
            ("deploy", self._collect_deploy),
            ("journal", self._collect_journal),
            ("vision", self._collect_vision),
            ("cognition", self._collect_cognition),
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fn): name for name, fn in modules_to_collect}
            for future in as_completed(futures, timeout=10):
                name = futures[future]
                try:
                    result = future.result(timeout=8)
                    if result:
                        os_stats[name] = result
                except (RuntimeError, TimeoutError, ValueError, TypeError, AttributeError):
                    pass

        self._merge_snapshot(snap, os_stats)
        snap.timestamp = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._last_snapshot = snap
            self._last_refresh = now
        with self._stats_lock:
            self._stats["snapshots_generated"] += 1

        return snap

    def _collect_health(self):
        os = self._shadow_os
        if os is None: return {}
        h = os.get_health()
        modules = h.get("modules", {})
        healthy = sum(1 for m in modules.values() if m.get("status") == "healthy")
        degraded = sum(1 for m in modules.values() if m.get("status") != "healthy")
        return {
            "status": h.get("status", "unknown"),
            "uptime": h.get("uptime_seconds", 0),
            "total_modules": len(modules),
            "healthy": healthy,
            "degraded": degraded,
            "modules": modules,
        }

    def _collect_notifications(self):
        os = self._shadow_os
        if not hasattr(os, 'notifications') or not os.notifications:
            return {}
        return {
            "unread": os.notifications.get_unread_count(),
            "critical": os.notifications.get_unread_critical(),
            "recent": os.notifications.get_notifications(limit=10),
        }

    def _collect_deploy(self):
        os = self._shadow_os
        if not hasattr(os, 'shadow_deploy') or not os.shadow_deploy:
            return {}
        stats = os.shadow_deploy.get_stats()
        reports = os.shadow_deploy.get_reports(alerts_only=True, limit=10)
        return {
            "active_shadows": stats.get("active_shadows", 0),
            "alerting_shadows": stats.get("alerting_shadows", 0),
            "alert_reports": reports,
        }

    def _collect_journal(self):
        os = self._shadow_os
        if not hasattr(os, 'journal') or not os.journal:
            return {}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entries = os.journal.get_entries(date=today, limit=20)
        return {"entries_today": len(entries), "recent": entries[:10]}

    def _collect_vision(self):
        os = self._shadow_os
        result = {}
        if hasattr(os, 'vision') and os.vision:
            try:
                stats = os.vision.get_stats()
                result["captures"] = stats.get("stats", {}).get("captures", 0)
            except (ValueError, TypeError, AttributeError):
                pass
        if hasattr(os, 'visual_bridge') and os.visual_bridge:
            try:
                bstats = os.visual_bridge.get_stats()
                result["events_processed"] = bstats.get("events_processed", 0)
            except (ValueError, TypeError, AttributeError):
                pass
        return result

    def _collect_cognition(self):
        os = self._shadow_os
        result = {}
        if hasattr(os, 'graph') and os.graph:
            try:
                result["cognitive_state"] = os.graph.state.mode.value if os.graph.state else "listen"
            except (AttributeError, TypeError):
                pass
        if hasattr(os, 'personality') and os.personality:
            try:
                profile = os.personality.get_profile()
                if profile:
                    result["traits"] = {
                        "openness": round(profile.get("openness", 0.5), 2),
                        "conscientiousness": round(profile.get("conscientiousness", 0.5), 2),
                        "extraversion": round(profile.get("extraversion", 0.5), 2),
                        "agreeableness": round(profile.get("agreeableness", 0.5), 2),
                        "neuroticism": round(profile.get("neuroticism", 0.5), 2),
                    }
            except (ValueError, TypeError, AttributeError):
                pass
        if hasattr(os, 'memory') and os.memory:
            try:
                mstats = os.memory.get_stats()
                result["memories_stored"] = mstats.get("total_atoms", 0)
            except (ValueError, TypeError, AttributeError):
                pass
        return result

    def _merge_snapshot(self, snap: DashboardSnapshot, data: Dict[str, Any]):
        health = data.get("health", {})
        notif = data.get("notifications", {})
        deploy = data.get("deploy", {})
        journal = data.get("journal", {})
        vision = data.get("vision", {})
        cog = data.get("cognition", {})

        snap.system_status = health.get("status", "degraded")
        snap.uptime_seconds = health.get("uptime", 0)
        snap.module_count = health.get("total_modules", 0)
        snap.healthy_count = health.get("healthy", 0)
        snap.degraded_count = health.get("degraded", 0)

        snap.unread_notifications = notif.get("unread", 0)
        snap.critical_alerts = notif.get("critical", 0)
        snap.active_shadows = deploy.get("active_shadows", 0)
        snap.alerting_shadows = deploy.get("alerting_shadows", 0)

        snap.journal_entries_today = journal.get("entries_today", 0)
        snap.memories_stored = cog.get("memories_stored", 0)
        snap.vision_captures = vision.get("captures", 0)
        snap.visual_events_processed = vision.get("events_processed", 0)

        snap.cognitive_state = cog.get("cognitive_state", "listen")
        snap.personality_traits = cog.get("traits", {})

        snap.recent_events = journal.get("recent", [])[:20]
        snap.top_alerts = []
        for r in deploy.get("alert_reports", []):
            snap.top_alerts.append({
                "report_id": r.get("report_id", ""),
                "shadow_id": r.get("shadow_id", ""),
                "alert_reason": r.get("alert_reason", ""),
            })
        for n in notif.get("recent", []):
            snap.top_alerts.append({
                "notification_id": n.get("notification_id", ""),
                "title": n.get("title", ""),
                "priority": n.get("priority", ""),
            })
        snap.top_alerts = snap.top_alerts[:10]

        snap.module_details = health.get("modules", {})

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._lock:
            stats["cached"] = self._last_snapshot is not None
        return stats
