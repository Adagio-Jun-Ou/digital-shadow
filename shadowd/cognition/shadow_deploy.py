#!/usr/bin/env python3
"""
ShadowDeploy - Deploy ShadowAI instances to remote devices.

Deploy a "shadow" (sensor node) to an Android phone or any device
to keep an eye and ear on things remotely. The deployed shadow:
- Captures periodic camera frames (scene awareness)
- Records environmental audio (sound event detection)
- Monitors device sensors (movement, light, location)
- Reports back to the main ShadowOS instance
- Works over Tailscale for secure P2P connectivity

Architecture:
  Android Phone (PWA) ──Tailscale──► Main ShadowOS Server
       │                                    │
       ├── Camera capture ──────────────► vision pipeline
       ├── Microphone samples ──────────► audio analysis  
       ├── Device sensors ──────────────► device_context
       └── Alerts/events ◄────────────── notification_bus

Deploy modes:
  - `sentry`: Stationary monitor (watch a room, door, elder care)
  - `patrol`: Mobile monitor (walk around, check on things)
  - `listen`: Audio-only (baby monitor, conversation detection)
"""

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.shadow_deploy")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


class DeployMode(str, Enum):
    SENTRY = "sentry"
    PATROL = "patrol"
    LISTEN = "listen"


class ShadowStatus(str, Enum):
    PROVISIONING = "provisioning"
    ACTIVE = "active"
    IDLE = "idle"
    OFFLINE = "offline"
    ALERTING = "alerting"
    TERMINATED = "terminated"


@dataclass
class DeployedShadow:
    """A shadow instance deployed to a remote device."""
    shadow_id: str = ""
    name: str = ""
    device_id: str = ""
    mode: DeployMode = DeployMode.SENTRY
    status: ShadowStatus = ShadowStatus.PROVISIONING
    location: str = ""
    auth_token: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: Optional[datetime] = None
    last_report: Optional[str] = None

    # Configuration
    camera_interval_seconds: float = 10.0
    mic_sample_seconds: float = 5.0
    report_interval_minutes: float = 5.0
    alert_on_faces: bool = True
    alert_on_motion: bool = True
    alert_on_sound: bool = True
    min_face_count_for_alert: int = 1

    # Stats
    total_reports: int = 0
    total_alerts: int = 0
    total_frames: int = 0
    uptime_seconds: float = 0.0

    def __post_init__(self):
        if not self.shadow_id:
            self.shadow_id = uuid.uuid4().hex[:12]
        self.name = _sanitize_content(self.name, 200)
        self.device_id = _sanitize(self.device_id, 100)
        self.location = _sanitize_content(self.location, 500)
        if not self.auth_token:
            self.auth_token = hashlib.sha256(os.urandom(32)).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shadow_id": self.shadow_id,
            "name": self.name,
            "device_id": self.device_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "location": self.location,
            "created_at": self.created_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "last_report": self.last_report,
            "camera_interval_seconds": self.camera_interval_seconds,
            "mic_sample_seconds": self.mic_sample_seconds,
            "report_interval_minutes": self.report_interval_minutes,
            "alert_on_faces": self.alert_on_faces,
            "alert_on_motion": self.alert_on_motion,
            "alert_on_sound": self.alert_on_sound,
            "min_face_count_for_alert": self.min_face_count_for_alert,
            "total_reports": self.total_reports,
            "total_alerts": self.total_alerts,
            "total_frames": self.total_frames,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


@dataclass
class ShadowReport:
    """A report from a deployed shadow."""
    report_id: str = ""
    shadow_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Sensor data
    visual_observation: Optional[str] = None
    scene_category: Optional[str] = None
    faces_detected: int = 0
    motion_detected: bool = False
    sound_event: Optional[str] = None
    sound_level_db: float = 0.0
    light_level: float = 0.0
    battery_pct: float = 100.0
    is_charging: bool = False
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    signal_strength: float = 1.0

    # Analysis
    alert_triggered: bool = False
    alert_reason: str = ""
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.report_id:
            self.report_id = uuid.uuid4().hex[:12]
        self.shadow_id = _sanitize(self.shadow_id, 20)
        if self.visual_observation:
            self.visual_observation = _sanitize_content(self.visual_observation, 5000)
        if self.scene_category:
            self.scene_category = _sanitize(self.scene_category, 50)
        if self.sound_event:
            self.sound_event = _sanitize_content(self.sound_event, 2000)
        self.importance = max(0.0, min(1.0, self.importance))
        self.sound_level_db = max(0.0, min(120.0, self.sound_level_db))
        self.light_level = max(0.0, min(1.0, self.light_level))
        self.battery_pct = max(0.0, min(100.0, self.battery_pct))
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "shadow_id": self.shadow_id,
            "timestamp": self.timestamp.isoformat(),
            "visual_observation": self.visual_observation,
            "scene_category": self.scene_category,
            "faces_detected": self.faces_detected,
            "motion_detected": self.motion_detected,
            "sound_event": self.sound_event,
            "sound_level_db": round(self.sound_level_db, 1),
            "light_level": round(self.light_level, 3),
            "battery_pct": round(self.battery_pct, 1),
            "is_charging": self.is_charging,
            "location": {"lat": self.location_lat, "lng": self.location_lng} if self.location_lat else None,
            "signal_strength": round(self.signal_strength, 2),
            "alert_triggered": self.alert_triggered,
            "alert_reason": self.alert_reason,
            "importance": round(self.importance, 3),
            "metadata": copy.deepcopy(self.metadata),
        }



class SoundCategory(str, Enum):
    """Audio event classification from deployed shadow microphone samples."""
    SILENCE = "silence"
    SPEECH = "speech"
    CRYING = "crying"
    ALARM = "alarm"
    BREAKING = "breaking"
    FOOTSTEPS = "footsteps"
    MACHINERY = "machinery"
    MUSIC = "music"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"


class AudioAnalyzer:
    """Heuristic audio event classifier from sound_level_db + duration patterns.
    
    Uses db thresholds and temporal patterns to classify ambient sounds
    without requiring a full ML model on the deployed device.
    """
    
    # db thresholds for different sound types
    _THRESHOLDS = {
        SoundCategory.SILENCE: (0, 25),
        SoundCategory.SPEECH: (30, 65),
        SoundCategory.MACHINERY: (40, 70),
        SoundCategory.MUSIC: (45, 80),
        SoundCategory.FOOTSTEPS: (20, 40),
        SoundCategory.CRYING: (50, 80),
        SoundCategory.ALARM: (65, 120),
        SoundCategory.BREAKING: (70, 120),
        SoundCategory.VEHICLE: (55, 90),
    }
    
    # Keywords for text-based audio descriptions  
    _SOUND_KEYWORDS = {
        SoundCategory.SPEECH: ["talking","voices","conversation","speaking","yelling","shouting"],
        SoundCategory.CRYING: ["cry","crying","sob","wailing","scream","screaming"],
        SoundCategory.ALARM: ["alarm","beep","beeping","siren","warning","alert"],
        SoundCategory.BREAKING: ["break","breaking","shatter","glass","crash","smash"],
        SoundCategory.FOOTSTEPS: ["step","steps","footstep","walking","footfall"],
        SoundCategory.MACHINERY: ["machine","engine","motor","hum","drone","fan","ac"],
        SoundCategory.MUSIC: ["music","song","singing","radio","melody","rhythm"],
        SoundCategory.VEHICLE: ["car","truck","bus","engine","horn","traffic","motorcycle"],
    }
    
    @staticmethod
    def classify(sound_level_db: float, description: str = "", duration_seconds: float = 5.0) -> SoundCategory:
        """Classify a sound event from db level + optional description."""
        if sound_level_db < 25:
            return SoundCategory.SILENCE
        
        # Try description-based classification first
        if description:
            desc_lower = description.lower()
            for category, keywords in AudioAnalyzer._SOUND_KEYWORDS.items():
                for kw in keywords:
                    if kw in desc_lower:
                        return category
        
        # Fall back to db-level classification
        if sound_level_db < 35:
            return SoundCategory.FOOTSTEPS if sound_level_db > 20 else SoundCategory.SILENCE
        if sound_level_db < 55:
            return SoundCategory.SPEECH
        if sound_level_db < 70:
            return SoundCategory.MACHINERY
        if sound_level_db < 85:
            return SoundCategory.ALARM
        return SoundCategory.BREAKING
    
    @staticmethod
    def get_alert_priority(sound_category: SoundCategory, sound_level_db: float) -> str:
        """Determine notification priority for a sound event."""
        critical = [SoundCategory.BREAKING, SoundCategory.ALARM, SoundCategory.CRYING]
        high = [SoundCategory.VEHICLE]
        if sound_category in critical:
            return "critical" if sound_level_db > 75 else "high"
        if sound_category in high:
            return "high" if sound_level_db > 70 else "normal"
        return "normal"
    
    @staticmethod
    def should_alert(sound_category: SoundCategory, sound_level_db: float,
                     min_alert_db: float = 65) -> bool:
        """Determine if this sound should trigger an alert."""
        always_alert = [SoundCategory.BREAKING, SoundCategory.ALARM, SoundCategory.CRYING]
        if sound_category in always_alert:
            return True
        return sound_level_db >= min_alert_db


class ShadowDeployEngine:
    """Manages deployment of shadow instances to remote devices."""

    def __init__(self, shadow_os=None, data_dir: Optional[str] = None):
        self._shadow_os = shadow_os
        self.data_dir = data_dir or os.path.join(
            os.path.expanduser("~"), "digital-shadow", "data", "deploy"
        )
        os.makedirs(self.data_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._shadows: Dict[str, DeployedShadow] = {}
        self._reports: deque = deque(maxlen=5000)
        self._report_history: deque = deque(maxlen=10000)
        self._heartbeat_history: Dict[str, deque] = {}

        self._MAX_SHADOWS = 50
        self._MAX_REPORTS_PER_SHADOW = 1000

        self._stats = {
            "shadows_deployed": 0,
            "shadows_active": 0,
            "reports_received": 0,
            "alerts_generated": 0,
            "total_frames": 0,
            "total_sound_events": 0,
        }

        self._load_state()

    def _load_state(self):
        state_file = os.path.join(self.data_dir, "deploy_state.json")
        try:
            with open(state_file) as f:
                data = json.load(f)
                for sdata in data.get("shadows", []):
                    shadow = DeployedShadow(
                        shadow_id=sdata.get("shadow_id", ""),
                        name=sdata.get("name", ""),
                        device_id=sdata.get("device_id", ""),
                        mode=DeployMode(sdata.get("mode", "sentry")),
                        status=ShadowStatus(sdata.get("status", "offline")),
                        location=sdata.get("location", ""),
                        auth_token=sdata.get("auth_token", ""),
                    )
                    shadow.status = ShadowStatus.OFFLINE
                    self._shadows[shadow.shadow_id] = shadow
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            pass

    def _save_state(self):
        state_file = os.path.join(self.data_dir, "deploy_state.json")
        try:
            data = {
                "shadows": [
                    {
                        "shadow_id": s.shadow_id,
                        "name": s.name,
                        "device_id": s.device_id,
                        "mode": s.mode.value,
                        "status": s.status.value,
                        "location": s.location,
                        "auth_token": s.auth_token,
                    }
                    for s in self._shadows.values()
                ]
            }
            tmp_path = state_file + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(data, f, default=str)
            os.replace(tmp_path, state_file)
        except (OSError, TypeError):
            pass

    def provision_shadow(self, name: str, device_id: str = "",
                         mode: str = "sentry", location: str = "",
                         camera_interval: float = 10.0,
                         mic_sample: float = 5.0,
                         report_interval: float = 5.0,
                         alert_on_faces: bool = True,
                         alert_on_motion: bool = True,
                         alert_on_sound: bool = True) -> DeployedShadow:
        try:
            deploy_mode = DeployMode(mode)
        except ValueError:
            deploy_mode = DeployMode.SENTRY

        shadow = DeployedShadow(
            name=name,
            device_id=device_id,
            mode=deploy_mode,
            location=location,
            camera_interval_seconds=camera_interval,
            mic_sample_seconds=mic_sample,
            report_interval_minutes=report_interval,
            alert_on_faces=alert_on_faces,
            alert_on_motion=alert_on_motion,
            alert_on_sound=alert_on_sound,
        )

        with self._lock:
            while len(self._shadows) >= self._MAX_SHADOWS:
                oldest_id = next(iter(self._shadows))
                evicted = self._shadows.pop(oldest_id, None)
                if evicted:
                    logger.warning("Shadow cap reached, evicted: %s", evicted.shadow_id)
            self._shadows[shadow.shadow_id] = shadow

        self._save_state()
        with self._stats_lock:
            self._stats["shadows_deployed"] += 1

        logger.info("Shadow provisioned: %s (%s mode)", shadow.shadow_id, shadow.mode.value)
        return shadow

    def deprovision_shadow(self, shadow_id: str) -> bool:
        shadow_id = _sanitize(shadow_id, 20)
        with self._lock:
            shadow = self._shadows.pop(shadow_id, None)
            if shadow:
                shadow.status = ShadowStatus.TERMINATED
                self._heartbeat_history.pop(shadow_id, None)
                self._save_state()
                logger.info("Shadow deprovisioned: %s", shadow_id)
                return True
        return False

    def heartbeat(self, shadow_id: str, auth_token: str) -> bool:
        shadow_id = _sanitize(shadow_id, 20)
        auth_token = _sanitize(auth_token, 100)
        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if not shadow or shadow.auth_token != auth_token:
                return False
            shadow.last_heartbeat = datetime.now(timezone.utc)
            if shadow.status in (ShadowStatus.PROVISIONING, ShadowStatus.IDLE, ShadowStatus.OFFLINE):
                shadow.status = ShadowStatus.ACTIVE
                with self._stats_lock:
                    self._stats["shadows_active"] += 1
            if shadow_id not in self._heartbeat_history:
                self._heartbeat_history[shadow_id] = deque(maxlen=1000)
            self._heartbeat_history[shadow_id].append(datetime.now(timezone.utc).isoformat())
            return True

    def submit_report(self, shadow_id: str, auth_token: str,
                      visual_observation: Optional[str] = None,
                      scene_category: Optional[str] = None,
                      faces_detected: int = 0, motion_detected: bool = False,
                      sound_event: Optional[str] = None, sound_level_db: float = 0.0,
                      light_level: float = 0.0, battery_pct: float = 100.0,
                      is_charging: bool = False, location_lat: Optional[float] = None,
                      location_lng: Optional[float] = None, signal_strength: float = 1.0,
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[ShadowReport]:
        shadow_id = _sanitize(shadow_id, 20)
        auth_token = _sanitize(auth_token, 100)

        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if not shadow or shadow.auth_token != auth_token:
                return None

            alert = False
            alert_reason = ""

            if shadow.alert_on_faces and faces_detected >= shadow.min_face_count_for_alert:
                alert = True
                alert_reason = f"Faces detected: {faces_detected}"
            if shadow.alert_on_motion and motion_detected:
                alert = True
                alert_reason += " | Motion detected" if alert_reason else "Motion detected"
            if shadow.alert_on_sound:
                sound_cat = AudioAnalyzer.classify(sound_level_db, sound_event or "")
                if AudioAnalyzer.should_alert(sound_cat, sound_level_db):
                    alert = True
                    reason = f"{sound_cat.value} sound ({round(sound_level_db, 1)}dB)"
                    alert_reason += " | " + reason if alert_reason else reason

            importance = 0.5
            if faces_detected > 0:
                importance = min(1.0, importance + faces_detected * 0.1)
            if motion_detected:
                importance = min(1.0, importance + 0.15)
            if alert:
                importance = min(1.0, importance + 0.25)

            report = ShadowReport(
                shadow_id=shadow_id,
                visual_observation=visual_observation,
                scene_category=scene_category,
                faces_detected=faces_detected,
                motion_detected=motion_detected,
                sound_event=sound_event,
                sound_level_db=sound_level_db,
                light_level=light_level,
                battery_pct=battery_pct,
                is_charging=is_charging,
                location_lat=location_lat,
                location_lng=location_lng,
                signal_strength=signal_strength,
                alert_triggered=alert,
                alert_reason=alert_reason,
                importance=importance,
                metadata=metadata or {},
            )

            self._reports.append(report)
            self._report_history.append(report.to_dict())
            shadow.total_reports += 1
            shadow.total_frames += 1 if visual_observation else 0
            shadow.last_report = report.report_id
            if alert:
                shadow.total_alerts += 1
                shadow.status = ShadowStatus.ALERTING
            shadow.last_heartbeat = datetime.now(timezone.utc)

        with self._stats_lock:
            self._stats["reports_received"] += 1
            if alert:
                self._stats["alerts_generated"] += 1
            if visual_observation:
                self._stats["total_frames"] += 1
            if sound_event:
                self._stats["total_sound_events"] += 1

        if alert and self._shadow_os:
            try:
                if hasattr(self._shadow_os, 'notifications') and self._shadow_os.notifications:
                    self._shadow_os.notifications.notify_critical(
                        title=f"[{shadow.name}] Shadow Alert",
                        body=f"{alert_reason}\n{visual_observation or ''}",
                        source_module="shadow_deploy",
                        category="shadow_alert",
                        metadata={"shadow_id": shadow_id, "report_id": report.report_id},
                    )
            except (ValueError, TypeError, AttributeError, RuntimeError):
                pass

        return report

    def get_shadow(self, shadow_id: str) -> Optional[Dict[str, Any]]:
        shadow_id = _sanitize(shadow_id, 20)
        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if shadow:
                return copy.deepcopy(shadow.to_dict())
        return None

    def list_shadows(self, status: Optional[str] = None,
                     mode: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            shadows = list(self._shadows.values())
        if status:
            shadows = [s for s in shadows if s.status.value == status]
        if mode:
            shadows = [s for s in shadows if s.mode.value == mode]
        shadows.sort(key=lambda s: s.created_at, reverse=True)
        return [s.to_dict() for s in shadows]

    def get_reports(self, shadow_id: Optional[str] = None,
                    limit: int = 50, alerts_only: bool = False) -> List[Dict[str, Any]]:
        limit = max(1, min(500, limit))
        with self._lock:
            reports = list(self._reports)
        if shadow_id:
            reports = [r for r in reports if r.shadow_id == shadow_id]
        if alerts_only:
            reports = [r for r in reports if r.alert_triggered]
        reports.sort(key=lambda r: r.timestamp, reverse=True)
        return [r.to_dict() for r in reports[:limit]]

    def get_shadow_dashboard(self, shadow_id: str, auth_token: str) -> Optional[Dict[str, Any]]:
        """Get dashboard data for a deployed shadow device (PWA endpoint)."""
        shadow_id = _sanitize(shadow_id, 20)
        auth_token = _sanitize(auth_token, 100)
        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if not shadow or shadow.auth_token != auth_token:
                return None

            recent_reports = [
                r for r in list(self._reports)[-20:]
                if r.shadow_id == shadow_id
            ]

        return {
            "shadow": shadow.to_dict(),
            "recent_reports": [r.to_dict() for r in recent_reports],
            "config": {
                "camera_interval_seconds": shadow.camera_interval_seconds,
                "mic_sample_seconds": shadow.mic_sample_seconds,
                "report_interval_minutes": shadow.report_interval_minutes,
                "alert_on_faces": shadow.alert_on_faces,
                "alert_on_motion": shadow.alert_on_motion,
                "alert_on_sound": shadow.alert_on_sound,
                "min_face_count_for_alert": shadow.min_face_count_for_alert,
            },
        }

    def update_config(self, shadow_id: str, auth_token: str,
                      camera_interval: Optional[float] = None,
                      mic_sample: Optional[float] = None,
                      report_interval: Optional[float] = None,
                      alert_on_faces: Optional[bool] = None,
                      alert_on_sound: Optional[bool] = None) -> bool:
        shadow_id = _sanitize(shadow_id, 20)
        auth_token = _sanitize(auth_token, 100)
        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if not shadow or shadow.auth_token != auth_token:
                return False
            if camera_interval is not None:
                shadow.camera_interval_seconds = max(1.0, min(300.0, camera_interval))
            if mic_sample is not None:
                shadow.mic_sample_seconds = max(1.0, min(30.0, mic_sample))
            if report_interval is not None:
                shadow.report_interval_minutes = max(1.0, min(60.0, report_interval))
            if alert_on_faces is not None:
                shadow.alert_on_faces = alert_on_faces
            if alert_on_sound is not None:
                shadow.alert_on_sound = alert_on_sound
        self._save_state()
        return True

    def classify_sound(self, sound_level_db: float, description: str = "",
                          duration_seconds: float = 5.0) -> Dict:
        """Classify a sound event from a deployed shadow and determine alert priority."""
        category = AudioAnalyzer.classify(sound_level_db, description, duration_seconds)
        priority = AudioAnalyzer.get_alert_priority(category, sound_level_db)
        should_alert = AudioAnalyzer.should_alert(category, sound_level_db)
        return {
            "sound_category": category.value,
            "alert_priority": priority,
            "should_alert": should_alert,
            "sound_level_db": round(sound_level_db, 1),
        }

    def mark_shadow_idle(self, shadow_id: str, auth_token: str) -> bool:
        shadow_id = _sanitize(shadow_id, 20)
        with self._lock:
            shadow = self._shadows.get(shadow_id)
            if not shadow or shadow.auth_token != auth_token:
                return False
            if shadow.status in (ShadowStatus.ACTIVE, ShadowStatus.ALERTING):
                shadow.status = ShadowStatus.IDLE
        return True

    def get_offline_shadows(self, max_seconds: float = 300.0) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_seconds)
        offline = []
        with self._lock:
            for s in self._shadows.values():
                if s.last_heartbeat is None or s.last_heartbeat < cutoff:
                    if s.status not in (ShadowStatus.OFFLINE, ShadowStatus.TERMINATED, ShadowStatus.PROVISIONING):
                        s.status = ShadowStatus.OFFLINE
                    offline.append(s.to_dict())
        return offline

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._shadows)
            active = sum(1 for s in self._shadows.values() if s.status == ShadowStatus.ACTIVE)
            alerting = sum(1 for s in self._shadows.values() if s.status == ShadowStatus.ALERTING)
            offline = sum(1 for s in self._shadows.values() if s.status == ShadowStatus.OFFLINE)
        with self._stats_lock:
            stats = dict(self._stats)
        stats.update({
            "total_shadows": total,
            "active_shadows": active,
            "alerting_shadows": alerting,
            "offline_shadows": offline,
        })
        return stats
