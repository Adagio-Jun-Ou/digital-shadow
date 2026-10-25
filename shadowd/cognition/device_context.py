#!/usr/bin/env python3
"""
DeviceContext - Multi-Device Passive Context Fusion Engine

Fuses passive data streams from iPhone, Apple Watch, and Linux laptop
into a unified real-time context model that powers ShadowAI's cognitive modules.

Architecture:
  DeviceStream → DeviceContextEngine → Cognitive Modules
                    │
                    ├── LocationContext (home/office/gym/transit)
                    ├── HealthContext (heart rate, sleep, stress, SpO2)
                    ├── ActivityContext (exercise, sedentary, sleep state)
                    ├── DigitalContext (screen time, app usage, notifications)
                    ├── SocialContext (nearby people, calls, messages)
                    └── EnvironmentContext (weather, noise, light, battery)

Data Sources:
  iPhone (via Shortcuts/webhook):
    - Location (GPS, significant locations)
    - Battery level, charging state
    - Active app, screen on/off
    - Calendar events
    - Notifications received
    - Call/messaging metadata (no content)
    - Steps, distance (HealthKit via Shortcuts)
    - Screen time summary

  Apple Watch (via iPhone Shortcuts relay):
    - Heart rate (current, resting, walking)
    - Heart rate variability (HRV)
    - Blood oxygen (SpO2)
    - Sleep analysis (in bed, asleep, REM, deep, core)
    - Exercise minutes, stand hours
    - Step count, flights climbed
    - Wrist temperature
    - ECG availability
    - Fall detection events

  Linux Laptop (via local system monitor):
    - Active window/app
    - CPU/memory/disk usage
    - Network connections (SSIDs, interfaces)
    - Bluetooth devices nearby
    - Screen lock state
    - Battery level
    - Active media (music/podcast)
    - Calendar/clock events (via calcurse/Thunderbird)
    - Browser tabs count
    - Terminal activity
"""

import re
import copy
import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger("shadowd.cognition.device_context")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_METADATA_KEYS = 20


def _sanitize(text: str, max_len: int = 5000) -> str:
    """Content-safe sanitize: preserves <> for comparison expressions."""
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    text = _SANITIZE_RE.sub('', text)
    if max_len > 0:
        text = text.strip()[:max_len]
    return text


def _sanitize_strict(text: str, max_len: int = 200) -> str:
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_len]


class DeviceType(str, Enum):
    IPHONE = "iphone"
    APPLE_WATCH = "apple_watch"
    LAPTOP = "laptop"
    TABLET = "tablet"
    UNKNOWN = "unknown"


class LocationType(str, Enum):
    HOME = "home"
    OFFICE = "office"
    GYM = "gym"
    TRANSIT = "transit"
    OUTDOORS = "outdoors"
    RESTAURANT = "restaurant"
    SHOPPING = "shopping"
    MEDICAL = "medical"
    SOCIAL = "social"
    UNKNOWN = "unknown"


class ActivityType(str, Enum):
    SLEEPING = "sleeping"
    SEDENTARY = "sedentary"
    WALKING = "walking"
    RUNNING = "running"
    CYCLING = "cycling"
    DRIVING = "driving"
    EXERCISING = "exercising"
    COOKING = "cooking"
    SOCIAL = "social"
    WORKING = "working"
    COMMUTING = "commuting"
    RELAXING = "relaxing"
    UNKNOWN = "unknown"


class StressLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class FocusLevel(str, Enum):
    DEEP_WORK = "deep_work"
    ACTIVE = "active"
    SCANNING = "scanning"
    DISTRACTED = "distracted"
    FATIGUED = "fatigued"
    RESTING = "resting"


class SleepPhase(str, Enum):
    AWAKE = "awake"
    LIGHT = "light"
    DEEP = "deep"
    REM = "rem"
    UNKNOWN = "unknown"


class NetworkType(str, Enum):
    WIFI_HOME = "wifi_home"
    WIFI_OFFICE = "wifi_office"
    WIFI_PUBLIC = "wifi_public"
    CELLULAR = "cellular"
    ETHERNET = "ethernet"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class NotificationType(str, Enum):
    MESSAGE = "message"
    EMAIL = "email"
    SOCIAL = "social"
    NEWS = "news"
    REMINDER = "reminder"
    ALERT = "alert"
    SYSTEM = "system"
    MARKETING = "marketing"
    UNKNOWN = "unknown"


@dataclass
class LocationPoint:
    latitude: float = 0.0
    longitude: float = 0.0
    accuracy: float = 0.0
    altitude: Optional[float] = None
    speed: Optional[float] = None
    location_type: LocationType = LocationType.UNKNOWN
    name: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "accuracy": round(self.accuracy, 3),
            "altitude": round(self.altitude, 3) if self.altitude is not None else None,
            "speed": round(self.speed, 3) if self.speed is not None else None,
            "location_type": self.location_type.value,
            "name": _sanitize_strict(self.name, 100),
            "timestamp": self.timestamp,
        }


@dataclass
class HealthSnapshot:
    heart_rate: Optional[int] = None
    heart_rate_resting: Optional[int] = None
    heart_rate_walking: Optional[int] = None
    hrv: Optional[float] = None
    spo2: Optional[float] = None
    wrist_temperature: Optional[float] = None
    step_count: int = 0
    flights_climbed: int = 0
    exercise_minutes: int = 0
    stand_hours: float = 0.0
    stress_level: StressLevel = StressLevel.LOW
    sleep_phase: SleepPhase = SleepPhase.UNKNOWN
    sleep_minutes_last_24h: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "heart_rate": self.heart_rate,
            "heart_rate_resting": self.heart_rate_resting,
            "heart_rate_walking": self.heart_rate_walking,
            "hrv": round(self.hrv, 2) if self.hrv is not None else None,
            "spo2": round(self.spo2, 2) if self.spo2 is not None else None,
            "wrist_temperature": round(self.wrist_temperature, 2) if self.wrist_temperature is not None else None,
            "step_count": self.step_count,
            "flights_climbed": self.flights_climbed,
            "exercise_minutes": self.exercise_minutes,
            "stand_hours": round(self.stand_hours, 2),
            "stress_level": self.stress_level.value,
            "sleep_phase": self.sleep_phase.value,
            "sleep_minutes_last_24h": self.sleep_minutes_last_24h,
            "timestamp": self.timestamp,
        }


@dataclass
class DigitalSnapshot:
    screen_on: bool = False
    active_app: str = ""
    active_app_category: str = ""
    screen_time_minutes_last_24h: int = 0
    notifications_last_hour: int = 0
    notification_types: List[NotificationType] = field(default_factory=list)
    browser_tabs_open: int = 0
    terminal_sessions: int = 0
    media_playing: bool = False
    media_title: str = ""
    media_type: str = ""
    calendar_next_event: Optional[Dict] = None
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "screen_on": self.screen_on,
            "active_app": _sanitize_strict(self.active_app, 100),
            "active_app_category": _sanitize_strict(self.active_app_category, 50),
            "screen_time_minutes_last_24h": self.screen_time_minutes_last_24h,
            "notifications_last_hour": self.notifications_last_hour,
            "notification_types": [t.value for t in self.notification_types[:10]],
            "browser_tabs_open": self.browser_tabs_open,
            "terminal_sessions": self.terminal_sessions,
            "media_playing": self.media_playing,
            "media_title": _sanitize_strict(self.media_title, 200),
            "media_type": _sanitize_strict(self.media_type, 50),
            "calendar_next_event": copy.deepcopy(self.calendar_next_event) if self.calendar_next_event else None,
            "timestamp": self.timestamp,
        }


@dataclass
class EnvironmentSnapshot:
    network_type: NetworkType = NetworkType.UNKNOWN
    network_ssid: str = ""
    connected_devices: int = 0
    nearby_bluetooth_devices: List[str] = field(default_factory=list)
    battery_level: Optional[float] = None
    battery_charging: bool = False
    laptop_cpu_percent: Optional[float] = None
    laptop_memory_percent: Optional[float] = None
    laptop_disk_percent: Optional[float] = None
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "network_type": self.network_type.value,
            "network_ssid": _sanitize_strict(self.network_ssid, 100),
            "connected_devices": self.connected_devices,
            "nearby_bluetooth_devices": [_sanitize_strict(d, 50) for d in self.nearby_bluetooth_devices[:20]],
            "battery_level": round(self.battery_level, 2) if self.battery_level is not None else None,
            "battery_charging": self.battery_charging,
            "laptop_cpu_percent": round(self.laptop_cpu_percent, 2) if self.laptop_cpu_percent is not None else None,
            "laptop_memory_percent": round(self.laptop_memory_percent, 2) if self.laptop_memory_percent is not None else None,
            "laptop_disk_percent": round(self.laptop_disk_percent, 2) if self.laptop_disk_percent is not None else None,
            "timestamp": self.timestamp,
        }


@dataclass
class UnifiedContext:
    location: LocationPoint = field(default_factory=LocationPoint)
    health: HealthSnapshot = field(default_factory=HealthSnapshot)
    digital: DigitalSnapshot = field(default_factory=DigitalSnapshot)
    environment: EnvironmentSnapshot = field(default_factory=EnvironmentSnapshot)
    primary_device: DeviceType = DeviceType.LAPTOP
    activity: ActivityType = ActivityType.UNKNOWN
    focus: FocusLevel = FocusLevel.SCANNING
    people_nearby: List[str] = field(default_factory=list)
    context_confidence: float = 0.0
    timestamp: str = ""
    raw_device_data: Dict[str, Any] = field(default_factory=dict)

    _MAX_PEOPLE = 20

    def to_dict(self) -> Dict:
        return {
            "location": self.location.to_dict(),
            "health": self.health.to_dict(),
            "digital": self.digital.to_dict(),
            "environment": self.environment.to_dict(),
            "primary_device": self.primary_device.value,
            "activity": self.activity.value,
            "focus": self.focus.value,
            "people_nearby": [_sanitize_strict(p, 100) for p in self.people_nearby[:self._MAX_PEOPLE]],
            "context_confidence": round(self.context_confidence, 3),
            "timestamp": self.timestamp,
        }


class DeviceContextEngine:
    """
    Multi-device passive context fusion.

    Receives data pushes from iPhone (via Shortcuts/webhook),
    Apple Watch (relayed through iPhone), and Linux laptop
    (via local system monitor), then fuses them into a unified
    context model that cognitive modules can query.

    Thread-safe, bounded history, input sanitization.
    """

    _MAX_HISTORY = 1000
    _MAX_DEVICES = 10
    _MAX_BLUETOOTH = 50
    _MAX_NOTIFICATION_HISTORY = 500
    _CONTEXT_REFRESH_INTERVAL = 30  # seconds
    _STALE_DATA_THRESHOLD = timedelta(minutes=10)

    def __init__(self):
        self._lock = threading.Lock()
        self._devices: Dict[str, Dict] = {}  # device_id -> {type, last_seen, data}
        self._location_history: deque = deque(maxlen=self._MAX_HISTORY)
        self._health_history: deque = deque(maxlen=self._MAX_HISTORY)
        self._digital_history: deque = deque(maxlen=self._MAX_HISTORY)
        self._environment_history: deque = deque(maxlen=self._MAX_HISTORY)
        self._notification_history: deque = deque(maxlen=self._MAX_NOTIFICATION_HISTORY)
        self._unified_context: Optional[UnifiedContext] = None
        self._people_map: Dict[str, str] = {}  # bt_mac -> name
        self._wifi_location_map: Dict[str, LocationType] = {}  # ssid -> type
        self._app_category_map: Dict[str, str] = {
            "code": "development", "vscode": "development", "vim": "development",
            "terminal": "development", "iterm": "development", "kitty": "development",
            "chrome": "browsing", "firefox": "browsing", "safari": "browsing",
            "slack": "communication", "messages": "communication", "mail": "communication",
            "spotify": "entertainment", "netflix": "entertainment", "youtube": "entertainment",
            "notion": "productivity", "obsidian": "productivity", "todoist": "productivity",
        }
        self._stats = {
            "iphone_updates": 0,
            "watch_updates": 0,
            "laptop_updates": 0,
            "context_refreshes": 0,
            "location_changes": 0,
            "activity_changes": 0,
            "health_alerts": 0,
            "stress_episodes": 0,
        }
        self._stats_lock = threading.Lock()

    def register_device(self, device_id: str, device_type: DeviceType,
                        name: str = "", capabilities: Optional[List[str]] = None) -> None:
        device_id = _sanitize_strict(device_id, 100)
        name = _sanitize_strict(name, 100)
        with self._lock:
            if len(self._devices) >= self._MAX_DEVICES and device_id not in self._devices:
                oldest_id = min(self._devices, key=lambda k: self._devices[k].get("last_seen", ""))
                del self._devices[oldest_id]
            self._devices[device_id] = {
                "type": device_type.value,
                "name": name,
                "capabilities": capabilities or [],
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "data": {},
            }

    def ingest_iphone(self, device_id: str, data: Dict) -> UnifiedContext:
        device_id = _sanitize_strict(device_id, 100)
        data = _sanitize_dict(data)
        with self._lock:
            with self._stats_lock:
                self._stats["iphone_updates"] += 1
            if device_id in self._devices:
                self._devices[device_id]["last_seen"] = datetime.now(timezone.utc).isoformat()
                self._devices[device_id]["data"].update(data)

        if "location" in data:
            loc_data = data["location"]
            loc = LocationPoint(
                latitude=float(loc_data.get("latitude", 0)),
                longitude=float(loc_data.get("longitude", 0)),
                accuracy=float(loc_data.get("accuracy", 0)),
                altitude=loc_data.get("altitude"),
                speed=loc_data.get("speed"),
                location_type=_classify_location(loc_data),
                name=_sanitize_strict(loc_data.get("name", ""), 100),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            with self._lock:
                self._location_history.append(loc)

        if "battery" in data:
            batt = data["battery"]
            if isinstance(batt, dict):
                with self._lock:
                    self._environment_history.append(EnvironmentSnapshot(
                        network_type=NetworkType.UNKNOWN,
                        battery_level=float(batt.get("level", 0)),
                        battery_charging=bool(batt.get("charging", False)),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))

        if "health" in data:
            with self._lock:
                self._health_history.append(self._parse_watch_health(data["health"]))

        if "notifications" in data:
            notifs = data["notifications"]
            if isinstance(notifs, list):
                for n in notifs[:20]:
                    n_dict = n if isinstance(n, dict) else {"app": str(n), "category": ""}
                    nt = _classify_notification(n_dict)
                    with self._lock:
                        self._notification_history.append({
                            "type": nt.value,
                            "app": _sanitize_strict(str(n_dict.get("app", "")), 50),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

        return self._refresh_context()

    def ingest_watch(self, device_id: str, data: Dict) -> UnifiedContext:
        device_id = _sanitize_strict(device_id, 100)
        data = _sanitize_dict(data)
        with self._lock:
            with self._stats_lock:
                self._stats["watch_updates"] += 1
            if device_id in self._devices:
                self._devices[device_id]["last_seen"] = datetime.now(timezone.utc).isoformat()
                self._devices[device_id]["data"].update(data)

        health = self._parse_watch_health(data)
        with self._lock:
            self._health_history.append(health)
            if health.heart_rate and health.heart_rate > 100:
                with self._stats_lock:
                    self._stats["health_alerts"] += 1

        return self._refresh_context()

    def ingest_laptop(self, device_id: str, data: Dict) -> UnifiedContext:
        device_id = _sanitize_strict(device_id, 100)
        data = _sanitize_dict(data)
        with self._lock:
            with self._stats_lock:
                self._stats["laptop_updates"] += 1
            if device_id in self._devices:
                self._devices[device_id]["last_seen"] = datetime.now(timezone.utc).isoformat()
                self._devices[device_id]["data"].update(data)

        digital = self._parse_laptop_digital(data)
        env = self._parse_laptop_environment(data)
        with self._lock:
            self._digital_history.append(digital)
            self._environment_history.append(env)

        if "active_window" in data or "screen_on" in data:
            pass  # already in digital

        return self._refresh_context()

    def get_context(self) -> Optional[UnifiedContext]:
        with self._lock:
            if self._unified_context is None:
                return None
            return copy.deepcopy(self._unified_context)

    def get_health_trends(self, minutes: int = 60) -> Dict:
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            recent = [h for h in self._health_history
                       if h.heart_rate is not None][:100]
        if not recent:
            return {"trend": "no_data", "data_points": 0}
        hrs = [h.heart_rate for h in recent if h.heart_rate]
        avg_hr = sum(hrs) / len(hrs) if hrs else 0
        hrvs = [h.hrv for h in recent if h.hrv is not None]
        avg_hrv = sum(hrvs) / len(hrvs) if hrvs else 0
        spo2s = [h.spo2 for h in recent if h.spo2 is not None]
        min_spo2 = min(spo2s) if spo2s else 0
        stress = _infer_stress(avg_hr, avg_hrv, min_spo2)
        return {
            "avg_heart_rate": round(avg_hr, 1),
            "avg_hrv": round(avg_hrv, 1),
            "min_spo2": round(min_spo2, 1) if spo2s else None,
            "stress_level": stress.value,
            "data_points": len(recent),
            "trend": "elevated" if avg_hr > 80 else "normal",
        }

    def get_digital_wellbeing(self, hours: int = 24) -> Dict:
        with self._lock:
            recent = list(self._digital_history)
            recent_notifs = list(self._notification_history)
        total_screen = sum(d.screen_time_minutes_last_24h for d in recent) if recent else 0
        total_notifs = len(recent_notifs)
        notif_types = defaultdict(int)
        for n in recent_notifs:
            notif_types[n.get("type", "unknown")] += 1
        distraction_score = _calculate_distraction_score(total_screen, total_notifs, recent)
        return {
            "screen_time_minutes": total_screen,
            "total_notifications": total_notifs,
            "notification_breakdown": dict(notif_types),
            "distraction_score": round(distraction_score, 2),
            "focus_quality": "poor" if distraction_score > 0.7 else "moderate" if distraction_score > 0.4 else "good",
        }

    def get_device_summary(self) -> Dict:
        with self._lock:
            devices = []
            for did, d in self._devices.items():
                devices.append({
                    "device_id": did,
                    "type": d["type"],
                    "name": d["name"],
                    "last_seen": d["last_seen"],
                    "stale": self._is_device_stale(d),
                })
            stats = dict(self._stats)
        return {
            "devices": devices,
            "online_count": sum(1 for d in self._devices.values() if not self._is_device_stale(d)),
            "stats": stats,
        }

    def get_stats(self) -> Dict:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._lock:
            devices_count = len(self._devices)
            loc_count = len(self._location_history)
            health_count = len(self._health_history)
            digital_count = len(self._digital_history)
            notif_count = len(self._notification_history)
        return {
            "stats": stats,
            "devices_registered": devices_count,
            "location_history": loc_count,
            "health_history": health_count,
            "digital_history": digital_count,
            "notification_history": notif_count,
        }

    def register_wifi_location(self, ssid: str, location_type: LocationType) -> None:
        ssid = _sanitize_strict(ssid, 100)
        with self._lock:
            self._wifi_location_map[ssid] = location_type

    def register_person(self, bt_mac: str, name: str) -> None:
        bt_mac = _sanitize_strict(bt_mac, 50)
        name = _sanitize_strict(name, 100)
        with self._lock:
            if len(self._people_map) >= self._MAX_BLUETOOTH:
                oldest = next(iter(self._people_map))
                del self._people_map[oldest]
            self._people_map[bt_mac] = name

    def _refresh_context(self) -> UnifiedContext:
        ctx = UnifiedContext()
        ctx.timestamp = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with self._stats_lock:
                self._stats["context_refreshes"] += 1
            latest_location = self._location_history[-1] if self._location_history else None
            latest_health = self._health_history[-1] if self._health_history else None
            latest_digital = self._digital_history[-1] if self._digital_history else None
            latest_env = self._environment_history[-1] if self._environment_history else None
            connected_devices = sum(1 for d in self._devices.values() if not self._is_device_stale(d))
            people = list(self._people_map.values())

        if latest_location:
            ctx.location = copy.deepcopy(latest_location)
        if latest_health:
            ctx.health = copy.deepcopy(latest_health)
        if latest_digital:
            ctx.digital = copy.deepcopy(latest_digital)
        if latest_env:
            ctx.environment = copy.deepcopy(latest_env)

        ctx.people_nearby = people
        ctx.primary_device = self._infer_primary_device()
        ctx.activity = self._infer_activity(ctx)
        ctx.focus = self._infer_focus(ctx)
        ctx.context_confidence = self._calculate_confidence(ctx, connected_devices)

        with self._lock:
            prev_activity = self._unified_context.activity if self._unified_context else None
            self._unified_context = ctx
            if prev_activity and ctx.activity != prev_activity:
                with self._stats_lock:
                    self._stats["activity_changes"] += 1

        return copy.deepcopy(ctx)

    def _is_device_stale(self, device: Dict) -> bool:
        last_seen_str = device.get("last_seen", "")
        try:
            last_seen = datetime.fromisoformat(last_seen_str)
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - last_seen) > self._STALE_DATA_THRESHOLD
        except (ValueError, TypeError):
            return True

    def _infer_primary_device(self) -> DeviceType:
        with self._lock:
            for did, d in self._devices.items():
                if not self._is_device_stale(d):
                    try:
                        return DeviceType(d["type"])
                    except ValueError:
                        continue
        return DeviceType.UNKNOWN

    def _infer_activity(self, ctx: UnifiedContext) -> ActivityType:
        if ctx.health.sleep_phase in (SleepPhase.DEEP, SleepPhase.REM, SleepPhase.LIGHT):
            return ActivityType.SLEEPING
        if ctx.health.exercise_minutes > 30:
            return ActivityType.EXERCISING
        if ctx.environment.network_type in (NetworkType.WIFI_HOME,) and not ctx.digital.screen_on:
            if ctx.health.step_count > 100:
                return ActivityType.COOKING
            return ActivityType.RELAXING
        if ctx.environment.network_type == NetworkType.WIFI_OFFICE and ctx.digital.screen_on:
            return ActivityType.WORKING
        if ctx.location.location_type == LocationType.TRANSIT or ctx.location.speed and ctx.location.speed > 5:
            return ActivityType.COMMUTING
        if ctx.health.step_count > 500 and ctx.digital.screen_on:
            return ActivityType.WALKING
        if ctx.digital.screen_on:
            return ActivityType.SEDENTARY
        return ActivityType.UNKNOWN

    def _infer_focus(self, ctx: UnifiedContext) -> FocusLevel:
        hr = ctx.health.heart_rate
        hrv = ctx.health.hrv
        spo2 = ctx.health.spo2
        stress = _infer_stress(
            hr if hr else 70,
            hrv if hrv else 50,
            spo2 if spo2 else 97,
        )
        if stress == StressLevel.CRITICAL:
            return FocusLevel.FATIGUED
        if ctx.digital.notifications_last_hour > 20:
            return FocusLevel.DISTRACTED
        if not ctx.digital.screen_on:
            return FocusLevel.RESTING
        if ctx.digital.active_app_category in ("development", "productivity") and ctx.digital.notifications_last_hour < 5:
            if stress == StressLevel.LOW:
                return FocusLevel.DEEP_WORK
            return FocusLevel.ACTIVE
        if stress == StressLevel.HIGH:
            return FocusLevel.DISTRACTED
        return FocusLevel.SCANNING

    def _calculate_confidence(self, ctx: UnifiedContext, device_count: int) -> float:
        confidence = 0.1
        if ctx.location.location_type != LocationType.UNKNOWN:
            confidence += 0.2
        if ctx.health.heart_rate is not None:
            confidence += 0.15
        if ctx.digital.timestamp:
            confidence += 0.15
        if ctx.environment.timestamp:
            confidence += 0.1
        if device_count >= 2:
            confidence += 0.15
        if device_count >= 3:
            confidence += 0.1
        if ctx.health.sleep_phase != SleepPhase.UNKNOWN:
            confidence += 0.05
        return min(1.0, confidence)

    def _parse_watch_health(self, data: Dict) -> HealthSnapshot:
        h = data.get("health", data)
        if not isinstance(h, dict):
            h = {}
        hr = None
        if "heart_rate" in h:
            try:
                hr = int(h["heart_rate"])
            except (ValueError, TypeError):
                pass
        hrv = None
        if "hrv" in h:
            try:
                hrv = float(h["hrv"])
            except (ValueError, TypeError):
                pass
        spo2 = None
        if "spo2" in h:
            try:
                spo2 = float(h["spo2"])
            except (ValueError, TypeError):
                pass
        wtemp = None
        if "wrist_temperature" in h:
            try:
                wtemp = float(h["wrist_temperature"])
            except (ValueError, TypeError):
                pass
        sleep_phase = SleepPhase.UNKNOWN
        if "sleep_phase" in h:
            try:
                sleep_phase = SleepPhase(h["sleep_phase"])
            except ValueError:
                pass
        stress = _infer_stress(hr or 70, hrv or 50, spo2 or 97)
        return HealthSnapshot(
            heart_rate=hr,
            hrv=hrv,
            spo2=spo2,
            wrist_temperature=wtemp,
            step_count=int(h.get("step_count", 0)),
            flights_climbed=int(h.get("flights_climbed", 0)),
            exercise_minutes=int(h.get("exercise_minutes", 0)),
            stand_hours=float(h.get("stand_hours", 0)),
            stress_level=stress,
            sleep_phase=sleep_phase,
            sleep_minutes_last_24h=int(h.get("sleep_minutes", 0)),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _parse_laptop_digital(self, data: Dict) -> DigitalSnapshot:
        screen_on = bool(data.get("screen_on", data.get("screen_locked", True) == False))
        active_app = str(data.get("active_window", data.get("active_app", "")))
        active_cat = self._app_category_map.get(active_app.lower(), "")
        browser_tabs = int(data.get("browser_tabs_open", 0))
        term_sessions = int(data.get("terminal_sessions", 0))
        media_playing = bool(data.get("media_playing", False))
        media_title = str(data.get("media_title", ""))
        media_type = str(data.get("media_type", ""))
        notifs_last_hour = int(data.get("notifications_last_hour", 0))
        calendar_next = data.get("calendar_next_event")
        if isinstance(calendar_next, dict):
            calendar_next = {k: _sanitize_strict(str(v), 200) for k, v in calendar_next.items() if len(str(v)) < 500}
        return DigitalSnapshot(
            screen_on=screen_on,
            active_app=_sanitize_strict(active_app, 100),
            active_app_category=_sanitize_strict(active_cat, 50),
            screen_time_minutes_last_24h=int(data.get("screen_time_minutes", 0)),
            notifications_last_hour=notifs_last_hour,
            browser_tabs_open=browser_tabs,
            terminal_sessions=term_sessions,
            media_playing=media_playing,
            media_title=_sanitize_strict(media_title, 200),
            media_type=_sanitize_strict(media_type, 50),
            calendar_next_event=calendar_next,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _parse_laptop_environment(self, data: Dict) -> EnvironmentSnapshot:
        ssid = str(data.get("wifi_ssid", ""))
        net_type = self._wifi_location_map.get(ssid, NetworkType.UNKNOWN)
        bt_devices = data.get("bluetooth_devices", [])
        if isinstance(bt_devices, list):
            bt_devices = [_sanitize_strict(str(d), 50) for d in bt_devices[:20]]
        else:
            bt_devices = []
        cpu = None
        if "cpu_percent" in data:
            try:
                cpu = float(data["cpu_percent"])
            except (ValueError, TypeError):
                pass
        mem = None
        if "memory_percent" in data:
            try:
                mem = float(data["memory_percent"])
            except (ValueError, TypeError):
                pass
        disk = None
        if "disk_percent" in data:
            try:
                disk = float(data["disk_percent"])
            except (ValueError, TypeError):
                pass
        battery = None
        if "battery_level" in data:
            try:
                battery = float(data["battery_level"])
            except (ValueError, TypeError):
                pass
        return EnvironmentSnapshot(
            network_type=net_type if isinstance(net_type, NetworkType) else NetworkType.UNKNOWN,
            network_ssid=_sanitize_strict(ssid, 100),
            connected_devices=int(data.get("connected_devices", 0)),
            nearby_bluetooth_devices=bt_devices,
            battery_level=battery,
            battery_charging=bool(data.get("battery_charging", False)),
            laptop_cpu_percent=cpu,
            laptop_memory_percent=mem,
            laptop_disk_percent=disk,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def _classify_location(loc_data: Dict) -> LocationType:
    if not isinstance(loc_data, dict):
        return LocationType.UNKNOWN
    name = str(loc_data.get("name", "")).lower()
    ssid = str(loc_data.get("wifi_ssid", "")).lower()
    if any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', name) for w in ["home", "house", "apartment"]):
        return LocationType.HOME
    if any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', name) for w in ["office", "work", "building"]):
        return LocationType.OFFICE
    if any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', name) for w in ["gym", "fitness", "sport"]):
        return LocationType.GYM
    if any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', ssid) for w in ["home", "my_wifi"]):
        return LocationType.HOME
    if any(re.search(r'(?:^|[\s_\-])' + re.escape(w) + r'(?:$|[\s_\-])', ssid) for w in ["office", "corp"]):
        return LocationType.OFFICE
    speed = loc_data.get("speed")
    if speed and float(speed) > 2:
        return LocationType.TRANSIT
    return LocationType.UNKNOWN


def _classify_notification(notif: Any) -> NotificationType:
    if not isinstance(notif, dict):
        return NotificationType.UNKNOWN
    app = str(notif.get("app", "")).lower()
    category = str(notif.get("category", "")).lower()
    app_map = {
        "message": NotificationType.MESSAGE,
        "messages": NotificationType.MESSAGE,
        "mail": NotificationType.EMAIL,
        "email": NotificationType.EMAIL,
        "gmail": NotificationType.EMAIL,
        "social": NotificationType.SOCIAL,
        "news": NotificationType.NEWS,
        "reminders": NotificationType.REMINDER,
        "reminder": NotificationType.REMINDER,
        "alert": NotificationType.ALERT,
        "system": NotificationType.SYSTEM,
        "marketing": NotificationType.MARKETING,
    }
    for key, ntype in app_map.items():
        if re.search(r'\b' + re.escape(key) + r'\b', app) or re.search(r'\b' + re.escape(key) + r'\b', category):
            return ntype
    return NotificationType.UNKNOWN


def _infer_stress(heart_rate: float, hrv: float, spo2: float) -> StressLevel:
    score = 0.0
    if heart_rate > 100:
        score += 0.3
    elif heart_rate > 80:
        score += 0.1
    if hrv < 30:
        score += 0.3
    elif hrv < 50:
        score += 0.1
    if spo2 < 94:
        score += 0.4
    elif spo2 < 97:
        score += 0.1
    if score >= 0.6:
        return StressLevel.CRITICAL
    if score >= 0.4:
        return StressLevel.HIGH
    if score >= 0.2:
        return StressLevel.MODERATE
    return StressLevel.LOW


def _calculate_distraction_score(screen_minutes: int, notifs: int, digital_snaps: list) -> float:
    score = 0.0
    score += min(1.0, screen_minutes / 480) * 0.3
    score += min(1.0, notifs / 50) * 0.4
    context_switches = 0
    if digital_snaps:
        prev_app = None
        for d in digital_snaps[-100:]:
            if d.active_app != prev_app and prev_app is not None:
                context_switches += 1
            prev_app = d.active_app
        if len(digital_snaps) > 1:
            score += min(1.0, context_switches / len(digital_snaps)) * 0.3
    return min(1.0, score)


def _sanitize_dict(data: Dict, max_depth: int = 5) -> Dict:
    if max_depth <= 0:
        return {}
    result = {}
    for k, v in list(data.items())[:_MAX_METADATA_KEYS * 2]:
        k = _sanitize_strict(str(k), 100)
        if isinstance(v, dict):
            result[k] = _sanitize_dict(v, max_depth - 1)
        elif isinstance(v, list):
            result[k] = [_sanitize_strict(str(i), 200) for i in v[:20]]
        elif isinstance(v, str):
            result[k] = _sanitize_strict(v, 5000)
        elif isinstance(v, (int, float, bool)):
            result[k] = v
        elif v is None:
            result[k] = None
        else:
            result[k] = _sanitize(str(v))
    return result