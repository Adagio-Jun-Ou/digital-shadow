#!/usr/bin/env python3
"""
PresenceDetection - Social & Environmental Presence Detection for Shadow AI

Tracks who and what is nearby, infers social dynamics, and manages privacy modes.
The shadow's "sixth sense" for ambient social awareness.

Architecture:
  PresenceDetection
  ├── Device Registry (capped at 100 devices)
  ├── Presence Log (FIFO, 5000 max)
  ├── Interaction Log (FIFO, 10000 max)
  ├── Environment Inference (alone/family/colleagues/social/public)
  ├── Privacy Mode Manager (private/social/public/focus)
  ├── Presence Pattern Detector (recurring patterns per person)
  └── Social Suggestion Engine (proactive social actions)
"""

import logging
import re
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.presence_detection")


class DeviceType(str, Enum):
    PHONE = "phone"
    LAPTOP = "laptop"
    TABLET = "tablet"
    DESKTOP = "desktop"
    SPEAKER = "speaker"
    WATCH = "watch"
    TV = "tv"
    OTHER = "other"


class PresenceStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    AWAY = "away"
    OFFLINE = "offline"


class EnvironmentType(str, Enum):
    ALONE = "alone"
    WITH_FAMILY = "with_family"
    WITH_COLLEAGUES = "with_colleagues"
    SOCIAL = "social"
    PUBLIC = "public"
    UNKNOWN = "unknown"


class PrivacyModeStatus(str, Enum):
    PRIVATE = "private"
    SOCIAL = "social"
    PUBLIC = "public"
    FOCUS = "focus"


@dataclass
class PresenceInfo:
    person: str
    devices: List[str]
    status: PresenceStatus
    last_seen: str
    estimated_activity: str

    def to_dict(self) -> Dict:
        return {
            "person": self.person,
            "devices": list(self.devices),
            "status": self.status.value,
            "last_seen": self.last_seen,
            "estimated_activity": self.estimated_activity,
        }


@dataclass
class EnvironmentContext:
    environment: EnvironmentType
    noise_level: str
    social_density: float
    present_people: List[str]
    privacy_mode: PrivacyModeStatus

    def to_dict(self) -> Dict:
        return {
            "environment": self.environment.value,
            "noise_level": self.noise_level,
            "social_density": round(self.social_density, 2),
            "present_people": list(self.present_people),
            "privacy_mode": self.privacy_mode.value,
        }


@dataclass
class PresencePattern:
    person: str
    pattern_type: str
    typical_days: List[str]
    typical_hours: List[int]
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "person": self.person,
            "pattern_type": self.pattern_type,
            "typical_days": list(self.typical_days),
            "typical_hours": list(self.typical_hours),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class SocialSuggestion:
    action: str
    person: Optional[str]
    reason: str
    urgency: str

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "person": self.person,
            "reason": self.reason,
            "urgency": self.urgency,
        }


class PresenceDetection:
    """
    Social & Environmental Presence Detection

    Tracks:
    - Who is nearby (device presence, activity signals)
    - Environmental context (time, location mode, activity level)
    - Social density (how many people around)
    - Presence patterns (who's typically around at this time)
    - Privacy modes (when to be less interactive)
    """

    _MAX_DEVICES = 100
    _MAX_OWNER_DEVICES = 500
    _MAX_METADATA_KEYS = 100
    _MAX_METADATA_VALUE_LENGTH = 500

    def __init__(self):
        self._lock = threading.Lock()
        self._devices: Dict[str, Dict] = {}
        self._presence_log: deque = deque(maxlen=5000)
        self._interactions: deque = deque(maxlen=10000)
        self._owner_devices: Dict[str, List[str]] = defaultdict(list)

    def register_device(
        self, device_id: str, owner: str, device_type: str = "unknown"
    ) -> str:
        device_id = re.sub(r'[<>\x00-\x1f\x7f]', '', device_id.strip())[:100]
        owner = re.sub(r'[<>\x00-\x1f\x7f]', '', owner.strip())[:100]
        device_type = re.sub(r'[<>\x00-\x1f\x7f]', '', device_type.strip())[:20]
        if not device_id or not owner:
            raise ValueError("device_id and owner must not be empty")
        valid_types = {dt.value for dt in DeviceType}
        if device_type not in valid_types and device_type != "unknown":
            device_type = "other"
        with self._lock:
            if len(self._devices) >= self._MAX_DEVICES and device_id not in self._devices:
                oldest_id = next(iter(self._devices))
                logger.warning("Evicting device %s to make room for %s", oldest_id, device_id)
                self._remove_device(oldest_id)
            self._devices[device_id] = {
                "id": device_id,
                "owner": owner,
                "device_type": device_type,
                "status": PresenceStatus.OFFLINE.value,
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            }
            if device_id not in self._owner_devices[owner]:
                self._owner_devices[owner].append(device_id)
            if len(self._owner_devices) > self._MAX_OWNER_DEVICES:
                excess_owners = list(self._owner_devices.keys())[:len(self._owner_devices) - self._MAX_OWNER_DEVICES]
                for ow in excess_owners:
                    dev_ids = self._owner_devices.pop(ow)
                    for did in dev_ids:
                        self._devices.pop(did, None)
                    logger.warning("Evicted owner '%s' with %d devices (owner cap exceeded)", ow, len(dev_ids))
        logger.info("Registered device %s for owner %s", device_id, owner)
        return device_id

    def update_presence(
        self, device_id: str, status: str = "active", metadata: Dict = None
    ) -> None:
        device_id = re.sub(r'[<>\x00-\x1f\x7f]', '', device_id.strip())[:100]
        valid_statuses = {ps.value for ps in PresenceStatus}
        if status not in valid_statuses:
            logger.warning("Invalid status '%s' for device %s", status, device_id)
            return
        if metadata and len(metadata) > self._MAX_METADATA_KEYS:
            metadata = {k: metadata[k] for k in list(metadata.keys())[:self._MAX_METADATA_KEYS]}
        if metadata:
            sanitized_meta = {}
            for k, v in metadata.items():
                key = re.sub(r'[<>\x00-\x1f\x7f]', '', str(k)).strip()[:50]
                if isinstance(v, str):
                    val = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v).strip()[:self._MAX_METADATA_VALUE_LENGTH]
                else:
                    val = v
                sanitized_meta[key] = val
            metadata = sanitized_meta
        with self._lock:
            if device_id not in self._devices:
                logger.warning("Unknown device %s", device_id)
                return
            now = datetime.now(timezone.utc).isoformat()
            self._devices[device_id]["status"] = status
            self._devices[device_id]["last_seen"] = now
            if metadata:
                self._devices[device_id]["metadata"].update(metadata)
            self._presence_log.append(
                {
                    "device_id": device_id,
                    "status": status,
                    "timestamp": now,
                    "owner": self._devices[device_id]["owner"],
                }
            )

    def get_present_people(self) -> List[PresenceInfo]:
        with self._lock:
            owner_data: Dict[str, Dict] = {}
            for device_id, dev in self._devices.items():
                owner = dev["owner"]
                if owner not in owner_data:
                    owner_data[owner] = {
                        "devices": [],
                        "best_status": PresenceStatus.OFFLINE,
                        "last_seen": dev["last_seen"],
                    }
                owner_data[owner]["devices"].append(device_id)
                try:
                    current = PresenceStatus(dev["status"])
                except ValueError:
                    current = PresenceStatus.OFFLINE
                if self._status_priority(current) > self._status_priority(
                    owner_data[owner]["best_status"]
                ):
                    owner_data[owner]["best_status"] = current
                if dev["last_seen"] > owner_data[owner]["last_seen"]:
                    owner_data[owner]["last_seen"] = dev["last_seen"]
        result = []
        for owner, data in owner_data.items():
            activity = self._estimate_activity(data["best_status"], data["last_seen"])
            result.append(
                PresenceInfo(
                    person=owner,
                    devices=data["devices"],
                    status=data["best_status"],
                    last_seen=data["last_seen"],
                    estimated_activity=activity,
                )
            )
        return result

    def detect_environment(self) -> EnvironmentContext:
        present = self.get_present_people()
        active_people = [
            p for p in present if p.status in (PresenceStatus.ACTIVE, PresenceStatus.IDLE)
        ]
        active_names = [p.person for p in active_people]
        num_active = len(active_people)
        if num_active == 0:
            env = EnvironmentType.ALONE
            noise = "quiet"
            density = 0.0
        elif num_active == 1:
            env = EnvironmentType.ALONE
            noise = "quiet"
            density = 1.0
        else:
            env = self._infer_multi_person_env(num_active, active_people)
            noise = self._infer_noise(num_active)
            density = min(10.0, float(num_active))
        privacy = self._infer_privacy_mode(env)
        return EnvironmentContext(
            environment=env,
            noise_level=noise,
            social_density=density,
            present_people=active_names,
            privacy_mode=privacy,
        )

    def get_privacy_mode(self) -> PrivacyModeStatus:
        env_ctx = self.detect_environment()
        return env_ctx.privacy_mode

    def record_interaction(
        self, person: str, interaction_type: str, duration_minutes: float = 0
    ) -> str:
        person = re.sub(r'[<>\x00-\x1f\x7f]', '', person.strip())[:100]
        interaction_type = re.sub(r'[<>\x00-\x1f\x7f]', '', interaction_type.strip())[:50]
        if not person:
            raise ValueError("person must not be empty")
        if duration_minutes < 0:
            duration_minutes = 0
        interaction_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": interaction_id,
            "person": person,
            "interaction_type": interaction_type,
            "duration_minutes": duration_minutes,
            "timestamp": now,
        }
        with self._lock:
            self._interactions.append(record)
        return interaction_id

    def get_presence_patterns(self, person: str = None) -> List[PresencePattern]:
        with self._lock:
            interactions = list(self._interactions)
            presence_entries = list(self._presence_log)
        all_persons = set()
        person_data: Dict[str, List] = defaultdict(list)
        for entry in interactions:
            p = entry["person"]
            all_persons.add(p)
            person_data[p].append(entry)
        for entry in presence_entries:
            p = entry.get("owner", "")
            if p:
                all_persons.add(p)
                person_data[p].append(entry)
        target_persons = [person] if person else list(all_persons)
        patterns = []
        for p in target_persons:
            entries = person_data.get(p, [])
            if not entries:
                continue
            p_patterns = self._extract_patterns(p, entries)
            patterns.extend(p_patterns)
        return patterns

    def suggest_social_actions(self) -> List[SocialSuggestion]:
        suggestions = []
        present = self.get_present_people()
        now = datetime.now(timezone.utc)
        offline_people = [
            p for p in present if p.status in (PresenceStatus.AWAY, PresenceStatus.OFFLINE)
        ]
        for p in offline_people:
            try:
                last = datetime.fromisoformat(p.last_seen)
                days_away = (now - last).days
                if days_away >= 3:
                    suggestions.append(
                        SocialSuggestion(
                            action=f"Reach out to {p.person}",
                            person=p.person,
                            reason=f"{p.person} hasn't been around in {days_away} days",
                            urgency="high",
                        )
                    )
                elif days_away >= 1:
                    suggestions.append(
                        SocialSuggestion(
                            action=f"Check in with {p.person}",
                            person=p.person,
                            reason=f"{p.person} hasn't been seen today",
                            urgency="medium",
                        )
                    )
            except (ValueError, TypeError):
                pass
        patterns = self.get_presence_patterns()
        for pat in patterns:
            if pat.confidence > 0.5:
                present_names = [pp.person for pp in present]
                if pat.person not in present_names:
                    current_day = now.strftime("%A").lower()
                    if current_day in pat.typical_days and now.hour in pat.typical_hours:
                        suggestions.append(
                            SocialSuggestion(
                                action=f"Expect {pat.person} soon",
                                person=pat.person,
                                reason=f"{pat.person} is typically here on {current_day} around this time",
                                urgency="low",
                            )
                        )
        env_ctx = self.detect_environment()
        if env_ctx.environment == EnvironmentType.ALONE:
            suggestions.append(
                SocialSuggestion(
                    action="Consider social interaction",
                    person=None,
                    reason="You've been alone for a while",
                    urgency="low",
                )
            )
        return suggestions

    def get_stats(self) -> Dict:
        with self._lock:
            total_devices = len(self._devices)
            active_devices = sum(
                1 for d in self._devices.values() if d["status"] == PresenceStatus.ACTIVE.value
            )
            owners = set(d["owner"] for d in self._devices.values())
            return {
                "total_devices": total_devices,
                "active_devices": active_devices,
                "total_people": len(owners),
                "presence_log_size": len(self._presence_log),
                "interaction_count": len(self._interactions),
                "max_devices": self._MAX_DEVICES,
            }

    def _remove_device(self, device_id: str) -> None:
        if device_id in self._devices:
            owner = self._devices[device_id]["owner"]
            del self._devices[device_id]
            if owner in self._owner_devices:
                self._owner_devices[owner] = [
                    d for d in self._owner_devices[owner] if d != device_id
                ]
                if not self._owner_devices[owner]:
                    del self._owner_devices[owner]

    @staticmethod
    def _status_priority(status: PresenceStatus) -> int:
        priorities = {
            PresenceStatus.ACTIVE: 3,
            PresenceStatus.IDLE: 2,
            PresenceStatus.AWAY: 1,
            PresenceStatus.OFFLINE: 0,
        }
        return priorities.get(status, 0)

    @staticmethod
    def _estimate_activity(status: PresenceStatus, last_seen: str) -> str:
        if status == PresenceStatus.ACTIVE:
            return "working"
        elif status == PresenceStatus.IDLE:
            return "leisure"
        elif status == PresenceStatus.AWAY:
            return "unknown"
        else:
            try:
                last = datetime.fromisoformat(last_seen)
                now = datetime.now(timezone.utc)
                hours = (now - last).total_seconds() / 3600
                if hours > 8:
                    return "sleeping"
            except (ValueError, TypeError):
                pass
            return "unknown"

    def _infer_multi_person_env(
        self, num_active: int, active_people: List[PresenceInfo]
    ) -> EnvironmentType:
        if num_active <= 3:
            return EnvironmentType.WITH_FAMILY
        elif num_active <= 6:
            return EnvironmentType.WITH_COLLEAGUES
        elif num_active <= 10:
            return EnvironmentType.SOCIAL
        else:
            return EnvironmentType.PUBLIC

    @staticmethod
    def _infer_noise(num_active: int) -> str:
        if num_active <= 2:
            return "quiet"
        elif num_active <= 5:
            return "moderate"
        else:
            return "loud"

    @staticmethod
    def _infer_privacy_mode(env: EnvironmentType) -> PrivacyModeStatus:
        now = datetime.now(timezone.utc)
        is_weekday = now.weekday() < 5
        is_work_hours = 9 <= now.hour < 17
        if env == EnvironmentType.ALONE and is_weekday and is_work_hours:
            return PrivacyModeStatus.FOCUS
        mapping = {
            EnvironmentType.ALONE: PrivacyModeStatus.PRIVATE,
            EnvironmentType.WITH_FAMILY: PrivacyModeStatus.SOCIAL,
            EnvironmentType.WITH_COLLEAGUES: PrivacyModeStatus.SOCIAL,
            EnvironmentType.SOCIAL: PrivacyModeStatus.SOCIAL,
            EnvironmentType.PUBLIC: PrivacyModeStatus.PUBLIC,
            EnvironmentType.UNKNOWN: PrivacyModeStatus.PRIVATE,
        }
        return mapping.get(env, PrivacyModeStatus.PRIVATE)

    def _extract_patterns(self, person: str, entries: List[Dict]) -> List[PresencePattern]:
        if len(entries) < 2:
            return []
        hour_counts: Dict[int, int] = defaultdict(int)
        day_counts: Dict[str, int] = defaultdict(int)
        parsed = 0
        for entry in entries:
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                hour_counts[ts.hour] += 1
                day_name = ts.strftime("%A").lower()
                day_counts[day_name] += 1
                parsed += 1
            except (ValueError, TypeError, KeyError):
                continue
        total = max(parsed, 1)
        typical_days = [d for d, c in day_counts.items() if c / total >= 0.3]
        typical_hours = [h for h, c in hour_counts.items() if c / total >= 0.2]
        if not typical_days and not typical_hours:
            pattern_type = "irregular"
            confidence = min(0.3, total / 20)
        elif typical_days and all(d in ["monday", "tuesday", "wednesday", "thursday", "friday"] for d in typical_days):
            if any(h < 12 for h in typical_hours):
                pattern_type = "weekday_morning"
            else:
                pattern_type = "weekday_evening"
            confidence = min(0.95, total / 10)
        elif all(d in ["saturday", "sunday"] for d in typical_days):
            pattern_type = "weekend"
            confidence = min(0.95, total / 10)
        else:
            pattern_type = "irregular"
            confidence = min(0.5, total / 15)
        if not typical_days:
            typical_days = sorted(day_counts.keys())
        if not typical_hours:
            typical_hours = sorted(hour_counts.keys())
        return [
            PresencePattern(
                person=person,
                pattern_type=pattern_type,
                typical_days=typical_days,
                typical_hours=typical_hours,
                confidence=round(confidence, 2),
            )
        ]