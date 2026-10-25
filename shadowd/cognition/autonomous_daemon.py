#!/usr/bin/env python3
"""
AutonomousDaemon — Always-On Processing Engine

ShadowAI is your shadow — always on, always listening, always processing.
It doesn't sleep. It doesn't have its own rhythm. It has yours.

It models YOUR vitals, YOUR schedule, YOUR patterns. When you're active,
it's fully engaged. When you drift away, it silently consolidates and
waits — ready the instant you return.

Architecture:
  AutonomousDaemon
  ├── UserRhythmModel (learns YOUR patterns from device data + activity)
  ├── MultiModalObserver (system metrics, filesystem, network, process)
  ├── AlwaysListeningPipeline (continuous observation + proactive insight)
  ├── SecurityGuardian (intrusion detection, anomaly response, zero-day hardening)
  └── PulseScheduler (configurable tick intervals per subsystem)

Security-first design:
  - All observations validated, sanitized, bounded
  - Rate-limited self-actions (no runaway loops)
  - Cryptographic integrity checks on config/state files
  - Resource caps on every subsystem
  - Audit logging on every autonomous decision
  - Kill switch for immediate shutdown
  - Principle of least privilege for file/network access
"""

import asyncio
import copy
from shadowd.event_bus import ShadowEvent, EventType
import hashlib
import os
import platform
import re
try:
    import resource
except ImportError:  # Windows
    resource = None
import subprocess
import threading
import time
import uuid
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("shadowd.cognition.autonomous_daemon")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_RE = _SANITIZE_CONTENT_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_TEXT = 10000
_MAX_STRICT = 200
_MAX_CONTENT = 10000


def _sanitize(text: str, max_len: int = _MAX_TEXT) -> str:
    """Content-safe sanitize — preserves <> and \\t\\n\\r (for descriptions, observations, reasons)."""
    return _SANITIZE_CONTENT_RE.sub('', str(text)).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = _MAX_CONTENT) -> str:
    """Content-safe sanitize — preserves <> and \\t\\n\\r (alias for clarity)."""
    return _SANITIZE_CONTENT_RE.sub('', str(text)).strip()[:max_len]


def _sanitize_strict(text: str, max_len: int = _MAX_STRICT) -> str:
    """Strict sanitize — strips <> and all control chars (for identifiers, titles, IDs)."""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()[:max_len]


class UserActivityState(str, Enum):
    PRESENT = "present"
    IDLE = "idle"
    AWAY = "away"
    UNKNOWN = "unknown"


class ObservationType(str, Enum):
    SYSTEM_METRICS = "system_metrics"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    PROCESS = "process"
    CLIPBOARD = "clipboard"
    SCHEDULE = "schedule"
    BROWSER = "browser"
    DEVICE = "device"
    ENVIRONMENT = "environment"
    SECURITY = "security"
    CUSTOM = "custom"


class AutonomousActionType(str, Enum):
    MEMORY_CONSOLIDATION = "memory_consolidation"
    PATTERN_ANALYSIS = "pattern_analysis"
    PROACTIVE_INSIGHT = "proactive_insight"
    HEALTH_CHECK = "health_check"
    SECURITY_SCAN = "security_scan"
    SKILL_EVOLUTION = "skill_evolution"
    CURIOSITY_DRIVE = "curiosity_drive"
    SELF_REPAIR = "self_repair"
    DREAM_CYCLE = "dream_cycle"
    NOTIFICATION = "notification"
    EVOLUTION_SYNC = "evolution_sync"
    ORACLE_SYNC = "oracle_sync"


class ThreatLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SecurityEventType(str, Enum):
    FILE_INTEGRITY = "file_integrity"
    PROCESS_ANOMALY = "process_anomaly"
    NETWORK_ANOMALY = "network_anomaly"
    RESOURCE_ABUSE = "resource_abuse"
    AUTH_FAILURE = "auth_failure"
    CONFIG_TAMPER = "config_tamper"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    ZERO_DAY_INDICATOR = "zero_day_indicator"


@dataclass
class AutonomousDecision:
    decision_id: str = ""
    action_type: AutonomousActionType = AutonomousActionType.HEALTH_CHECK
    reason: str = ""
    confidence: float = 0.0
    risk_level: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    executed: bool = False
    outcome: Optional[str] = None

    def __post_init__(self):
        if not self.decision_id:
            self.decision_id = f"ad_{uuid.uuid4().hex[:12]}"
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.risk_level = max(0.0, min(1.0, self.risk_level))
        self.reason = _sanitize_content(self.reason, 2000)
        if self.outcome is not None:
            self.outcome = _sanitize_content(self.outcome, 1000)
        if self.context:
            sanitized = {}
            for k, v in list(self.context.items())[:50]:
                sk = _sanitize_strict(str(k), 100)
                if isinstance(v, str):
                    sanitized[sk] = _sanitize_content(v, 500)
                elif isinstance(v, (int, float, bool)):
                    sanitized[sk] = v
                else:
                    sanitized[sk] = _sanitize_content(str(v), 200)
            self.context = sanitized

    def to_dict(self) -> Dict:
        return {
            "decision_id": self.decision_id,
            "action_type": self.action_type.value,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "risk_level": round(self.risk_level, 3),
            "context": copy.deepcopy(self.context) if self.context else {},
            "timestamp": self.timestamp,
            "executed": self.executed,
            "outcome": self.outcome,
        }


@dataclass
class SecurityEvent:
    event_id: str = ""
    event_type: SecurityEventType = SecurityEventType.SUSPICIOUS_BEHAVIOR
    threat_level: ThreatLevel = ThreatLevel.NONE
    description: str = ""
    source: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False
    response_taken: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"se_{uuid.uuid4().hex[:12]}"
        self.description = _sanitize_content(self.description, 1000)
        self.source = _sanitize_strict(self.source)
        if self.details:
            sanitized = {}
            for k, v in list(self.details.items())[:50]:
                sk = _sanitize_strict(str(k), 100)
                if isinstance(v, str):
                    sanitized[sk] = _sanitize_content(v, 500)
                elif isinstance(v, (int, float, bool)):
                    sanitized[sk] = v
                elif isinstance(v, dict):
                    nested = {}
                    for nk, nv in list(v.items())[:50]:
                        nk_s = _sanitize_strict(str(nk), 100)
                        if isinstance(nv, str):
                            nested[nk_s] = _sanitize_content(nv, 500)
                        elif isinstance(nv, (int, float, bool)):
                            nested[nk_s] = nv
                        else:
                            nested[nk_s] = copy.deepcopy(nv) if isinstance(nv, dict) else _sanitize_content(str(nv), 200)
                    sanitized[sk] = nested
                else:
                    sanitized[sk] = _sanitize_content(str(v), 200)
            self.details = sanitized

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "threat_level": int(self.threat_level),
            "description": self.description,
            "source": self.source,
            "details": copy.deepcopy(self.details),
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "response_taken": self.response_taken,
        }


@dataclass
class ObservationPulse:
    pulse_id: str = ""
    observation_type: ObservationType = ObservationType.SYSTEM_METRICS
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.pulse_id:
            self.pulse_id = f"op_{uuid.uuid4().hex[:12]}"
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.source = _sanitize_strict(self.source)
        sanitized_data = {}
        for k, v in list(self.data.items())[:50]:
            sk = _sanitize_strict(str(k), 100)
            if isinstance(v, str):
                sanitized_data[sk] = _sanitize_content(v, 1000)
            elif isinstance(v, (int, float, bool)):
                sanitized_data[sk] = v
            elif isinstance(v, dict):
                inner = {}
                for ik, iv in list(v.items())[:20]:
                    inner[_sanitize_strict(str(ik), 50)] = str(iv)[:200] if isinstance(iv, str) else iv
                sanitized_data[sk] = inner
            elif isinstance(v, list):
                sanitized_data[sk] = v[:100]
        self.data = sanitized_data

    def to_dict(self) -> Dict:
        return {
            "pulse_id": self.pulse_id,
            "observation_type": self.observation_type.value,
            "data": copy.deepcopy(self.data),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass
class FileIntegrityEntry:
    path: str = ""
    sha256: str = ""
    last_checked: str = ""
    modified: bool = False

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "sha256": self.sha256[:16] + "...",
            "last_checked": self.last_checked,
            "modified": self.modified,
        }


@dataclass
class RhythmSample:
    timestamp: str = ""
    activity_state: UserActivityState = UserActivityState.UNKNOWN
    heart_rate: Optional[float] = None
    hrv: Optional[float] = None
    step_count: Optional[int] = None
    sleep_hours: Optional[float] = None
    stress_level: Optional[float] = None
    screen_on: Optional[bool] = None
    location_type: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        self.location_type = _sanitize_strict(self.location_type, 50)
        if self.heart_rate is not None:
            self.heart_rate = max(0.0, min(300.0, float(self.heart_rate)))
        if self.hrv is not None:
            self.hrv = max(0.0, min(500.0, float(self.hrv)))
        if self.stress_level is not None:
            self.stress_level = max(0.0, min(1.0, float(self.stress_level)))
        if self.sleep_hours is not None:
            self.sleep_hours = max(0.0, min(24.0, float(self.sleep_hours)))

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "activity_state": self.activity_state.value,
            "heart_rate": round(self.heart_rate, 1) if self.heart_rate is not None else None,
            "hrv": round(self.hrv, 2) if self.hrv is not None else None,
            "step_count": self.step_count,
            "sleep_hours": round(self.sleep_hours, 2) if self.sleep_hours is not None else None,
            "stress_level": round(self.stress_level, 3) if self.stress_level is not None else None,
            "screen_on": self.screen_on,
            "location_type": self.location_type,
        }


class UserRhythmModel:
    """
    Learns and models the USER's natural patterns.
    ShadowAI is a shadow of you — it understands YOUR rhythm,
    not an artificial one it imposes on itself.

    Tracks:
    - When you're typically active vs away
    - Your heart rate / HRV / stress baselines
    - Your sleep patterns
    - Your location transitions (home/work/gym)
    - Your screen-on patterns
    """

    _MAX_SAMPLES = 10000
    _MAX_HOURLY_STATS = 168

    def __init__(self):
        self._samples: deque = deque(maxlen=self._MAX_SAMPLES)
        self._hourly_activity: Dict[int, Dict[str, int]] = {}
        self._last_sample_time: float = 0
        self._last_activity: UserActivityState = UserActivityState.UNKNOWN
        self._activity_transitions: deque = deque(maxlen=500)
        self._location_history: deque = deque(maxlen=200)
        self._avg_heart_rate: Optional[float] = None
        self._avg_hrv: Optional[float] = None
        self._avg_sleep_hours: Optional[float] = None
        self._avg_stress: Optional[float] = None
        self._lock = threading.Lock()
        self._sample_count = 0
        self._hr_samples: List[float] = []
        self._hrv_samples: List[float] = []
        self._stress_samples: List[float] = []
        self._sleep_samples: List[float] = []

    def ingest(self, sample: RhythmSample):
        with self._lock:
            self._samples.append(sample)
            self._sample_count += 1
            try:
                hour = datetime.fromisoformat(sample.timestamp).hour
            except (ValueError, TypeError):
                hour = datetime.now(timezone.utc).hour
            if hour not in self._hourly_activity:
                self._hourly_activity[hour] = {"present": 0, "idle": 0, "away": 0}
            state = sample.activity_state.value
            if state in self._hourly_activity[hour]:
                self._hourly_activity[hour][state] += 1
            if sample.heart_rate is not None:
                self._hr_samples.append(sample.heart_rate)
                if len(self._hr_samples) > 1000:
                    self._hr_samples = self._hr_samples[-1000:]
                self._avg_heart_rate = round(sum(self._hr_samples) / len(self._hr_samples), 1)
            if sample.hrv is not None:
                self._hrv_samples.append(sample.hrv)
                if len(self._hrv_samples) > 1000:
                    self._hrv_samples = self._hrv_samples[-1000:]
                self._avg_hrv = round(sum(self._hrv_samples) / len(self._hrv_samples), 1)
            if sample.stress_level is not None:
                self._stress_samples.append(sample.stress_level)
                if len(self._stress_samples) > 1000:
                    self._stress_samples = self._stress_samples[-1000:]
                self._avg_stress = round(sum(self._stress_samples) / len(self._stress_samples), 3)
            if sample.sleep_hours is not None:
                self._sleep_samples.append(sample.sleep_hours)
                if len(self._sleep_samples) > 100:
                    self._sleep_samples = self._sleep_samples[-100:]
                self._avg_sleep_hours = round(sum(self._sleep_samples) / len(self._sleep_samples), 1)
            if sample.location_type:
                self._location_history.append(sample.location_type)
            if sample.activity_state != self._last_activity:
                self._activity_transitions.append({
                    "from": self._last_activity.value,
                    "to": sample.activity_state.value,
                    "timestamp": sample.timestamp,
                })
                self._last_activity = sample.activity_state
            self._last_sample_time = time.monotonic()

    def get_user_state(self) -> UserActivityState:
        with self._lock:
            if self._last_sample_time == 0:
                return UserActivityState.UNKNOWN
            elapsed = time.monotonic() - self._last_sample_time
            if elapsed < 300:
                return self._last_activity
            elif elapsed < 1800:
                return UserActivityState.IDLE
            else:
                return UserActivityState.AWAY

    def get_likelihood_active_at_hour(self, hour: int) -> float:
        with self._lock:
            stats = self._hourly_activity.get(hour, {})
            total = sum(stats.values())
            if total == 0:
                return 0.5
            return round(stats.get("present", 0) / total, 3)

    def get_baselines(self) -> Dict:
        with self._lock:
            return {
                "avg_heart_rate": round(self._avg_heart_rate, 1) if self._avg_heart_rate else None,
                "avg_hrv": round(self._avg_hrv, 1) if self._avg_hrv else None,
                "avg_sleep_hours": round(self._avg_sleep_hours, 1) if self._avg_sleep_hours else None,
                "avg_stress": round(self._avg_stress, 3) if self._avg_stress else None,
                "current_state": self._last_activity.value,
                "last_sample_age_seconds": round(time.monotonic() - self._last_sample_time, 1) if self._last_sample_time else None,
            }

    def get_activity_schedule(self) -> Dict[int, float]:
        with self._lock:
            result = {}
            for h in range(24):
                stats = self._hourly_activity.get(h, {})
                total = sum(stats.values())
                result[h] = round(stats.get("present", 0) / total, 3) if total > 0 else 0.5
            return result

    def get_recent_transitions(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            return copy.deepcopy(list(self._activity_transitions)[-limit:])

    def get_current_stress(self) -> Optional[float]:
        with self._lock:
            if self._stress_samples:
                return round(self._stress_samples[-1], 3)
            return self._avg_stress

    def is_stress_elevated(self) -> bool:
        with self._lock:
            if not self._stress_samples or self._avg_stress is None:
                return False
            latest = self._stress_samples[-1] if self._stress_samples else 0
            return latest > self._avg_stress + 0.2

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "samples_total": self._sample_count,
                "current_state": self._last_activity.value,
                "avg_heart_rate": round(self._avg_heart_rate, 1) if self._avg_heart_rate else 0,
                "avg_hrv": round(self._avg_hrv, 1) if self._avg_hrv else 0,
                "avg_sleep_hours": round(self._avg_sleep_hours, 1) if self._avg_sleep_hours else 0,
                "avg_stress": round(self._avg_stress, 3) if self._avg_stress else 0,
                "hourly_stats_entries": len(self._hourly_activity),
                "transitions_tracked": len(self._activity_transitions),
                "locations_tracked": len(self._location_history),
            }


class MultiModalObserver:
    """
    Observes the system through multiple modalities:
    - CPU/memory/disk metrics
    - Filesystem changes
    - Network connections
    - Process list
    - Clipboard (if permitted)
    - Schedule/calendar
    """

    _MAX_OBSERVATIONS = 5000
    _MAX_METRICS_HISTORY = 1000
    _MAX_WATCHED_PATHS = 50

    def __init__(self):
        self._observations: deque = deque(maxlen=self._MAX_OBSERVATIONS)
        self._metrics_history: deque = deque(maxlen=self._MAX_METRICS_HISTORY)
        self._file_hashes: Dict[str, FileIntegrityEntry] = {}
        self._watched_paths: Set[str] = set()
        self._lock = threading.Lock()
        self._last_process_snapshot: List[Dict] = []
        self._last_network_snapshot: List[Dict] = []
        self._observation_count = 0

    def observe_system_metrics(self) -> ObservationPulse:
        data = {}
        try:
            data["cpu_count"] = os.cpu_count() or 0
            data["platform"] = _sanitize_strict(platform.system(), 50)
            data["python_version"] = _sanitize_strict(platform.python_version(), 20)
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            data["load_avg_1m"] = round(load_avg[0], 2)
            data["load_avg_5m"] = round(load_avg[1], 2)
            data["load_avg_15m"] = round(load_avg[2], 2)
        except (OSError, ValueError):
            pass
        try:
            if resource is not None:
                ru = resource.getrusage(resource.RUSAGE_SELF)
                data["ru_maxrss_mb"] = round(ru.ru_maxrss / 1024, 1)
                data["ru_utime"] = round(ru.ru_utime, 2)
                data["ru_stime"] = round(ru.ru_stime, 2)
        except (OSError, ValueError, AttributeError):
            pass
        try:
            if hasattr(os, "statvfs"):
                disk = os.statvfs("/")
                data["disk_total_gb"] = round(disk.f_blocks * disk.f_frsize / (1024**3), 1)
                data["disk_free_gb"] = round(disk.f_bavail * disk.f_frsize / (1024**3), 1)
                data["disk_usage_pct"] = round((1 - disk.f_bavail / max(1, disk.f_blocks)) * 100, 1)
            else:
                import shutil
                usage = shutil.disk_usage(os.path.abspath(os.sep))
                data["disk_total_gb"] = round(usage.total / (1024**3), 1)
                data["disk_free_gb"] = round(usage.free / (1024**3), 1)
                data["disk_usage_pct"] = round((usage.used / max(1, usage.total)) * 100, 1)
        except (OSError, AttributeError):
            pass
        pulse = ObservationPulse(
            observation_type=ObservationType.SYSTEM_METRICS,
            data=data,
            confidence=0.95,
            source="multimodal_observer",
        )
        self._record(pulse)
        return pulse

    def observe_processes(self) -> ObservationPulse:
        processes = []
        try:
            result = subprocess.run(
                ["ps", "aux", "--sort=-%mem"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[:50]:
                    parts = line.split(None, 10)
                    if len(parts) >= 11:
                        processes.append({
                            "user": _sanitize_strict(parts[0], 20),
                            "pid": parts[1],
                            "cpu_pct": parts[2],
                            "mem_pct": parts[3],
                            "command": _sanitize_content(parts[10], 200),
                        })
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        with self._lock:
            self._last_process_snapshot = processes
        pulse = ObservationPulse(
            observation_type=ObservationType.PROCESS,
            data={"processes": processes, "count": len(processes)},
            confidence=0.85,
            source="multimodal_observer",
        )
        self._record(pulse)
        return pulse

    def observe_network(self) -> ObservationPulse:
        connections = []
        try:
            result = subprocess.run(
                ["ss", "-tunap"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[:100]:
                    line = _sanitize_content(line, 200)
                    if line and not line.startswith("Netid"):
                        connections.append(line)
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            pass
        with self._lock:
            self._last_network_snapshot = [{"raw": c} for c in connections]
        pulse = ObservationPulse(
            observation_type=ObservationType.NETWORK,
            data={"connections": len(connections), "samples": connections[:20]},
            confidence=0.8,
            source="multimodal_observer",
        )
        self._record(pulse)
        return pulse

    def observe_filesystem(self, watch_paths: Optional[List[str]] = None) -> ObservationPulse:
        changes = []
        with self._lock:
            paths_to_check = list(watch_paths or self._watched_paths)[:50]
        for p in paths_to_check:
            try:
                if not os.path.exists(p):
                    continue
                current_hash = self._hash_file(p)
                with self._lock:
                    if p in self._file_hashes:
                        if self._file_hashes[p].sha256 != current_hash:
                            self._file_hashes[p].modified = True
                            self._file_hashes[p].sha256 = current_hash
                            self._file_hashes[p].last_checked = datetime.now(timezone.utc).isoformat()
                            changes.append({"path": p, "change": "modified"})
                        else:
                            self._file_hashes[p].last_checked = datetime.now(timezone.utc).isoformat()
                    else:
                        self._file_hashes[p] = FileIntegrityEntry(
                            path=p, sha256=current_hash,
                            last_checked=datetime.now(timezone.utc).isoformat(), modified=False,
                        )
            except (OSError, PermissionError):
                pass
        pulse = ObservationPulse(
            observation_type=ObservationType.FILESYSTEM,
            data={"changes": changes, "watched": len(paths_to_check)},
            confidence=0.9 if changes else 0.95,
            source="multimodal_observer",
        )
        self._record(pulse)
        return pulse

    _DANGEROUS_PATHS = frozenset([
        '/etc/shadow', '/etc/passwd', '/etc/ssh', '/root/.ssh',
        '/etc/gshadow', '/etc/hosts.equiv', '/etc/NetworkManager',
    ])

    def add_watch_path(self, path: str) -> bool:
        path = _sanitize_strict(path, 500)
        if not path or not os.path.exists(path):
            return False
        resolved = os.path.realpath(path)
        for dangerous in self._DANGEROUS_PATHS:
            if resolved == dangerous or resolved.startswith(dangerous + '/'):
                logger.warning("Blocked watch path: %s (matches dangerous path)", path.replace(chr(10), ' ').replace(chr(13), ' '))
                return False
        with self._lock:
            if len(self._watched_paths) >= self._MAX_WATCHED_PATHS:
                logger.warning("Watch path limit reached (%s)", self._MAX_WATCHED_PATHS)
                return False
            self._watched_paths.add(path)
            if path not in self._file_hashes:
                try:
                    self._file_hashes[path] = FileIntegrityEntry(
                        path=path, sha256=self._hash_file(path),
                        last_checked=datetime.now(timezone.utc).isoformat(), modified=False,
                    )
                except (OSError, PermissionError):
                    return False
        return True

    def remove_watch_path(self, path: str) -> bool:
        with self._lock:
            self._watched_paths.discard(path)
            self._file_hashes.pop(path, None)
        return True

    def get_watched_paths(self) -> List[str]:
        with self._lock:
            return list(self._watched_paths)

    def get_file_integrity(self) -> List[Dict]:
        with self._lock:
            return [e.to_dict() for e in self._file_hashes.values()]

    def get_recent_observations(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return [o.to_dict() for o in list(self._observations)[-limit:]]

    def get_observation_count(self) -> int:
        with self._lock:
            return self._observation_count

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "observations_total": self._observation_count,
                "observations_in_buffer": len(self._observations),
                "watched_paths": len(self._watched_paths),
                "file_hashes_tracked": len(self._file_hashes),
                "last_process_count": len(self._last_process_snapshot),
                "last_network_count": len(self._last_network_snapshot),
            }

    def _record(self, pulse: ObservationPulse):
        with self._lock:
            self._observations.append(pulse)
            if pulse.observation_type == ObservationType.SYSTEM_METRICS:
                self._metrics_history.append(pulse)
            self._observation_count += 1

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.sha256()
        max_bytes = 10 * 1024 * 1024
        total = 0
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
        return h.hexdigest()


class SecurityGuardian:
    """
    Zero-day protection, exploit prevention, and self-monitoring.

    Security model:
    - File integrity monitoring (detect config tampering)
    - Process anomaly detection (unexpected new processes)
    - Network anomaly detection (unexpected connections)
    - Resource abuse detection (memory/CPU spikes)
    - Auth failure tracking
    - Config tamper detection
    - Automatic threat response (escalate, isolate, alert)
    - Audit trail for all security decisions
    """

    _MAX_EVENTS = 5000
    _MAX_AUTH_FAILURES = 1000
    _MAX_BASELINES = 20

    def __init__(self):
        self._events: deque = deque(maxlen=self._MAX_EVENTS)
        self._auth_failures: deque = deque(maxlen=self._MAX_AUTH_FAILURES)
        self._process_baseline: Set[str] = set()
        self._network_baseline: Set[str] = set()
        self._allowed_network_remote_ports: Set[int] = {443, 80, 8443, 8444, 8000, 8001, 8765, 8766, 4096, 3000}
        self._threat_level: ThreatLevel = ThreatLevel.NONE
        self._lock = threading.Lock()
        self._scan_count = 0
        self._auto_response_enabled = True
        self._kill_switch_engaged = False
        self._stats = {
            "scans_run": 0,
            "threats_detected": 0,
            "auto_responses": 0,
            "auth_failures_total": 0,
            "critical_events": 0,
        }
        self._stats_lock = threading.Lock()

    def record_auth_failure(self, source: str, detail: str = "") -> SecurityEvent:
        event = SecurityEvent(
            event_type=SecurityEventType.AUTH_FAILURE,
            threat_level=ThreatLevel.LOW,
            description=f"Auth failure from {_sanitize_strict(source, 100)}",
            source="security_guardian",
            details={"source": _sanitize_strict(source, 100), "detail": _sanitize_content(detail, 500)},
        )
        with self._lock:
            self._auth_failures.append(event)
            def _within_5min(e: SecurityEvent) -> bool:
                try:
                    ts = datetime.fromisoformat(e.timestamp)
                except (ValueError, TypeError):
                    return False
                return (datetime.now(timezone.utc) - ts).total_seconds() < 300

            recent_failures = sum(1 for e in self._auth_failures if _within_5min(e))
            if recent_failures >= 10:
                event.threat_level = ThreatLevel.HIGH
                event.description = f"Brute force detected: {recent_failures} failures in 5 minutes"
            elif recent_failures >= 5:
                event.threat_level = ThreatLevel.MEDIUM
            self._events.append(event)
            with self._stats_lock:
                self._stats["auth_failures_total"] += 1
            if event.threat_level >= ThreatLevel.MEDIUM:
                with self._stats_lock:
                    self._stats["critical_events"] += 1
                self._update_threat_level(event.threat_level)
        return event

    def scan_processes(self, current_processes: List[Dict]) -> List[SecurityEvent]:
        events = []
        current_names = set()
        for p in current_processes:
            cmd = p.get("command", "")
            if cmd:
                base = cmd.split()[0].split('/')[-1] if '/' in cmd.split()[0] else cmd.split()[0]
                current_names.add(_sanitize_strict(base, 100))

        with self._lock:
            if not self._process_baseline:
                self._process_baseline = current_names
                self._scan_count += 1
                with self._stats_lock:
                    self._stats["scans_run"] += 1
                return events

            new_processes = current_names - self._process_baseline
            suspicious_patterns = re.compile(
                r'(ncat|nc\s|netcat|reverse.shell|bindshell|keylog|rootkit|backdoor|cryptominer|xmrig|'
                r'miners|hack2|exploit|payload|meterpreter|beacon|c2_|cobalt)',
                re.IGNORECASE
            )
            for proc_name in new_processes:
                if suspicious_patterns.search(proc_name):
                    event = SecurityEvent(
                        event_type=SecurityEventType.PROCESS_ANOMALY,
                        threat_level=ThreatLevel.CRITICAL,
                        description=f"SUSPICIOUS PROCESS DETECTED: {proc_name}",
                        source="security_guardian",
                        details={"process_name": _sanitize_strict(proc_name, 100), "pid": "unknown"},
                    )
                    events.append(event)
                    self._events.append(event)
                    with self._stats_lock:
                        self._stats["threats_detected"] += 1
                    with self._stats_lock:
                        self._stats["critical_events"] += 1
                    self._update_threat_level(ThreatLevel.CRITICAL)
            if len(new_processes) > 5:
                event = SecurityEvent(
                    event_type=SecurityEventType.PROCESS_ANOMALY,
                    threat_level=ThreatLevel.MEDIUM,
                    description=f"Process burst: {len(new_processes)} new processes",
                    source="security_guardian",
                    details={"new_count": len(new_processes)},
                )
                events.append(event)
                self._events.append(event)
                with self._stats_lock:
                    self._stats["threats_detected"] += 1
            self._process_baseline.update(current_names)
            if len(self._process_baseline) > 500:
                self._process_baseline = set(list(self._process_baseline)[-500:])
            self._scan_count += 1
            with self._stats_lock:
                self._stats["scans_run"] += 1
        return events

    def scan_network(self, connections: List[str]) -> List[SecurityEvent]:
        events = []
        suspicious_port_patterns = re.compile(
            r'[:\.](4444|5555|6666|6667|6668|6669|31337|1234|6697|9999)[\s:>-]',
        )
        for conn in connections:
            if suspicious_port_patterns.search(conn):
                event = SecurityEvent(
                    event_type=SecurityEventType.NETWORK_ANOMALY,
                    threat_level=ThreatLevel.HIGH,
                    description=f"Suspicious port in connection: {_sanitize_content(conn, 200)}",
                    source="security_guardian",
                    details={"connection": _sanitize_content(conn, 200)},
                )
                events.append(event)
                with self._lock:
                    self._events.append(event)
                    with self._stats_lock:
                        self._stats["threats_detected"] += 1
                    with self._stats_lock:
                        self._stats["critical_events"] += 1
                    self._update_threat_level(ThreatLevel.HIGH)
        with self._lock:
            self._scan_count += 1
            with self._stats_lock:
                self._stats["scans_run"] += 1
        return events

    def check_resource_abuse(self, metrics: Dict) -> Optional[SecurityEvent]:
        load_1m = metrics.get("load_avg_1m", 0)
        cpu_count = metrics.get("cpu_count", 1)
        if cpu_count > 0 and load_1m > cpu_count * 4:
            event = SecurityEvent(
                event_type=SecurityEventType.RESOURCE_ABUSE,
                threat_level=ThreatLevel.HIGH if load_1m > cpu_count * 8 else ThreatLevel.MEDIUM,
                description=f"CPU overload: load={load_1m}, cores={cpu_count}",
                source="security_guardian",
                details=copy.deepcopy(metrics),
            )
            with self._lock:
                self._events.append(event)
                with self._stats_lock:
                    self._stats["threats_detected"] += 1
            return event
        memory_pct = metrics.get("disk_usage_pct", 0)
        if memory_pct > 95:
            event = SecurityEvent(
                event_type=SecurityEventType.RESOURCE_ABUSE,
                threat_level=ThreatLevel.HIGH,
                description=f"Disk nearly full: {memory_pct}%",
                source="security_guardian",
                details=copy.deepcopy(metrics),
            )
            with self._lock:
                self._events.append(event)
                with self._stats_lock:
                    self._stats["threats_detected"] += 1
            return event
        return None

    def check_file_integrity(self, integrity_entries: List[Dict]) -> List[SecurityEvent]:
        events = []
        for entry in integrity_entries:
            if entry.get("modified"):
                event = SecurityEvent(
                    event_type=SecurityEventType.FILE_INTEGRITY,
                    threat_level=ThreatLevel.MEDIUM,
                    description=f"File modified: {_sanitize_strict(entry.get('path', 'unknown'), 200)}",
                    source="security_guardian",
                    details={"path": _sanitize_strict(entry.get("path", ""), 200)},
                )
                events.append(event)
                with self._lock:
                    self._events.append(event)
                    with self._stats_lock:
                        self._stats["threats_detected"] += 1
        return events

    def detect_zero_day_indicators(self, context: Dict) -> List[SecurityEvent]:
        events = []
        indicators = [
            ("unexpected_child_process", lambda c: bool(re.search(r'python.*-c.*import\s', str(c.get("command", "")), re.IGNORECASE))),
            ("suspicious_env_modification", lambda c: any(re.search(r'\b' + re.escape(k) + r'\b', str(c)) for k in ["LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "__ITEM"])),
            ("shell_injection_pattern", lambda c: bool(re.search(r';\s*rm\s|&&\s*curl\s|`\s*wget', str(c), re.IGNORECASE))),
            ("privilege_escalation", lambda c: bool(re.search(r'sudo\s+su|sudo\s+-i|pkexec', str(c), re.IGNORECASE))),
        ]
        for name, detector in indicators:
            try:
                if detector(context):
                    event = SecurityEvent(
                        event_type=SecurityEventType.ZERO_DAY_INDICATOR,
                        threat_level=ThreatLevel.CRITICAL,
                        description=f"Zero-day indicator: {name}",
                        source="security_guardian",
                        details={"indicator": _sanitize_strict(name, 100)},
                    )
                    events.append(event)
                    with self._lock:
                        self._events.append(event)
                        with self._stats_lock:
                            self._stats["threats_detected"] += 1
                        with self._stats_lock:
                            self._stats["critical_events"] += 1
                        self._update_threat_level(ThreatLevel.CRITICAL)
            except (TypeError, ValueError, re.error):
                pass
        return events

    def engage_kill_switch(self) -> Dict:
        with self._lock:
            self._kill_switch_engaged = True
            self._update_threat_level(ThreatLevel.CRITICAL)
            logger.critical("KILL SWITCH ENGAGED - ShadowOS autonomous operations halted")
            return {
                "action": "kill_switch_engaged",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "threat_level": int(self._threat_level),
            }

    def disengage_kill_switch(self) -> Dict:
        with self._lock:
            self._kill_switch_engaged = False
            self._threat_level = ThreatLevel.NONE
            return {
                "action": "kill_switch_disengaged",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def is_kill_switch_engaged(self) -> bool:
        with self._lock:
            return self._kill_switch_engaged

    def get_threat_level(self) -> ThreatLevel:
        with self._lock:
            return self._threat_level

    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return [e.to_dict() for e in list(self._events)[-limit:]]

    def get_stats(self) -> Dict:
        with self._lock:
            with self._stats_lock:
                stats_copy = copy.deepcopy(self._stats)
            return {
                "threat_level": int(self._threat_level),
                "kill_switch": self._kill_switch_engaged,
                "auto_response_enabled": self._auto_response_enabled,
                "scans_run": self._scan_count,
                **stats_copy,
            }

    def _update_threat_level(self, new_level: ThreatLevel):
        if new_level > self._threat_level:
            self._threat_level = new_level
            if self._auto_response_enabled and new_level >= ThreatLevel.CRITICAL:
                self._kill_switch_engaged = True
                with self._stats_lock:
                    self._stats["auto_responses"] += 1


class AutonomousDaemon:
    """
    Always-on autonomous processing engine.

    ShadowAI is your shadow — it's always there, always processing.
    No artificial sleep cycles. No imposed rhythms.
    It models YOUR patterns and adapts to YOU.

    When you're present → full observation, proactive insights, real-time response.
    When you're idle   → background consolidation, dreaming, pattern mining.
    When you're away   → deep consolidation, security hardening, maintenance.
    The instant you return → fully engaged again.

    It never sleeps. It's a shadow. Shadows don't rest.
    """

    _MAX_DECISIONS = 2000
    _MAX_AUTONOMOUS_HISTORY = 5000
    _MIN_TICK_INTERVAL = 5.0
    _MAX_TICK_INTERVAL = 60.0

    def __init__(self, base_tick_seconds: float = 15.0):
        self.observer = MultiModalObserver()
        self.guardian = SecurityGuardian()
        self.user_rhythm = UserRhythmModel()
        self._base_tick = max(self._MIN_TICK_INTERVAL, min(base_tick_seconds, self._MAX_TICK_INTERVAL))
        self._running = False
        self._daemon_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._decisions: deque = deque(maxlen=self._MAX_DECISIONS)
        self._history: deque = deque(maxlen=self._MAX_AUTONOMOUS_HISTORY)
        self._last_observation_time: float = 0
        self._last_security_scan: float = 0
        self._last_consolidation: float = 0
        self._last_dream_cycle: float = 0
        self._last_proactive_insight: float = 0
        self._last_pattern_mining: float = 0
        self._last_evolution_sync: float = 0
        self._last_oracle_sync: float = 0
        self._cycles_completed = 0
        self._start_time: Optional[float] = None
        self._shadow_os = None
        self._stats = {
            "observations_made": 0,
            "autonomous_actions": 0,
            "security_scans": 0,
            "dream_cycles": 0,
            "consolidation_runs": 0,
            "curiosity_drives": 0,
            "proactive_insights": 0,
            "pattern_mining_runs": 0,
            "evolution_syncs": 0,
            "oracle_syncs": 0,
            "threats_responded": 0,
            "errors": 0,
        }
        self._action_rate_limits: Dict[str, float] = {
            "security_scan": 30.0,
            "observation": 15.0,
            "consolidation": 120.0,
            "dream_cycle": 300.0,
            "curiosity_drive": 60.0,
            "skill_evolution": 300.0,
            "proactive_insight": 45.0,
            "pattern_mining": 90.0,
            "evolution_sync": 120.0,
            "oracle_sync": 180.0,
        }
        self._last_action_time: Dict[str, float] = {}
        self._stats_lock = threading.Lock()

    def set_shadow_os(self, shadow_os):
        self._shadow_os = shadow_os

    def get_user_state(self) -> UserActivityState:
        return self.user_rhythm.get_user_state()

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._daemon_task = asyncio.create_task(self._run_loop())
        logger.info("AutonomousDaemon started — tick=%ss, always on", self._base_tick)

    async def stop(self):
        self._running = False
        if self._daemon_task:
            self._daemon_task.cancel()
            try:
                await self._daemon_task
            except asyncio.CancelledError:
                pass
        logger.info("AutonomousDaemon stopped")

    def is_running(self) -> bool:
        return self._running

    async def _run_loop(self):
        while self._running:
            try:
                if self.guardian.is_kill_switch_engaged():
                    await asyncio.sleep(self._base_tick * 4)
                    continue
                user_state = self.get_user_state()
                await self._tick(user_state)
                await asyncio.sleep(self._base_tick)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Daemon tick error: %s", type(e).__name__)
                with self._lock:
                    with self._stats_lock:
                        self._stats["errors"] += 1
                await asyncio.sleep(self._base_tick * 2)

    async def _tick(self, user_state: UserActivityState):
        now = time.monotonic()
        pulse_data = {}
        try:
            pulse = await asyncio.get_running_loop().run_in_executor(
                None, self.observer.observe_system_metrics
            )
            pulse_data = pulse.data
            self._process_observation_pulse(pulse)
        except (OSError, ValueError, RuntimeError) as e:
            logger.debug("System metrics error: %s", type(e).__name__)

        if now - self._last_security_scan >= self._rate_limit("security_scan"):
            await self._run_security_scan(pulse_data)

        if user_state in (UserActivityState.PRESENT, UserActivityState.IDLE):
            if now - self._last_observation_time >= self._rate_limit("observation"):
                try:
                    proc_pulse = await asyncio.get_running_loop().run_in_executor(
                        None, self.observer.observe_processes
                    )
                    self._process_observation_pulse(proc_pulse)
                    net_pulse = await asyncio.get_running_loop().run_in_executor(
                        None, self.observer.observe_network
                    )
                    self._process_observation_pulse(net_pulse)
                except (OSError, ValueError, RuntimeError, ConnectionError) as e:
                    logger.debug("Observation error: %s", type(e).__name__)
                self._last_observation_time = now

        if user_state == UserActivityState.PRESENT:
            if now - self._last_action_time.get("curiosity_drive", 0) >= self._rate_limit("curiosity_drive"):
                await self._run_curiosity_drive()
            if now - self._last_proactive_insight >= self._rate_limit("proactive_insight"):
                await self._run_proactive_insight()
            if now - self._last_evolution_sync >= self._rate_limit("evolution_sync"):
                await self._run_evolution_sync()

        if user_state == UserActivityState.IDLE:
            if now - self._last_action_time.get("pattern_mining", 0) >= self._rate_limit("pattern_mining"):
                await self._run_pattern_mining()

        if user_state in (UserActivityState.IDLE, UserActivityState.AWAY):
            if now - self._last_consolidation >= self._rate_limit("consolidation"):
                await self._run_consolidation()
            if now - self._last_dream_cycle >= self._rate_limit("dream_cycle"):
                await self._run_dream_cycle()
            if now - self._last_oracle_sync >= self._rate_limit("oracle_sync"):
                await self._run_oracle_sync()

        with self._lock:
            self._cycles_completed += 1
            
            # Auto-notify the main bus about security events
            if self._shadow_os and hasattr(self._shadow_os, 'journal'):
                try:
                    self._shadow_os.journal.record(
                        "observation",
                        f"Daemon cycle {self._cycles_completed} | state={user_state.value}",
                        category="system", importance=0.3,
                        tags=["daemon", "tick", "automated"]
                    )
                except (ValueError, TypeError, AttributeError, RuntimeError):
                    pass

    def _process_observation_pulse(self, pulse: ObservationPulse):
        with self._lock:
            with self._stats_lock:
                self._stats["observations_made"] += 1
        if self._shadow_os:
            try:
                eb = getattr(self._shadow_os, 'event_bus', None)
                if eb and hasattr(eb, 'publish'):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(eb.publish(ShadowEvent(
                            source="autonomous_daemon",
                            event_type=EventType.OBSERVATION,
                            data=copy.deepcopy(pulse.data),
                            confidence=pulse.confidence,
                            metadata={"observation_type": pulse.observation_type.value, "source": pulse.source},
                        )))
                    except RuntimeError:
                        pass
            except (AttributeError, ValueError, RuntimeError):
                pass

    async def _run_security_scan(self, metrics: Dict):
        self._last_security_scan = time.monotonic()
        events = []
        resource_event = self.guardian.check_resource_abuse(metrics)
        if resource_event:
            events.append(resource_event)
        integrity = self.observer.get_file_integrity()
        events.extend(self.guardian.check_file_integrity(integrity))
        if self._shadow_os:
            try:
                proc_pulse = await asyncio.get_running_loop().run_in_executor(
                    None, self.observer.observe_processes
                )
                processes = proc_pulse.data.get("processes", [])
                events.extend(self.guardian.scan_processes(processes))
            except (OSError, ValueError, RuntimeError):
                pass
            try:
                net_pulse = await asyncio.get_running_loop().run_in_executor(
                    None, self.observer.observe_network
                )
                connections = net_pulse.data.get("samples", [])
                events.extend(self.guardian.scan_network(connections))
            except (OSError, ValueError, RuntimeError):
                pass
        with self._lock:
            with self._stats_lock:
                self._stats["security_scans"] += 1
            if events:
                with self._stats_lock:
                    self._stats["threats_responded"] += len(events)

    async def _run_consolidation(self):
        self._last_consolidation = time.monotonic()
        if self._shadow_os:
            try:
                dream = getattr(self._shadow_os, 'dream', None)
                if dream and hasattr(dream, 'consolidate'):
                    await asyncio.get_running_loop().run_in_executor(
                        None, dream.consolidate
                    )
                memory = getattr(self._shadow_os, 'memory', None)
                if memory and hasattr(memory, 'compact'):
                    await asyncio.get_running_loop().run_in_executor(
                        None, memory.compact
                    )
            except (AttributeError, RuntimeError, ValueError, OSError):
                pass
        decision = AutonomousDecision(
            action_type=AutonomousActionType.MEMORY_CONSOLIDATION,
            reason="Background consolidation while user idle/away",
            confidence=0.9,
            risk_level=0.05,
        )
        decision.executed = True
        decision.outcome = "completed"
        self._record_decision(decision)
        with self._lock:
            with self._stats_lock:
                self._stats["consolidation_runs"] += 1

    async def _run_dream_cycle(self):
        self._last_dream_cycle = time.monotonic()
        if self._shadow_os:
            try:
                dream = getattr(self._shadow_os, 'dream', None)
                if dream and hasattr(dream, 'consolidate'):
                    await asyncio.get_running_loop().run_in_executor(
                        None, dream.consolidate
                    )
            except (AttributeError, RuntimeError, ValueError, OSError):
                pass
        decision = AutonomousDecision(
            action_type=AutonomousActionType.DREAM_CYCLE,
            reason="Deep consolidation while user away",
            confidence=0.85,
            risk_level=0.02,
        )
        decision.executed = True
        decision.outcome = "completed"
        self._record_decision(decision)
        with self._lock:
            with self._stats_lock:
                self._stats["dream_cycles"] += 1

    async def _run_curiosity_drive(self):
        self._last_action_time["curiosity_drive"] = time.monotonic()
        decision = AutonomousDecision(
            action_type=AutonomousActionType.CURIOSITY_DRIVE,
            reason="Curiosity drive: exploring unobserved patterns",
            confidence=0.6,
            risk_level=0.1,
        )
        if self._shadow_os and decision.risk_level < 0.3:
            try:
                tr = getattr(self._shadow_os, 'temporal_reasoning', None)
                if tr and hasattr(tr, 'detect_patterns'):
                    patterns = tr.detect_patterns("system")
                    decision.context["patterns_found"] = len(patterns)
                    decision.executed = True
                    decision.outcome = f"explored: {len(patterns)} temporal patterns"
            except (AttributeError, RuntimeError, ValueError, OSError):
                decision.outcome = "skipped: error"
        self._record_decision(decision)
        with self._lock:
            with self._stats_lock:
                self._stats["curiosity_drives"] += 1
            with self._stats_lock:
                self._stats["autonomous_actions"] += 1

    async def _run_proactive_insight(self):
        self._last_proactive_insight = time.monotonic()
        decision = AutonomousDecision(
            action_type=AutonomousActionType.PROACTIVE_INSIGHT,
            reason="Proactive insight: user is present, generating relevant observations",
            confidence=0.7,
            risk_level=0.08,
        )
        if self._shadow_os:
            try:
                ad = getattr(self._shadow_os, 'anomaly_detector', None)
                if ad and hasattr(ad, 'get_stats'):
                    stress = self.user_rhythm.get_current_stress()
                    if stress is not None and stress > 0.6:
                        decision.context["stress_alert"] = True
                        decision.outcome = "user stress elevated — prioritizing calming insights"
                    else:
                        anomalies = ad.get_stats()
                        unack = anomalies.get("unacknowledged_anomalies", 0)
                        if unack > 0:
                            decision.context["anomaly_alert"] = True
                            decision.outcome = f"{unack} unacknowledged anomalies — surfacing"
                decision.executed = True
            except (AttributeError, RuntimeError, ValueError, KeyError):
                decision.outcome = "skipped: error"
        self._record_decision(decision)
        with self._lock:
            with self._stats_lock:
                self._stats["proactive_insights"] += 1
            with self._stats_lock:
                self._stats["autonomous_actions"] += 1

    async def _run_pattern_mining(self):
        self._last_action_time["pattern_mining"] = time.monotonic()
        decision = AutonomousDecision(
            action_type=AutonomousActionType.PATTERN_ANALYSIS,
            reason="Pattern mining: user idle, mining cross-domain patterns",
            confidence=0.75,
            risk_level=0.05,
        )
        if self._shadow_os:
            try:
                recall = getattr(self._shadow_os, 'recall_synthesis', None)
                if recall and hasattr(recall, 'find_cross_domain_patterns'):
                    patterns = recall.find_cross_domain_patterns()
                    decision.context["cross_domain_patterns"] = len(patterns)
                    decision.executed = True
                    decision.outcome = f"mined: {len(patterns)} cross-domain patterns"
            except (AttributeError, RuntimeError, ValueError, KeyError):
                decision.outcome = "skipped: error"
        self._record_decision(decision)
        with self._lock:
            with self._stats_lock:
                self._stats["pattern_mining_runs"] += 1
            with self._stats_lock:
                self._stats["autonomous_actions"] += 1

    async def _run_evolution_sync(self):
        self._last_evolution_sync = time.monotonic()
        synced = []
        if self._shadow_os:
            try:
                sync_fn = getattr(self._shadow_os, 'sync_evolution_from_device', None)
                if sync_fn and callable(sync_fn):
                    synced = sync_fn()
            except (AttributeError, RuntimeError, ValueError):
                pass
        decision = AutonomousDecision(
            action_type=AutonomousActionType.EVOLUTION_SYNC,
            reason="Periodic sync of device context into evolution guide",
            confidence=0.8,
            risk_level=0.0,
        )
        decision.executed = True
        decision.outcome = f"synced: {', '.join(synced) if synced else 'none'}"
        self._record_decision(decision)
        with self._lock:
            with self._stats_lock:
                self._stats["evolution_syncs"] += 1
            with self._stats_lock:
                self._stats["autonomous_actions"] += 1

    async def _run_oracle_sync(self):
        self._last_oracle_sync = time.monotonic()
        synced = []
        if self._shadow_os:
            try:
                sync_fn = getattr(self._shadow_os, 'sync_oracle_from_cognition', None)
                if sync_fn and callable(sync_fn):
                    synced = sync_fn()
            except (AttributeError, RuntimeError, ValueError):
                pass
        decision = AutonomousDecision(
            action_type=AutonomousActionType.ORACLE_SYNC,
            reason="Periodic sync of temporal/anomaly data into oracle bridge",
            confidence=0.7,
            risk_level=0.0,
        )
        decision.executed = True
        decision.outcome = f"synced: {', '.join(synced) if synced else 'none'}"
        self._record_decision(decision)
        with self._lock:
            with self._stats_lock:
                self._stats["oracle_syncs"] += 1
            with self._stats_lock:
                self._stats["autonomous_actions"] += 1

    def _rate_limit(self, action: str) -> float:
        return self._action_rate_limits.get(action, 300.0)

    def _record_decision(self, decision: AutonomousDecision):
        with self._lock:
            self._decisions.append(decision)
            self._history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": decision.action_type.value,
                "confidence": decision.confidence,
                "risk_level": decision.risk_level,
            })

    def ingest_user_vitals(self, sample: RhythmSample):
        self.user_rhythm.ingest(sample)

    def get_status(self) -> Dict:
        with self._lock:
            uptime = time.monotonic() - self._start_time if self._start_time else 0
            with self._stats_lock:
                stats_copy = copy.deepcopy(self._stats)
            return {
                "running": self._running,
                "tick_interval_seconds": self._base_tick,
                "user_state": self.get_user_state().value,
                "user_rhythm_baselines": self.user_rhythm.get_baselines(),
                "cycles_completed": self._cycles_completed,
                "uptime_seconds": round(uptime, 1),
                "kill_switch": self.guardian.is_kill_switch_engaged(),
                "observer_stats": self.observer.get_stats(),
                "guardian_stats": self.guardian.get_stats(),
                "stats": stats_copy,
                "recent_decisions": [d.to_dict() for d in list(self._decisions)[-10:]],
            }

    def get_health(self) -> Dict:
        with self._lock:
            cycles = self._cycles_completed
        return {
            "status": "running" if self._running else "stopped",
            "kill_switch": self.guardian.is_kill_switch_engaged(),
            "threat_level": int(self.guardian.get_threat_level()),
            "user_state": self.get_user_state().value,
            "cycles_completed": cycles,
            "observer_status": "active" if self._running else "idle",
            "guardian_status": "active" if self._running else "idle",
        }

    def get_decisions(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return [d.to_dict() for d in list(self._decisions)[-limit:]]

    def get_security_events(self, limit: int = 50) -> List[Dict]:
        return self.guardian.get_recent_events(limit)

    def get_observations(self, limit: int = 50) -> List[Dict]:
        return self.observer.get_recent_observations(limit)

    def add_watch_path(self, path: str) -> bool:
        return self.observer.add_watch_path(path)

    def remove_watch_path(self, path: str) -> bool:
        return self.observer.remove_watch_path(path)

    def engage_kill_switch(self) -> Dict:
        return self.guardian.engage_kill_switch()

    def disengage_kill_switch(self) -> Dict:
        return self.guardian.disengage_kill_switch()

    def force_observe(self, obs_type: ObservationType = ObservationType.SYSTEM_METRICS) -> Dict:
        if obs_type == ObservationType.SYSTEM_METRICS:
            pulse = self.observer.observe_system_metrics()
        elif obs_type == ObservationType.PROCESS:
            pulse = self.observer.observe_processes()
        elif obs_type == ObservationType.NETWORK:
            pulse = self.observer.observe_network()
        elif obs_type == ObservationType.FILESYSTEM:
            pulse = self.observer.observe_filesystem()
        else:
            pulse = self.observer.observe_system_metrics()
        return pulse.to_dict()