#!/usr/bin/env python3
"""
Security Utilities for Shadow AI

Provides:
- AuditLogger: Centralized security event logging
- RateLimiter: Per-key rate limiting with configurable windows
- InputValidator: Reusable validation helpers
- sanitize_path: Path traversal protection
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("shadowd.security_utils")


@dataclass
class AuditEvent:
    timestamp: datetime
    event_type: str
    severity: str
    source: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity,
            "source": self.source,
            "message": self.message,
            "metadata": copy.deepcopy(self.metadata),
        }


class AuditLogger:
    """
    Centralized security audit logger.
    
    Records security events with severity levels and persistence.
    """

    MAX_EVENTS = 10000
    _VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low", "info"})
    _MAX_METADATA_KEYS = 50

    def __init__(self, log_dir: Optional[Path] = None):
        self.events: List[AuditEvent] = []
        self._lock = Lock()
        self.log_dir = log_dir
        self._counters: Dict[str, int] = defaultdict(int)

    def log(
        self,
        event_type: str,
        severity: str,
        source: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ):
        """Log a security event."""
        if severity not in self._VALID_SEVERITIES:
            severity = "info"
        if metadata and len(metadata) > self._MAX_METADATA_KEYS:
            metadata = dict(list(metadata.items())[:self._MAX_METADATA_KEYS])
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type[:100],
            severity=severity,
            source=source[:100],
            message=message[:500],
            metadata=metadata or {},
        )
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.MAX_EVENTS:
                self.events = self.events[-self.MAX_EVENTS:]
            self._counters[event_type] += 1
        if severity in ("critical", "high"):
            logger.warning(f"[AUDIT] {severity.upper()} {source}: {message}")
        else:
            logger.info(f"[AUDIT] {severity} {source}: {message}")

    def get_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get filtered audit events."""
        with self._lock:
            filtered = self.events
            if event_type:
                filtered = [e for e in filtered if e.event_type == event_type]
            if severity:
                filtered = [e for e in filtered if e.severity == severity]
            return [e.to_dict() for e in filtered[-limit:]]

    def get_stats(self) -> Dict:
        """Get audit statistics."""
        with self._lock:
            return {
                "total_events": len(self.events),
                "by_type": dict(self._counters),
                "by_severity": {
                    s: sum(1 for e in self.events if e.severity == s)
                    for s in {"critical", "high", "medium", "low", "info"}
                },
            }


class RateLimiter:
    """
    Thread-safe rate limiter with configurable windows.
    
    Uses sliding window algorithm with per-key tracking.
    """

    DEFAULT_MAX_KEYS = 10000

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
        max_keys: int = DEFAULT_MAX_KEYS,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str) -> bool:
        """Check if request is allowed. Returns True if allowed."""
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            self._requests[key] = [
                t for t in self._requests[key] if t > cutoff
            ]
            # Clean up keys with empty timestamp lists
            empty_keys = [k for k, v in self._requests.items() if not v]
            for k in empty_keys:
                del self._requests[k]

            if len(self._requests[key]) >= self.max_requests:
                return False
            self._requests[key].append(now)

            if len(self._requests) > self.max_keys:
                oldest_key = min(
                    self._requests, key=lambda k: self._requests[k][0] if self._requests[k] else float('inf')
                )
                del self._requests[oldest_key]

        return True

    def reset(self, key: Optional[str] = None):
        """Reset rate limit for a key or all keys."""
        with self._lock:
            if key:
                self._requests.pop(key, None)
            else:
                self._requests.clear()


class InputValidator:
    """Reusable input validation helpers."""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000, strict: bool = False) -> str:
        """Sanitize a string input.
        
        Args:
            value: String to sanitize
            max_length: Maximum length (default 1000)
            strict: If True, also strip <> characters (for identifiers)
        """
        if not isinstance(value, str):
            raise ValueError("Expected string")
        value = value.strip()
        if strict:
            value = re.sub(r'[<>\x00-\x1f\x7f]', '', value)
        else:
            value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
        if len(value) > max_length:
            value = value[:max_length]
        return value

    @staticmethod
    def validate_id(value: str, max_length: int = 64) -> str:
        """Validate an ID field (alphanumeric + hyphens/underscores only)."""
        if not isinstance(value, str):
            raise ValueError("ID must be a string")
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError("ID contains invalid characters")
        if len(value) > max_length:
            raise ValueError(f"ID too long (max {max_length})")
        return value

    @staticmethod
    def validate_path_sandbox(path: str, allowed_bases: List[Path]) -> Path:
        """Validate path is within allowed bases."""
        resolved = Path(path).expanduser().resolve()
        for base in allowed_bases:
            try:
                resolved.relative_to(base.resolve())
                return resolved
            except ValueError:
                continue
        raise ValueError("Path is outside allowed directories")

    @staticmethod
    def validate_dict_keys(data: Dict, max_keys: int = 50, max_key_length: int = 64) -> Dict:
        """Validate dictionary keys are safe."""
        if len(data) > max_keys:
            raise ValueError(f"Too many keys (max {max_keys})")
        for key in data:
            if not isinstance(key, str):
                raise ValueError("Dictionary keys must be strings")
            if len(key) > max_key_length:
                raise ValueError(f"Key exceeds maximum length ({max_key_length})")
            if re.search(r'[^\w.-]', key):
                raise ValueError("Key contains invalid characters")
        return data


_global_audit = AuditLogger()
_global_rate_limiter: Dict[str, RateLimiter] = {}
_rate_limiter_lock = Lock()


def get_audit_logger() -> AuditLogger:
    return _global_audit


def get_rate_limiter(name: str = "default", **kwargs) -> RateLimiter:
    with _rate_limiter_lock:
        if name not in _global_rate_limiter:
            _global_rate_limiter[name] = RateLimiter(**kwargs)
        return _global_rate_limiter[name]


__all__ = [
    "AuditEvent",
    "AuditLogger",
    "RateLimiter",
    "InputValidator",
    "get_audit_logger",
    "get_rate_limiter",
]