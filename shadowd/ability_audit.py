"""
Ability invocation audit log.

JSONL audit file recording every shikigami invocation: timestamp,
ability, query hash, success, latency, confidence. Rotates daily so a
single file never grows unbounded.

Disabled by default. Enable by setting SHADOW_ABILITY_AUDIT=1 (or by
passing enabled=True when constructing AuditLog directly in tests).

Designed to never raise: an audit failure must not break the user's
request. All errors are logged and silently swallowed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger("shadowd.audit")

DEFAULT_AUDIT_DIR = Path.home() / "digital-shadow" / "data" / "audit"


def _hash_query(query: str, max_chars: int = 4096) -> str:
    if not isinstance(query, str):
        query = str(query)
    digest = hashlib.sha256(query[:max_chars].encode("utf-8")).hexdigest()
    return digest[:16]  # short enough to fit in a log line


def _redact_obs(obs: List[Dict]) -> int:
    """Return only the count — never persist observation contents."""
    return len(obs) if isinstance(obs, list) else 0


class AuditLog:
    """Append-only JSONL audit. Rotates by UTC date.

    Files older than `retention_days` are automatically removed at most once
    per UTC day. Audit failures (including cleanup) never propagate.
    """

    def __init__(
        self,
        audit_dir: Path = DEFAULT_AUDIT_DIR,
        enabled: bool = True,
        retention_days: int = 30,
    ):
        self.audit_dir = Path(audit_dir)
        self.enabled = enabled
        self.retention_days = max(1, int(retention_days))
        self._lock = threading.Lock()
        self._last_cleanup_date: Optional[str] = None
        if self.enabled:
            try:
                self.audit_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error("Could not create audit dir: %s", type(e).__name__)
                self.enabled = False

    def _path_for(self, ts: datetime) -> Path:
        return self.audit_dir / f"abilities-{ts.strftime('%Y-%m-%d')}.jsonl"

    def record(
        self,
        ability: str,
        query: str,
        success: bool,
        latency_s: float,
        confidence: float = 0.0,
        observation_count: int = 0,
        error_type: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        now = datetime.now(timezone.utc)
        entry = {
            "ts": now.isoformat(),
            "ability": str(ability)[:64],
            "query_hash": _hash_query(query),
            "query_len": min(len(query) if isinstance(query, str) else 0, 100000),
            "success": bool(success),
            "latency_s": round(float(latency_s), 4),
            "confidence": round(float(confidence), 4),
            "obs_count": int(observation_count),
        }
        if error_type:
            entry["error_type"] = str(error_type)[:64]
        path = self._path_for(now)
        line = json.dumps(entry, separators=(",", ":"))
        try:
            with self._lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            # Cleanup throttled to once per UTC date so the write hot path
            # isn't slowed by a directory listing on every invocation.
            self._maybe_cleanup(now)
        except OSError as e:
            # Never propagate — audit must not break the request path.
            logger.error("Audit write failed: %s", type(e).__name__)

    def _maybe_cleanup(self, now: datetime) -> None:
        today = now.strftime("%Y-%m-%d")
        if self._last_cleanup_date == today:
            return
        self._last_cleanup_date = today
        try:
            self.cleanup_old(now=now)
        except Exception as e:  # noqa: BLE001 - cleanup must never raise
            logger.error("Audit cleanup failed: %s", type(e).__name__)

    def cleanup_old(self, now: Optional[datetime] = None) -> int:
        """Delete audit files older than self.retention_days. Returns count
        of files removed."""
        if not self.enabled:
            return 0
        now = now or datetime.now(timezone.utc)
        removed = 0
        try:
            with self._lock:
                for path in self.audit_dir.glob("abilities-*.jsonl"):
                    # Filename format: abilities-YYYY-MM-DD.jsonl
                    name = path.stem  # abilities-YYYY-MM-DD
                    parts = name.split("-")
                    if len(parts) != 4:  # ["abilities", "YYYY", "MM", "DD"]
                        continue
                    try:
                        file_date = datetime(
                            int(parts[1]), int(parts[2]), int(parts[3]),
                            tzinfo=timezone.utc,
                        )
                    except (ValueError, TypeError):
                        continue
                    age_days = (now - file_date).days
                    if age_days > self.retention_days:
                        try:
                            path.unlink()
                            removed += 1
                        except OSError as e:
                            logger.error("Audit unlink failed: %s", type(e).__name__)
        except OSError as e:
            logger.error("Audit cleanup scan failed: %s", type(e).__name__)
        return removed

    def read_recent(self, lines: int = 100) -> List[Dict[str, Any]]:
        """Read the most recent N entries from today's file."""
        if not self.enabled:
            return []
        path = self._path_for(datetime.now(timezone.utc))
        if not path.exists():
            return []
        try:
            with self._lock:
                text = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Audit read failed: %s", type(e).__name__)
            return []
        out: List[Dict[str, Any]] = []
        for raw in text.splitlines()[-lines:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out

    def list_log_files(self) -> List[str]:
        if not self.enabled:
            return []
        try:
            return sorted(
                str(p) for p in self.audit_dir.glob("abilities-*.jsonl")
            )
        except OSError as e:
            logger.error("Audit list failed: %s", type(e).__name__)
            return []


# Module-level singleton, configured from env.
_AUDIT_INSTANCE: Optional[AuditLog] = None
_AUDIT_LOCK = threading.Lock()


def _is_enabled_from_env() -> bool:
    return os.environ.get("SHADOW_ABILITY_AUDIT", "").lower() in ("1", "true", "yes", "on")


def get_audit() -> AuditLog:
    """Return the module-level AuditLog singleton."""
    global _AUDIT_INSTANCE
    with _AUDIT_LOCK:
        if _AUDIT_INSTANCE is None:
            _AUDIT_INSTANCE = AuditLog(enabled=_is_enabled_from_env())
        return _AUDIT_INSTANCE


def reset_for_tests(
    audit_dir: Optional[Path] = None,
    enabled: bool = True,
    retention_days: int = 30,
) -> AuditLog:
    """Replace the singleton — only intended for tests."""
    global _AUDIT_INSTANCE
    with _AUDIT_LOCK:
        target = audit_dir or DEFAULT_AUDIT_DIR
        _AUDIT_INSTANCE = AuditLog(
            audit_dir=target, enabled=enabled, retention_days=retention_days,
        )
        return _AUDIT_INSTANCE
