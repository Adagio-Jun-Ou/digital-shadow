"""
Extra observers: process snapshot, browser history, calendar.

Each is opt-in (controlled by env var) and best-effort: missing tooling /
permissions / database files just turn the observer into a no-op.

Wired in alongside the core observers by `observers.start_all_observers`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("shadowd.observers_extra")

_SANITIZE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_TEXT = 500


def _sanitize(text: str, max_len: int = _MAX_TEXT) -> str:
    return _SANITIZE_RE.sub("", text)[:max_len]


def _should_stop() -> bool:
    from .observers import _CANCEL_EVENT
    return _CANCEL_EVENT is not None and _CANCEL_EVENT.is_set()


# ---------- Process activity ---------------------------------------------------

async def observe_processes(shadow, interval: float = 60.0):
    """Emit `process_change` observations for newly-started / -ended processes
    matching a small interest list (editors, browsers, IDEs)."""
    if os.environ.get("SHADOW_OBSERVE_PROCESSES", "true").lower() in ("0", "false", "no"):
        return
    try:
        import psutil  # noqa: F401
    except ImportError:
        logger.info("psutil not installed; process observer disabled")
        return

    interesting = re.compile(
        r"^(code|cursor|chrome|chromium|firefox|safari|vim|nvim|emacs|"
        r"sublime|atom|nano|tmux|python|node|cargo|rustc|go|claude)",
        re.IGNORECASE,
    )
    last_set: Set[str] = set()
    while not _should_stop():
        try:
            import psutil
            current = set()
            for p in psutil.process_iter(["name"]):
                try:
                    name = p.info.get("name") or ""
                    if interesting.match(name):
                        current.add(name.lower())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            added = current - last_set
            removed = last_set - current
            for name in added:
                await shadow.learn({
                    "type": "process_change",
                    "content": {"event": "started", "name": _sanitize(name, 100)},
                    "source": "process_observer",
                })
            for name in removed:
                await shadow.learn({
                    "type": "process_change",
                    "content": {"event": "ended", "name": _sanitize(name, 100)},
                    "source": "process_observer",
                })
            last_set = current
        except Exception as e:
            logger.debug("process observer error: %s", type(e).__name__)
        await asyncio.sleep(interval)


# ---------- Browser history ----------------------------------------------------

_CHROMIUM_HISTORY_LOCATIONS = [
    Path.home() / ".config/google-chrome/Default/History",
    Path.home() / ".config/chromium/Default/History",
    Path.home() / ".config/BraveSoftware/Brave-Browser/Default/History",
    Path.home() / "Library/Application Support/Google/Chrome/Default/History",
]

_FIREFOX_PROFILES = Path.home() / ".mozilla/firefox"


def _find_history_files() -> List[Path]:
    """Return paths to browser history sqlite files that exist + are readable."""
    found: List[Path] = []
    for p in _CHROMIUM_HISTORY_LOCATIONS:
        if p.exists() and os.access(p, os.R_OK):
            found.append(p)
    if _FIREFOX_PROFILES.exists():
        for profile in _FIREFOX_PROFILES.iterdir():
            cand = profile / "places.sqlite"
            if cand.exists() and os.access(cand, os.R_OK):
                found.append(cand)
    return found


async def observe_browser_history(shadow, interval: float = 300.0):
    """Poll browser history sqlite files and emit `web_visit` observations
    for new entries since the last check. Browsers hold a write lock on the
    live DB, so we copy it first."""
    if os.environ.get("SHADOW_OBSERVE_BROWSER", "true").lower() in ("0", "false", "no"):
        return

    history_files = _find_history_files()
    if not history_files:
        logger.info("no browser history files found; browser observer disabled")
        return

    last_seen: Dict[str, int] = {}  # path -> last microseconds seen
    while not _should_stop():
        for src in history_files:
            try:
                # Snapshot the DB to a temp file so the browser's lock
                # doesn't block our read.
                with tempfile.NamedTemporaryFile(
                    prefix="shadow_hist_", suffix=".sqlite", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                shutil.copy2(src, tmp_path)
                try:
                    conn = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True, timeout=2)
                    conn.row_factory = sqlite3.Row
                    is_firefox = src.name == "places.sqlite"
                    cursor = conn.cursor()
                    if is_firefox:
                        # Firefox uses microseconds since epoch.
                        since = last_seen.get(str(src), 0)
                        cursor.execute(
                            "SELECT url, title, visit_date FROM moz_places "
                            "JOIN moz_historyvisits ON moz_places.id = place_id "
                            "WHERE visit_date > ? ORDER BY visit_date DESC LIMIT 50",
                            (since,),
                        )
                    else:
                        # Chromium uses microseconds since 1601 (WebKit time);
                        # last_seen for chromium stays in WebKit epoch.
                        since = last_seen.get(str(src), 0)
                        cursor.execute(
                            "SELECT url, title, last_visit_time FROM urls "
                            "WHERE last_visit_time > ? "
                            "ORDER BY last_visit_time DESC LIMIT 50",
                            (since,),
                        )
                    rows = cursor.fetchall()
                    conn.close()
                except sqlite3.Error as e:
                    logger.debug("history sqlite error: %s", type(e).__name__)
                    rows = []
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                for row in rows[:25]:
                    url = _sanitize(row[0] or "", 500)
                    title = _sanitize(row[1] or "", 200)
                    ts = row[2] or 0
                    if ts > last_seen.get(str(src), 0):
                        last_seen[str(src)] = ts
                    if not url:
                        continue
                    await shadow.learn({
                        "type": "web_visit",
                        "content": {
                            "url": url,
                            "title": title,
                            "browser": "firefox" if is_firefox else "chromium",
                        },
                        "source": "browser_observer",
                    })
            except (OSError, shutil.SameFileError) as e:
                logger.debug("browser observer copy failed: %s", type(e).__name__)
        await asyncio.sleep(interval)


# ---------- Calendar (.ics files) ----------------------------------------------

_CALENDAR_LOCATIONS = [
    Path.home() / ".local/share/calendar",
    Path.home() / ".calendar",
    Path.home() / "Documents/calendar",
]


async def observe_calendar(shadow, interval: float = 600.0):
    """Scan .ics files in standard calendar locations and emit
    `calendar_event` observations for events happening in the next 24 hours."""
    if os.environ.get("SHADOW_OBSERVE_CALENDAR", "true").lower() in ("0", "false", "no"):
        return

    seen: Set[str] = set()
    while not _should_stop():
        ics_files: List[Path] = []
        for loc in _CALENDAR_LOCATIONS:
            if loc.exists():
                ics_files.extend(loc.rglob("*.ics"))
        if not ics_files:
            await asyncio.sleep(interval)
            continue

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=24)
        for path in ics_files[:50]:
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for event in _parse_ics_events(text):
                uid = event.get("uid") or f"{path}:{event.get('dtstart')}"
                if uid in seen:
                    continue
                try:
                    dt = datetime.fromisoformat(event["dtstart"].replace("Z", "+00:00"))
                except (ValueError, KeyError, AttributeError):
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if not (now <= dt <= cutoff):
                    continue
                seen.add(uid)
                if len(seen) > 1000:
                    # Periodically prune.
                    seen = set(list(seen)[-500:])
                await shadow.learn({
                    "type": "calendar_event",
                    "content": {
                        "title": _sanitize(event.get("summary", ""), 200),
                        "starts_at": event["dtstart"],
                        "location": _sanitize(event.get("location", ""), 200),
                    },
                    "source": "calendar_observer",
                })
        await asyncio.sleep(interval)


_DT_RE = re.compile(r"(\d{8})T(\d{6})Z?")


def _parse_ics_events(text: str) -> List[Dict[str, str]]:
    """Tiny ICS parser. Handles the subset of fields we care about
    (UID, SUMMARY, DTSTART, LOCATION). Not a full RFC 5545 implementation."""
    events: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "BEGIN:VEVENT":
            current = {}
            continue
        if line == "END:VEVENT":
            if current is not None:
                events.append(current)
            current = None
            continue
        if current is None:
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.split(";", 1)[0].upper()  # drop params like DTSTART;TZID=...
        if key == "UID":
            current["uid"] = value
        elif key == "SUMMARY":
            current["summary"] = value
        elif key == "LOCATION":
            current["location"] = value
        elif key == "DTSTART":
            m = _DT_RE.match(value)
            if m:
                d, t = m.group(1), m.group(2)
                current["dtstart"] = (
                    f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}+00:00"
                )
            else:
                current["dtstart"] = value
    return events
