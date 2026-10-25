#!/usr/bin/env python3
"""
Digital Shadow Observers - Watch and learn from all activities.
"""

import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger("shadowd.observers")

_CANCEL_EVENT: Optional[asyncio.Event] = None

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_BASH_LINES = 500
_MAX_FS_DEPTH = 5
_MAX_FS_FILES = 1000
_MAX_FILE_CACHE = 500
_MAX_GIT_REPOS = 50
_MAX_COMMIT_MSG = 200
_MAX_WINDOW_LEN = 200

_SENSITIVE_PATTERNS = re.compile(
    r'(?i)(password|passwd|secret|token|api[_-]?key|bearer|'
    r'private[_-]?key|\bssh\b|ssh-keygen|ssh-add|gpg|openssl|pass |vault|'
    r'aws[_-]?secret|aws[_-]?access|credential|credit|card[_-]?num|ssn|\.pem|\.key)',
    re.IGNORECASE
)


def _sanitize(text: str, max_len: int = 500) -> str:
    """Strip control chars (preserve tab/newline/CR) for content fields."""
    return _SANITIZE_RE.sub('', text)[:max_len]


def _sanitize_strict(text: str, max_len: int = 200) -> str:
    """Strip control chars AND angle brackets for identifier fields."""
    return _SANITIZE_STRICT_RE.sub('', text)[:max_len]


def _is_sensitive_command(command: str) -> bool:
    """Check if a command contains sensitive patterns."""
    return bool(_SENSITIVE_PATTERNS.search(command))


def _should_stop() -> bool:
    return _CANCEL_EVENT is not None and _CANCEL_EVENT.is_set()


async def _run_cmd(args: list, timeout: int = 2) -> Optional[str]:
    """Run a subprocess command without blocking the event loop."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode == 0:
            return stdout.decode().strip()
    except (asyncio.TimeoutError, FileNotFoundError, OSError):
        pass
    return None


async def observe_terminal(shadow):
    """Observe terminal/bash history"""
    bash_history = Path.home() / ".bash_history"
    last_size = 0

    if bash_history.exists():
        last_size = bash_history.stat().st_size

    while not _should_stop():
        try:
            if bash_history.exists():
                current_size = bash_history.stat().st_size
                if current_size < last_size:
                    last_size = 0
                if current_size > last_size:
                    with open(bash_history, "r") as f:
                        f.seek(last_size)
                        new_lines = f.readlines(_MAX_BASH_LINES)
                        for line in new_lines:
                            stripped = _sanitize_strict(line.strip(), 500)
                            if stripped and not _is_sensitive_command(stripped):
                                await shadow.learn({
                                    "type": "terminal_command",
                                    "content": {"command": stripped},
                                    "source": "bash_history"
                                })
                    last_size = current_size
        except OSError:
            logger.debug("Terminal observer error: OSError")
        await asyncio.sleep(2)


async def observe_filesystem(shadow):
    """Observe file system changes using watchfiles or inotify"""
    try:
        from watchfiles import awatch
    except ImportError:
        logger.warning("watchfiles not installed, using polling for FS observation")
        await _observe_filesystem_poll(shadow)
        return

    home = Path.home()
    watch_dirs = [home / "digital-shadow", home / "projects", home / "Documents"]

    watch_dirs = [d for d in watch_dirs if d.exists()]
    if not watch_dirs:
        return

    # Skip paths that would create a self-observation feedback loop or that
    # are pure noise (caches, logs, tempfiles).
    _IGNORE_SUBSTRINGS = (
        ".git", "__pycache__", "node_modules", ".pyc", ".pyo",
        "/data/logs/", "/data/cache/", "/data/checkpoints/", "/data/",
        "/.pytest_cache/", "/.hypothesis/", "/venv/",
        ".tmp", ".swp", ".lock", ".json.tmp",
    )

    _last_event_time = 0.0
    _event_cooldown = 2.0  # throttle to 1 event per 2 seconds

    async for changes in awatch(*watch_dirs, stop_event=_CANCEL_EVENT):
        for change_type, path in changes:
            spath = str(path)
            if any(s in spath for s in _IGNORE_SUBSTRINGS):
                continue
            now = time.monotonic()
            if now - _last_event_time < _event_cooldown:
                continue
            _last_event_time = now
            await shadow.learn({
                "type": "file_change",
                "content": {
                    "change_type": _sanitize_strict(str(change_type)),
                    "path": _sanitize_strict(spath)
                },
                "source": "filesystem_watcher"
            })


async def _observe_filesystem_poll(shadow):
    """Fallback polling-based file observation"""
    home = Path.home()
    watch_dirs = [home / "digital-shadow", home / "projects", home / "Documents"]
    watch_dirs = [d for d in watch_dirs if d.exists()]
    if not watch_dirs:
        logger.warning("No directories to watch for filesystem changes")
        return

    def _rglob_limited(root: Path, max_depth: int = _MAX_FS_DEPTH, max_files: int = _MAX_FS_FILES):
        results = []
        for path in root.rglob("*"):
            if len(results) >= max_files:
                break
            try:
                if path.is_file() and not any(s in str(path) for s in (
                    ".git", "__pycache__", "node_modules", ".pyc", ".pyo",
                    "/data/logs/", "/data/cache/", "/data/checkpoints/",
                    "/.pytest_cache/", "/.hypothesis/", "/venv/",
                    ".tmp", ".swp", ".lock",
                )):
                    if len(str(path.relative_to(root)).split(os.sep)) - 1 <= max_depth:
                        results.append(path)
            except (OSError, ValueError):
                pass
        return results

    file_cache: dict[str, float] = {}
    for wdir in watch_dirs:
        for fpath in _rglob_limited(wdir):
            try:
                file_cache[str(fpath)] = fpath.stat().st_mtime
            except OSError:
                pass

    while not _should_stop():
        try:
            for wdir in watch_dirs:
                if _should_stop():
                    break
                for fpath in _rglob_limited(wdir):
                    if _should_stop():
                        break
                    key = _sanitize_strict(str(fpath))
                    try:
                        mtime = fpath.stat().st_mtime
                    except OSError:
                        continue
                    if key in file_cache:
                        if mtime != file_cache[key]:
                            file_cache[key] = mtime
                            await shadow.learn({
                                "type": "file_change",
                                "content": {"change_type": "modified", "path": key},
                                "source": "filesystem_poll"
                            })
                    else:
                        file_cache[key] = mtime
                        await shadow.learn({
                            "type": "file_change",
                            "content": {"change_type": "created", "path": key},
                            "source": "filesystem_poll"
                        })
            if len(file_cache) > _MAX_FILE_CACHE:
                oldest = sorted(file_cache.items(), key=lambda x: x[1])[:len(file_cache) - _MAX_FILE_CACHE]
                for k, _ in oldest:
                    del file_cache[k]
        except (OSError, ValueError, KeyError) as e:
            logger.debug("Filesystem poll error: %s", type(e).__name__)
        await asyncio.sleep(15)


async def observe_clipboard(shadow):
    """Observe clipboard content (text only, for privacy)"""
    last_content = ""

    while not _should_stop():
        output = await _run_cmd(["xclip", "-selection", "clipboard", "-o"], timeout=1)
        if output and output != last_content and len(output) < 1000:
            if not _is_sensitive_command(output):
                await shadow.learn({
                    "type": "clipboard",
                    "content": {"preview": _sanitize(output[:200])},
                    "source": "clipboard_watcher"
                })
                last_content = output
        await asyncio.sleep(3)


async def observe_active_window(shadow):
    """Observe currently active window (gracefully skips if xdotool unavailable)"""
    last_window = ""
    xdotool_available = None

    while not _should_stop():
        if xdotool_available is None:
            output = await _run_cmd(["which", "xdotool"], timeout=2)
            xdotool_available = output is not None
            if not xdotool_available:
                logger.info("xdotool not found; window observer disabled")
                return

        output = await _run_cmd(["xdotool", "getwindowfocus", "getwindowname"], timeout=1)
        if output and output != last_window:
            await shadow.learn({
                "type": "window_focus",
                "content": {"window": _sanitize_strict(output, _MAX_WINDOW_LEN)},
                "source": "window_watcher"
            })
            last_window = output
        await asyncio.sleep(5)


async def observe_git_activity(shadow):
    """Observe git commits and changes"""
    last_check: dict = {}
    WATCH_DIRS = [
        Path.home() / "digital-shadow",
        Path.home() / "projects",
    ]

    while not _should_stop():
        try:
            git_repos = []
            for watch_dir in WATCH_DIRS:
                if watch_dir.exists():
                    git_repos.extend(list(watch_dir.glob("*/*/.git"))[:5])
            for repo in git_repos[:10]:
                repo_path = repo.parent
                if "node_modules" in str(repo_path) or ".cache" in str(repo_path):
                    continue

                output = await _run_cmd(
                    ["git", "-C", str(repo_path), "log", "-1", "--format=%H|%s"],
                    timeout=2,
                )
                if output:
                    last_commit = last_check.get(str(repo_path))
                    if output != last_commit:
                        parts = output.split("|", 1)
                        commit_hash = parts[0][:8] if parts else "unknown"
                        commit_msg = _sanitize(parts[1], _MAX_COMMIT_MSG) if len(parts) > 1 else ""
                        await shadow.learn({
                            "type": "git_commit",
                            "content": {
                                "repo": _sanitize_strict(str(repo_path)),
                                "commit": commit_hash,
                                "message": commit_msg
                            },
                            "source": "git_watcher"
                        })
                    last_check[str(repo_path)] = output
            if len(last_check) > _MAX_GIT_REPOS:
                items = list(last_check.items())
                last_check = dict(items[-_MAX_GIT_REPOS:])
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug("Git observer error: %s", type(e).__name__)
        await asyncio.sleep(30)


# Lightweight registry so /observers/status can report which observer tasks
# are alive and which silently died (e.g. xdotool missing).
_OBSERVER_TASKS: Dict[str, asyncio.Task] = {}


def observer_status() -> Dict[str, Dict]:
    """Return per-observer task state for the /observers/status endpoint."""
    out: Dict[str, Dict] = {}
    for name, task in _OBSERVER_TASKS.items():
        if task.done():
            exc = task.exception() if not task.cancelled() else None
            out[name] = {
                "alive": False,
                "cancelled": task.cancelled(),
                "error": f"{type(exc).__name__}: {exc}" if exc else None,
            }
        else:
            out[name] = {"alive": True, "cancelled": False, "error": None}
    return out


async def start_all_observers(shadow):
    """Start all background observers with cancellation support."""
    global _CANCEL_EVENT, _OBSERVER_TASKS
    _CANCEL_EVENT = asyncio.Event()

    from .observers_extra import (
        observe_processes, observe_browser_history, observe_calendar,
    )

    logger.info("Starting observers: terminal, filesystem, clipboard, window, git, "
                "processes, browser, calendar")

    _OBSERVER_TASKS = {
        "terminal": asyncio.create_task(observe_terminal(shadow), name="obs-terminal"),
        "filesystem": asyncio.create_task(observe_filesystem(shadow), name="obs-filesystem"),
        "clipboard": asyncio.create_task(observe_clipboard(shadow), name="obs-clipboard"),
        "window": asyncio.create_task(observe_active_window(shadow), name="obs-window"),
        "git": asyncio.create_task(observe_git_activity(shadow), name="obs-git"),
        "processes": asyncio.create_task(observe_processes(shadow), name="obs-processes"),
        "browser": asyncio.create_task(observe_browser_history(shadow), name="obs-browser"),
        "calendar": asyncio.create_task(observe_calendar(shadow), name="obs-calendar"),
    }
    tasks = list(_OBSERVER_TASKS.values())

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        _CANCEL_EVENT.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)