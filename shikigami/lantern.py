"""
Lantern — Project state tracker.

Scans file_change observations to identify which projects you have on the
go, classify them as active / idle / dormant, and surface stale ones that
might be worth revisiting before you forget the context.

Key heuristics:
- Project root = the first directory under $HOME (or after /Users/.../) where
  files share a common ancestor. We don't try to follow git roots because
  observations don't carry that signal; the heuristic is good enough.
- Active:  touched within the last 7 days
- Idle:    touched 7-30 days ago
- Dormant: not touched for >30 days but had >=3 changes total
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple


_HOME = os.path.expanduser("~")
_SAFE_NAME_RE = re.compile(r'[^\w\-.]+')
# Tokens that are usually a *parent* of a project, not the project itself.
_PARENT_HINTS = {
    "src", "lib", "app", "apps", "packages", "projects", "code", "work",
    "workspace", "repos", "github", "gitlab", "dev", "tmp", "Documents",
    "Desktop", "Downloads",
}
# POSIX home prefixes. Observations may originate on a Linux/macOS device and
# be processed on Windows, so path heuristics cannot rely on the local $HOME.
_POSIX_HOME_RE = re.compile(r"^/(?:home|Users)/[^/]+/")


@dataclass
class ProjectStats:
    name: str
    root: str
    change_count: int
    file_count: int
    first_seen: str
    last_seen: str
    days_since: float
    status: str  # active / idle / dormant
    weekly_frequency: float  # changes per week (over project lifespan)
    top_files: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class LanternReport:
    total_changes: int
    projects: List[ProjectStats]
    active: List[str]
    idle: List[str]
    dormant: List[str]
    revisit_suggestions: List[str]


def _parse_ts(o: Dict) -> Optional[datetime]:
    ts = o.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _split_path_segments(path: str) -> List[str]:
    """Split a filesystem path into segments for either separator.

    Observations can be produced on a different OS than the one processing
    them (e.g. Linux paths synced from a laptop to a Windows host), so paths
    must be parsed with both '/' and '\\' regardless of the local platform.
    """
    normalized = path.replace("\\", "/")
    return [seg for seg in normalized.split("/") if seg and seg != "."]


def _path_basename(path: str) -> str:
    segments = _split_path_segments(path)
    return segments[-1] if segments else ""



def _extract_path(obs: Dict) -> Optional[str]:
    """Find a filesystem path inside an observation if there is one."""
    otype = obs.get("type", "")
    if otype not in ("file_change", "file_event", "file_save", "edit"):
        return None
    c = obs.get("content")
    if not isinstance(c, dict):
        return None
    for k in ("path", "file", "filename", "filepath"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _is_git_dir(path: str) -> bool:
    """True when *path* contains git metadata.

    Accepts a ``.git`` file (worktrees/submodules) or a ``.git`` directory
    containing ``HEAD``/``config``. Bare, empty ``.git`` folders are ignored,
    matching git's own ``is_git_directory`` validation — stray directories
    (e.g. ``~/.git`` left behind by unrelated tooling) are not repos.
    """
    git_path = os.path.join(path, ".git")
    if os.path.isfile(git_path):
        return True
    if not os.path.isdir(git_path):
        return False
    return (
        os.path.isfile(os.path.join(git_path, "HEAD"))
        or os.path.isfile(os.path.join(git_path, "config"))
    )


def _git_repo_root(path: str) -> Optional[str]:
    """Walk up the path looking for a .git/ directory. Return the basename
    of the repo root, or None if no .git ancestor exists."""
    try:
        if not path or not isinstance(path, str):
            return None
        # Paths may come from another OS, so normalise separators before
        # walking the real filesystem.
        local = os.path.normpath(path.replace("\\", os.sep).replace("/", os.sep))
        d = os.path.dirname(local)
        # Limit upward walk to 32 levels to avoid pathological cases.
        for _ in range(32):
            if not d or d == os.sep:
                return None
            if _is_git_dir(d):
                return os.path.basename(d)
            parent = os.path.dirname(d)
            if parent == d:
                return None
            d = parent
        return None
    except OSError:
        return None


def _project_root(path: str, use_git: bool = True) -> str:
    """Heuristically map a file path to its 'project' root.

    If `use_git=True` and a `.git/` directory exists anywhere up the path,
    that repo's basename wins. Otherwise falls back to a path-segment
    heuristic that skips known parent-folder names.

    Examples:
        /home/user/digital-shadow/shikigami/foo.py → digital-shadow
        /home/user/projects/blog/posts/x.md      → blog
        /home/user/Documents/notes/2026/a.md     → notes
        /tmp/scratch.py                          → scratch.py
    """
    if not path or not isinstance(path, str):
        return "unknown"
    # Strip trailing whitespace and quotes.
    p = path.strip().strip("'\"")
    # Prefer actual git repo when one exists — most accurate signal.
    if use_git:
        git_root = _git_repo_root(p)
        if git_root:
            return git_root
    # Anchor to $HOME if possible. Compare on normalised separators so paths
    # produced on another OS still resolve their project token.
    home = _HOME.replace("\\", "/").rstrip("/")
    normalized = p.replace("\\", "/")
    if home and normalized.lower().startswith(home.lower() + "/"):
        parts = _split_path_segments(normalized[len(home) + 1:])
    else:
        # POSIX-style home paths from a remote device (Linux laptop, phone).
        posix_match = _POSIX_HOME_RE.match(normalized)
        if posix_match:
            parts = _split_path_segments(normalized[posix_match.end():])
        else:
            parts = _split_path_segments(p)
    if not parts:
        return "unknown"
    # Skip over parent-style segments to find the real project token.
    for seg in parts:
        if seg in _PARENT_HINTS:
            continue
        if seg.startswith("."):
            continue
        return seg
    # All segments looked like parents — use the last one as best guess.
    return parts[-1]


def _clean_name(name: str) -> str:
    name = _SAFE_NAME_RE.sub("_", name)[:60]
    return name or "unknown"


def compute_lantern(
    observations: List[Dict],
    now: Optional[datetime] = None,
    active_days: int = 7,
    idle_days: int = 30,
    use_git: bool = True,
) -> LanternReport:
    now = now or datetime.now(timezone.utc)
    total_changes = 0

    # Per-project: list of (ts, path).
    bucket: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
    # Memoize project-root lookups — git detection touches the filesystem.
    root_cache: Dict[str, str] = {}

    def _root_for(path: str) -> str:
        # Use the parent path as the cache key — same dir, same .git/.
        segments = _split_path_segments(path)
        d = "/".join(segments[:-1]) or path
        cached = root_cache.get(d)
        if cached is not None:
            return cached
        r = _clean_name(_project_root(path, use_git=use_git))
        root_cache[d] = r
        return r

    for o in observations:
        path = _extract_path(o)
        if not path:
            continue
        t = _parse_ts(o)
        if t is None:
            continue
        total_changes += 1
        root = _root_for(path)
        bucket[root].append((t, path))

    projects: List[ProjectStats] = []
    for root, hits in bucket.items():
        hits.sort(key=lambda x: x[0])
        first = hits[0][0]
        last = hits[-1][0]
        days_since = round((now - last).total_seconds() / 86400.0, 2)
        lifespan_days = max(1.0, (last - first).total_seconds() / 86400.0)
        weekly = round(len(hits) / lifespan_days * 7.0, 2)
        if days_since <= active_days:
            status = "active"
        elif days_since <= idle_days:
            status = "idle"
        else:
            status = "dormant"
        # Top 3 most-touched files within this project.
        file_counter = Counter(path for _, path in hits)
        top_files = file_counter.most_common(3)
        projects.append(ProjectStats(
            name=root,
            root=root,
            change_count=len(hits),
            file_count=len({path for _, path in hits}),
            first_seen=first.isoformat(),
            last_seen=last.isoformat(),
            days_since=days_since,
            status=status,
            weekly_frequency=weekly,
            top_files=top_files,
        ))

    # Sort: active first (by recency), then idle (by recency), then dormant (by change count desc).
    projects.sort(key=lambda p: (
        0 if p.status == "active" else 1 if p.status == "idle" else 2,
        p.days_since if p.status != "dormant" else -p.change_count,
    ))

    active = [p.name for p in projects if p.status == "active"]
    idle = [p.name for p in projects if p.status == "idle"]
    dormant = [p.name for p in projects if p.status == "dormant"]

    # Suggestions: dormant projects with high prior activity (worth revisiting).
    revisit: List[str] = []
    dormant_ranked = sorted(
        (p for p in projects if p.status == "dormant" and p.change_count >= 3),
        key=lambda p: -p.change_count,
    )
    for p in dormant_ranked[:3]:
        revisit.append(
            f"{p.name} — {p.change_count} changes, last touched {int(p.days_since)}d ago"
        )

    return LanternReport(
        total_changes=total_changes,
        projects=projects,
        active=active,
        idle=idle,
        dormant=dormant,
        revisit_suggestions=revisit,
    )
