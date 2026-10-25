"""
Time Witness — categorize observation history into time buckets and report
where the hours actually went.

Activity categories (each a regex on observation type + text):
- coding      git, terminal commands, file changes
- communication  messages, email, calls
- browsing    web visits, clipboard URLs
- meetings    calendar events
- media       audio playing
- system      process changes, fs events not in projects
- idle        no activity periods

We sum per-category duration over the last N days (default 7) using a
simple "each observation represents some attention" heuristic with a
per-category time-weight, then report rank, fraction, and trend.
"""

from __future__ import annotations

import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple


# Each rule: (label, applies_to_type_regex_or_None, applies_to_text_regex_or_None, weight_minutes)
# Weight = minutes attributed to this observation. Conservative defaults.
_RULES: List[Tuple[str, Optional[re.Pattern], Optional[re.Pattern], float]] = [
    ("coding", re.compile(r"^(git_commit|terminal_command|file_change|process_change)$"), None, 1.5),
    ("coding", None, re.compile(r"\b(git|commit|push|merge|build|compile|pytest|docker)\b", re.I), 1.0),
    ("communication", re.compile(r"^(message|email|chat|call|communication)$"), None, 2.0),
    ("browsing", re.compile(r"^(web_visit|clipboard)$"), None, 1.0),
    ("meetings", re.compile(r"^(calendar_event|meeting)$"), None, 30.0),  # meetings are chunky
    ("media", None, re.compile(r"\b(audio playing|music|video|youtube|netflix)\b", re.I), 5.0),
    ("system", re.compile(r"^(window_focus|sensor_reading|location)$"), None, 0.5),
]


@dataclass
class TimeBucket:
    label: str
    minutes: float
    fraction: float          # of total tracked time
    events: int
    delta_vs_prev: float     # change vs the prior same-length window
    rank: int                # 1 = most time


@dataclass
class TimeReport:
    window_days: int
    total_minutes: float
    total_events: int
    buckets: List[TimeBucket]
    idle_minutes: float
    coverage: float          # fraction of window minutes accounted for
    top_label: Optional[str]


def _parse_ts(o: Dict) -> Optional[datetime]:
    ts = o.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _text_of(obs: Dict) -> str:
    c = obs.get("content")
    parts = [str(obs.get("type", ""))]
    if isinstance(c, dict):
        for v in c.values():
            if isinstance(v, str):
                parts.append(v)
    elif isinstance(c, str):
        parts.append(c)
    return " ".join(parts)


def _classify(obs: Dict) -> Tuple[str, float]:
    """Return (label, minutes) for the observation. 'other' / 0 if no rule fired."""
    otype = str(obs.get("type", ""))
    text = _text_of(obs)
    for label, type_re, text_re, weight in _RULES:
        if type_re and not type_re.match(otype):
            continue
        if text_re and not text_re.search(text):
            continue
        return label, weight
    return "other", 0.5  # tiny bit of time for every obs


def compute_time_report(
    observations: List[Dict],
    window_days: int = 7,
    now: Optional[datetime] = None,
) -> TimeReport:
    now = now or datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)
    prev_start = now - timedelta(days=window_days * 2)

    cur_buckets: Dict[str, List[float]] = defaultdict(list)  # label → list of minutes
    prev_buckets: Dict[str, float] = defaultdict(float)
    cur_events_in_window = 0

    for o in observations:
        t = _parse_ts(o)
        if t is None:
            continue
        label, mins = _classify(o)
        if t >= window_start:
            cur_buckets[label].append(mins)
            cur_events_in_window += 1
        elif t >= prev_start:
            prev_buckets[label] += mins

    total_minutes = sum(sum(v) for v in cur_buckets.values())
    window_minutes = window_days * 24 * 60.0
    idle = max(0.0, window_minutes - total_minutes)
    coverage = min(1.0, total_minutes / window_minutes if window_minutes else 0.0)

    bucket_list: List[TimeBucket] = []
    ranked = sorted(cur_buckets.items(), key=lambda kv: -sum(kv[1]))
    for rank, (label, mins_list) in enumerate(ranked, start=1):
        mins = sum(mins_list)
        bucket_list.append(TimeBucket(
            label=label,
            minutes=round(mins, 1),
            fraction=round(mins / total_minutes, 3) if total_minutes else 0.0,
            events=len(mins_list),
            delta_vs_prev=round(mins - prev_buckets.get(label, 0.0), 1),
            rank=rank,
        ))

    top = bucket_list[0].label if bucket_list else None
    return TimeReport(
        window_days=window_days,
        total_minutes=round(total_minutes, 1),
        total_events=cur_events_in_window,
        buckets=bucket_list,
        idle_minutes=round(idle, 1),
        coverage=round(coverage, 3),
        top_label=top,
    )


def format_bar(fraction: float, width: int = 20) -> str:
    """Tiny ASCII bar for fractions in [0, 1]."""
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)
