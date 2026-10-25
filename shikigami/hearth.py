"""
Hearth — Location patterns and transitions.

Uses location-bearing observations (WiFi SSID, GPS, calendar location, etc.)
to compute:
- Time spent per labeled location (home/work/cafe/travel/unknown)
- Transition graph (where do you go after location X)
- Anomalous days (unusual location for time-of-day)
- Most-frequent location for each hour-of-day
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple


# Substrings that classify a network/location name into a label.
# Order matters — first match wins. "School" comes before "work" so that
# "MIT Campus" lands in school, not work (both used to claim "campus").
_LOCATION_PATTERNS: Dict[str, List[str]] = {
    "school": ["school", "univ", "college", "campus", "academy"],
    "home": ["home", "house", "myhome", "residence", "apt", "apartment"],
    "work": ["office", "work", "corp", "company", "hq", "biz"],
    "cafe": ["cafe", "coffee", "starbucks", "espresso", "dunkin", "peet"],
    "public": ["public", "free", "guest", "library", "mall", "airport", "hotel"],
    "gym": ["gym", "fitness", "yoga", "crossfit"],
}


@dataclass
class LocationVisit:
    label: str
    raw: str
    start: str
    duration_min: float


@dataclass
class HearthReport:
    total_observations: int
    labeled_observations: int
    time_per_label_min: Dict[str, float]
    hour_label_mode: Dict[int, str]  # hour → most frequent label
    top_transitions: List[Tuple[str, str, int]]  # (from, to, count)
    recent_visits: List[LocationVisit]
    anomalies: List[str]


def _parse_ts(o: Dict) -> Optional[datetime]:
    ts = o.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _extract_location_hint(obs: Dict) -> Optional[str]:
    """Return a free-text location hint from an observation, or None."""
    otype = obs.get("type", "")
    c = obs.get("content")
    if not isinstance(c, dict):
        return None
    # Direct location fields.
    for k in ("location", "place", "where"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # WiFi observations.
    if otype in ("wifi", "wifi_scan", "wifi_change"):
        for k in ("ssid", "network", "name"):
            v = c.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        nets = c.get("networks")
        if isinstance(nets, list) and nets:
            first = nets[0]
            if isinstance(first, dict):
                v = first.get("ssid") or first.get("name")
                if isinstance(v, str):
                    return v
            elif isinstance(first, str):
                return first
    # GPS observations — fall back to city.
    if otype in ("gps", "location"):
        for k in ("city", "label"):
            v = c.get(k)
            if isinstance(v, str):
                return v
    # Calendar events frequently include a location.
    if otype == "calendar_event":
        v = c.get("location")
        if isinstance(v, str) and v:
            return v
    return None


def classify(raw: str) -> str:
    if not raw:
        return "unknown"
    r = raw.lower()
    for label, kws in _LOCATION_PATTERNS.items():
        for kw in kws:
            if kw in r:
                return label
    return "other"


def compute_hearth(
    observations: List[Dict],
    now: Optional[datetime] = None,
    visit_gap_min: int = 30,
) -> HearthReport:
    """Build a HearthReport from the observation history."""
    now = now or datetime.now(timezone.utc)
    raw_points: List[Tuple[datetime, str, str]] = []  # (ts, label, raw)
    n_obs = 0
    for o in observations:
        n_obs += 1
        t = _parse_ts(o)
        if t is None:
            continue
        hint = _extract_location_hint(o)
        if hint is None:
            continue
        raw_points.append((t, classify(hint), hint))
    raw_points.sort(key=lambda p: p[0])

    # Group into visits — contiguous runs of the same label with no gap > visit_gap_min.
    visits: List[LocationVisit] = []
    if raw_points:
        cur_label, cur_raw, cur_start, cur_end = raw_points[0][1], raw_points[0][2], raw_points[0][0], raw_points[0][0]
        for t, label, raw in raw_points[1:]:
            gap = (t - cur_end).total_seconds() / 60.0
            if label == cur_label and gap <= visit_gap_min * 4:
                cur_end = t
            else:
                visits.append(LocationVisit(
                    label=cur_label, raw=cur_raw,
                    start=cur_start.isoformat(),
                    duration_min=round((cur_end - cur_start).total_seconds() / 60.0, 1),
                ))
                cur_label, cur_raw, cur_start, cur_end = label, raw, t, t
        visits.append(LocationVisit(
            label=cur_label, raw=cur_raw,
            start=cur_start.isoformat(),
            duration_min=round((cur_end - cur_start).total_seconds() / 60.0, 1),
        ))

    # Time per label.
    time_per_label: Dict[str, float] = defaultdict(float)
    for v in visits:
        time_per_label[v.label] += max(1.0, v.duration_min)
    time_per_label = {k: round(v, 1) for k, v in time_per_label.items()}

    # Hour-of-day → most common label.
    hour_labels: Dict[int, Counter] = defaultdict(Counter)
    for t, label, _ in raw_points:
        hour_labels[t.hour][label] += 1
    hour_mode = {h: counter.most_common(1)[0][0] for h, counter in hour_labels.items()}

    # Transition counts.
    transitions: Counter = Counter()
    for prev, curr in zip(visits, visits[1:]):
        if prev.label != curr.label:
            transitions[(prev.label, curr.label)] += 1
    top_trans = [(a, b, n) for (a, b), n in transitions.most_common(5)]

    # Anomalies: current hour has a mode, but the latest visit's label differs.
    anomalies: List[str] = []
    if visits and hour_mode:
        last = visits[-1]
        h = now.hour
        expected = hour_mode.get(h)
        if expected and expected != last.label and last.label != "unknown":
            anomalies.append(
                f"At hour {h:02d} you're usually at '{expected}' but recently at '{last.label}'"
            )

    return HearthReport(
        total_observations=n_obs,
        labeled_observations=len(raw_points),
        time_per_label_min=time_per_label,
        hour_label_mode=hour_mode,
        top_transitions=top_trans,
        recent_visits=visits[-10:],
        anomalies=anomalies,
    )
