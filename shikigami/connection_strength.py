"""
Connection strength — per-person relationship scoring from interaction
frequency, recency, sentiment, and reciprocity.

Inputs: tracked `people` dict (name → metadata) + observations.
Output: per-person score in [0, 1] + trend label.

Score components (weighted):
- recency:   0.35  — how recently you've interacted
- frequency: 0.30  — total interactions in last 30/90 days
- sentiment: 0.20  — average tone of recent messages
- reciprocity: 0.15 — incoming vs outgoing balance (when known)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple


@dataclass
class Connection:
    name: str
    relation: str
    score: float                  # [0, 1] aggregate
    components: Dict[str, float]  # individual subscores
    trend: str                    # "rising" | "stable" | "declining" | "dormant"
    interactions_30d: int
    interactions_90d: int
    last_contact: Optional[str]
    days_since_contact: Optional[int]


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
    c = obs.get("content", {})
    if isinstance(c, dict):
        for k in ("text", "body", "message", "preview", "content"):
            v = c.get(k)
            if isinstance(v, str):
                return v
    elif isinstance(c, str):
        return c
    return ""


def _person_in_text(name: str, text: str) -> bool:
    if not name or not text:
        return False
    return bool(re.search(r'\b' + re.escape(name.lower()) + r'\b', text.lower()))


def _outgoing_marker(text: str) -> Optional[bool]:
    """Heuristic: True for outgoing (user → person), False for incoming.
    Returns None if unsure."""
    t = text.lower()
    if re.search(r"\b(?:said|replied|wrote|messaged|asked) (?:to me|me)\b", t):
        return False
    if re.search(r"\b(?:i (?:said|told|asked|sent|wrote|messaged))\b", t):
        return True
    return None


_INTERACTION_TYPES = frozenset({
    "clipboard", "message", "email", "communication", "chat",
    "call", "meeting", "note", "device_nearby",
})


def score_connection(
    name: str,
    relation: str,
    observations: List[Dict],
    now: Optional[datetime] = None,
) -> Connection:
    now = now or datetime.now(timezone.utc)
    name_l = name.lower()
    cutoff_30 = now - timedelta(days=30)
    cutoff_90 = now - timedelta(days=90)

    matches: List[Tuple[datetime, str, str]] = []  # (ts, text, type)
    for o in observations:
        t = _parse_ts(o)
        if t is None:
            continue
        if str(o.get("type") or "") not in _INTERACTION_TYPES:
            continue
        text = _text_of(o)
        if not text:
            continue
        if not _person_in_text(name, text):
            continue
        matches.append((t, text, str(o.get("type") or "")))
    matches.sort(key=lambda m: m[0])

    interactions_30d = sum(1 for t, *_ in matches if t >= cutoff_30)
    interactions_90d = sum(1 for t, *_ in matches if t >= cutoff_90)

    last_dt = matches[-1][0] if matches else None
    days_since = (now - last_dt).days if last_dt else None

    # Component 1 — recency.
    if last_dt is None:
        recency = 0.0
    else:
        d = (now - last_dt).total_seconds() / 86400.0  # days
        recency = max(0.0, min(1.0, 1.0 - d / 30.0))

    # Component 2 — frequency.
    freq_30 = min(1.0, interactions_30d / 10.0)
    freq_90 = min(1.0, interactions_90d / 25.0)
    frequency = 0.6 * freq_30 + 0.4 * freq_90

    # Component 3 — sentiment.
    sentiment = 0.5
    try:
        from .sentiment import analyze_messages
        recent_texts = [text for t, text, _ in matches if t >= cutoff_30]
        if recent_texts:
            agg = analyze_messages(recent_texts)
            # Map [-1, 1] sentiment to [0, 1]: neutral = 0.5.
            sentiment = max(0.0, min(1.0, 0.5 + agg["avg_sentiment"] * 0.5))
    except Exception:
        pass

    # Component 4 — reciprocity.
    out_count = in_count = 0
    for _, text, _ in matches:
        dir_ = _outgoing_marker(text)
        if dir_ is True:
            out_count += 1
        elif dir_ is False:
            in_count += 1
    total_directed = out_count + in_count
    if total_directed >= 4:
        # Reciprocity is highest when the split is close to 50/50.
        ratio = min(out_count, in_count) / max(out_count, in_count, 1)
        reciprocity = ratio
    else:
        reciprocity = 0.5  # not enough info — neutral

    components = {
        "recency": round(recency, 3),
        "frequency": round(frequency, 3),
        "sentiment": round(sentiment, 3),
        "reciprocity": round(reciprocity, 3),
    }
    score = (
        0.35 * recency + 0.30 * frequency
        + 0.20 * sentiment + 0.15 * reciprocity
    )

    # Trend: compare 30d vs the older 30d window.
    prev_window = sum(
        1 for t, *_ in matches
        if cutoff_90 <= t < cutoff_30
    )
    if days_since is None or days_since > 60:
        # No interactions in the history, or last contact > 60 days ago.
        trend = "dormant"
    elif interactions_30d > prev_window * 1.5:
        trend = "rising"
    elif prev_window > 0 and interactions_30d < prev_window * 0.6:
        trend = "declining"
    else:
        trend = "stable"

    return Connection(
        name=name,
        relation=relation,
        score=round(score, 3),
        components=components,
        trend=trend,
        interactions_30d=interactions_30d,
        interactions_90d=interactions_90d,
        last_contact=last_dt.isoformat() if last_dt else None,
        days_since_contact=days_since,
    )


def score_all_connections(
    people: Dict[str, Dict],
    observations: List[Dict],
    now: Optional[datetime] = None,
) -> List[Connection]:
    """Score every tracked person, sorted by score descending."""
    result = []
    for name, info in (people or {}).items():
        relation = (info or {}).get("relation", "")
        result.append(score_connection(name, relation, observations, now=now))
    result.sort(key=lambda c: -c.score)
    return result


def trend_marker(trend: str) -> str:
    return {"rising": "↑", "declining": "↓", "stable": "→", "dormant": "⌛"}.get(trend, "•")
