"""
Focus blocks — detect deep-work sessions in observation history.

A "focus block" is a contiguous span where:
- the user does mostly one *kind* of activity (entropy below threshold)
- with no long idle gaps (< max_gap minutes)

We compute current focus state ("in flow" / "scattered" / "idle") plus a
history of past focus blocks for trend analysis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


@dataclass
class FocusBlock:
    start: str
    end: str
    duration_min: float
    dominant_type: str
    purity: float       # 0..1; how much of the block was the dominant type
    n_events: int


def _parse_ts(o: Dict) -> Optional[datetime]:
    ts = o.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def find_focus_blocks(
    observations: List[Dict],
    max_gap_minutes: int = 10,
    min_block_minutes: int = 15,
    min_purity: float = 0.6,
) -> List[FocusBlock]:
    """Return focus blocks chronologically. A block ends when there's a gap
    longer than max_gap_minutes or activity type entropy spikes."""
    events: List[tuple] = []  # (timestamp, type)
    for o in observations:
        t = _parse_ts(o)
        if t is None:
            continue
        events.append((t, str(o.get("type") or "unknown")))
    events.sort(key=lambda e: e[0])

    blocks: List[FocusBlock] = []
    if len(events) < 2:
        return blocks

    cur_start = events[0][0]
    cur_events: List[str] = [events[0][1]]
    cur_last = events[0][0]

    def _finalize():
        nonlocal cur_start, cur_events, cur_last
        if len(cur_events) < 2:
            return
        duration = (cur_last - cur_start).total_seconds() / 60.0
        if duration < min_block_minutes:
            return
        type_counts = Counter(cur_events)
        dominant, dom_count = type_counts.most_common(1)[0]
        purity = dom_count / len(cur_events)
        if purity < min_purity:
            return
        blocks.append(FocusBlock(
            start=cur_start.isoformat(),
            end=cur_last.isoformat(),
            duration_min=round(duration, 1),
            dominant_type=dominant,
            purity=round(purity, 3),
            n_events=len(cur_events),
        ))

    for t, otype in events[1:]:
        gap = (t - cur_last).total_seconds() / 60.0
        if gap > max_gap_minutes:
            _finalize()
            cur_start = t
            cur_events = [otype]
        else:
            cur_events.append(otype)
        cur_last = t
    _finalize()
    return blocks


def current_focus_state(
    observations: List[Dict],
    now: Optional[datetime] = None,
    idle_minutes: int = 15,
) -> Dict:
    """Classify the current focus state based on recent activity:
    'in_flow'  — last ≥3 events same type, ≤5min apart
    'scattered'— recent activity but with type switching
    'idle'     — no event in `idle_minutes` minutes
    """
    now = now or datetime.now(timezone.utc)
    recent = []
    for o in observations:
        t = _parse_ts(o)
        if t and (now - t).total_seconds() < idle_minutes * 60:
            recent.append((t, str(o.get("type") or "unknown")))
    recent.sort(key=lambda e: e[0])

    if not recent:
        return {"state": "idle", "since_last_event_min": None,
                "recent_types": [], "dominant": None}

    last = recent[-1][0]
    if (now - last).total_seconds() > idle_minutes * 60:
        return {"state": "idle", "since_last_event_min": (now - last).total_seconds() / 60.0,
                "recent_types": [r[1] for r in recent], "dominant": None}

    types = [r[1] for r in recent[-6:]]
    counts = Counter(types)
    dominant, dom_count = counts.most_common(1)[0]
    purity = dom_count / len(types)
    state = "in_flow" if purity >= 0.66 and len(types) >= 3 else "scattered"
    return {
        "state": state,
        "since_last_event_min": round((now - last).total_seconds() / 60.0, 1),
        "recent_types": types,
        "dominant": dominant if state == "in_flow" else None,
        "purity": round(purity, 3),
    }


def summarize_focus(blocks: List[FocusBlock]) -> Dict:
    if not blocks:
        return {"total_blocks": 0, "total_focused_minutes": 0,
                "longest_block_minutes": 0, "favorite_focus_type": None}
    total = sum(b.duration_min for b in blocks)
    longest = max(b.duration_min for b in blocks)
    # Weighted by duration so a 60min focus on X outweighs a 15min one.
    weighted = Counter()
    for b in blocks:
        weighted[b.dominant_type] += b.duration_min
    favorite = weighted.most_common(1)[0][0] if weighted else None
    return {
        "total_blocks": len(blocks),
        "total_focused_minutes": round(total, 1),
        "longest_block_minutes": round(longest, 1),
        "favorite_focus_type": favorite,
    }
