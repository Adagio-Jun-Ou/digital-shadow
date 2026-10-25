"""
Habit Tracker — detects recurring behavioral patterns in observations.

A *habit candidate* is a (type, time-bucket) pair where the user does the same
kind of activity on the same days/hours regularly. We score:

- frequency       — how often the habit fires
- consistency     — how aligned the firings are around a time bucket
- streak          — longest unbroken run of expected firings
- gap_days        — days since the last firing (for "broken streak" alerts)

Time buckets: weekday-of-week + hour-of-day (e.g. "Monday 09:00").
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class Habit:
    activity_type: str
    bucket: str          # e.g. "Mon 09" — weekday + hour
    frequency: int       # total firings
    consistency: float   # in [0, 1]: how concentrated in this bucket
    avg_per_week: float
    streak: int          # current consecutive-week streak
    longest_streak: int
    last_seen: Optional[str]
    days_since: int      # int.MAX_VALUE if never
    days_observed: int   # span of observations


def _parse_ts(o: Dict) -> Optional[datetime]:
    ts = o.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def detect_habits(
    observations: List[Dict],
    min_frequency: int = 3,
    min_consistency: float = 0.4,
    now: Optional[datetime] = None,
) -> List[Habit]:
    """Find habits in `observations`. Returns at most one habit per activity
    type (the strongest bucket), sorted by frequency desc."""
    now = now or datetime.now(timezone.utc)
    if not observations:
        return []

    # First, bin observations by (type, bucket).
    bins: Dict[str, Dict[str, List[datetime]]] = defaultdict(lambda: defaultdict(list))
    by_type_total: Counter = Counter()
    earliest: Optional[datetime] = None

    for o in observations:
        t = _parse_ts(o)
        if t is None:
            continue
        otype = str(o.get("type") or "unknown")
        bucket = f"{WEEKDAY_NAMES[t.weekday()]} {t.hour:02d}"
        bins[otype][bucket].append(t)
        by_type_total[otype] += 1
        if earliest is None or t < earliest:
            earliest = t

    if not bins or earliest is None:
        return []
    days_observed = max(1, (now - earliest).days)

    habits: List[Habit] = []
    for otype, buckets in bins.items():
        total = by_type_total[otype]
        if total < min_frequency:
            continue
        # Pick the strongest bucket for this activity type.
        top_bucket, top_times = max(buckets.items(), key=lambda kv: len(kv[1]))
        bucket_count = len(top_times)
        consistency = bucket_count / total
        if consistency < min_consistency or bucket_count < min_frequency:
            continue

        sorted_times = sorted(top_times)
        last = sorted_times[-1]
        days_since = (now.date() - last.date()).days

        # Streak: consecutive weeks in which the habit fired in this bucket.
        weeks_with_fire = sorted({_iso_week_id(t) for t in sorted_times})
        longest_streak, current_streak = _compute_streak(weeks_with_fire, _iso_week_id(now))

        weeks_span = max(1, days_observed / 7.0)
        habits.append(Habit(
            activity_type=otype,
            bucket=top_bucket,
            frequency=bucket_count,
            consistency=round(consistency, 3),
            avg_per_week=round(bucket_count / weeks_span, 2),
            streak=current_streak,
            longest_streak=longest_streak,
            last_seen=last.isoformat(),
            days_since=days_since,
            days_observed=days_observed,
        ))

    habits.sort(key=lambda h: -h.frequency)
    return habits


def _iso_week_id(dt: datetime) -> tuple:
    iso = dt.isocalendar()
    # `iso.year, iso.week` for python 3.9+
    return (getattr(iso, "year", iso[0]), getattr(iso, "week", iso[1]))


def _compute_streak(weeks_sorted: List[tuple], current_week: tuple) -> tuple:
    """Given a sorted list of (year, week) where the habit fired and the
    current (year, week), return (longest_streak, current_streak)."""
    if not weeks_sorted:
        return 0, 0
    longest = 1
    cur = 1
    for prev, this in zip(weeks_sorted, weeks_sorted[1:]):
        if _weeks_consecutive(prev, this):
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    # The "current" streak is alive only if the last week observed is the
    # current week or the previous one.
    last = weeks_sorted[-1]
    if last == current_week or _weeks_consecutive(last, current_week):
        current = cur
    else:
        current = 0
    return longest, current


def _weeks_consecutive(a: tuple, b: tuple) -> bool:
    """True if (year, week) b directly follows a."""
    if a == b:
        return True
    if a[0] == b[0] and b[1] == a[1] + 1:
        return True
    # Year wraparound: last week of year N → week 1 of year N+1.
    if b[0] == a[0] + 1 and a[1] >= 52 and b[1] == 1:
        return True
    return False


def find_broken_streaks(habits: List[Habit], grace_days: int = 2) -> List[Habit]:
    """Return habits whose streak just broke (last seen > 1 week ago but
    within recent enough history to flag)."""
    return [h for h in habits
            if h.streak == 0 and 7 <= h.days_since <= 30 and h.longest_streak >= 3]


def format_habit(h: Habit) -> str:
    """Human-readable one-liner for a habit."""
    streak_str = f"streak={h.streak}w" if h.streak else f"BROKEN (was {h.longest_streak}w)"
    return (
        f"{h.activity_type:<20s} @{h.bucket}  "
        f"{h.frequency}x  {h.avg_per_week:.1f}/wk  "
        f"consistency={h.consistency:.0%}  {streak_str}  "
        f"last={h.days_since}d ago"
    )
