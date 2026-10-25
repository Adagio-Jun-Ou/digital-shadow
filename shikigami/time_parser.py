"""
Natural-language time window parser for Perfect Memory.

Recognizes relative ranges like:
- "today" / "yesterday"
- "this week" / "last week"
- "this month" / "last month"
- "last N hours/days/weeks/months"
- "past hour" / "past 24 hours"
- "in the last hour"
- "since yesterday"

Returns a (start, end) datetime range in UTC, or None if no time expression
matched. Pure Python — no dateparser dependency.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple


_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60,
}


def _to_int(word: str) -> Optional[int]:
    word = word.lower().strip()
    if word.isdigit():
        return int(word)
    return _NUMBER_WORDS.get(word)


def _unit_to_seconds(unit: str) -> Optional[int]:
    unit = unit.lower().rstrip("s")
    return {
        "second": 1, "minute": 60, "hour": 3600, "day": 86400,
        "week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400,
    }.get(unit)


def parse_time_window(text: str, now: Optional[datetime] = None) -> Optional[Tuple[datetime, datetime]]:
    """Return (start, end) UTC tuple if `text` contains a recognized time window,
    else None. `now` defaults to current UTC time."""
    if not text:
        return None
    now = now or datetime.now(timezone.utc)
    text = text.lower().strip()

    # "today" / "yesterday" — entire day in UTC.
    if re.search(r"\btoday\b", text):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)
    if re.search(r"\byesterday\b", text):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)

    # "this week" — Monday 00:00 → now.
    if re.search(r"\bthis week\b", text):
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now
    if re.search(r"\blast week\b", text):
        this_week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        last_week_start = this_week_start - timedelta(days=7)
        return last_week_start, this_week_start

    # "this month" / "last month".
    if re.search(r"\bthis month\b", text):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, now
    if re.search(r"\blast month\b", text):
        first_of_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Subtract a day to land in last month, then snap to its 1st.
        last_month_end = first_of_this
        last_month_start = (first_of_this - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return last_month_start, last_month_end

    # "past hour" / "past 24 hours" / "last 3 days" / "in the last 2 weeks"
    m = re.search(
        r"\b(?:past|last|previous)\s+(?:the\s+)?(?:(\d+|[a-z]+)\s+)?"
        r"(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
        text,
    )
    if not m:
        m = re.search(
            r"\bin\s+the\s+(?:last|past)\s+(?:(\d+|[a-z]+)\s+)?"
            r"(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
            text,
        )
    if m:
        n_raw, unit = m.group(1), m.group(2)
        n = _to_int(n_raw) if n_raw else 1
        secs = _unit_to_seconds(unit)
        if n is not None and secs:
            return now - timedelta(seconds=n * secs), now

    # "since yesterday" / "since last week"
    if re.search(r"\bsince yesterday\b", text):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now
    if re.search(r"\bsince last week\b", text):
        start = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now

    return None


def in_window(obs_timestamp: str, window: Tuple[datetime, datetime]) -> bool:
    """Return True if observation's ISO-8601 timestamp falls within `window`."""
    try:
        ts = datetime.fromisoformat(obs_timestamp.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return window[0] <= ts <= window[1]
