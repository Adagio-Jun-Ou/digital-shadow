"""
Tide — Emotional sentiment trend across time.

Aggregates sentiment from text-bearing observations (notes, messages,
clipboard, journal, reflection) into weekly buckets, surfaces:
- Overall mood and slope (rising/falling/stable)
- Best and worst weeks
- Longest positive / negative streaks
- Dominant emotional words for each phase
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .sentiment import analyze_sentiment


_TEXT_TYPES = {
    "note", "message", "clipboard", "journal", "reflection",
    "diary", "comment", "tweet", "slack_message",
}


@dataclass
class WeekBucket:
    iso_week: str  # YYYY-WW format
    start: str
    end: str
    sample_count: int
    avg_sentiment: float
    label: str
    top_positive: List[str] = field(default_factory=list)
    top_negative: List[str] = field(default_factory=list)


@dataclass
class TideReport:
    total_samples: int
    weeks: List[WeekBucket]
    overall_avg: float
    overall_label: str
    slope: str           # rising / falling / stable
    slope_value: float   # linear-fit slope in sentiment per week
    best_week: Optional[str]
    worst_week: Optional[str]
    longest_positive_streak: int
    longest_negative_streak: int
    summary: str


def _parse_ts(o: Dict) -> Optional[datetime]:
    ts = o.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _extract_text(obs: Dict) -> Optional[str]:
    if obs.get("type") not in _TEXT_TYPES:
        return None
    c = obs.get("content")
    if not isinstance(c, dict):
        return None
    for k in ("text", "body", "content", "message", "note"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _iso_week_key(dt: datetime) -> Tuple[str, datetime, datetime]:
    iso_year, iso_week, _ = dt.isocalendar()
    # Monday of that ISO week.
    monday = datetime.fromisocalendar(iso_year, iso_week, 1).replace(tzinfo=timezone.utc)
    sunday = monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return f"{iso_year}-W{iso_week:02d}", monday, sunday


def _label_for(score: float) -> str:
    if score >= 0.4: return "very_positive"
    if score >= 0.1: return "positive"
    if score <= -0.4: return "very_negative"
    if score <= -0.1: return "negative"
    return "neutral"


def _linear_slope(values: List[float]) -> float:
    """Ordinary-least-squares slope for evenly-spaced points."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    num = sum((xs[i] - mx) * (values[i] - my) for i in range(n))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def compute_tide(observations: List[Dict]) -> TideReport:
    samples: List[Tuple[datetime, float, List[str], List[str]]] = []
    for o in observations:
        text = _extract_text(o)
        if not text:
            continue
        t = _parse_ts(o)
        if t is None:
            continue
        s = analyze_sentiment(text)
        samples.append((t, s.score, s.positive_terms, s.negative_terms))

    if not samples:
        return TideReport(
            total_samples=0, weeks=[], overall_avg=0.0, overall_label="neutral",
            slope="stable", slope_value=0.0, best_week=None, worst_week=None,
            longest_positive_streak=0, longest_negative_streak=0,
            summary="No text-bearing observations yet — Tide needs notes, messages, or clipboard entries to read.",
        )

    samples.sort(key=lambda s: s[0])

    # Bucket per ISO week.
    week_buckets: Dict[str, Dict] = defaultdict(lambda: {
        "start": None, "end": None, "scores": [],
        "pos_words": Counter(), "neg_words": Counter(),
    })
    for t, score, pos, neg in samples:
        key, monday, sunday = _iso_week_key(t)
        b = week_buckets[key]
        if b["start"] is None or monday < b["start"]:
            b["start"] = monday
        if b["end"] is None or sunday > b["end"]:
            b["end"] = sunday
        b["scores"].append(score)
        b["pos_words"].update(pos)
        b["neg_words"].update(neg)

    weeks: List[WeekBucket] = []
    for key in sorted(week_buckets.keys()):
        b = week_buckets[key]
        avg = sum(b["scores"]) / len(b["scores"])
        weeks.append(WeekBucket(
            iso_week=key,
            start=b["start"].isoformat() if b["start"] else "",
            end=b["end"].isoformat() if b["end"] else "",
            sample_count=len(b["scores"]),
            avg_sentiment=round(avg, 4),
            label=_label_for(avg),
            top_positive=[w for w, _ in b["pos_words"].most_common(5)],
            top_negative=[w for w, _ in b["neg_words"].most_common(5)],
        ))

    overall = sum(s[1] for s in samples) / len(samples)
    slope_val = _linear_slope([w.avg_sentiment for w in weeks])
    if slope_val > 0.05:
        slope = "rising"
    elif slope_val < -0.05:
        slope = "falling"
    else:
        slope = "stable"

    best = max(weeks, key=lambda w: w.avg_sentiment) if weeks else None
    worst = min(weeks, key=lambda w: w.avg_sentiment) if weeks else None

    # Streaks of positive/negative sample scores in chronological order.
    pos_streak = neg_streak = 0
    longest_pos = longest_neg = 0
    for _, score, _, _ in samples:
        if score > 0.05:
            pos_streak += 1
            neg_streak = 0
            longest_pos = max(longest_pos, pos_streak)
        elif score < -0.05:
            neg_streak += 1
            pos_streak = 0
            longest_neg = max(longest_neg, neg_streak)
        else:
            pos_streak = 0
            neg_streak = 0

    summary_parts: List[str] = []
    summary_parts.append(f"Overall mood: {_label_for(overall)} ({overall:+.2f})")
    summary_parts.append(f"Trend: {slope}")
    if best and worst and best.iso_week != worst.iso_week:
        summary_parts.append(f"Best: {best.iso_week} ({best.avg_sentiment:+.2f})")
        summary_parts.append(f"Worst: {worst.iso_week} ({worst.avg_sentiment:+.2f})")

    return TideReport(
        total_samples=len(samples),
        weeks=weeks,
        overall_avg=round(overall, 4),
        overall_label=_label_for(overall),
        slope=slope,
        slope_value=round(slope_val, 4),
        best_week=best.iso_week if best else None,
        worst_week=worst.iso_week if worst else None,
        longest_positive_streak=longest_pos,
        longest_negative_streak=longest_neg,
        summary=". ".join(summary_parts),
    )
