"""
Energy curve — models the user's circadian-ish energy from observation density.

The premise: hours where lots of activity happens are high-energy; hours of
quiet are low-energy. We bucket observations by hour-of-day, smooth with a
3-hour window, normalize to [0, 1], and report:

- current hour's energy level (low/medium/high) and percentile
- peak hour and trough hour
- 6-hour forecast (energy expected in the next 6 hours)
- weekly variation (weekday vs weekend if both present)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


@dataclass
class EnergyReport:
    current_hour: int
    current_level: str            # "low" | "medium" | "high"
    current_score: float          # 0.0 - 1.0
    peak_hour: int
    trough_hour: int
    hourly_curve: List[float]     # 24 entries, 0..1
    forecast_6h: List[Dict]       # [{hour, level, score}, …]
    sample_size: int              # # of observations contributing
    days_observed: int


def _parse_ts(o: Dict) -> Optional[datetime]:
    ts = o.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _level(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def compute_energy_curve(observations: List[Dict], now: Optional[datetime] = None) -> EnergyReport:
    now = now or datetime.now(timezone.utc)
    hourly: Dict[int, int] = defaultdict(int)
    earliest: Optional[datetime] = None
    n = 0

    for o in observations:
        t = _parse_ts(o)
        if t is None:
            continue
        hourly[t.hour] += 1
        n += 1
        if earliest is None or t < earliest:
            earliest = t

    days_observed = max(1, (now - earliest).days) if earliest else 0

    raw = [hourly.get(h, 0) for h in range(24)]
    # Smooth with a 3-hour moving average so single-hour spikes don't dominate.
    smoothed = [
        (raw[(h - 1) % 24] + raw[h] + raw[(h + 1) % 24]) / 3.0
        for h in range(24)
    ]

    peak = max(smoothed) if smoothed else 0
    trough = min(smoothed) if smoothed else 0
    span = max(peak - trough, 1e-9)
    curve = [(v - trough) / span for v in smoothed]

    peak_hour = smoothed.index(peak) if smoothed else now.hour
    trough_hour = smoothed.index(trough) if smoothed else (peak_hour + 12) % 24

    current_hour = now.hour
    current_score = curve[current_hour] if curve else 0.0

    forecast = []
    for offset in range(1, 7):
        h = (current_hour + offset) % 24
        s = curve[h] if curve else 0.0
        forecast.append({"hour": h, "score": round(s, 3), "level": _level(s)})

    return EnergyReport(
        current_hour=current_hour,
        current_level=_level(current_score),
        current_score=round(current_score, 3),
        peak_hour=peak_hour,
        trough_hour=trough_hour,
        hourly_curve=[round(v, 3) for v in curve],
        forecast_6h=forecast,
        sample_size=n,
        days_observed=days_observed,
    )


def render_curve_ascii(curve: List[float], current_hour: int) -> str:
    """ASCII bar chart for inclusion in CLI/text output."""
    if not curve:
        return "(no data)"
    bars = " ▁▂▃▄▅▆▇█"
    lines = []
    line = ""
    for h, v in enumerate(curve):
        bar = bars[min(len(bars) - 1, int(v * (len(bars) - 1)))]
        marker = "*" if h == current_hour else " "
        line += f"{bar}{marker}"
    lines.append(line)
    lines.append(" 0   3   6   9   12  15  18  21")
    return "\n".join(lines)
