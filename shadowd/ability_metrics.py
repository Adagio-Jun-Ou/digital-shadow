"""
Per-ability invocation metrics. Tiny, in-memory, thread-safe.

Tracks for each ability:
- total invocations
- success / failure counts
- latency samples (last 100)
- avg / p50 / p95

Exposed via /metrics and a Python API for tests.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Dict, List


_LOCK = threading.Lock()
_SAMPLES_PER_ABILITY = 100

_counts: Dict[str, int] = defaultdict(int)
_successes: Dict[str, int] = defaultdict(int)
_failures: Dict[str, int] = defaultdict(int)
_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_SAMPLES_PER_ABILITY))
_last_invoked: Dict[str, float] = {}


def record(ability: str, success: bool, duration_seconds: float) -> None:
    with _LOCK:
        _counts[ability] += 1
        if success:
            _successes[ability] += 1
        else:
            _failures[ability] += 1
        _latencies[ability].append(duration_seconds)
        _last_invoked[ability] = time.time()


def snapshot() -> Dict[str, Dict[str, float]]:
    """Return a dict keyed by ability name with computed stats. Safe to call
    from any thread."""
    out: Dict[str, Dict[str, float]] = {}
    with _LOCK:
        for ability in _counts:
            latencies = sorted(_latencies[ability])
            n = len(latencies)
            avg = sum(latencies) / n if n else 0.0
            p50 = latencies[n // 2] if n else 0.0
            p95 = latencies[min(n - 1, max(0, int(n * 0.95)))] if n else 0.0
            out[ability] = {
                "count": _counts[ability],
                "successes": _successes[ability],
                "failures": _failures[ability],
                "avg_latency_s": round(avg, 4),
                "p50_latency_s": round(p50, 4),
                "p95_latency_s": round(p95, 4),
                "last_invoked_ts": _last_invoked.get(ability, 0.0),
                "sample_count": n,
            }
    return out


def prometheus_lines() -> List[str]:
    """Render as Prometheus exposition format."""
    snap = snapshot()
    lines: List[str] = []
    lines.append("# HELP shadow_ability_invocations Total ability invocations")
    lines.append("# TYPE shadow_ability_invocations counter")
    for ability, stats in snap.items():
        lines.append(f'shadow_ability_invocations{{ability="{ability}"}} {stats["count"]}')
    lines.append("# HELP shadow_ability_successes Successful ability invocations")
    lines.append("# TYPE shadow_ability_successes counter")
    for ability, stats in snap.items():
        lines.append(f'shadow_ability_successes{{ability="{ability}"}} {stats["successes"]}')
    lines.append("# HELP shadow_ability_failures Failed ability invocations")
    lines.append("# TYPE shadow_ability_failures counter")
    for ability, stats in snap.items():
        lines.append(f'shadow_ability_failures{{ability="{ability}"}} {stats["failures"]}')
    lines.append("# HELP shadow_ability_latency_p95 95th-percentile latency in seconds")
    lines.append("# TYPE shadow_ability_latency_p95 gauge")
    for ability, stats in snap.items():
        lines.append(f'shadow_ability_latency_p95{{ability="{ability}"}} {stats["p95_latency_s"]}')
    return lines


def reset() -> None:
    """Wipe all counters (for tests)."""
    with _LOCK:
        _counts.clear()
        _successes.clear()
        _failures.clear()
        _latencies.clear()
        _last_invoked.clear()
