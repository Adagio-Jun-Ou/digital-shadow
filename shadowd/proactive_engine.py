"""
Proactive Insight Engine.

Runs in the background and periodically scans recent observations / tracked
people for things worth alerting about — without waiting for the user to call
an ability. Broadcasts findings on the daemon's WebSocket `insights` topic.

Current triggers:
- Manipulation: any recent message scoring above MANIPULATION_THRESHOLD
- Sentiment shift: a person's avg sentiment swings hard in either direction
- Activity anomaly: Omni Sense's anomaly_detector fires
- High-value council guidance: when many similar past decisions point one way

Cheap to run (no LLM calls). Tunable via env: SHADOW_PROACTIVE_INTERVAL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger("shadowd.proactive")

_INTERVAL_S = int(os.environ.get("SHADOW_PROACTIVE_INTERVAL", "60"))
_MANIPULATION_THRESHOLD = float(os.environ.get("SHADOW_MANIPULATION_THRESHOLD", "0.35"))
_SENTIMENT_SHIFT_THRESHOLD = float(os.environ.get("SHADOW_SENTIMENT_SHIFT", "0.4"))

# Per-insight cooldown so we don't spam the same alert. Key: insight signature,
# value: last-emit timestamp.
_RECENT_EMIT: Dict[str, float] = {}
_COOLDOWN_S = int(os.environ.get("SHADOW_INSIGHT_COOLDOWN", "300"))

# Bounded ring of insight signatures to keep cooldown table small.
_SIG_RING: deque = deque(maxlen=256)


def _should_emit(signature: str) -> bool:
    now = time.time()
    last = _RECENT_EMIT.get(signature, 0.0)
    if now - last < _COOLDOWN_S:
        return False
    _RECENT_EMIT[signature] = now
    _SIG_RING.append(signature)
    # Drop any keys that fell out of the ring.
    if len(_RECENT_EMIT) > 512:
        keep = set(_SIG_RING)
        for k in list(_RECENT_EMIT.keys()):
            if k not in keep:
                del _RECENT_EMIT[k]
    return True


def _extract_text(obs: Dict) -> str:
    content = obs.get("content", {})
    if isinstance(content, dict):
        for k in ("text", "body", "message", "preview", "content"):
            v = content.get(k)
            if isinstance(v, str):
                return v
    elif isinstance(content, str):
        return content
    return ""


async def _scan_manipulation(shadow) -> List[Dict]:
    """Look at the most recent messages/clipboard and flag manipulation."""
    try:
        from shikigami.sentiment import detect_manipulation
    except ImportError:
        return []

    insights = []
    obs = shadow.get_observations(50)
    msg_types = {"clipboard", "message", "email", "communication", "chat"}
    for o in obs:
        if o.get("type") not in msg_types:
            continue
        text = _extract_text(o)
        if not text:
            continue
        result = detect_manipulation(text)
        if result.score < _MANIPULATION_THRESHOLD:
            continue
        # Dedupe by a stable hash of the text + tactic set.
        sig = f"manip:{hash(text + ','.join(sorted(result.tactics)))}"
        if not _should_emit(sig):
            continue
        insights.append({
            "kind": "manipulation",
            "severity": "high" if result.score > 0.6 else "medium",
            "score": result.score,
            "tactics": list(result.tactics.keys()),
            "explanation": result.explanation,
            "source_type": o.get("type"),
            "preview": text[:200],
            "timestamp": o.get("timestamp"),
        })
    return insights


async def _scan_sentiment_shift(shadow) -> List[Dict]:
    """Flag sharp swings in recent message tone toward a tracked person."""
    try:
        from shikigami.sentiment import analyze_messages
    except ImportError:
        return []
    if not getattr(shadow, "shadow_ai", None):
        return []

    insights = []
    obs = shadow.get_observations(200)
    people = list(getattr(shadow.shadow_ai, "people", {}).keys())
    for name in people:
        name_l = name.lower()
        msgs = []
        for o in obs:
            t = o.get("type")
            if t not in {"clipboard", "message", "email", "communication", "chat"}:
                continue
            text = _extract_text(o)
            if not text:
                continue
            if name_l in text.lower():
                msgs.append(text)
        if len(msgs) < 4:
            continue
        agg = analyze_messages(msgs)
        if agg["sentiment_trend"] not in ("improving", "declining"):
            continue
        # Compare halves: only emit if the magnitude is meaningful.
        first_half = msgs[: len(msgs) // 2]
        second_half = msgs[len(msgs) // 2 :]
        first_avg = analyze_messages(first_half)["avg_sentiment"]
        second_avg = analyze_messages(second_half)["avg_sentiment"]
        delta = second_avg - first_avg
        if abs(delta) < _SENTIMENT_SHIFT_THRESHOLD:
            continue
        sig = f"sent:{name}:{round(delta, 1)}"
        if not _should_emit(sig):
            continue
        insights.append({
            "kind": "sentiment_shift",
            "severity": "medium",
            "person": name,
            "trend": agg["sentiment_trend"],
            "delta": round(delta, 3),
            "from_avg": round(first_avg, 3),
            "to_avg": round(second_avg, 3),
            "sample_size": len(msgs),
        })
    return insights


async def _scan_council_alignment(shadow) -> List[Dict]:
    """Surface Shadow Council's recommendation when there's enough data to be
    confident — useful for the dashboard to show standing guidance."""
    try:
        from shikigami.council_synthesis import synthesize
    except ImportError:
        return []
    if not getattr(shadow, "shadow_ai", None):
        return []

    registry = getattr(shadow.shadow_ai, "registry", None)
    tracker = getattr(registry, "decision_tracker", None)
    if tracker is None or len(getattr(tracker, "decisions", [])) < 5:
        return []

    verdict = synthesize(list(tracker.decisions), "current direction")
    rec = verdict["recommendation"]
    if not rec.get("action") or rec.get("confidence", 0) < 0.4 or rec.get("lift", 0) < 0.5:
        return []

    sig = f"council:{rec['action']}:{round(rec['lift'], 1)}"
    if not _should_emit(sig):
        return []
    return [{
        "kind": "council_recommendation",
        "severity": "low",
        "recommendation": rec["action"],
        "confidence": rec["confidence"],
        "lift": rec["lift"],
        "reasoning": rec["reasoning"],
    }]


async def _scan_anticipation(shadow) -> List[Dict]:
    """Bridge: pull anticipatory signals from the cognition v2 engine and
    surface them as insights. Only emits signals with confidence > 0.65."""
    try:
        from .cognition.anticipatory_engine import (
            AnticipatoryEngine, AnticipationContext,
        )
    except ImportError:
        return []

    # Reuse a singleton on the shadow object so the engine accumulates context
    # across iterations.
    engine = getattr(shadow, "_anticipatory_engine", None)
    if engine is None:
        engine = AnticipatoryEngine()
        shadow._anticipatory_engine = engine

    from datetime import datetime, timezone
    obs = shadow.get_observations(50)
    recent_types = [o.get("type", "") for o in obs[-15:] if o.get("type")]
    active_domains: List[str] = []
    if any(t in ("git_commit", "terminal_command", "file_change") for t in recent_types):
        active_domains.append("coding")
    if any(t in ("clipboard", "message", "communication", "email") for t in recent_types):
        active_domains.append("communication")
    if any(t == "window_focus" for t in recent_types):
        active_domains.append("desktop")

    now = datetime.now(timezone.utc)
    context = AnticipationContext(
        user_state={"observation_count": len(obs)},
        active_domains=active_domains,
        time_of_day=now.hour,
        day_of_week=now.weekday(),
        recent_actions=recent_types[-5:],
        timestamp=now.isoformat(),
    )

    try:
        signals = engine.predict(context)
    except Exception as e:
        logger.debug("anticipatory predict failed: %s", type(e).__name__)
        return []

    insights = []
    for s in signals[:3]:
        if getattr(s, "confidence", 0) < 0.65:
            continue
        sig_key = f"anticipate:{getattr(s, 'prediction', '?')}:{round(s.confidence, 2)}"
        if not _should_emit(sig_key):
            continue
        insights.append({
            "kind": "anticipation",
            "severity": "low",
            "prediction": getattr(s, "prediction", ""),
            "confidence": round(getattr(s, "confidence", 0.0), 3),
            "reasoning": getattr(s, "reasoning", ""),
            "suggested_action": getattr(s, "suggested_action", ""),
            "trigger": getattr(s, "trigger", ""),
        })
    return insights


async def _scan_anomaly_baseline(shadow) -> List[Dict]:
    """Flag wall-clock activity that's an outlier vs the user's hour-of-day
    baseline. (E.g. lots of activity at 4 AM is unusual for daytime workers.)"""
    insights = []
    obs = shadow.get_observations(200)
    if len(obs) < 30:
        return []
    # Bucket recent obs by hour.
    from collections import Counter
    from datetime import datetime, timezone
    hours = Counter()
    for o in obs:
        ts = o.get("timestamp")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        hours[t.hour] += 1
    if not hours:
        return []
    now_hour = datetime.now(timezone.utc).hour
    # Z-score against the bucket distribution.
    counts = list(hours.values())
    mean = sum(counts) / len(counts)
    if mean == 0:
        return []
    var = sum((c - mean) ** 2 for c in counts) / len(counts)
    std = var ** 0.5 if var > 0 else 1.0
    z = (hours[now_hour] - mean) / std
    if z > 2.0:  # current hour is an outlier upward
        sig = f"anomaly:hour_burst:{now_hour}"
        if _should_emit(sig):
            insights.append({
                "kind": "activity_anomaly",
                "severity": "low",
                "hour": now_hour,
                "z_score": round(z, 2),
                "current_hour_count": hours[now_hour],
                "mean_count": round(mean, 1),
            })
    return insights


# Ring buffer of insights emitted in this process — exposed via
# /insights/recent so HTTP-only clients can fetch without a WebSocket.
_INSIGHT_HISTORY: deque = deque(maxlen=100)


def recent_insights(limit: int = 20) -> List[Dict]:
    return list(_INSIGHT_HISTORY)[-limit:]


async def run_proactive_loop(shadow):
    """Main loop. Cancelled when the daemon shuts down."""
    logger.info("Proactive engine starting (interval=%ds, threshold=%.2f)",
                _INTERVAL_S, _MANIPULATION_THRESHOLD)
    # Brief startup wait so observers can buffer some data first.
    await asyncio.sleep(min(30, _INTERVAL_S // 2))

    while True:
        try:
            scanners = (
                _scan_manipulation,
                _scan_sentiment_shift,
                _scan_council_alignment,
                _scan_anomaly_baseline,
                _scan_anticipation,
            )
            all_insights: List[Dict] = []
            for scanner in scanners:
                try:
                    found = await scanner(shadow)
                    all_insights.extend(found)
                except Exception as e:
                    logger.warning("scanner %s failed: %s", scanner.__name__, type(e).__name__)

            for insight in all_insights:
                insight["emitted_at"] = time.time()
                _INSIGHT_HISTORY.append(insight)
                logger.info("proactive insight: %s (%s)",
                            insight.get("kind"), insight.get("severity"))
                try:
                    await shadow.broadcast_insight(insight)
                except Exception as e:
                    logger.warning("broadcast failed: %s", type(e).__name__)

        except asyncio.CancelledError:
            logger.info("Proactive engine stopping")
            raise
        except Exception as e:
            logger.warning("proactive loop error: %s", type(e).__name__)

        await asyncio.sleep(_INTERVAL_S)
