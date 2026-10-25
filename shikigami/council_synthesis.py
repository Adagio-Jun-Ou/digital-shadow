"""
Shadow Council cross-decision synthesis.

Goes beyond "list past decisions" to extract the *pattern of choices that work
for you*: classifies choices into action categories, weighs recent decisions
more heavily, surfaces success differentials, and flags counterexamples.

Input: list of decision dicts with keys
    timestamp, situation, choice, outcome, satisfaction (1-10), tags
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# Action-type taxonomy. Each category has tokens / phrases that signal it.
# A choice is classified by accumulating per-category hits.
_ACTION_CATEGORIES: Dict[str, List[re.Pattern]] = {
    "accept": [
        re.compile(r"\b(accept\w*|agree\w*|approv\w+|said? yes|went ahead|commit\w*|sign\w*|take it|took it)\b", re.I),
    ],
    "negotiate": [
        re.compile(r"\b(negotiat\w*|counter[\s-]?propos\w*|counter[\s-]?offer\w*|push\w* back|haggl\w+|bargain\w*|propos\w+|ask\w* for more)\b", re.I),
    ],
    "decline": [
        re.compile(r"\b(declin\w+|reject\w*|refus\w+|said? no|turn\w* down|pass\w* on|opt\w* out)\b", re.I),
    ],
    "wait": [
        re.compile(r"\b(wait\w*|delay\w*|defer\w*|postpon\w+|held? off|hold off|sleep\w* on|think\w* about|paused?)\b", re.I),
    ],
    "consult": [
        re.compile(r"\b(consult\w*|ask\w+ (?:them|him|her|someone)|check\w* with|talk\w* to|sought? advice|second opinion)\b", re.I),
    ],
    "research": [
        re.compile(r"\b(research\w*|investigat\w+|stud\w+|analyz\w+|review\w*|explor\w+|look\w+ into|read up)\b", re.I),
    ],
    "act_quickly": [
        re.compile(r"\b(rush\w*|hurr\w+|jump\w*|immediately|right away|on the spot|impulsiv\w*)\b", re.I),
    ],
    "ignore": [
        re.compile(r"\b(ignor\w+|skip\w*|drop\w*|abandon\w*|let go|forg\w+ about)\b", re.I),
    ],
    "compromise": [
        re.compile(r"\b(compromis\w+|meet halfway|middle ground|split the difference|met them halfway)\b", re.I),
    ],
    "delegate": [
        re.compile(r"\b(delegat\w+|hand\w* off|pass\w* to|let \w+ handle|outsourc\w+)\b", re.I),
    ],
}

# Half-life for recency weighting, in days. A 90-day-old decision counts about
# half as much as today's; this is fast enough that the council stays current
# but slow enough that decisions from last quarter still matter.
_RECENCY_HALFLIFE_DAYS = 90.0


def classify_choice(choice_text: str) -> str:
    """Return the action category that best fits `choice_text`, or 'other'."""
    if not choice_text:
        return "other"
    scores: Dict[str, int] = {}
    for category, patterns in _ACTION_CATEGORIES.items():
        for p in patterns:
            scores[category] = scores.get(category, 0) + len(p.findall(choice_text))
    if not scores or max(scores.values()) == 0:
        return "other"
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _recency_weight(timestamp_str: Optional[str], now: datetime) -> float:
    """Exponential decay weight in (0, 1] for an ISO-8601 timestamp."""
    if not timestamp_str:
        return 0.5
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0.5
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / _RECENCY_HALFLIFE_DAYS)


def _split_decisions_by_time(decs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split into older-half / recent-half (preserves order)."""
    if len(decs) < 4:
        return [], decs[:]
    mid = len(decs) // 2
    sorted_decs = sorted(decs, key=lambda d: d.get("timestamp", ""))
    return sorted_decs[:mid], sorted_decs[mid:]


def synthesize(
    decisions: List[Dict],
    situation: str,
    similar: Optional[List[Dict]] = None,
) -> Dict:
    """
    Build the council's verdict.

    Returns a dict with:
        action_stats: per-action {n, weighted_n, avg_satisfaction, success_rate, sample}
        recommendation: {action, confidence, lift, reasoning}
        trend: {category: {old_avg, recent_avg, direction}}
        counterfactual: closest past decision (warning or affirming case)
    """
    if not decisions:
        return {
            "action_stats": {},
            "recommendation": {"action": None, "confidence": 0.0,
                               "lift": 0.0, "reasoning": "no past decisions tracked"},
            "trend": {},
            "counterfactual": None,
        }

    now = datetime.now(timezone.utc)
    similar = similar or []
    similar_ids = {d.get("id") for d in similar}

    # If we have similar decisions, prioritize them — but still pull global
    # action-stats from the full history so rare categories aren't punished
    # for not appearing in the similar set.
    pool = similar if similar else decisions

    action_stats: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0, "weighted_n": 0.0,
        "satisfactions": [], "weighted_satisfaction_sum": 0.0,
        "samples": [],
    })

    for dec in pool:
        choice = dec.get("choice", "")
        category = classify_choice(choice)
        sat = dec.get("satisfaction")
        if sat is None:
            continue
        weight = _recency_weight(dec.get("timestamp"), now)
        # If the decision is in the "similar" set, give it an extra boost.
        if dec.get("id") in similar_ids:
            weight *= 1.5

        stats = action_stats[category]
        stats["n"] += 1
        stats["weighted_n"] += weight
        stats["satisfactions"].append(sat)
        stats["weighted_satisfaction_sum"] += weight * sat
        if len(stats["samples"]) < 3:
            stats["samples"].append({
                "timestamp": dec.get("timestamp", "")[:10],
                "situation": (dec.get("situation") or "")[:80],
                "choice": choice[:80],
                "satisfaction": sat,
            })

    # Compute summary metrics.
    summary: Dict[str, Dict] = {}
    global_satisfactions = []
    for category, stats in action_stats.items():
        if stats["weighted_n"] == 0:
            continue
        avg = stats["weighted_satisfaction_sum"] / stats["weighted_n"]
        success_rate = sum(1 for s in stats["satisfactions"] if s >= 7) / len(stats["satisfactions"])
        summary[category] = {
            "n": stats["n"],
            "weighted_n": round(stats["weighted_n"], 3),
            "avg_satisfaction": round(avg, 2),
            "success_rate": round(success_rate, 3),
            "samples": stats["samples"],
        }
        global_satisfactions.extend(stats["satisfactions"])

    # Recommendation: action with highest weighted satisfaction *and* enough
    # support to be more than a single data point.
    recommendation = {"action": None, "confidence": 0.0, "lift": 0.0,
                      "reasoning": "insufficient data"}
    if summary:
        # Wilson-style confidence: shrink toward global mean when n is small.
        global_mean = (sum(global_satisfactions) / len(global_satisfactions)
                       if global_satisfactions else 5.0)
        ranked = []
        for category, stats in summary.items():
            n = stats["weighted_n"]
            # Bayesian shrinkage toward global mean — strong support pulls
            # less, weak support pulls harder.
            shrunk = (stats["avg_satisfaction"] * n + global_mean * 3) / (n + 3)
            ranked.append((category, shrunk, stats))
        ranked.sort(key=lambda x: -x[1])
        winner_cat, winner_score, winner_stats = ranked[0]
        lift = winner_score - global_mean
        confidence = min(0.95, winner_stats["weighted_n"] / 5.0)
        recommendation = {
            "action": winner_cat,
            "confidence": round(confidence, 3),
            "lift": round(lift, 2),
            "reasoning": (
                f"{winner_stats['n']} past decisions of type '{winner_cat}' "
                f"averaged {winner_stats['avg_satisfaction']}/10 satisfaction "
                f"({winner_stats['success_rate']*100:.0f}% success rate); "
                f"+{lift:.1f} over your overall baseline of {global_mean:.1f}"
            ),
        }

    # Trend: did each action category get better or worse?
    older, recent = _split_decisions_by_time(decisions)
    trend: Dict[str, Dict] = {}
    if older and recent:
        for category in summary.keys():
            old_sats = [d.get("satisfaction", 5) for d in older
                        if classify_choice(d.get("choice", "")) == category
                        and d.get("satisfaction") is not None]
            new_sats = [d.get("satisfaction", 5) for d in recent
                        if classify_choice(d.get("choice", "")) == category
                        and d.get("satisfaction") is not None]
            if not old_sats or not new_sats:
                continue
            old_avg = sum(old_sats) / len(old_sats)
            new_avg = sum(new_sats) / len(new_sats)
            delta = new_avg - old_avg
            direction = "improving" if delta > 0.5 else "declining" if delta < -0.5 else "stable"
            trend[category] = {
                "old_avg": round(old_avg, 2),
                "recent_avg": round(new_avg, 2),
                "delta": round(delta, 2),
                "direction": direction,
            }

    # Counterfactual: the most similar past decision (worst-case or best-case
    # highlight — whichever is more informative).
    counterfactual = None
    if similar:
        ranked_similar = sorted(
            similar,
            key=lambda d: abs((d.get("satisfaction") or 5) - 5),
            reverse=True,
        )
        top = ranked_similar[0]
        sat = top.get("satisfaction") or 5
        polarity = "affirming" if sat >= 7 else "warning" if sat <= 4 else "neutral"
        counterfactual = {
            "polarity": polarity,
            "timestamp": (top.get("timestamp") or "")[:10],
            "situation": (top.get("situation") or "")[:120],
            "choice": (top.get("choice") or "")[:120],
            "outcome": (top.get("outcome") or "")[:200],
            "satisfaction": sat,
        }

    return {
        "action_stats": summary,
        "recommendation": recommendation,
        "trend": trend,
        "counterfactual": counterfactual,
    }
