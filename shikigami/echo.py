"""
Echo — Counterfactual ("what if") analysis.

Looks at recent decision points and surfaces alternative paths drawn from
the user's own decision history. Different from Shadow Council in that
Echo speaks in past tense ("what would have happened") rather than future
("what should I do") — it's the reflective sibling.

Strategy:
1. Identify recent decisions from a) DecisionTracker, b) observation
   types that act like decisions (commit messages, calendar invites,
   accept/reject patterns in messages).
2. For each, find past decisions in similar situations that had different
   choices.
3. Compute counterfactual deltas: avg satisfaction of alternative choices
   vs. the chosen one, sample sizes for both sides.
4. Rank by "regret pressure" — magnitude of satisfaction delta × sample
   count of the alternative.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional


_DECISION_PATTERNS = [
    re.compile(r"\b(accepted|rejected|declined|chose|decided|picked|skipped|cancell?ed|approved|denied)\b", re.I),
    re.compile(r"\b(committed|merged|reverted|rolled back|pushed)\b", re.I),
    re.compile(r"\bgit (commit|revert|reset|merge|push|rebase)\b", re.I),
]


@dataclass
class DecisionPoint:
    """A moment where a meaningful choice was made."""
    timestamp: str
    situation: str
    choice: str
    satisfaction: Optional[float] = None
    source: str = "tracker"  # tracker / observation
    tags: List[str] = field(default_factory=list)


@dataclass
class Counterfactual:
    decision: DecisionPoint
    alt_choice: str
    chosen_avg_sat: float
    alt_avg_sat: float
    chosen_sample_n: int
    alt_sample_n: int
    delta: float
    regret_pressure: float
    narrative: str


@dataclass
class EchoReport:
    decisions_seen: int
    counterfactuals: List[Counterfactual]
    top_regret: Optional[Counterfactual]
    top_relief: Optional[Counterfactual]
    summary: str


def _looks_like_decision(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _DECISION_PATTERNS)


def _parse_ts(ts: Any) -> Optional[datetime]:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _bag_of_words(text: str) -> set:
    return set(re.findall(r"[A-Za-z']{3,}", (text or "").lower()))


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _extract_decisions_from_observations(
    observations: List[Dict], lookback_days: int = 30,
) -> List[DecisionPoint]:
    """Pull decision-like observations out of recent history."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    out: List[DecisionPoint] = []
    for o in observations:
        t = _parse_ts(o.get("timestamp"))
        if t is None or t < cutoff:
            continue
        c = o.get("content")
        if not isinstance(c, dict):
            continue
        # Look in any text-bearing field.
        text = ""
        for k in ("text", "body", "message", "command", "title", "subject"):
            v = c.get(k)
            if isinstance(v, str):
                text = v
                break
        if not _looks_like_decision(text):
            continue
        out.append(DecisionPoint(
            timestamp=t.isoformat(),
            situation=text[:300],
            choice=text[:120],
            satisfaction=None,
            source="observation",
        ))
    return out


def _gather_tracker_decisions(decision_tracker) -> List[DecisionPoint]:
    if decision_tracker is None:
        return []
    out: List[DecisionPoint] = []
    decisions = getattr(decision_tracker, "decisions", None) or []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        out.append(DecisionPoint(
            timestamp=d.get("timestamp", ""),
            situation=str(d.get("situation", ""))[:300],
            choice=str(d.get("choice", ""))[:120],
            satisfaction=(float(d["satisfaction"])
                          if d.get("satisfaction") is not None else None),
            source="tracker",
            tags=list(d.get("tags") or []),
        ))
    return out


def _find_alternatives(
    decision: DecisionPoint, history: List[DecisionPoint],
) -> Dict[str, List[DecisionPoint]]:
    """Group past decisions by choice for situations similar to this one."""
    target = _bag_of_words(decision.situation)
    target_tags = set(decision.tags)
    groups: Dict[str, List[DecisionPoint]] = defaultdict(list)
    for h in history:
        if h.timestamp >= decision.timestamp:
            continue  # only look backwards
        if h.satisfaction is None:
            continue
        sim_text = _overlap(target, _bag_of_words(h.situation))
        sim_tags = (len(target_tags & set(h.tags)) / max(1, len(target_tags | set(h.tags)))
                    if (target_tags or h.tags) else 0.0)
        score = max(sim_text, sim_tags)
        if score < 0.15:
            continue
        groups[h.choice or "unknown"].append(h)
    return groups


def compute_echo(
    observations: List[Dict],
    decision_tracker=None,
    lookback_days: int = 30,
) -> EchoReport:
    history = _gather_tracker_decisions(decision_tracker)
    recent_observation_decisions = _extract_decisions_from_observations(
        observations, lookback_days=lookback_days,
    )
    # Combine all decisions; recent ones get counterfactuals run on them.
    all_decisions = history + recent_observation_decisions
    all_decisions.sort(key=lambda d: d.timestamp)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    recent = [d for d in all_decisions if d.timestamp >= cutoff]

    counterfactuals: List[Counterfactual] = []
    for dec in recent:
        groups = _find_alternatives(dec, history)
        if not groups:
            continue
        chosen_group = groups.get(dec.choice, [])
        chosen_sats = [d.satisfaction for d in chosen_group if d.satisfaction is not None]
        chosen_avg = sum(chosen_sats) / len(chosen_sats) if chosen_sats else (dec.satisfaction or 5.0)

        for alt_choice, alt_group in groups.items():
            if alt_choice == dec.choice:
                continue
            alt_sats = [d.satisfaction for d in alt_group if d.satisfaction is not None]
            if not alt_sats:
                continue
            alt_avg = sum(alt_sats) / len(alt_sats)
            delta = alt_avg - chosen_avg
            regret_pressure = abs(delta) * (len(alt_sats) ** 0.5)
            narrative = (
                f"If you had chosen \"{alt_choice[:60]}\" instead "
                f"(seen {len(alt_sats)}× before, avg satisfaction {alt_avg:.1f}/10) — "
                f"vs your actual choice \"{dec.choice[:60]}\" "
                f"(avg {chosen_avg:.1f}/10), the historical delta is {delta:+.1f}."
            )
            counterfactuals.append(Counterfactual(
                decision=dec,
                alt_choice=alt_choice,
                chosen_avg_sat=round(chosen_avg, 2),
                alt_avg_sat=round(alt_avg, 2),
                chosen_sample_n=len(chosen_sats),
                alt_sample_n=len(alt_sats),
                delta=round(delta, 3),
                regret_pressure=round(regret_pressure, 3),
                narrative=narrative,
            ))

    counterfactuals.sort(key=lambda c: -c.regret_pressure)
    top_regret = next((c for c in counterfactuals if c.delta > 0), None)
    top_relief = next((c for c in counterfactuals if c.delta < 0), None)

    if not counterfactuals:
        summary = "No counterfactuals available — need more decision history with satisfaction scores."
    else:
        parts = [f"{len(counterfactuals)} counterfactuals from {len(recent)} recent decisions."]
        if top_regret:
            parts.append(f"Biggest regret pressure: {top_regret.alt_choice[:50]} (Δ{top_regret.delta:+.1f}).")
        if top_relief:
            parts.append(f"Biggest relief: avoided {top_relief.alt_choice[:50]} (Δ{top_relief.delta:+.1f}).")
        summary = " ".join(parts)

    return EchoReport(
        decisions_seen=len(all_decisions),
        counterfactuals=counterfactuals[:20],
        top_regret=top_regret,
        top_relief=top_relief,
        summary=summary,
    )
