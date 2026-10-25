"""
Cross-ability narrative weaver.

Takes the structured outputs from multiple abilities (each is an
AbilityResult.data dict + ability name) and produces a single woven
narrative — cross-signal themes, contradictions, and a ranked set of
recommended actions.

Doesn't replace the council endpoint's current behavior; it augments it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Map ability → (signal_extractor, recommendation_extractor)
# Each extractor takes the ability's data dict and returns a list of strings.


def _coerce(v) -> Optional[float]:
    if isinstance(v, (int, float)) and v == v:  # NaN check
        return float(v)
    return None


def _signals_from(ability: str, data: Dict[str, Any]) -> List[str]:
    """Return short signal strings derived from an ability's data."""
    out: List[str] = []
    if not isinstance(data, dict):
        return out

    if ability == "tide":
        slope = data.get("slope")
        label = data.get("overall_label")
        if slope and label:
            out.append(f"mood is {label} and {slope}")
        if data.get("longest_negative_streak", 0) >= 5:
            out.append(f"long negative streak ({data['longest_negative_streak']})")
    elif ability == "lantern":
        active = data.get("active") or []
        idle = data.get("idle") or []
        dormant = data.get("dormant") or []
        if dormant:
            out.append(f"{len(dormant)} dormant projects: {', '.join(dormant[:3])}")
        if not active and (idle or dormant):
            out.append("no active projects — context loss risk")
        suggestions = data.get("revisit_suggestions") or []
        for s in suggestions[:1]:
            out.append(f"revisit candidate: {s}")
    elif ability == "compass":
        # Compass returns goal alignment.
        score = _coerce(data.get("alignment_score"))
        trend = data.get("trend")
        if score is not None:
            out.append(f"goal alignment {score:.2f} ({trend or 'unknown'})")
        drifting = data.get("drifting_goals") or []
        for g in drifting[:2]:
            out.append(f"goal drifting: {g}")
    elif ability == "hearth":
        anomalies = data.get("anomalies") or []
        for a in anomalies[:1]:
            out.append(f"location anomaly: {a}")
    elif ability == "time_witness":
        # Categories report.
        totals = data.get("total_minutes") or {}
        if isinstance(totals, dict) and totals:
            top = max(totals.items(), key=lambda kv: _coerce(kv[1]) or 0)
            if _coerce(top[1]):
                out.append(f"top time category: {top[0]} ({int(top[1])}min)")
    elif ability == "focus_sight":
        if data.get("longest_block_min"):
            out.append(f"longest focus block: {int(data['longest_block_min'])}min")
        if data.get("fragmentation_score"):
            f = _coerce(data["fragmentation_score"])
            if f and f > 0.6:
                out.append(f"high fragmentation ({f:.2f})")
    elif ability == "energy_flow":
        peak = data.get("peak_hour")
        trough = data.get("trough_hour")
        if peak is not None and trough is not None:
            out.append(f"energy peak {peak}h / trough {trough}h")
    elif ability == "habit_tracker":
        habits = data.get("habits") or []
        if habits:
            out.append(f"{len(habits)} habits tracked")
    elif ability == "mirror":
        rising = data.get("rising") or []
        for theme in rising[:2]:
            out.append(f"rising theme: {theme}")
        avg = _coerce(data.get("avg_self_sentiment"))
        if avg is not None and avg < -0.2:
            out.append(f"harsh self-talk ({avg:+.2f})")
    elif ability == "pattern_sight":
        biases = data.get("biases") or []
        for b in biases[:2]:
            out.append(f"possible bias: {b}")
        correlations = data.get("correlations") or []
        if correlations:
            out.append(f"{len(correlations)} cross-signal correlations")
    elif ability == "echo":
        if data.get("counterfactual_count", 0) > 0:
            cr = data.get("top_regret")
            if isinstance(cr, dict) and cr.get("delta"):
                out.append(f"top regret: {cr['alt_choice'][:50]} (Δ{cr['delta']:+.1f})")
    elif ability == "social_radar":
        if data.get("manipulation_score"):
            ms = _coerce(data["manipulation_score"])
            if ms and ms > 0.3:
                out.append(f"manipulation signal in messages ({ms:.2f})")
    elif ability == "intent_seer":
        intent = data.get("predicted_action")
        if intent:
            out.append(f"likely next: {intent}")
    elif ability == "connection":
        weakest = data.get("weakest_links") or []
        for w in weakest[:1]:
            if isinstance(w, dict):
                out.append(f"weak connection: {w.get('name', '?')}")
            else:
                out.append(f"weak connection: {w}")
    elif ability == "oracle":
        prob = _coerce(data.get("probability"))
        if prob is not None:
            out.append(f"oracle p={prob:.2f}")

    return out


@dataclass
class NarrativeRecommendation:
    title: str
    rationale: str
    triggering_abilities: List[str] = field(default_factory=list)
    priority: int = 5  # 1 (most urgent) to 10 (gentle nudge)


@dataclass
class Contradiction:
    description: str
    abilities: List[str]


@dataclass
class WovenNarrative:
    summary: str
    signals: List[str]
    recommendations: List[NarrativeRecommendation]
    health_score: float       # 0..1 — rough "overall state" gauge
    themes: List[str]
    contradictions: List[Contradiction] = field(default_factory=list)


# Cross-ability rules. Each rule looks at the combined signal set and emits
# a recommendation when its preconditions are present.
def _build_recommendations(
    ability_data: Dict[str, Dict],
) -> List[NarrativeRecommendation]:
    recs: List[NarrativeRecommendation] = []

    tide = ability_data.get("tide") or {}
    lantern = ability_data.get("lantern") or {}
    compass = ability_data.get("compass") or {}
    mirror = ability_data.get("mirror") or {}
    focus = ability_data.get("focus_sight") or {}
    energy = ability_data.get("energy_flow") or {}
    echo = ability_data.get("echo") or {}
    hearth = ability_data.get("hearth") or {}

    # 1. Falling mood + harsh self-talk → suggest reflection break.
    if tide.get("slope") == "falling" and (_coerce(mirror.get("avg_self_sentiment")) or 0) < -0.1:
        recs.append(NarrativeRecommendation(
            title="Take a reflective break",
            rationale=(
                "Tide shows a falling mood trend while Mirror is picking up "
                "harsher self-talk. Step away from the screen, write down "
                "what's heavy, separate facts from inner critic."
            ),
            triggering_abilities=["tide", "mirror"],
            priority=2,
        ))

    # 2. Dormant projects + low compass alignment → revisit-or-prune.
    dormant = lantern.get("dormant") or []
    alignment = _coerce(compass.get("alignment_score"))
    if len(dormant) >= 3 and (alignment is None or alignment < 0.4):
        recs.append(NarrativeRecommendation(
            title="Audit dormant projects",
            rationale=(
                f"{len(dormant)} projects haven't been touched in 30+ days "
                f"and goal alignment is low ({alignment if alignment is not None else '?'}). "
                "Decide what to revive vs. archive so cognitive load drops."
            ),
            triggering_abilities=["lantern", "compass"],
            priority=3,
        ))

    # 3. High fragmentation + low energy at certain hours → schedule focus blocks.
    frag = _coerce(focus.get("fragmentation_score"))
    if frag and frag > 0.55 and energy.get("peak_hour") is not None:
        recs.append(NarrativeRecommendation(
            title=f"Protect a deep-work block around hour {energy['peak_hour']}",
            rationale=(
                f"Focus Sight reports fragmentation at {frag:.2f} (high) and "
                f"Energy Flow places your peak at {energy['peak_hour']}h. "
                "Carve a no-meeting block there."
            ),
            triggering_abilities=["focus_sight", "energy_flow"],
            priority=4,
        ))

    # 4. Significant echo regret + mood falling → name and re-decide.
    cr = echo.get("top_regret") if isinstance(echo.get("top_regret"), dict) else None
    if cr and (_coerce(cr.get("delta")) or 0) >= 2.0 and tide.get("slope") in ("falling", "stable"):
        recs.append(NarrativeRecommendation(
            title="Reopen one past decision deliberately",
            rationale=(
                f"Echo shows a counterfactual with delta {cr['delta']:+.1f}: "
                f"alt {cr['alt_choice'][:60]}. Consider whether the current "
                "decision can still be reversed or partially undone."
            ),
            triggering_abilities=["echo", "tide"],
            priority=5,
        ))

    # 5. Location anomaly + falling alignment → check whether routine is drifting.
    if (hearth.get("anomalies") and alignment is not None and alignment < 0.5):
        recs.append(NarrativeRecommendation(
            title="Check whether your routine is drifting",
            rationale=(
                "Hearth saw an unusual location-for-time-of-day signal and "
                "Compass alignment is below 0.5. Routine drift often "
                "precedes goal drift."
            ),
            triggering_abilities=["hearth", "compass"],
            priority=6,
        ))

    # 6. Burnout signal: tide falling + mirror anxiety/doubt rising +
    #    compass declining + most projects dormant. Highest priority.
    rising_themes = set(mirror.get("rising") or [])
    burnout_themes = {"anxiety", "doubt", "frustration", "burnout", "stuck", "exhausted"}
    burnout_signals = 0
    if tide.get("slope") == "falling":
        burnout_signals += 1
    if rising_themes & burnout_themes:
        burnout_signals += 1
    if alignment is not None and alignment < 0.35:
        burnout_signals += 1
    if (len(lantern.get("dormant") or []) >= 3
        and len(lantern.get("active") or []) <= 1):
        burnout_signals += 1
    if (_coerce(mirror.get("avg_self_sentiment")) or 0) < -0.25:
        burnout_signals += 1

    if burnout_signals >= 3:
        # Find which themes triggered to surface in the rationale.
        triggering_themes = sorted(rising_themes & burnout_themes)
        triggered_by = []
        if tide.get("slope") == "falling":
            triggered_by.append("falling mood")
        if triggering_themes:
            triggered_by.append("rising " + "/".join(triggering_themes[:2]))
        if alignment is not None and alignment < 0.35:
            triggered_by.append(f"goal alignment {alignment:.2f}")
        if len(lantern.get("dormant") or []) >= 3 and len(lantern.get("active") or []) <= 1:
            triggered_by.append(f"{len(lantern.get('dormant') or [])} dormant projects")

        recs.append(NarrativeRecommendation(
            title="🚨 Burnout risk — pause and reassess",
            rationale=(
                f"{burnout_signals} burnout-correlated signals are co-active: "
                + ", ".join(triggered_by) + ". "
                "This is the cross-signal pattern that historically precedes "
                "extended low periods. Cancel the next non-essential thing on "
                "your calendar today, and write down what's actually heavy."
            ),
            triggering_abilities=[a for a in ["tide", "mirror", "compass", "lantern"]
                                  if a in ability_data],
            priority=1,
        ))

    return recs


def _health_score(ability_data: Dict[str, Dict]) -> float:
    """0..1 gauge from cross-signal cues. Heuristic, not diagnostic."""
    score = 0.5
    tide = ability_data.get("tide") or {}
    mirror = ability_data.get("mirror") or {}
    compass = ability_data.get("compass") or {}
    lantern = ability_data.get("lantern") or {}
    focus = ability_data.get("focus_sight") or {}

    sent = _coerce(tide.get("overall_avg"))
    if sent is not None:
        score += 0.15 * sent  # +0.15 if very positive, -0.15 if very negative
    if tide.get("slope") == "rising":
        score += 0.05
    elif tide.get("slope") == "falling":
        score -= 0.05

    selftalk = _coerce(mirror.get("avg_self_sentiment"))
    if selftalk is not None:
        score += 0.1 * selftalk

    alignment = _coerce(compass.get("alignment_score"))
    if alignment is not None:
        score += 0.1 * (alignment - 0.5)

    if (lantern.get("active") or []):
        score += 0.03

    frag = _coerce(focus.get("fragmentation_score"))
    if frag is not None:
        score -= 0.1 * (frag - 0.5)

    return round(max(0.0, min(1.0, score)), 3)


def _detect_contradictions(ability_data: Dict[str, Dict]) -> List[Contradiction]:
    """Surface pairs of abilities that report opposite signals — these are
    the most informative cross-ability moments."""
    out: List[Contradiction] = []
    tide = ability_data.get("tide") or {}
    mirror = ability_data.get("mirror") or {}
    compass = ability_data.get("compass") or {}
    focus = ability_data.get("focus_sight") or {}
    energy = ability_data.get("energy_flow") or {}

    # 1. Mood rising in tide but compass alignment declining → outward mood
    #    masking inner drift, or vice-versa.
    align_trend = compass.get("trend")
    if tide.get("slope") == "rising" and align_trend == "declining":
        out.append(Contradiction(
            description="Mood is rising but goal alignment is declining — "
                        "feeling better while drifting from what matters.",
            abilities=["tide", "compass"],
        ))
    if tide.get("slope") == "falling" and align_trend == "rising":
        out.append(Contradiction(
            description="Mood is falling but goal alignment is rising — "
                        "doing the right work, but it's costing you energy.",
            abilities=["tide", "compass"],
        ))

    # 2. Mirror reports constructive self-talk but tide is very_negative —
    #    your inner narrator is kinder than your day-to-day shows.
    selftalk = _coerce(mirror.get("avg_self_sentiment"))
    if selftalk is not None and selftalk > 0.2 and tide.get("overall_label") == "very_negative":
        out.append(Contradiction(
            description="Self-talk reads constructive but overall mood reads "
                        "very negative — the dissonance might be worth a journal entry.",
            abilities=["mirror", "tide"],
        ))

    # 3. Focus Sight reports flow but Energy Flow says trough.
    state = focus.get("current_state")
    cur_hour = energy.get("trough_hour")
    if (state == "in_flow" and cur_hour is not None
        and isinstance(cur_hour, (int, float))):
        # Compare to "now" hour if available.
        now_h = energy.get("current_hour")
        if isinstance(now_h, (int, float)) and abs(now_h - cur_hour) < 1:
            out.append(Contradiction(
                description=f"You're in-flow at your energy trough (hour {int(cur_hour)}) — "
                            "rare and worth protecting, but watch for crash later.",
                abilities=["focus_sight", "energy_flow"],
            ))

    return out


def _detect_themes(signals: List[str]) -> List[str]:
    """Pick out recurring tokens across signal strings."""
    tokens: Counter = Counter()
    for s in signals:
        for tok in s.lower().split():
            if len(tok) >= 4 and tok.isalpha():
                tokens[tok] += 1
    return [t for t, n in tokens.most_common(5) if n >= 2]


def weave(ability_outputs: Dict[str, Dict]) -> WovenNarrative:
    """ability_outputs: {ability_name: data_dict}. Returns a WovenNarrative."""
    signals: List[str] = []
    for ab, data in ability_outputs.items():
        signals.extend(f"[{ab}] {s}" for s in _signals_from(ab, data))

    recommendations = _build_recommendations(ability_outputs)
    health = _health_score(ability_outputs)
    themes = _detect_themes(signals)
    contradictions = _detect_contradictions(ability_outputs)

    if not signals:
        summary = "No usable signals were extracted from the consulted abilities."
    else:
        parts = [f"{len(signals)} cross-ability signals from {len(ability_outputs)} abilities."]
        parts.append(f"Health gauge: {health:.2f}/1.0.")
        if themes:
            parts.append(f"Recurring threads: {', '.join(themes[:3])}.")
        if contradictions:
            parts.append(f"{len(contradictions)} cross-signal contradiction(s).")
        if recommendations:
            parts.append(f"{len(recommendations)} cross-signal recommendations.")
        summary = " ".join(parts)

    # Sort recommendations by priority (lower = more urgent).
    recommendations.sort(key=lambda r: r.priority)

    return WovenNarrative(
        summary=summary,
        signals=signals,
        recommendations=recommendations,
        health_score=health,
        themes=themes,
        contradictions=contradictions,
    )
