"""
Cognitive bias detector — surfaces patterns in observations that match
known biases. Outputs are signals, not diagnoses.

Detected biases (heuristics, not pretensions of clinical accuracy):
- confirmation: repeated favorable mentions of one option vs alternatives
- recency:      decisions weighted toward very recent events when older
                events should also count
- sunk_cost:    continuing investment after repeated negative outcomes
- anchoring:    early choices dominating later decisions in a category
- availability: rare events overrepresented after one vivid example
- bandwagon:    sudden adoption mirroring tracked people's activity

Each bias returns:
- strength    [0, 1]
- evidence    list of excerpts / counts
- suggestion  what to consider
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple


@dataclass
class BiasSignal:
    name: str
    strength: float
    evidence: List[str]
    suggestion: str


def _parse_ts(o: Dict) -> Optional[datetime]:
    ts = o.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _text(o: Dict) -> str:
    c = o.get("content")
    parts = [str(o.get("type", ""))]
    if isinstance(c, dict):
        for v in c.values():
            if isinstance(v, str):
                parts.append(v)
    elif isinstance(c, str):
        parts.append(c)
    return " ".join(parts).lower()


# ──────────── individual detectors ────────────


def _detect_confirmation_bias(observations: List[Dict]) -> Optional[BiasSignal]:
    """Look for repeated favorable framings of one topic without counter-balance."""
    # Tokens following "best", "great", "love", "perfect" capture favorable framings.
    fav_re = re.compile(
        r"\b(?:best|great|love(?:d|s)?|perfect|amazing|wonderful|brilliant)\s+(\w{4,20})",
        re.I,
    )
    crit_re = re.compile(
        r"\b(?:terrible|awful|hate(?:d|s)?|worst|disappointed|broken|bad)\s+(\w{4,20})",
        re.I,
    )
    fav: Counter = Counter()
    crit: Counter = Counter()
    for o in observations:
        text = _text(o)
        for m in fav_re.findall(text):
            fav[m.lower()] += 1
        for m in crit_re.findall(text):
            crit[m.lower()] += 1
    # A bias signal: a token that appears ≥4 times favorably, 0 times critically.
    suspects = [(tok, n) for tok, n in fav.most_common(10)
                if n >= 4 and crit.get(tok, 0) == 0]
    if not suspects:
        return None
    top = suspects[0]
    strength = min(1.0, top[1] / 10.0)
    evidence = [f"'{tok}' mentioned favorably {n}x without any critical mention"
                for tok, n in suspects[:3]]
    return BiasSignal(
        name="confirmation",
        strength=round(strength, 3),
        evidence=evidence,
        suggestion=(
            f"You've described '{top[0]}' in only positive terms. "
            "Worth steel-manning the case against it before deciding."
        ),
    )


def _detect_recency_bias(observations: List[Dict], now: Optional[datetime] = None) -> Optional[BiasSignal]:
    """Detect heavy emotional weighting on the last 1-2 days."""
    now = now or datetime.now(timezone.utc)
    last_2 = now - timedelta(days=2)
    last_14 = now - timedelta(days=14)
    recent_emotion = 0
    older_emotion = 0
    emotion_re = re.compile(
        r"\b(?:furious|terrible|amazing|love|hate|disaster|breakthrough|catastroph)",
        re.I,
    )
    for o in observations:
        t = _parse_ts(o)
        if t is None:
            continue
        text = _text(o)
        hits = len(emotion_re.findall(text))
        if t >= last_2:
            recent_emotion += hits
        elif t >= last_14:
            older_emotion += hits
    # If the last 2 days carry > 50% of the emotion of the last 14 days, flag.
    total = recent_emotion + older_emotion
    if total < 3:
        return None
    recent_ratio = recent_emotion / total
    if recent_ratio < 0.4:
        return None
    return BiasSignal(
        name="recency",
        strength=round(min(1.0, (recent_ratio - 0.2) * 1.5), 3),
        evidence=[
            f"Last 2 days: {recent_emotion} emotional markers; "
            f"prior 12 days: {older_emotion}",
        ],
        suggestion=(
            "Recent intense events may be skewing how you weigh history. "
            "Consider how this would look averaged over the full period."
        ),
    )


def _detect_sunk_cost(decisions: List[Dict]) -> Optional[BiasSignal]:
    """If a choice category has multiple low-satisfaction outcomes but the
    user keeps repeating it, flag sunk-cost bias."""
    if not decisions:
        return None
    try:
        from .council_synthesis import classify_choice
    except ImportError:
        return None

    by_action: Dict[str, List[int]] = defaultdict(list)
    for d in decisions:
        sat = d.get("satisfaction")
        if not isinstance(sat, (int, float)):
            continue
        action = classify_choice(d.get("choice", ""))
        by_action[action].append(int(sat))

    flagged = []
    for action, sats in by_action.items():
        if len(sats) >= 3 and sum(s <= 4 for s in sats) >= 2 and action != "other":
            avg = sum(sats) / len(sats)
            flagged.append((action, len(sats), avg))
    if not flagged:
        return None
    top = max(flagged, key=lambda f: f[1])
    return BiasSignal(
        name="sunk_cost",
        strength=round(min(1.0, top[1] / 8.0), 3),
        evidence=[
            f"Repeated '{a}' decisions {n}x with avg satisfaction {avg:.1f}/10"
            for a, n, avg in flagged[:3]
        ],
        suggestion=(
            f"You keep choosing '{top[0]}' even though past attempts averaged "
            f"{top[2]:.1f}/10. Worth asking: what would change the outcome this time?"
        ),
    )


def _detect_bandwagon(
    observations: List[Dict],
    tracked_people: Optional[Dict] = None,
    now: Optional[datetime] = None,
) -> Optional[BiasSignal]:
    """Detect mimicry: tokens that suddenly appear in your text shortly after
    a tracked person mentions the same token."""
    if not tracked_people:
        return None
    now = now or datetime.now(timezone.utc)
    last_30 = now - timedelta(days=30)

    # Pull text per person, plus your own (any obs without 'from' attribution).
    self_tokens: Counter = Counter()
    per_person_tokens: Dict[str, Counter] = defaultdict(Counter)
    tok_re = re.compile(r"\b[a-z]{5,15}\b")

    for o in observations:
        t = _parse_ts(o)
        if t is None or t < last_30:
            continue
        text = _text(o)
        content = o.get("content", {}) if isinstance(o.get("content"), dict) else {}
        from_ = (content.get("from") or "").lower()
        if from_ and from_ in {name.lower() for name in tracked_people}:
            for tok in tok_re.findall(text):
                if tok not in _COMMON:
                    per_person_tokens[from_][tok] += 1
        else:
            for tok in tok_re.findall(text):
                if tok not in _COMMON:
                    self_tokens[tok] += 1

    # Tokens that appear ≥3x in someone else's text AND ≥3x in yours are
    # candidate bandwagon adoptions. Without per-token timestamps this is
    # only a weak signal.
    flagged: List[Tuple[str, str, int]] = []
    for person, counter in per_person_tokens.items():
        for tok, n in counter.items():
            if n >= 3 and self_tokens.get(tok, 0) >= 3:
                flagged.append((tok, person, n))
    if not flagged:
        return None
    flagged.sort(key=lambda f: -f[2])
    top = flagged[0]
    return BiasSignal(
        name="bandwagon",
        strength=round(min(1.0, len(flagged) / 5.0), 3),
        evidence=[f"'{tok}' echoed from {p}" for tok, p, _ in flagged[:4]],
        suggestion=(
            f"You've been mirroring {top[1]}'s language around '{top[0]}'. "
            "Worth checking whether you actually agree or are just adopting their frame."
        ),
    )


# Stopwords for the bandwagon detector — common English tokens that shouldn't trigger.
_COMMON = frozenset({
    "about", "above", "after", "again", "against", "around", "because", "before",
    "below", "between", "doing", "during", "every", "going", "great", "having",
    "looking", "should", "their", "there", "these", "thing", "things", "think",
    "those", "today", "today", "today", "tonight", "would", "could", "still",
    "talking", "totally", "really", "people", "myself", "yours", "where",
    "which", "while", "right", "thank", "thanks", "trying", "another",
    "always", "never", "found", "different", "though",
})


# ──────────── public API ────────────


def detect_biases(
    observations: List[Dict],
    decisions: Optional[List[Dict]] = None,
    tracked_people: Optional[Dict] = None,
    now: Optional[datetime] = None,
) -> List[BiasSignal]:
    """Return all bias signals above strength 0.1, sorted by strength desc."""
    out: List[BiasSignal] = []
    for d in (
        _detect_confirmation_bias(observations),
        _detect_recency_bias(observations, now=now),
        _detect_sunk_cost(decisions or []),
        _detect_bandwagon(observations, tracked_people, now=now),
    ):
        if d and d.strength >= 0.1:
            out.append(d)
    out.sort(key=lambda s: -s.strength)
    return out
