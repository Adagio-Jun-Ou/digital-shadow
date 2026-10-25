"""
Lightweight sentiment + manipulation analysis for Social Radar.

Lexicon-based (VADER-style) — no external models, no network calls. Catches
the patterns the README promises: emotional tone, urgency markers, pressure
language, guilt trips, false-urgency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


# Compact AFINN-style lexicon. Score is intensity in [-3, +3].
# Curated to cover common workplace / personal communication tone.
_LEXICON: Dict[str, int] = {
    # positive
    "love": 3, "amazing": 3, "excellent": 3, "wonderful": 3, "brilliant": 3,
    "fantastic": 3, "outstanding": 3, "superb": 3, "perfect": 3, "awesome": 3,
    "great": 2, "good": 2, "happy": 2, "glad": 2, "pleased": 2, "thanks": 2,
    "appreciate": 2, "grateful": 2, "support": 2, "agree": 2, "yes": 1,
    "okay": 1, "ok": 1, "fine": 1, "nice": 1, "sure": 1, "helpful": 2,
    # negative
    "hate": -3, "terrible": -3, "awful": -3, "horrible": -3, "disaster": -3,
    "furious": -3, "outraged": -3, "betrayed": -3, "useless": -3,
    "bad": -2, "angry": -2, "frustrated": -2, "annoyed": -2, "upset": -2,
    "disappointed": -2, "concerned": -2, "worried": -2, "stressed": -2,
    "issue": -1, "problem": -1, "concern": -1, "wrong": -1, "fail": -2,
    "broken": -2, "unhappy": -2, "sorry": -1, "no": -1, "never": -1,
    "delayed": -1, "blocked": -1, "stuck": -1,
}

# Intensifiers multiply the absolute score of the next sentiment-bearing word.
_INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "extremely": 1.8, "absolutely": 1.7,
    "completely": 1.6, "totally": 1.6, "utterly": 1.8, "super": 1.4,
    "incredibly": 1.7, "deeply": 1.5,
}
_DOWNTONERS = {
    "slightly": 0.6, "somewhat": 0.7, "a bit": 0.7, "kinda": 0.7,
    "sort of": 0.7, "rather": 0.85,
}
_NEGATORS = {"not", "no", "never", "n't", "without", "neither", "nor", "none"}

_TOKEN_RE = re.compile(r"[A-Za-z']+")

# Manipulation pattern catalog — each entry is (label, compiled regex, weight).
# Weight is "how strongly this single hit pushes the manipulation score" — the
# detector accumulates and saturates so a single phrase isn't a smoking gun.
_MANIPULATION_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    ("urgency", re.compile(r"\b(asap|right now|immediately|urgent|emergency|drop everything)\b", re.I), 0.25),
    ("urgency", re.compile(r"\b(today only|deadline|hurry|no time|quickly)\b", re.I), 0.15),
    ("pressure", re.compile(r"\b(you must|you have to|you need to|you should|no choice)\b", re.I), 0.20),
    ("pressure", re.compile(r"\b(or else|otherwise|don'?t make me|don'?t let me down)\b", re.I), 0.25),
    ("guilt", re.compile(r"\b(after all I'?ve done|owe me|do this for me|disappointed in you)\b", re.I), 0.30),
    ("guilt", re.compile(r"\b(if you really cared|if you loved|i thought you)\b", re.I), 0.25),
    ("false_dichotomy", re.compile(r"\b(either you|you'?re (with|against)|only two options|no other way)\b", re.I), 0.20),
    ("flattery_baiting", re.compile(r"\b(you'?re the only one|nobody else can|you'?re the best|i need you specifically)\b", re.I), 0.15),
    ("scarcity", re.compile(r"\b(last chance|won'?t happen again|once in a lifetime|limited time)\b", re.I), 0.15),
    ("authority", re.compile(r"\b(everyone agrees|the team thinks|management said|the boss wants)\b", re.I), 0.10),
    ("isolation", re.compile(r"\b(don'?t tell anyone|just between us|keep this quiet|our little secret)\b", re.I), 0.30),
]

# Heavy punctuation amplifies emotional intensity, mirroring VADER's behavior.
_EXCLAIM_RE = re.compile(r"!{2,}")
_CAPS_RE = re.compile(r"\b[A-Z]{4,}\b")


@dataclass
class SentimentResult:
    score: float           # in [-1, 1]
    label: str             # "very_negative" .. "very_positive"
    positive_terms: List[str]
    negative_terms: List[str]
    intensity: float       # absolute magnitude before clipping


@dataclass
class ManipulationResult:
    score: float           # in [0, 1]
    tactics: Dict[str, List[str]]  # label -> matched phrases
    explanation: str


def _label_for(score: float) -> str:
    if score >= 0.5:
        return "very_positive"
    if score >= 0.15:
        return "positive"
    if score <= -0.5:
        return "very_negative"
    if score <= -0.15:
        return "negative"
    return "neutral"


def analyze_sentiment(text: str) -> SentimentResult:
    """Score sentiment in [-1, 1] with positive/negative term extraction."""
    if not text:
        return SentimentResult(0.0, "neutral", [], [], 0.0)

    raw = text
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)

    score = 0.0
    positive_terms: List[str] = []
    negative_terms: List[str] = []

    for i, tok in enumerate(tokens):
        base = _LEXICON.get(tok)
        if base is None:
            continue

        # Look back up to 2 tokens for modifiers.
        multiplier = 1.0
        negated = False
        for j in (1, 2):
            if i - j < 0:
                break
            prev = tokens[i - j]
            if prev in _NEGATORS or prev.endswith("n't"):
                negated = True
            elif prev in _INTENSIFIERS:
                multiplier *= _INTENSIFIERS[prev]
            elif prev in _DOWNTONERS:
                multiplier *= _DOWNTONERS[prev]

        val = base * multiplier
        if negated:
            val = -val * 0.7  # negation softens magnitude rather than mirroring

        score += val
        if val > 0:
            positive_terms.append(tok)
        elif val < 0:
            negative_terms.append(tok)

    # Punctuation / shout amplification.
    score *= 1.0 + 0.1 * len(_EXCLAIM_RE.findall(raw))
    score *= 1.0 + 0.05 * len(_CAPS_RE.findall(raw))

    intensity = abs(score)
    # Squash into [-1, 1] with a smooth diminishing-returns curve.
    normalized = score / (abs(score) + 4.0) if score else 0.0
    normalized = max(-1.0, min(1.0, normalized))

    return SentimentResult(
        score=round(normalized, 4),
        label=_label_for(normalized),
        positive_terms=positive_terms[:10],
        negative_terms=negative_terms[:10],
        intensity=round(intensity, 3),
    )


def detect_manipulation(text: str) -> ManipulationResult:
    """Detect manipulation tactics; score saturates at 1.0."""
    if not text:
        return ManipulationResult(0.0, {}, "no text")

    raw_score = 0.0
    tactics: Dict[str, List[str]] = {}
    for label, pattern, weight in _MANIPULATION_PATTERNS:
        for m in pattern.finditer(text):
            phrase = m.group(0)
            tactics.setdefault(label, []).append(phrase)
            raw_score += weight

    # Saturating curve so one strong pattern can't max the score, but several
    # weaker patterns add up convincingly.
    score = 1.0 - 0.5 ** raw_score if raw_score > 0 else 0.0
    score = round(min(1.0, score), 4)

    if not tactics:
        explanation = "no manipulation patterns detected"
    else:
        top = sorted(tactics.items(), key=lambda kv: -len(kv[1]))[:3]
        explanation = "; ".join(f"{lbl}: {phrases[0]!r}" for lbl, phrases in top)

    return ManipulationResult(score=score, tactics=tactics, explanation=explanation)


def analyze_messages(messages: Iterable[str]) -> Dict[str, object]:
    """Aggregate sentiment + manipulation across multiple messages."""
    msgs = [m for m in messages if m and isinstance(m, str)]
    if not msgs:
        return {
            "count": 0, "avg_sentiment": 0.0, "sentiment_label": "neutral",
            "sentiment_trend": "flat", "manipulation_score": 0.0,
            "manipulation_tactics": {}, "examples": [],
        }

    sentiments = [analyze_sentiment(m) for m in msgs]
    manipulations = [detect_manipulation(m) for m in msgs]

    avg_sent = sum(s.score for s in sentiments) / len(sentiments)
    # Trend: compare second-half mean to first-half mean.
    if len(sentiments) >= 4:
        mid = len(sentiments) // 2
        delta = (sum(s.score for s in sentiments[mid:]) / (len(sentiments) - mid)
                 - sum(s.score for s in sentiments[:mid]) / mid)
        if delta > 0.1:
            trend = "improving"
        elif delta < -0.1:
            trend = "declining"
        else:
            trend = "flat"
    else:
        trend = "insufficient_data"

    # Aggregate manipulation tactics across all messages.
    all_tactics: Dict[str, List[str]] = {}
    for mr in manipulations:
        for label, phrases in mr.tactics.items():
            all_tactics.setdefault(label, []).extend(phrases)

    avg_manip = sum(m.score for m in manipulations) / len(manipulations)
    max_manip_idx = max(range(len(manipulations)), key=lambda i: manipulations[i].score)
    examples = []
    if manipulations[max_manip_idx].score > 0.2:
        examples.append({
            "message": msgs[max_manip_idx][:200],
            "score": manipulations[max_manip_idx].score,
            "tactics": list(manipulations[max_manip_idx].tactics.keys()),
        })

    return {
        "count": len(msgs),
        "avg_sentiment": round(avg_sent, 4),
        "sentiment_label": _label_for(avg_sent),
        "sentiment_trend": trend,
        "manipulation_score": round(avg_manip, 4),
        "manipulation_tactics": {k: list(set(v))[:5] for k, v in all_tactics.items()},
        "examples": examples,
    }
