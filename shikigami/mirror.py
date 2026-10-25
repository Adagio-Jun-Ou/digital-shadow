"""
Mirror — Self-reflection ability.

Scans the user's own first-person text (clipboard, notes, journal observations)
for:
- self-referential statements ("I feel…", "I'm …", "I keep …")
- recurring inner-state themes (anxiety, motivation, frustration, …)
- self-talk valence (constructive vs negative self-talk)
- topic drift over time

This is text-only inference — no biometric inputs — so it surfaces patterns
the user already wrote down. Useful for "what have I been telling myself
this week?"
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple


# Inner-state buckets. Each is a list of stem patterns.
_THEMES: Dict[str, List[str]] = {
    "anxiety":     [r"\banx", r"\bworri", r"\bnervous", r"\bafraid", r"\bscared", r"\bdread"],
    "motivation":  [r"\bmotivat", r"\bexcited", r"\bdriven", r"\binspir", r"\beager"],
    "frustration": [r"\bfrustrat", r"\bangry", r"\bannoyed", r"\bpissed", r"\bfed up"],
    "burnout":     [r"\bburnt? out", r"\bexhaust", r"\btired", r"\bdrained", r"\bwiped"],
    "growth":      [r"\blearn", r"\bgrow", r"\bimprov", r"\bbetter at", r"\bprogress"],
    "doubt":       [r"\bdoubt", r"\bnot sure", r"\bworth(less)?", r"\bimposter", r"\bfraud"],
    "gratitude":   [r"\bgrate", r"\bthank", r"\bblessed", r"\bappreciat"],
    "stuck":       [r"\bstuck", r"\bblocked", r"\bspinning", r"\bgoing nowhere", r"\bplateau"],
    "purpose":     [r"\bpurpose", r"\bmeaning", r"\bwhy am i", r"\bwhat'?s the point"],
}

# "I/me" first-person indicators — required for a sentence to count as self-talk.
_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|i'll|i'd|me|my|myself|mine)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ThemeHit:
    theme: str
    excerpt: str
    timestamp: str
    sentiment: float  # [-1, 1] of the surrounding sentence


@dataclass
class MirrorReport:
    total_sentences: int
    self_talk_sentences: int
    self_talk_ratio: float
    avg_self_sentiment: float       # average sentiment of self-talk
    theme_counts: Dict[str, int]
    rising_themes: List[str]        # themes more common this week vs previous
    declining_themes: List[str]
    sample_hits: List[ThemeHit]
    sentences_by_week: Dict[str, int]


def _parse_ts(o: Dict) -> Optional[datetime]:
    ts = o.get("timestamp")
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _text_of(obs: Dict) -> str:
    """Pull longer-form first-person text. Limited to obs types that plausibly
    contain user's own words."""
    if str(obs.get("type", "")) not in {"clipboard", "note", "journal", "message", "reflection"}:
        return ""
    c = obs.get("content")
    if isinstance(c, dict):
        for k in ("text", "body", "preview", "content", "message"):
            v = c.get(k)
            if isinstance(v, str) and len(v) > 5:
                return v
    elif isinstance(c, str):
        return c
    return ""


def _matches_theme(text: str) -> List[str]:
    t = text.lower()
    out = []
    for theme, patterns in _THEMES.items():
        for p in patterns:
            if re.search(p, t):
                out.append(theme)
                break
    return out


def _sentence_sentiment(text: str) -> float:
    """Light wrapper over shikigami.sentiment when available; falls back to 0."""
    try:
        from .sentiment import analyze_sentiment
        return analyze_sentiment(text).score
    except Exception:
        return 0.0


def reflect(
    observations: List[Dict],
    now: Optional[datetime] = None,
    sample_per_theme: int = 2,
) -> MirrorReport:
    now = now or datetime.now(timezone.utc)
    cutoff_recent = now - timedelta(days=7)
    cutoff_prev = now - timedelta(days=14)

    total_sent = 0
    self_sent_count = 0
    self_sentiment_sum = 0.0
    theme_counts: Counter = Counter()
    theme_counts_recent: Counter = Counter()
    theme_counts_prev: Counter = Counter()
    samples: Dict[str, List[ThemeHit]] = defaultdict(list)
    sentences_by_week: Dict[str, int] = defaultdict(int)

    for o in observations:
        t = _parse_ts(o)
        text = _text_of(o)
        if not text or t is None:
            continue
        ts_iso = t.isoformat()
        week_key = f"{t.isocalendar()[0]}-W{t.isocalendar()[1]:02d}"
        for sent in _SENT_SPLIT.split(text):
            sent = sent.strip()
            if len(sent) < 6:
                continue
            total_sent += 1
            sentences_by_week[week_key] += 1
            if not _FIRST_PERSON.search(sent):
                continue
            self_sent_count += 1
            score = _sentence_sentiment(sent)
            self_sentiment_sum += score
            for theme in _matches_theme(sent):
                theme_counts[theme] += 1
                if t >= cutoff_recent:
                    theme_counts_recent[theme] += 1
                elif t >= cutoff_prev:
                    theme_counts_prev[theme] += 1
                if len(samples[theme]) < sample_per_theme:
                    samples[theme].append(ThemeHit(
                        theme=theme,
                        excerpt=sent[:200],
                        timestamp=ts_iso,
                        sentiment=round(score, 3),
                    ))

    self_ratio = self_sent_count / total_sent if total_sent else 0.0
    avg_sent = self_sentiment_sum / self_sent_count if self_sent_count else 0.0

    rising = []
    declining = []
    for theme in set(list(theme_counts_recent) + list(theme_counts_prev)):
        r = theme_counts_recent.get(theme, 0)
        p = theme_counts_prev.get(theme, 0)
        if r > p * 1.5 and r >= 2:
            rising.append(theme)
        elif p > r * 1.5 and p >= 2:
            declining.append(theme)

    sample_list: List[ThemeHit] = []
    for theme in [t for t, _ in theme_counts.most_common(6)]:
        sample_list.extend(samples.get(theme, []))

    return MirrorReport(
        total_sentences=total_sent,
        self_talk_sentences=self_sent_count,
        self_talk_ratio=round(self_ratio, 3),
        avg_self_sentiment=round(avg_sent, 3),
        theme_counts=dict(theme_counts),
        rising_themes=rising,
        declining_themes=declining,
        sample_hits=sample_list,
        sentences_by_week=dict(sentences_by_week),
    )
