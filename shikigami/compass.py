"""
Compass — Goal alignment scoring over time.

Goals here are explicit intentions tracked separately from generic decisions.
A goal has:
- title          (the intention)
- keywords       (tokens we look for in observations to count progress)
- created_at     (when set)
- target_date    (optional deadline)
- status         (active | paused | done | abandoned)
- last_progress  (computed)

The Compass:
- Counts recent observations that touch the goal's keywords
- Compares to the baseline before the goal was set
- Computes alignment score in [0, 1] from frequency + recency
- Detects drift (declining mentions over the goal's lifetime)
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_DEFAULT_DB = Path.home() / "digital-shadow" / "data" / "goals.json"


@dataclass
class Goal:
    goal_id: str
    title: str
    keywords: List[str]
    created_at: str
    status: str = "active"
    target_date: Optional[str] = None
    notes: str = ""

    def matches(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        for kw in self.keywords:
            if kw and re.search(r"\b" + re.escape(kw.lower()) + r"\b", t):
                return True
        return False


@dataclass
class GoalProgress:
    goal: Goal
    mentions_total: int
    mentions_last_7d: int
    mentions_last_30d: int
    mentions_pre_goal: int
    age_days: int
    days_to_target: Optional[int]
    alignment: float        # [0, 1]
    trend: str              # "rising" | "stable" | "declining" | "dormant"
    last_mention: Optional[str]


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
    c = obs.get("content")
    parts = [str(obs.get("type", ""))]
    if isinstance(c, dict):
        for k in ("text", "body", "message", "preview", "command", "title", "summary"):
            v = c.get(k)
            if isinstance(v, str):
                parts.append(v)
    elif isinstance(c, str):
        parts.append(c)
    return " ".join(parts)


class CompassStore:
    """Tiny persistent store for goals. Thread-safe JSON file."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _DEFAULT_DB
        self._lock = threading.Lock()
        self.goals: Dict[str, Goal] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.db_path.exists():
                data = json.loads(self.db_path.read_text())
                if isinstance(data, list):
                    for d in data:
                        if isinstance(d, dict) and d.get("goal_id"):
                            self.goals[d["goal_id"]] = Goal(**{
                                k: d[k] for k in d if k in Goal.__dataclass_fields__
                            })
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self.goals = {}

    def _save(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.db_path.with_suffix(".tmp")
            tmp.write_text(json.dumps([asdict(g) for g in self.goals.values()], indent=2))
            tmp.replace(self.db_path)
        except OSError:
            pass

    def add(self, title: str, keywords: List[str], target_date: Optional[str] = None,
            notes: str = "") -> Goal:
        kws = [k.strip().lower() for k in keywords if k.strip()][:20]
        goal = Goal(
            goal_id=f"goal_{uuid.uuid4().hex[:10]}",
            title=title.strip()[:200],
            keywords=kws,
            created_at=datetime.now(timezone.utc).isoformat(),
            target_date=target_date,
            notes=notes[:1000],
        )
        with self._lock:
            self.goals[goal.goal_id] = goal
            self._save()
        return goal

    def set_status(self, goal_id: str, status: str) -> Optional[Goal]:
        with self._lock:
            g = self.goals.get(goal_id)
            if not g:
                return None
            g.status = status
            self._save()
            return g

    def remove(self, goal_id: str) -> bool:
        with self._lock:
            if goal_id in self.goals:
                del self.goals[goal_id]
                self._save()
                return True
        return False

    def list(self) -> List[Goal]:
        with self._lock:
            return list(self.goals.values())


def _compute_alignment(mentions_30d: int, pre_goal_per_30d: float, days_to_target: Optional[int]) -> float:
    """Alignment in [0, 1]. Higher when mentions are more frequent than baseline
    and a target date provides urgency."""
    # Frequency component — normalize against expected baseline.
    baseline = max(0.5, pre_goal_per_30d)
    freq = min(1.0, mentions_30d / (3 * baseline))
    # Urgency boost when target is near (within 30 days).
    urgency = 1.0
    if days_to_target is not None:
        if days_to_target <= 0:
            urgency = 0.5  # past due
        elif days_to_target <= 7:
            urgency = 1.1
        elif days_to_target <= 30:
            urgency = 1.05
    return max(0.0, min(1.0, freq * urgency))


def _compute_trend(mentions_last_7d: int, mentions_30d: int, age_days: int) -> str:
    if age_days < 1:
        return "stable"
    expected_in_last_7 = mentions_30d * (7 / 30.0)
    if mentions_last_7d == 0 and mentions_30d == 0:
        return "dormant"
    if mentions_last_7d > expected_in_last_7 * 1.5 + 1:
        return "rising"
    if mentions_last_7d < expected_in_last_7 * 0.5:
        return "declining"
    return "stable"


def score_goal(goal: Goal, observations: List[Dict],
               now: Optional[datetime] = None) -> GoalProgress:
    now = now or datetime.now(timezone.utc)
    try:
        created = datetime.fromisoformat(goal.created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        created = now

    cutoff_7 = now - timedelta(days=7)
    cutoff_30 = now - timedelta(days=30)

    matches: List[datetime] = []
    pre_goal_matches: List[datetime] = []
    for o in observations:
        t = _parse_ts(o)
        if t is None:
            continue
        if not goal.matches(_text_of(o)):
            continue
        if t >= created:
            matches.append(t)
        else:
            pre_goal_matches.append(t)
    matches.sort()
    last = matches[-1] if matches else None

    age_days = max(1, (now - created).days)
    pre_goal_per_30d = len(pre_goal_matches) / max(1, len(pre_goal_matches) and
                                                    (created - min(pre_goal_matches)).days / 30.0 or 1.0)

    days_to_target: Optional[int] = None
    if goal.target_date:
        try:
            tgt = datetime.fromisoformat(goal.target_date.replace("Z", "+00:00"))
            if tgt.tzinfo is None:
                tgt = tgt.replace(tzinfo=timezone.utc)
            days_to_target = (tgt.date() - now.date()).days
        except (ValueError, TypeError):
            pass

    mentions_7d = sum(1 for t in matches if t >= cutoff_7)
    mentions_30d = sum(1 for t in matches if t >= cutoff_30)
    alignment = _compute_alignment(mentions_30d, pre_goal_per_30d, days_to_target)
    trend = _compute_trend(mentions_7d, mentions_30d, age_days)

    return GoalProgress(
        goal=goal,
        mentions_total=len(matches),
        mentions_last_7d=mentions_7d,
        mentions_last_30d=mentions_30d,
        mentions_pre_goal=len(pre_goal_matches),
        age_days=age_days,
        days_to_target=days_to_target,
        alignment=round(alignment, 3),
        trend=trend,
        last_mention=last.isoformat() if last else None,
    )


def trend_marker(trend: str) -> str:
    return {"rising": "↑", "declining": "↓", "stable": "→", "dormant": "⌛"}.get(trend, "•")
