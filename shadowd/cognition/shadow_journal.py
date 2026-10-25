#!/usr/bin/env python3
"""
ShadowJournal - Daily Activity Log for ShadowOS.

A chronological record of everything the ShadowAI observed, thought,
decided, and acted upon each day. Like a ship's log, but for an AI
shadow that follows you through your digital and physical life.

Features:
- Daily entries with timeline of events
- Automatic summarization at end of day
- Search across days by keyword, module, or activity type
- Weekly/monthly digests with patterns and trends
- Export as structured JSON or human-readable text
"""

import copy
import json
import logging
import os
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.shadow_journal")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


class EntryType(str, Enum):
    OBSERVATION = "observation"
    DECISION = "decision"
    ACTION = "action"
    ALERT = "alert"
    INSIGHT = "insight"
    SUMMARY = "summary"
    MILESTONE = "milestone"


class ActivityCategory(str, Enum):
    WORK = "work"
    HEALTH = "health"
    SOCIAL = "social"
    LEARNING = "learning"
    CREATIVITY = "creativity"
    SECURITY = "security"
    SYSTEM = "system"
    VISUAL = "visual"
    GENERAL = "general"


@dataclass
class JournalEntry:
    entry_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry_type: EntryType = EntryType.OBSERVATION
    category: ActivityCategory = ActivityCategory.GENERAL
    source_module: str = ""
    content: str = ""
    summary: str = ""
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = uuid.uuid4().hex[:12]
        self.source_module = _sanitize(self.source_module, 100)
        self.content = _sanitize_content(self.content, 10000)
        self.summary = _sanitize_content(self.summary, 2000)
        self.importance = max(0.0, min(1.0, self.importance))
        self.tags = [_sanitize(t, 50) for t in self.tags[:20]]
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "entry_type": self.entry_type.value,
            "category": self.category.value,
            "source_module": self.source_module,
            "content": self.content,
            "summary": self.summary,
            "importance": round(self.importance, 3),
            "tags": list(self.tags),
            "metadata": copy.deepcopy(self.metadata),
        }

    @property
    def date_key(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d")


@dataclass
class DailySummary:
    date: str
    total_entries: int
    categories: Dict[str, int]
    top_entries: List[Dict[str, Any]]
    mood_summary: str
    productivity_score: float
    key_events: List[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        self.date = _sanitize(self.date, 20)
        self.mood_summary = _sanitize_content(self.mood_summary, 2000)
        self.key_events = [_sanitize_content(e, 500) for e in self.key_events[:20]]
        self.productivity_score = max(0.0, min(1.0, self.productivity_score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "total_entries": self.total_entries,
            "categories": dict(self.categories),
            "top_entries": [copy.deepcopy(e) for e in self.top_entries[:10]],
            "mood_summary": self.mood_summary,
            "productivity_score": round(self.productivity_score, 3),
            "key_events": list(self.key_events),
            "generated_at": self.generated_at.isoformat(),
        }


class ShadowJournal:
    def __init__(self, shadow_os=None, data_dir: Optional[str] = None):
        self._shadow_os = shadow_os
        self.data_dir = data_dir or os.path.join(
            os.path.expanduser("~"), "digital-shadow", "data", "journal"
        )
        os.makedirs(self.data_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._entries: Dict[str, JournalEntry] = {}
        self._entry_order: deque = deque(maxlen=100000)
        self._date_index: Dict[str, List[str]] = {}
        self._tag_index: Dict[str, set] = {}
        self._summaries: Dict[str, DailySummary] = {}

        self._MAX_ENTRIES = 100000
        self._MAX_ENTRIES_PER_DAY = 10000

        self._stats: Dict[str, int] = {
            "entries_recorded": 0,
            "summaries_generated": 0,
            "daily_average": 0,
        }

        self._load_state()

    def _load_state(self):
        state_file = os.path.join(self.data_dir, "journal_state.json")
        try:
            with open(state_file) as f:
                data = json.load(f)
                for edata in data.get("entries", [])[-1000:]:
                    try:
                        ts = datetime.fromisoformat(edata.get("timestamp", ""))
                    except (ValueError, TypeError):
                        ts = datetime.now(timezone.utc)
                    entry = JournalEntry(
                        entry_id=edata.get("entry_id", ""),
                        timestamp=ts,
                        entry_type=EntryType(edata.get("entry_type", "observation")),
                        category=ActivityCategory(edata.get("category", "general")),
                        source_module=edata.get("source_module", ""),
                        content=edata.get("content", ""),
                        summary=edata.get("summary", ""),
                        importance=edata.get("importance", 0.5),
                        tags=edata.get("tags", []),
                        metadata=edata.get("metadata", {}),
                    )
                    self._entries[entry.entry_id] = entry
                    self._entry_order.append(entry.entry_id)
                    dk = entry.date_key
                    if dk not in self._date_index:
                        self._date_index[dk] = []
                    self._date_index[dk].append(entry.entry_id)
                self._stats["entries_recorded"] = len(self._entries)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    def _save_state(self):
        try:
            with self._lock:
                entries = [e.to_dict() for e in list(self._entries.values())[-5000:]]
            tmp = os.path.join(self.data_dir, "journal_state.json.tmp")
            with open(tmp, 'w') as f:
                json.dump({"entries": entries}, f, default=str)
            os.replace(tmp, os.path.join(self.data_dir, "journal_state.json"))
        except (OSError, TypeError):
            pass

    def record(self, entry_type: str, content: str, category: str = "general",
               source_module: str = "shadow_os", importance: float = 0.5,
               summary: str = "", tags: Optional[List[str]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> str:
        try:
            etype = EntryType(entry_type)
        except ValueError:
            etype = EntryType.OBSERVATION
        try:
            cat = ActivityCategory(category)
        except ValueError:
            cat = ActivityCategory.GENERAL

        entry = JournalEntry(
            entry_type=etype, category=cat, source_module=source_module,
            content=content, summary=summary or content[:200],
            importance=importance, tags=tags or [], metadata=metadata or {},
        )

        dk = entry.date_key
        with self._lock:
            self._entries[entry.entry_id] = entry
            self._entry_order.append(entry.entry_id)
            if dk not in self._date_index:
                self._date_index[dk] = []
            if len(self._date_index[dk]) < self._MAX_ENTRIES_PER_DAY:
                self._date_index[dk].append(entry.entry_id)
            for tag in entry.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(entry.entry_id)
            while len(self._entries) > self._MAX_ENTRIES:
                oldest = self._entry_order.popleft()
                removed = self._entries.pop(oldest, None)
                if removed:
                    rdk = removed.date_key
                    if rdk in self._date_index:
                        try:
                            self._date_index[rdk].remove(oldest)
                        except ValueError:
                            pass

        with self._stats_lock:
            self._stats["entries_recorded"] += 1

        if self._stats["entries_recorded"] % 10 == 0:
            self._save_state()
        return entry.entry_id

    def record_observation(self, content: str, source_module: str = "shadow_os",
                           category: str = "general", importance: float = 0.5,
                           tags: Optional[List[str]] = None) -> str:
        return self.record("observation", content=content, source_module=source_module,
                           category=category, importance=importance, tags=tags)

    def record_decision(self, content: str, source_module: str = "decision_journal",
                        importance: float = 0.7, tags: Optional[List[str]] = None) -> str:
        return self.record("decision", content=content, source_module=source_module,
                           category="work", importance=importance, tags=tags)

    def record_alert(self, content: str, source_module: str = "security_guardian",
                     tags: Optional[List[str]] = None) -> str:
        return self.record("alert", content=content, source_module=source_module,
                           category="security", importance=0.9, tags=tags)

    def record_insight(self, content: str, source_module: str = "oracle_bridge",
                       importance: float = 0.8, tags: Optional[List[str]] = None) -> str:
        return self.record("insight", content=content, source_module=source_module,
                           category="system", importance=importance, tags=tags)

    def get_entries(self, date: Optional[str] = None, entry_type: Optional[str] = None,
                    category: Optional[str] = None, tags: Optional[List[str]] = None,
                    limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        limit = max(1, min(500, limit))
        with self._lock:
            if date and date in self._date_index:
                eids = self._date_index[date][-limit - offset:]
                entries = [self._entries[eid] for eid in reversed(eids) if eid in self._entries]
            else:
                ids = list(self._entry_order)
                entries = [self._entries.get(eid) for eid in reversed(ids[-limit - offset:])]
                entries = [e for e in entries if e is not None]

        if entry_type:
            entries = [e for e in entries if e.entry_type.value == entry_type]
        if category:
            entries = [e for e in entries if e.category.value == category]
        if tags:
            entries = [e for e in entries if any(t in e.tags for t in tags)]

        return [e.to_dict() for e in entries[:limit]]

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        query = _sanitize_content(query.lower(), 500)
        results = []
        with self._lock:
            ids = list(self._entry_order)
            for eid in reversed(ids):
                entry = self._entries.get(eid)
                if not entry:
                    continue
                score = 0
                if query in entry.content.lower():
                    score += 2
                if query in entry.summary.lower():
                    score += 1
                for tag in entry.tags:
                    if query in tag.lower():
                        score += 1
                if score > 0:
                    results.append((entry, score))
                if len(results) >= limit * 3:
                    break
        results.sort(key=lambda x: x[1], reverse=True)
        return [e.to_dict() for e, _ in results[:limit]]

    def generate_daily_summary(self, date: Optional[str] = None) -> DailySummary:
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            eids = self._date_index.get(date, [])[:self._MAX_ENTRIES_PER_DAY]
            entries = [self._entries[eid] for eid in eids if eid in self._entries]

        if not entries:
            return DailySummary(date=date, total_entries=0, categories={},
                                top_entries=[], mood_summary="No activity recorded",
                                productivity_score=0.0, key_events=[])

        categories = {}
        for e in entries:
            cat = e.category.value
            categories[cat] = categories.get(cat, 0) + 1

        entries.sort(key=lambda e: e.importance, reverse=True)
        top = entries[:10]
        important = [e for e in entries if e.importance > 0.7]
        alerts = [e for e in entries if e.entry_type == EntryType.ALERT]

        work_entries = sum(1 for e in entries if e.category == ActivityCategory.WORK)
        total = len(entries)
        productivity = min(1.0, (work_entries / max(1, total)) * 0.7 +
                           (len(important) / max(1, total)) * 0.3)

        key_events = [e.summary or e.content[:100] for e in top[:5]]
        if alerts:
            key_events.insert(0, f"SECURITY: {len(alerts)} alert(s) today")

        summary = DailySummary(
            date=date,
            total_entries=len(entries),
            categories=categories,
            top_entries=[e.to_dict() for e in top],
            mood_summary=f"{len(entries)} entries across {len(categories)} categories. "
                         f"{len(alerts)} alerts, {len(important)} notable events.",
            productivity_score=productivity,
            key_events=key_events,
        )

        with self._lock:
            self._summaries[date] = summary
        self._save_state()

        with self._stats_lock:
            self._stats["summaries_generated"] += 1

        return summary

    def get_weekly_digest(self) -> Dict[str, Any]:
        today = datetime.now(timezone.utc)
        days = [(today - __import__('datetime').timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(6, -1, -1)]
        summaries = {}
        total_entries = 0
        for d in days:
            try:
                s = self._summaries.get(d) or self.generate_daily_summary(d)
                summaries[d] = s.to_dict()
                total_entries += s.total_entries
            except (ValueError, TypeError, AttributeError):
                summaries[d] = {"date": d, "total_entries": 0}
        return {
            "start_date": days[0],
            "end_date": days[-1],
            "total_entries": total_entries,
            "daily_breakdown": summaries,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._entries)
            days = len(self._date_index)
        with self._stats_lock:
            stats = dict(self._stats)
        stats.update({
            "total_entries": total,
            "total_days": days,
            "avg_per_day": round(total / max(1, days), 1),
        })
        return stats

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        entry_id = _sanitize(entry_id, 20)
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry:
                return copy.deepcopy(entry.to_dict())
        return None

    def export_journal(self, start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       fmt: str = "json") -> str:
        with self._lock:
            entries = list(self._entries.values())

        if start_date:
            entries = [e for e in entries if e.date_key >= start_date]
        if end_date:
            entries = [e for e in entries if e.date_key <= end_date]
        entries.sort(key=lambda e: e.timestamp)

        if fmt == "json":
            return json.dumps([e.to_dict() for e in entries], indent=2, default=str)
        lines = ["ShadowOS Activity Log", "=" * 40]
        for e in entries:
            ts = e.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[{ts}] [{e.entry_type.value.upper()}] {e.summary or e.content[:200]}")
        return "\n".join(lines)
