#!/usr/bin/env python3
"""
NarrativeMemory - Life Story Construction Engine

Constructs coherent narratives from fragmented memories, building a
life story that connects events into themes, arcs, and chapters.

Architecture:
  - NarrativeChapter: themed collection of events with emotional arc
  - NarrativeArc: multi-chapter storyline with progression tracking
  - NarrativeMemory: main engine that manages chapters, arcs, and themes
  - Theme detection, emotional arc scoring, narrative coherence
  - Thread-safe, bounded history, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("shadowd.cognition.narrative_memory")


class EmotionTone(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    GROWTH = "growth"
    RESILIENCE = "resilience"


class ThemeCategory(str, Enum):
    CAREER = "career"
    RELATIONSHIPS = "relationships"
    HEALTH = "health"
    GROWTH = "growth"
    CREATIVITY = "creativity"
    ADVERSITY = "adversity"
    ACHIEVEMENT = "achievement"
    LEARNING = "learning"
    IDENTITY = "identity"
    COMMUNITY = "community"


_POSITIVE_WORDS = {
    "achieve", "success", "happy", "love", "joy", "celebrate", "proud",
    "accomplish", "thrive", "grow", "learn", "create", "connect", "inspire",
    "hope", "grateful", "excited", "confident", "strong", "heal", "overcome",
    "breakthrough", "discover", "build", "progress", "win", "advance",
    "promoted", "promotion", "great", "wonderful", "amazing", "fantastic",
    "excellent", "good", "best", "better", "positive", "reward",
}
_NEGATIVE_WORDS = {
    "fail", "failed", "failing", "loss", "sad", "hurt", "pain", "struggle",
    "struggled", "crisis", "fear", "anxiety", "stress", "stressful",
    "conflict", "reject", "miss", "blame", "regret", "worry", "angry",
    "frustrate", "frustrated", "doubt", "lonely", "grief", "sick",
    "damage", "setback", "decline", "hardship", "trouble", "difficult",
    "terrible", "awful", "worst", "bad", "negative",
}
_GROWTH_WORDS = {
    "learn", "grow", "evolve", "transform", "overcome", "adapt", "improve",
    "develop", "understand", "realize", "discover", "change", "progress",
    "mature", "reflect", "insight", "resilience", "persevere", "recover",
    "skill", "developed", "grew",
}
_THEME_KEYWORDS = {
    ThemeCategory.CAREER: {"work", "job", "career", "promotion", "promoted", "project", "team", "role", "profession", "business", "salary", "manager", "company", "office"},
    ThemeCategory.RELATIONSHIPS: {"friend", "family", "partner", "love", "trust", "connect", "relationship", "marriage", "bond", "support", "together", "spouse"},
    ThemeCategory.HEALTH: {"health", "exercise", "diet", "sleep", "doctor", "wellness", "fitness", "mental", "therapy", "recovery", "run", "running", "workout"},
    ThemeCategory.GROWTH: {"learn", "learned", "grow", "grew", "evolve", "transform", "overcome", "adapt", "improve", "improved", "develop", "developed", "understand", "realize", "discover", "discovered", "change", "progress", "mature", "reflect", "insight", "resilience", "persevere", "recover", "skill", "skills"},
    ThemeCategory.CREATIVITY: {"create", "created", "art", "music", "write", "writing", "design", "build", "built", "imagine", "imagination", "innovation", "innovative", "craft", "express", "expression"},
    ThemeCategory.ADVERSITY: {"struggle", "struggled", "crisis", "loss", "hardship", "challenge", "challenged", "obstacle", "setback", "trauma", "grief", "fail", "failed", "failing", "difficult", "hard"},
    ThemeCategory.ACHIEVEMENT: {"achieve", "achieved", "success", "successful", "accomplish", "accomplished", "win", "won", "goal", "milestone", "complete", "completed", "earn", "earned", "attain", "conquer"},
    ThemeCategory.LEARNING: {"study", "studying", "research", "read", "reading", "class", "course", "teach", "taught", "understand", "understanding", "insight", "knowledge", "curiosity", "explore", "exam", "school", "education"},
    ThemeCategory.IDENTITY: {"identity", "values", "beliefs", "purpose", "meaning", "meaningful", "authentic", "self", "character", "principles", "voice", "who"},
    ThemeCategory.COMMUNITY: {"community", "volunteer", "help", "helped", "serve", "contribute", "contribution", "neighbor", "group", "belong", "belonging", "collaborate", "share", "sharing"},
}

_MAX_EVENT_LENGTH = 5000
_MAX_CHAPTERS = 500
_MAX_ARCS = 100
_MAX_EVENTS_PER_CHAPTER = 200


@dataclass
class NarrativeEvent:
    event_id: str
    content: str
    timestamp: str
    themes: List[str] = field(default_factory=list)
    emotion: str = "neutral"
    importance: float = 0.5
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "content": self.content[:200],
            "timestamp": self.timestamp,
            "themes": list(self.themes[:5]),
            "emotion": self.emotion,
            "importance": round(self.importance, 3),
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }


@dataclass
class NarrativeChapter:
    chapter_id: str
    title: str
    theme: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    emotion_arc: str = "neutral"
    events: List[NarrativeEvent] = field(default_factory=list)
    summary: str = ""
    coherence_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title[:100],
            "theme": self.theme,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "emotion_arc": self.emotion_arc,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
            "summary": self.summary[:300],
            "coherence_score": round(self.coherence_score, 3),
        }


@dataclass
class NarrativeArc:
    arc_id: str
    title: str
    themes: List[str] = field(default_factory=list)
    chapters: List[str] = field(default_factory=list)
    progression: float = 0.0
    resolution: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "arc_id": self.arc_id,
            "title": self.title[:100],
            "themes": list(self.themes[:5]),
            "chapters": list(self.chapters),
            "chapter_count": len(self.chapters),
            "progression": round(self.progression, 3),
            "resolution": self.resolution,
        }


@dataclass
class NarrativeSummary:
    summary_id: str
    total_chapters: int
    total_events: int
    dominant_themes: List[str]
    emotional_trajectory: str
    arcs: List[str]
    key_events: List[str]
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "summary_id": self.summary_id,
            "total_chapters": self.total_chapters,
            "total_events": self.total_events,
            "dominant_themes": list(self.dominant_themes[:5]),
            "emotional_trajectory": self.emotional_trajectory,
            "arcs": list(self.arcs),
            "key_events": list(self.key_events),
            "arc_count": len(self.arcs),
            "key_event_count": len(self.key_events),
            "timestamp": self.timestamp,
        }


class NarrativeMemory:
    _MAX_CHAPTERS = _MAX_CHAPTERS
    _MAX_ARCS = _MAX_ARCS
    _MAX_EVENTS_PER_CHAPTER = _MAX_EVENTS_PER_CHAPTER

    def __init__(self):
        self._lock = threading.Lock()
        self._events: deque = deque(maxlen=5000)
        self._chapters: Dict[str, NarrativeChapter] = {}
        self._arcs: Dict[str, NarrativeArc] = {}
        self._theme_counts: Dict[str, int] = {}

    def _sanitize(self, text: str) -> str:
        text = re.sub(r"[<>\x00-\x1f\x7f]", "", text).strip()
        return text[:_MAX_EVENT_LENGTH]

    @staticmethod
    def _sanitize_content(text: str) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
        return text[:_MAX_EVENT_LENGTH]

    def _detect_emotion(self, text: str) -> str:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        pos = len(words & _POSITIVE_WORDS)
        neg = len(words & _NEGATIVE_WORDS)
        growth = len(words & _GROWTH_WORDS)

        if growth > 0 and neg > 0:
            return EmotionTone.RESILIENCE.value
        if growth > 0 and neg == 0:
            return EmotionTone.GROWTH.value
        if pos > 0 and neg > 0:
            return EmotionTone.MIXED.value
        if pos > neg:
            return EmotionTone.POSITIVE.value
        if neg > pos:
            return EmotionTone.NEGATIVE.value
        return EmotionTone.NEUTRAL.value

    def _detect_themes(self, text: str) -> List[str]:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        theme_scores = {}
        for theme, keywords in _THEME_KEYWORDS.items():
            overlap = len(words & keywords)
            if overlap > 0:
                theme_scores[theme.value] = overlap
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, s in sorted_themes[:3]]

    def add_event(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> NarrativeEvent:
        content = self._sanitize_content(content)
        importance = max(0.0, min(1.0, importance))
        if metadata and len(metadata) > 20:
            metadata = {k: v for k, v in list(metadata.items())[:20]}
        if metadata:
            metadata = {str(k)[:50]: (str(v)[:200] if isinstance(v, str) else v) for k, v in metadata.items()}
        themes = self._detect_themes(content)
        emotion = self._detect_emotion(content)

        event = NarrativeEvent(
            event_id=uuid.uuid4().hex[:12],
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            themes=themes,
            emotion=emotion,
            importance=importance,
            metadata=metadata or {},
        )

        with self._lock:
            if len(self._events) >= 5000:
                logger.warning("Events deque at capacity (5000), oldest event evicted")
            self._events.append(event)
            for theme in themes:
                self._theme_counts[theme] = self._theme_counts.get(theme, 0) + 1

        return copy.deepcopy(event)

    def create_chapter(
        self,
        title: str,
        theme: str,
        summary: str = "",
    ) -> NarrativeChapter:
        title = self._sanitize(title)[:200]
        theme = self._sanitize(theme)[:200]
        summary = self._sanitize_content(summary)[:1000]

        chapter = NarrativeChapter(
            chapter_id=uuid.uuid4().hex[:12],
            title=title,
            theme=theme,
            summary=summary,
            start_date=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            if len(self._chapters) >= self._MAX_CHAPTERS:
                oldest_id = next(iter(self._chapters))
                logger.warning("Evicting oldest chapter %s", oldest_id)
                del self._chapters[oldest_id]
            self._chapters[chapter.chapter_id] = chapter

        return copy.deepcopy(chapter)

    def add_event_to_chapter(self, chapter_id: str, event: NarrativeEvent) -> bool:
        with self._lock:
            chapter = self._chapters.get(chapter_id)
            if chapter is None:
                return False
            if len(chapter.events) >= self._MAX_EVENTS_PER_CHAPTER:
                logger.warning("Chapter %s at max events (%d), evicting oldest", chapter_id, self._MAX_EVENTS_PER_CHAPTER)
                chapter.events = chapter.events[1:]
            chapter.events.append(event)
            chapter.end_date = event.timestamp
            self._update_chapter_arc(chapter)
            return True

    def _update_chapter_arc(self, chapter: NarrativeChapter) -> None:
        if not chapter.events:
            chapter.emotion_arc = "neutral"
            chapter.coherence_score = 0.0
            return

        emotions = [e.emotion for e in chapter.events]
        positive = sum(1 for e in emotions if e in ("positive", "growth", "resilience"))
        negative = sum(1 for e in emotions if e == "negative")
        total = len(emotions)

        if positive > negative * 2:
            chapter.emotion_arc = "ascending"
        elif negative > positive * 2:
            chapter.emotion_arc = "descending"
        elif positive > 0 and negative > 0:
            chapter.emotion_arc = "mixed"
        else:
            chapter.emotion_arc = "steady"

        if total > 2:
            theme_overlap = 0
            for i in range(1, len(chapter.events)):
                shared = set(chapter.events[i].themes) & set(chapter.events[i - 1].themes)
                theme_overlap += len(shared)
            chapter.coherence_score = min(1.0, (theme_overlap / max(total - 1, 1)) * 0.5 + 0.3)
        else:
            chapter.coherence_score = 0.5

    def create_arc(
        self,
        title: str,
        themes: Optional[List[str]] = None,
    ) -> NarrativeArc:
        title = self._sanitize(title)[:200]
        themes = [self._sanitize(t)[:200] for t in (themes or [])[:5]]

        arc = NarrativeArc(
            arc_id=uuid.uuid4().hex[:12],
            title=title,
            themes=themes[:5],
        )

        with self._lock:
            if len(self._arcs) >= self._MAX_ARCS:
                oldest_id = next(iter(self._arcs))
                logger.warning("Evicting oldest arc %s", oldest_id)
                del self._arcs[oldest_id]
            self._arcs[arc.arc_id] = arc

        return copy.deepcopy(arc)

    def add_chapter_to_arc(self, arc_id: str, chapter_id: str) -> bool:
        arc_id = self._sanitize(arc_id)[:50]
        chapter_id = self._sanitize(chapter_id)[:50]
        with self._lock:
            arc = self._arcs.get(arc_id)
            chapter = self._chapters.get(chapter_id)
            if arc is None or chapter is None:
                return False
            if chapter_id not in arc.chapters:
                arc.chapters.append(chapter_id)
            arc.progression = min(1.0, len(arc.chapters) / 5.0)
            if chapter.emotion_arc in ("ascending", "resilience"):
                arc.progression = min(1.0, arc.progression + 0.1)
            return True

    def resolve_arc(self, arc_id: str, resolution: str) -> bool:
        resolution = self._sanitize_content(resolution)[:500]
        with self._lock:
            arc = self._arcs.get(arc_id)
            if arc is None:
                return False
            arc.resolution = resolution
            arc.progression = 1.0
            return True

    def get_chapter(self, chapter_id: str) -> Optional[NarrativeChapter]:
        with self._lock:
            chapter = self._chapters.get(chapter_id)
            return copy.deepcopy(chapter) if chapter else None

    def get_arc(self, arc_id: str) -> Optional[NarrativeArc]:
        with self._lock:
            arc = self._arcs.get(arc_id)
            return copy.deepcopy(arc) if arc else None

    def list_chapters(self, theme: Optional[str] = None) -> List[NarrativeChapter]:
        with self._lock:
            chapters = [copy.deepcopy(c) for c in self._chapters.values()]
        if theme:
            chapters = [c for c in chapters if c.theme == theme]
        return sorted(chapters, key=lambda c: c.start_date or "")

    def list_arcs(self) -> List[NarrativeArc]:
        with self._lock:
            return [copy.deepcopy(a) for a in self._arcs.values()]

    def get_dominant_themes(self, limit: int = 5) -> List[Tuple[str, int]]:
        with self._lock:
            sorted_themes = sorted(self._theme_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_themes[:limit]

    def generate_summary(self) -> NarrativeSummary:
        with self._lock:
            total_events = len(self._events)
            total_chapters = len(self._chapters)

            dominant = [t for t, _ in sorted(self._theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]]

            if total_events > 0:
                recent = list(self._events)[-min(20, total_events):]
                emotions = [e.emotion for e in recent]
                pos = sum(1 for e in emotions if e in ("positive", "growth", "resilience"))
                neg = sum(1 for e in emotions if e == "negative")
                if pos > neg * 2:
                    trajectory = "ascending"
                elif neg > pos * 2:
                    trajectory = "descending"
                else:
                    trajectory = "steady"
            else:
                trajectory = "neutral"

            key_events = [
                e.event_id for e in sorted(
                    self._events, key=lambda e: e.importance, reverse=True
                )[:10]
            ] if total_events > 0 else []

            arc_ids = list(self._arcs.keys())

        return NarrativeSummary(
            summary_id=uuid.uuid4().hex[:12],
            total_chapters=total_chapters,
            total_events=total_events,
            dominant_themes=dominant,
            emotional_trajectory=trajectory,
            arcs=arc_ids,
            key_events=key_events,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_chapters_for_arc(self, arc_id: str) -> List[NarrativeChapter]:
        with self._lock:
            arc = self._arcs.get(arc_id)
            if arc is None:
                return []
            return [copy.deepcopy(self._chapters[cid]) for cid in arc.chapters if cid in self._chapters]

    def search_events(self, query: str, limit: int = 20) -> List[NarrativeEvent]:
        query = self._sanitize_content(query).lower()
        query_words = set(re.findall(r"\b\w+\b", query))
        if not query_words:
            return []

        with self._lock:
            events = list(self._events)

        scored = []
        for event in events:
            content_words = set(re.findall(r"\b\w+\b", event.content.lower()))
            theme_words = set()
            for t in event.themes:
                theme_words.update(re.findall(r"\b\w+\b", t.lower()))
            overlap = len(query_words & (content_words | theme_words))
            if overlap == 0 and event.importance < 0.5:
                continue
            if overlap > 0:
                scored.append((overlap + event.importance, event))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [copy.deepcopy(e) for _, e in scored[:limit]]

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_events": len(self._events),
                "total_chapters": len(self._chapters),
                "total_arcs": len(self._arcs),
                "theme_count": len(self._theme_counts),
                "top_themes": dict(list(self._theme_counts.items())[:10]),
            }