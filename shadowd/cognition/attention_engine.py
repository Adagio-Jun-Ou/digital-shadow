#!/usr/bin/env python3
"""
AttentionEngine — Focus, Attention, and Cognitive Resource Management

Manages the shadow's attentional resources:
- Priority-based focus management (what deserves attention now)
- Distraction detection and recovery
- Attention span tracking and fatigue modeling
- Cognitive resource allocation across competing demands
- Deep work mode optimization
"""

import copy
import logging
import re
import threading
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.attention_engine")


class AttentionLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class FocusState(str, Enum):
    DEEP_WORK = "deep_work"
    ACTIVE = "active"
    SCANNING = "scanning"
    DISTRACTED = "distracted"
    FATIGUED = "fatigued"
    IDLE = "idle"


class DistractionType(str, Enum):
    NOTIFICATION = "notification"
    INTERRUPTION = "interruption"
    CONTEXT_SWITCH = "context_switch"
    MIND_WANDERING = "mind_wandering"
    OVERSTIMULATION = "overstimulation"
    FATIGUE = "fatigue"


@dataclass
class AttentionItem:
    item_id: str
    topic: str
    level: AttentionLevel
    domain: str
    urgency_score: float
    importance_score: float
    created_at: str
    last_focused_at: Optional[str]
    focus_count: int
    metadata: Dict

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["level"] = self.level.value
        d["urgency_score"] = round(self.urgency_score, 3)
        d["importance_score"] = round(self.importance_score, 3)
        d["metadata"] = copy.deepcopy(self.metadata)
        return d


@dataclass
class FocusSession:
    session_id: str
    topic: str
    started_at: str
    ended_at: Optional[str]
    duration_minutes: float
    interruptions: int
    depth_score: float

    def to_dict(self) -> Dict:
        return copy.deepcopy(asdict(self))


@dataclass
class DistractionEvent:
    event_id: str
    distraction_type: DistractionType
    description: str
    timestamp: str
    recovery_time_seconds: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["distraction_type"] = self.distraction_type.value
        d["recovery_time_seconds"] = round(self.recovery_time_seconds, 3)
        return d


@dataclass
class AttentionProfile:
    avg_focus_span_minutes: float
    max_focus_span_minutes: float
    total_distractions: int
    total_interruptions: int
    deep_work_sessions: int
    current_state: FocusState
    cognitive_load: float
    fatigue_level: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["current_state"] = self.current_state.value
        d["cognitive_load"] = round(self.cognitive_load, 3)
        d["fatigue_level"] = round(self.fatigue_level, 3)
        d["avg_focus_span_minutes"] = round(self.avg_focus_span_minutes, 2)
        d["max_focus_span_minutes"] = round(self.max_focus_span_minutes, 2)
        return d


class AttentionEngine:
    _MAX_ATTENTION_ITEMS = 500
    _MAX_FOCUS_SESSIONS = 1000
    _MAX_DISTRACTIONS = 500
    _MAX_TOPIC_LENGTH = 500
    _MAX_DOMAIN_LENGTH = 100
    _COGNITIVE_LOAD_DECAY = 0.02
    _FATIGUE_RATE = 0.005
    _FATIGUE_RECOVERY_RATE = 0.02
    _DEEP_WORK_THRESHOLD_MINUTES = 25

    def __init__(self):
        self._lock = threading.Lock()
        self._attention_items: Dict[str, AttentionItem] = {}
        self._focus_sessions: deque = deque(maxlen=self._MAX_FOCUS_SESSIONS)
        self._distractions: deque = deque(maxlen=self._MAX_DISTRACTIONS)
        self._current_focus: Optional[str] = None
        self._current_session: Optional[FocusSession] = None
        self._cognitive_load: float = 0.0
        self._fatigue_level: float = 0.0
        self._stats = {
            "items_tracked": 0,
            "focus_sessions": 0,
            "distractions_detected": 0,
            "deep_work_sessions": 0,
        }
        self._stats_lock = threading.Lock()

    def _sanitize(self, text: str, max_len: int = 500) -> str:
        text = re.sub(r"[<>\x00-\x1f\x7f]", "", text).strip()
        return text[:max_len]

    @staticmethod
    def _sanitize_content(text: str, max_len: int = 500) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
        return text[:max_len]

    def add_attention_item(
        self,
        topic: str,
        level: str = "medium",
        domain: str = "general",
        urgency: float = 0.5,
        importance: float = 0.5,
        metadata: Dict = None,
    ) -> str:
        topic = self._sanitize_content(topic, self._MAX_TOPIC_LENGTH)
        if not topic:
            raise ValueError("topic must not be empty")
        try:
            att_level = AttentionLevel(level)
        except ValueError:
            raise ValueError("invalid attention level")
        domain = self._sanitize(domain, self._MAX_DOMAIN_LENGTH) or "general"
        urgency = max(0.0, min(1.0, urgency))
        importance = max(0.0, min(1.0, importance))
        if metadata:
            metadata = {str(k)[:100]: str(v)[:500] for k, v in list(metadata.items())[:20]}
        now = datetime.now(timezone.utc).isoformat()
        item_id = uuid.uuid4().hex[:12]
        item = AttentionItem(
            item_id=item_id,
            topic=topic,
            level=att_level,
            domain=domain,
            urgency_score=urgency,
            importance_score=importance,
            created_at=now,
            last_focused_at=None,
            focus_count=0,
            metadata=metadata or {},
        )
        with self._lock:
            if len(self._attention_items) >= self._MAX_ATTENTION_ITEMS:
                lowest_id = min(
                    self._attention_items,
                    key=lambda k: self._attention_items[k].urgency_score * 0.5
                    + self._attention_items[k].importance_score * 0.5,
                )
                logger.warning("Evicting attention item %s to make room", lowest_id)
                del self._attention_items[lowest_id]
            self._attention_items[item_id] = item
            with self._stats_lock:
                self._stats["items_tracked"] += 1
        logger.info("Added attention item %s (%s, %s)", item_id, topic[:50], level)
        return item_id

    def start_focus(self, topic: str) -> str:
        topic = self._sanitize_content(topic, self._MAX_TOPIC_LENGTH)
        if not topic:
            raise ValueError("topic must not be empty")
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        with self._lock:
            if self._current_session and not self._current_session.ended_at:
                self._end_focus_unlocked(now, interrupted=True)
            session = FocusSession(
                session_id=session_id,
                topic=topic,
                started_at=now.isoformat(),
                ended_at=None,
                duration_minutes=0.0,
                interruptions=0,
                depth_score=0.0,
            )
            self._current_session = session
            self._current_focus = topic
            self._cognitive_load = min(1.0, self._cognitive_load + 0.3)
        logger.info("Started focus session %s on %s", session_id, topic[:50])
        return session_id

    def end_focus(self, session_id: str = None) -> Optional[FocusSession]:
        now = datetime.now(timezone.utc)
        with self._lock:
            if not self._current_session:
                return None
            if session_id and self._current_session.session_id != session_id:
                return None
            return self._end_focus_unlocked(now, interrupted=False)

    def _end_focus_unlocked(self, now: datetime, interrupted: bool = False) -> Optional[FocusSession]:
        session = self._current_session
        if not session:
            return None
        session.ended_at = now.isoformat()
        try:
            duration = (now - datetime.fromisoformat(session.started_at)).total_seconds() / 60
        except (ValueError, TypeError):
            duration = 0.0
        session.duration_minutes = round(max(0.0, duration), 2)
        depth = min(1.0, duration / 60.0) * (1.0 - session.interruptions * 0.1)
        session.depth_score = round(max(0.0, min(1.0, depth)), 3)
        self._focus_sessions.append(session)
        self._current_focus = None
        self._current_session = None
        if session.duration_minutes >= self._DEEP_WORK_THRESHOLD_MINUTES:
            with self._stats_lock:
                self._stats["deep_work_sessions"] += 1
        with self._stats_lock:
            self._stats["focus_sessions"] += 1
        self._fatigue_level = min(1.0, self._fatigue_level + duration * self._FATIGUE_RATE)
        self._cognitive_load = max(0.0, self._cognitive_load - 0.3)
        logger.info(
            "Ended focus session %s (%.1f min, depth %.2f%s)",
            session.session_id,
            session.duration_minutes,
            session.depth_score,
            ", interrupted" if interrupted else "",
        )
        return copy.deepcopy(session)

    def record_distraction(
        self,
        distraction_type: str,
        description: str = "",
        recovery_time_seconds: float = 0,
    ) -> str:
        try:
            dtype = DistractionType(distraction_type)
        except ValueError:
            raise ValueError("invalid distraction type")
        description = self._sanitize_content(description, 1000)
        recovery_time_seconds = max(0.0, min(3600.0, recovery_time_seconds))
        event_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        event = DistractionEvent(
            event_id=event_id,
            distraction_type=dtype,
            description=description,
            timestamp=now,
            recovery_time_seconds=recovery_time_seconds,
        )
        with self._lock:
            self._distractions.append(event)
            with self._stats_lock:
                self._stats["distractions_detected"] += 1
            if self._current_session:
                self._current_session.interruptions += 1
            self._cognitive_load = min(1.0, self._cognitive_load + 0.1)
            self._fatigue_level = min(1.0, self._fatigue_level + 0.02)
        logger.info("Recorded distraction %s (%s)", event_id, distraction_type)
        return event_id

    def get_priorities(self, limit: int = 10) -> List[AttentionItem]:
        limit = max(1, min(100, limit))
        with self._lock:
            items_snapshot = list(self._attention_items.values())
        level_weight = {
            AttentionLevel.CRITICAL: 5.0,
            AttentionLevel.HIGH: 4.0,
            AttentionLevel.MEDIUM: 3.0,
            AttentionLevel.LOW: 2.0,
            AttentionLevel.BACKGROUND: 1.0,
        }
        scored = []
        for item in items_snapshot:
            score = (
                level_weight.get(item.level, 2.0) * 0.3
                + item.urgency_score * 0.35
                + item.importance_score * 0.35
            )
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [copy.deepcopy(item) for _, item in scored[:limit]]

    def detect_focus_state(self, device_context=None) -> FocusState:
        with self._lock:
            fatigue = self._fatigue_level
            load = self._cognitive_load
            current = self._current_focus
            sessions = list(self._focus_sessions)
            distractions = list(self._distractions)

        if device_context is not None:
            ctx = device_context.get_context()
            if ctx is not None:
                if ctx.health.stress_level.value in ("critical", "high"):
                    fatigue = min(1.0, fatigue + 0.3)
                if ctx.health.heart_rate is not None and ctx.health.heart_rate > 100:
                    fatigue = min(1.0, fatigue + 0.15)
                if ctx.health.sleep_minutes_last_24h > 0 and ctx.health.sleep_minutes_last_24h < 300:
                    fatigue = min(1.0, fatigue + 0.2)
                if ctx.digital.notifications_last_hour > 20:
                    load = min(1.0, load + 0.3)
                elif ctx.digital.notifications_last_hour > 10:
                    load = min(1.0, load + 0.15)
                if ctx.focus.value in ("deep_work", "active", "distracted", "fatigued", "resting"):
                    device_focus_map = {
                        "deep_work": FocusState.DEEP_WORK,
                        "active": FocusState.ACTIVE,
                        "distracted": FocusState.DISTRACTED,
                        "fatigued": FocusState.FATIGUED,
                        "resting": FocusState.IDLE,
                        "scanning": FocusState.SCANNING,
                    }
                    device_inferred = device_focus_map.get(ctx.focus.value)
                    if device_inferred in (FocusState.FATIGUED, FocusState.DISTRACTED):
                        return device_inferred

        if fatigue > 0.8:
            return FocusState.FATIGUED

        recent_distractions = 0
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        for d in distractions:
            if d.timestamp >= cutoff:
                recent_distractions += 1

        if recent_distractions >= 5:
            return FocusState.DISTRACTED

        if recent_distractions >= 3:
            return FocusState.SCANNING

        if current and load > 0.5:
            if sessions:
                recent = [s for s in sessions if s.ended_at and s.depth_score > 0.7]
                if len(recent) >= 2:
                    return FocusState.DEEP_WORK
            return FocusState.ACTIVE

        if load < 0.2 and not current:
            return FocusState.IDLE

        return FocusState.SCANNING

    def suggest_attention_action(self, device_context=None) -> Dict:
        state = self.detect_focus_state(device_context=device_context)
        profile = self.get_attention_profile()
        priorities = self.get_priorities(limit=3)

        suggestions = []
        device_ctx_data = None
        if device_context is not None:
            device_ctx_data = device_context.get_context()

        if state == FocusState.FATIGUED:
            suggestions.append("Take a break — cognitive fatigue is high")
            suggestions.append("Consider a 15-20 minute rest before resuming")
            if device_ctx_data and device_ctx_data.health.sleep_minutes_last_24h > 0 and device_ctx_data.health.sleep_minutes_last_24h < 300:
                suggestions.append(f"Sleep deficit detected ({device_ctx_data.health.sleep_minutes_last_24h // 60}h) — prioritize rest")
        elif state == FocusState.DISTRACTED:
            suggestions.append("Minimize notifications and interruptions")
            suggestions.append("Set a 25-minute deep work timer")
            if device_ctx_data and device_ctx_data.digital.notifications_last_hour > 10:
                suggestions.append(f"High notification count ({device_ctx_data.digital.notifications_last_hour}/hr) — consider Do Not Disturb")
        elif state == FocusState.SCANNING:
            if priorities:
                top = priorities[0]
                suggestions.append(f"Focus on: {top.topic} ({top.level.value} priority)")
        elif state == FocusState.DEEP_WORK:
            suggestions.append("Maintain focus — you're in deep work mode")
            suggestions.append("Protect this time from interruptions")
        elif state == FocusState.IDLE:
            if priorities:
                top = priorities[0]
                suggestions.append(f"Start working on: {top.topic}")
            else:
                suggestions.append("No pending attention items — add tasks to track")

        return {
            "current_state": state.value,
            "suggestions": suggestions[:5],
            "top_priority": priorities[0].to_dict() if priorities else None,
            "cognitive_load": round(profile.cognitive_load, 3),
            "fatigue_level": round(profile.fatigue_level, 3),
        }

    def recover_focus(self, minutes: float = 5.0) -> None:
        minutes = max(0.0, min(120.0, minutes))
        with self._lock:
            self._fatigue_level = max(0.0, self._fatigue_level - minutes * self._FATIGUE_RECOVERY_RATE)
            self._cognitive_load = max(0.0, self._cognitive_load - minutes * self._COGNITIVE_LOAD_DECAY)

    def get_attention_profile(self) -> AttentionProfile:
        with self._lock:
            sessions = list(self._focus_sessions)
            distractions = list(self._distractions)
            cognitive_load = self._cognitive_load
            fatigue_level = self._fatigue_level

        if sessions:
            durations = [s.duration_minutes for s in sessions if s.duration_minutes > 0]
            avg_span = sum(durations) / len(durations) if durations else 0.0
            max_span = max(durations) if durations else 0.0
        else:
            avg_span = 0.0
            max_span = 0.0

        interruption_count = sum(s.interruptions for s in sessions)

        deep_work = self._stats.get("deep_work_sessions", 0)
        state = self.detect_focus_state()

        return AttentionProfile(
            avg_focus_span_minutes=round(avg_span, 2),
            max_focus_span_minutes=round(max_span, 2),
            total_distractions=len(distractions),
            total_interruptions=interruption_count,
            deep_work_sessions=deep_work,
            current_state=state,
            cognitive_load=round(cognitive_load, 3),
            fatigue_level=round(fatigue_level, 3),
        )

    def get_stats(self) -> Dict:
        with self._stats_lock:
            focus_sessions = self._stats["focus_sessions"]
            deep_work_sessions = self._stats["deep_work_sessions"]
            distractions_detected = self._stats["distractions_detected"]
        with self._lock:
            current_focus = self._current_focus
            cognitive_load = self._cognitive_load
            fatigue_level = self._fatigue_level
            items_count = len(self._attention_items)
        return {
            "items_tracked": items_count,
            "focus_sessions": focus_sessions,
            "deep_work_sessions": deep_work_sessions,
            "distractions_detected": distractions_detected,
            "cognitive_load": round(cognitive_load, 3),
            "fatigue_level": round(fatigue_level, 3),
            "current_focus": current_focus,
            "max_items": self._MAX_ATTENTION_ITEMS,
        }