#!/usr/bin/env python3
"""
VisualObservationBridge - Connects VisionPerception outputs to the cognitive pipeline.

Routes visual observations into:
- EmotionalIntelligence (scene sentiment)
- ThoughtSense (intent from visual context)
- AttentionEngine (visual events → focus shifts)
- DarkMemory (store visual observations)
- NotificationBus (person detection alerts)
"""

import copy
import logging
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.visual_bridge")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


@dataclass
class VisualCognitiveEvent:
    """A visual observation processed through cognitive modules."""
    event_id: str = ""
    source_description: str = ""
    scene_category: str = ""
    detected_objects: List[str] = field(default_factory=list)
    faces_detected: int = 0
    text_present: bool = False
    emotional_analysis: Optional[Dict[str, Any]] = None
    thought_analysis: Optional[Dict[str, Any]] = None
    attention_triggered: bool = False
    notification_sent: bool = False
    memory_stored: bool = False
    importance: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.event_id:
            import uuid
            self.event_id = uuid.uuid4().hex[:12]
        self.source_description = _sanitize_content(self.source_description, 5000)
        self.scene_category = _sanitize(self.scene_category, 50)
        self.detected_objects = [_sanitize(o, 100) for o in self.detected_objects[:50]]
        self.importance = max(0.0, min(1.0, self.importance))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_description": self.source_description[:500] if self.source_description else "",
            "scene_category": self.scene_category,
            "detected_objects": list(self.detected_objects[:10]),
            "faces_detected": self.faces_detected,
            "text_present": self.text_present,
            "emotional_analysis": copy.deepcopy(self.emotional_analysis) if self.emotional_analysis else None,
            "thought_analysis": copy.deepcopy(self.thought_analysis) if self.thought_analysis else None,
            "attention_triggered": self.attention_triggered,
            "notification_sent": self.notification_sent,
            "memory_stored": self.memory_stored,
            "importance": round(self.importance, 3),
            "timestamp": self.timestamp,
        }


class VisualObservationBridge:
    """Bridges VisionPerception outputs into the cognitive pipeline.

    Optional hooks for custom handling:
      - on_alert(event): called when faces or social scene triggers attention
      - on_error(module_name, error): called when a cognitive module fails
    """
    """Bridges VisionPerception outputs into the cognitive pipeline."""

    def __init__(self, shadow_os=None):
        self._shadow_os = shadow_os
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._events: deque = deque(maxlen=2000)
        self._recent_descriptions: deque = deque(maxlen=100)

        self.on_alert = None
        self.on_error = None

        self._stats = {
            "events_processed": 0,
            "emotions_detected": 0,
            "thoughts_analyzed": 0,
            "attention_triggers": 0,
            "notifications_sent": 0,
            "memories_stored": 0,
        }

    def process_visual_observation(self, description: str, scene_category: str = "",
                                   objects: Optional[List[str]] = None,
                                   faces: int = 0, has_text: bool = False,
                                   importance: float = 0.5) -> VisualCognitiveEvent:
        event = VisualCognitiveEvent(
            source_description=description,
            scene_category=scene_category,
            detected_objects=objects or [],
            faces_detected=faces,
            text_present=has_text,
            importance=importance,
        )

        if self._shadow_os is None:
            return event

        if hasattr(self._shadow_os, 'emotional_intel') and self._shadow_os.emotional_intel:
            try:
                analysis = self._shadow_os.emotional_intel.detect_emotion(description)
                if analysis and hasattr(analysis, 'to_dict'):
                    event.emotional_analysis = analysis.to_dict()
                    with self._stats_lock:
                        self._stats["emotions_detected"] += 1
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                if self.on_error:
                    try: self.on_error("emotional_intelligence", e)
                    except (RuntimeError, ValueError, TypeError, AttributeError): pass

        if hasattr(self._shadow_os, 'thought_sense') and self._shadow_os.thought_sense:
            try:
                analysis = self._shadow_os.thought_sense.analyze(description)
                if analysis and hasattr(analysis, 'to_dict'):
                    event.thought_analysis = analysis.to_dict()
                    with self._stats_lock:
                        self._stats["thoughts_analyzed"] += 1
            except (ValueError, TypeError, AttributeError, RuntimeError):
                pass

        if hasattr(self._shadow_os, 'attention') and self._shadow_os.attention:
            try:
                if faces > 0 or scene_category in ("social", "transport", "medical"):
                    self._shadow_os.attention.add_attention_item(
                        topic=f"Visual: {description[:80]}",
                        level="high" if faces > 2 else "medium",
                        domain="visual",
                        urgency=0.7 if faces > 2 else 0.5,
                        importance=importance,
                    )
                    event.attention_triggered = True
                    if self.on_alert:
                        try: self.on_alert(event)
                        except (RuntimeError, ValueError, TypeError, AttributeError): pass
                    with self._stats_lock:
                        self._stats["attention_triggers"] += 1
            except (ValueError, TypeError, AttributeError, RuntimeError):
                pass

        if hasattr(self._shadow_os, 'memory') and self._shadow_os.memory:
            try:
                import asyncio
                from .dark_memory import AtomType, MemoryTier
                def _store_sync():
                    try:
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(
                            self._shadow_os.memory.store(
                                content=description,
                                atom_type=AtomType.OBSERVATION,
                                tier=MemoryTier.WORKING,
                                domain="visual",
                                importance=importance,
                            )
                        )
                        loop.close()
                        return True
                    except (RuntimeError, ValueError, TypeError, AttributeError):
                        return False
                result = _store_sync()
                if result:
                    event.memory_stored = True
                    with self._stats_lock:
                        self._stats["memories_stored"] += 1
            except (ValueError, TypeError, AttributeError, RuntimeError):
                pass

        if hasattr(self._shadow_os, 'notifications') and self._shadow_os.notifications:
            try:
                if faces > 0 or scene_category in ("social", "workspace"):
                    self._shadow_os.notifications.notify_visual_observation(
                        description=description, faces=faces, category="visual"
                    )
                    event.notification_sent = True
                    if self.on_alert:
                        try: self.on_alert(event)
                        except (RuntimeError, ValueError, TypeError, AttributeError): pass
                    with self._stats_lock:
                        self._stats["notifications_sent"] += 1
            except (ValueError, TypeError, AttributeError, RuntimeError):
                pass

        with self._lock:
            self._events.append(event)
            self._recent_descriptions.append(description)

        with self._stats_lock:
            self._stats["events_processed"] += 1

        return event

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(500, limit))
        with self._lock:
            events = list(self._events)[-limit:]
        return [e.to_dict() for e in events]

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        event_id = _sanitize(event_id, 20)
        with self._lock:
            for e in self._events:
                if e.event_id == event_id:
                    return copy.deepcopy(e.to_dict())
        return None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            event_count = len(self._events)
        with self._stats_lock:
            stats = dict(self._stats)
        stats["total_events"] = event_count
        return stats
